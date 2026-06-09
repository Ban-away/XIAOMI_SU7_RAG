# -*- coding: utf-8 -*-
"""
生成本地可答轨迹数据

功能：
  从 QA 对文件采样问题，走本地检索管线（BM25），
  生成只有 <search_local> 的轨迹（不含 <search_web>/<read_page>），
  教 RL 模型"本地够用时不要联网"。

支持多个 QA 数据源，默认加载全部可用数据。

运行：
  python src/rl/build_local_trajectories.py
  python src/rl/build_local_trajectories.py --sample 200    # 限制采样数量
  python src/rl/build_local_trajectories.py --dry-run       # 只跑 5 条验证
"""

import os
import re
import sys
import json
import hashlib
import argparse
from tqdm import tqdm

# ── 项目路径 ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# ── 静默 tqdm（防止子库弹进度条）──────────────────────────
os.environ["TQDM_DISABLE"] = "1"

from src.retriever.bm25_retriever import BM25
from src.rl.format_converter import (
    SYSTEM_PROMPT,
    to_sft_target,
    to_sft_format,
    to_grpo_format,
)

# ── 路径配置 ────────────────────────────────────────────────
QA_FILES = [
    os.path.join(BASE_DIR, "data/qa_pairs/test_qa_pair_verify.json"),
    os.path.join(BASE_DIR, "data/qa_pairs/train_qa_pair.json"),
]
OUTPUT_PATH    = os.path.join(BASE_DIR, "data/rl_data/local_trajectories.json")
GRPO_OUTPUT    = os.path.join(BASE_DIR, "data/rl_data/local_trajectories_grpo.jsonl")
SFT_OUTPUT     = os.path.join(BASE_DIR, "data/rl_data/local_trajectories_sft.json")

LOCAL_TOPK = 3   # 本地检索条数


# ────────────────────────────────────────────────────────────
# 检索模块（复用 data_builder 的 LocalSearchTool）
# ────────────────────────────────────────────────────────────

class LocalSearchTool:
    """BM25 粗召回（用于生成训练数据，不需要精排）"""

    def __init__(self):
        print("[INFO] 加载 BM25 检索组件...")
        self.bm25 = BM25(docs=None, retrieve=True)
        self.bm25.retrieve_topk("测试", topk=1)
        print("[INFO] BM25 加载完成")

    def search(self, query: str, topk: int = LOCAL_TOPK) -> list:
        return self.bm25.retrieve_topk(query, topk=topk)


# ────────────────────────────────────────────────────────────
# 轨迹生成
# ────────────────────────────────────────────────────────────

def build_local_trajectory(question: str, docs: list, gold_answer: str) -> str:
    """
    生成本地可答轨迹，格式：
      <search_local>关键词</search_local>
      <information>检索结果</information>
      <answer>答案</answer>
    不含 <search_web> 和 <read_page>。
    """
    keywords = re.sub(r"[？?！!。，,的了是在有和与或]", "", question)[:20]
    info_text = _format_docs(docs)

    return (
        f"<search_local>{keywords}</search_local>\n"
        f"<information>{info_text}</information>\n"
        f"<answer>{gold_answer}</answer>"
    )


def _format_docs(docs: list) -> str:
    if not docs:
        return "本地知识库中未检索到相关内容。"
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "")
        suffix = f"（第{page}页）" if page else ""
        parts.append(f"[{i}]{suffix} {doc.page_content[:300]}")
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────
# 数据加载：合并多个 QA 源文件并去重
# ────────────────────────────────────────────────────────────

def load_all_questions() -> list[dict]:
    """加载所有可用的 QA 对文件，按 question 去重"""
    all_data: list[dict] = []
    for path in QA_FILES:
        if not os.path.exists(path):
            print(f"[WARN] 数据文件不存在，跳过: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        print(f"[INFO] 加载 {len(data)} 条: {os.path.basename(path)}")
        all_data.extend(data)

    # 按 question 去重（保留后加载的覆盖先加载的）
    seen: dict[str, dict] = {}
    for item in all_data:
        q = item.get("question", "").strip()
        if q:
            seen[q] = item
    unique = list(seen.values())
    print(f"[INFO] 去重后共 {len(unique)} 条唯一问题")
    return unique


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成本地可答轨迹数据")
    parser.add_argument("--sample", type=int, default=0, help="采样问题数量（0=全部）")
    parser.add_argument("--dry-run", action="store_true", help="只处理前 5 条")
    args = parser.parse_args()

    # ── 加载问题 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📦 生成本地可答轨迹数据")
    print("=" * 60)

    all_questions = load_all_questions()

    # 采样或全量
    if args.dry_run:
        sampled = all_questions[:5]
    elif args.sample > 0:
        import random
        random.seed(42)
        sampled = random.sample(all_questions, min(args.sample, len(all_questions)))
    else:
        sampled = all_questions

    print(f"[INFO] 实际处理 {len(sampled)} 条")

    # ── 初始化检索 ────────────────────────────────────────
    local_tool = LocalSearchTool()

    # ── 生成轨迹 ─────────────────────────────────────────
    results = []
    for item in tqdm(sampled, desc="生成轨迹", ncols=80):
        question = item["question"].strip()
        gold     = item.get("answer", "").strip()
        if not gold:
            continue

        # 本地检索
        docs = local_tool.search(question)

        # 生成轨迹
        trajectory = build_local_trajectory(question, docs, gold)

        unique_id = item.get("unique_id", hashlib.md5(question.encode("utf-8")).hexdigest())
        results.append({
            "id":          unique_id,
            "unique_id":   unique_id,
            "category":    "local_manual",
            "category_zh": "手册本地问答",
            "question":    question,
            "trajectory":  trajectory,
            "local_docs_count": len(docs),
            "web_search_used": False,
            "sft_format":  to_sft_format(question, trajectory, data_source="local_only"),
            "grpo_format": to_grpo_format(
                question, trajectory,
                data_source="local_only",
                reward_type="local_answer_quality",
            ),
        })

    # ── 保存输出 ──────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 完整数据
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # SFT 格式
    sft_data = [r["sft_format"] for r in results]
    with open(SFT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    # GRPO 格式
    with open(GRPO_OUTPUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r["grpo_format"], ensure_ascii=False) + "\n")

    print(f"\n[INFO] 生成 {len(results)} 条本地可答轨迹")
    print(f"[INFO] 完整数据: {OUTPUT_PATH}")
    print(f"[INFO] SFT格式:  {SFT_OUTPUT}")
    print(f"[INFO] GRPO格式: {GRPO_OUTPUT}")

    # ── 合并网络兜底 + 本地可答数据 ──────────────────────
    merge_combined_data()


def merge_combined_data():
    """合并网络兜底 + 本地可答轨迹为统一训练集"""
    web_grpo_path   = os.path.join(BASE_DIR, "data/rl_data/web_fallback_trajectories_grpo.jsonl")
    local_grpo_path = GRPO_OUTPUT
    web_sft_path    = os.path.join(BASE_DIR, "data/rl_data/web_fallback_trajectories_sft.json")
    local_sft_path  = SFT_OUTPUT

    combined_grpo = os.path.join(BASE_DIR, "data/rl_data/combined_trajectories_grpo.jsonl")
    combined_sft  = os.path.join(BASE_DIR, "data/rl_data/combined_trajectories_sft.json")

    # 合并 GRPO 数据
    all_grpo = []
    for path, label in [(web_grpo_path, "网络兜底"), (local_grpo_path, "本地可答")]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                count = 0
                for line in f:
                    all_grpo.append(json.loads(line))
                    count += 1
            print(f"[INFO] {label} GRPO: {count} 条")
        else:
            print(f"[WARN] {label} GRPO 不存在: {path}")

    with open(combined_grpo, "w", encoding="utf-8") as f:
        for item in all_grpo:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[INFO] 合并 GRPO: {len(all_grpo)} 条 → {combined_grpo}")

    # 合并 SFT 数据
    all_sft = []
    for path, label in [(web_sft_path, "网络兜底"), (local_sft_path, "本地可答")]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                all_sft.extend(data)
            print(f"[INFO] {label} SFT: {len(data)} 条")
        else:
            print(f"[WARN] {label} SFT 不存在: {path}")

    with open(combined_sft, "w", encoding="utf-8") as f:
        json.dump(all_sft, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 合并 SFT: {len(all_sft)} 条 → {combined_sft}")

    print("\n" + "=" * 60)
    print(f"✅ 训练数据准备完成，共 {len(all_grpo)} 条 GRPO / {len(all_sft)} 条 SFT")
    print("=" * 60)


if __name__ == "__main__":
    main()
