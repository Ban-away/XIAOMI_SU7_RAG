# -*- coding: utf-8 -*-
"""
生成本地可答轨迹数据

功能：
  从 test_qa_pair_verify.json 采样问题，走本地检索管线（BM25+Milvus+重排），
  生成只有 <search_local> 的轨迹（不含 <search_web>/<read_page>），
  教 RL 模型"本地够用时不要联网"。

运行：
  python src/rl/build_local_trajectories.py
  python src/rl/build_local_trajectories.py --sample 100    # 采样数量
  python src/rl/build_local_trajectories.py --dry-run       # 只跑 5 条验证
"""

import os
import re
import sys
import json
import hashlib
import argparse
import threading
from tqdm import tqdm

# ── 项目路径 ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# ── 静默 tqdm（防止子库弹进度条）──────────────────────────
os.environ["TQDM_DISABLE"] = "1"

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.reranker.minicpm_reranker import MiniCPMReRanker
from src.constant import bge_reranker_minicpm_path
from src.utils import merge_docs

# ── 路径配置 ────────────────────────────────────────────────
QUESTIONS_PATH = os.path.join(BASE_DIR, "data/qa_pairs/test_qa_pair_verify.json")
OUTPUT_PATH    = os.path.join(BASE_DIR, "data/rl_data/local_trajectories.json")
GRPO_OUTPUT    = os.path.join(BASE_DIR, "data/rl_data/local_trajectories_grpo.jsonl")
SFT_OUTPUT     = os.path.join(BASE_DIR, "data/rl_data/local_trajectories_sft.json")

# ── 系统提示词（与 data_builder.py 一致）────────────────────
SYSTEM_PROMPT = """你是小米SU7车型的专业问答助手，服务范围严格限定在小米SU7相关问题。

回答问题时可以调用以下工具：
- 本地知识库检索（优先）：<search_local>检索关键词</search_local>
- 网络搜索（本地信息不足时）：<search_web>检索关键词</search_web>
- 页面深度阅读（搜索结果不够详细时）：<read_page>URL地址</read_page>

工具返回格式：<information>检索结果内容</information>

最终答案格式：<answer>答案内容</answer>

注意：
1. 优先调用本地知识库，本地无结果或信息严重不足时再调用网络搜索
2. 网络搜索结果中包含"网址："字段，可选择最有价值的页面用 <read_page> 深入阅读，最多读取2个页面
3. 与小米SU7无关的问题（闲聊、百科、娱乐等），直接输出 <answer>很抱歉，我只能回答小米SU7相关问题。</answer>
4. 网络搜索结果来源于互联网，答案中需注明"根据网络信息"
5. 涉及页码引用时格式为【页码】"""

LOCAL_TOPK = 3   # 本地检索条数


# ────────────────────────────────────────────────────────────
# 检索模块（复用 data_builder 的 LocalSearchTool）
# ────────────────────────────────────────────────────────────

class LocalSearchTool:
    """封装现有本地检索栈（Milvus + BM25 + MiniCPM精排）"""

    def __init__(self):
        print("[INFO] 加载本地检索组件...")
        self.bm25     = BM25(docs=None, retrieve=True)
        self.milvus   = MilvusRetriever(docs=None, retrieve=True)
        self.reranker = MiniCPMReRanker(
            model_path=bge_reranker_minicpm_path,
            cutoff_layers=28
        )
        self._lock = threading.Lock()
        self.milvus.retrieve_topk("测试", topk=1)
        print("[INFO] 本地检索组件加载完成")

    def search(self, query: str, topk: int = LOCAL_TOPK) -> list:
        with self._lock:
            bm25_docs = self.bm25.retrieve_topk(query, topk=topk)
        milvus_docs = self.milvus.retrieve_topk(query, topk=topk * 2)
        merged = merge_docs(bm25_docs, milvus_docs)
        if not merged:
            return []
        with self._lock:
            ranked = self.reranker.rank(query, merged[:10], topk=topk)
        return ranked

    def format_docs(self, docs: list) -> str:
        if not docs:
            return "本地知识库中未检索到相关内容。"
        parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "")
            suffix = f"（第{page}页）" if page else ""
            parts.append(f"[{i}]{suffix} {doc.page_content[:300]}")
        return "\n".join(parts)


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
    # 提取搜索关键词：取问句中的核心词
    keywords = re.sub(r"[？?！!。，,的了是在有和与或]", "", question)[:20]

    info_text = LocalSearchTool.format_docs(None, docs) if False else _format_docs(docs)

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
# 格式转换
# ────────────────────────────────────────────────────────────

def to_sft_format(question: str, trajectory: str) -> dict:
    answer_match = re.search(r"<answer>(.*?)</answer>", trajectory, re.DOTALL)
    answer_text  = answer_match.group(1).strip() if answer_match else ""
    return {
        "instruction": question,
        "input":       "",
        "output":      trajectory,
        "answer":      answer_text,
        "system":      SYSTEM_PROMPT,
        "data_source": "local_only",
    }


def to_grpo_format(question: str, trajectory: str) -> dict:
    answer_match = re.search(r"<answer>(.*?)</answer>", trajectory, re.DOTALL)
    answer_text  = answer_match.group(1).strip() if answer_match else ""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "completion":  trajectory,
        "answer":      answer_text,
        "data_source": "local_only",
        "reward_type": "local_answer_quality",
    }


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成本地可答轨迹数据")
    parser.add_argument("--sample", type=int, default=100, help="采样问题数量")
    parser.add_argument("--dry-run", action="store_true", help="只处理前 5 条")
    args = parser.parse_args()

    # ── 加载问题 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📦 生成本地可答轨迹数据")
    print("=" * 60)

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)

    print(f"[INFO] 加载 {len(all_questions)} 条测试问题")

    # 随机采样
    import random
    random.seed(42)
    sampled = random.sample(all_questions, min(args.sample, len(all_questions)))
    if args.dry_run:
        sampled = sampled[:5]

    print(f"[INFO] 采样 {len(sampled)} 条")

    # ── 初始化检索 ────────────────────────────────────────
    local_tool = LocalSearchTool()

    # ── 生成轨迹 ─────────────────────────────────────────
    results = []
    for item in tqdm(sampled, desc="生成轨迹", ncols=80):
        question = item["question"].strip()
        gold     = item["answer"].strip()

        # 本地检索
        docs = local_tool.search(question)

        # 生成轨迹
        trajectory = build_local_trajectory(question, docs, gold)

        unique_id = hashlib.md5(question.encode("utf-8")).hexdigest()
        results.append({
            "id":          item.get("unique_id", unique_id),
            "unique_id":   unique_id,
            "category":    "local_manual",
            "category_zh": "手册本地问答",
            "question":    question,
            "trajectory":  trajectory,
            "local_docs_count": len(docs),
            "web_search_used": False,
            "sft_format":  to_sft_format(question, trajectory),
            "grpo_format": to_grpo_format(question, trajectory),
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
    web_grpo_path  = os.path.join(BASE_DIR, "data/rl_data/web_fallback_trajectories_grpo.jsonl")
    local_grpo_path = GRPO_OUTPUT
    web_sft_path   = os.path.join(BASE_DIR, "data/rl_data/web_fallback_trajectories_sft.json")
    local_sft_path = SFT_OUTPUT

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
