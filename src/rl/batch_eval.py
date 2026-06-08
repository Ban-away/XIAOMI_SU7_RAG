# -*- coding: utf-8 -*-
"""
RL 模型批量评测脚本

功能：
  用与 final_score.py 相同的评测指标，评估 RL 优化后模型的问答质量。
  支持：
    - 语义相似度 + 关键词加权评分（text2vec）
    - RAGAs 上下文召回率 / 精确率
    - RL 6 维奖励函数评分
    - 与 baseline 对比

运行：
  # 完整评测（676条手册问答）
  python src/rl/batch_eval.py --vllm-url http://localhost:8000/v1

  # 快速验证（前5条）
  python src/rl/batch_eval.py --dry-run

  # 断点续传
  python src/rl/batch_eval.py --resume

  # 跳过 RAGAs（省 API 费用）
  python src/rl/batch_eval.py --skip-ragas

前置条件：
  - vLLM 已启动（RL 模型）
  - 检索环境就绪（BM25 + Milvus 索引）
  - RAGAs 需要 DOUBAO_API_KEY 等环境变量
"""

import os
import re
import sys
import json
import argparse
import threading
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import OpenAI
from text2vec import SentenceModel, semantic_search

from src.rl.infer_rl import run_rl_inference
from src.rl.environment import RetrievalEnvironment
from src.constant import qwen3_8b_tune_model_name, text2vec_model_path

# ── 路径配置 ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_QUESTIONS = os.path.join(BASE_DIR, "data/qa_pairs/test_qa_pair_verify.json")
OUTPUT_PATH       = os.path.join(BASE_DIR, "data/rl_data/rl_eval_results.json")
CKPT_PATH         = os.path.join(BASE_DIR, "data/rl_data/rl_eval_ckpt.jsonl")

# Baseline 指标（final_score.py 的评测结果）
BASELINE_SCORES = {
    "semantic_keyword_score":               0.8965,
    "context_recall":                       0.9386,
    "llm_context_precision_with_reference": 0.9488,
}

# ── 标签正则 ────────────────────────────────────────────────
_RE_INFORMATION = re.compile(r"<information>(.*?)</information>", re.DOTALL)
_RE_ANSWER      = re.compile(r"<answer>(.*?)</answer>",            re.DOTALL)


# ────────────────────────────────────────────────────────────
# 评分函数（移植自 final_score.py，保持逻辑一致）
# ────────────────────────────────────────────────────────────

def _fuzzy_keyword_match(kw: str, text: str) -> bool:
    """关键词匹配：精确匹配 或 字符级模糊匹配（>=60% 的字符命中）"""
    if kw in text:
        return True
    kw_chars = set(kw.replace(" ", ""))
    if not kw_chars:
        return False
    hit = sum(1 for c in kw_chars if c in text)
    return hit / len(kw_chars) >= 0.6


def score_one(gold: str, pred: str, keywords: list, sim_model) -> float:
    """
    计算单条样本的语义相似度 + 关键词加权评分。
    逻辑与 final_score.py 的 report_score() 完全一致。
    """
    # 无答案情况
    if gold == "无答案" and pred != gold:
        return 0.0
    if gold == "无答案" and pred == gold:
        return 1.0

    # 语义相似度
    semantic_score = semantic_search(
        sim_model.encode([gold]), sim_model.encode(pred), top_k=1
    )[0][0]["score"]

    # 关键词匹配
    valid_keywords = [kw for kw in keywords if _fuzzy_keyword_match(kw, gold)]
    if valid_keywords:
        join_keywords = [kw for kw in valid_keywords if _fuzzy_keyword_match(kw, pred)]
        kw_hit_rate   = len(join_keywords) / len(valid_keywords)
        keyword_score = 1.0 if kw_hit_rate > 0.3 else kw_hit_rate
    else:
        keyword_score = 0.0

    weighted = 0.3 * keyword_score + 0.7 * semantic_score
    score = max(semantic_score, weighted) if valid_keywords else semantic_score

    # 短答案精确匹配保底
    if len(gold) <= 20 and gold.strip() in pred:
        score = max(score, 0.90)
    elif 4 <= len(pred.strip()) <= 30 and pred.strip() in gold:
        score = max(score, 0.90)
    elif len(gold) <= 50:
        gold_chars = set(gold.replace(" ", ""))
        pred_chars = set(pred.replace(" ", ""))
        overlap = len(gold_chars & pred_chars) / max(len(gold_chars), 1)
        if overlap > 0.6:
            score = max(score, 0.80)

    return score


# ────────────────────────────────────────────────────────────
# 轨迹提取
# ────────────────────────────────────────────────────────────

def extract_context_from_trajectory(trajectory: str) -> str:
    """从 RL 轨迹中提取所有 <information> 块拼接为上下文"""
    blocks = _RE_INFORMATION.findall(trajectory)
    if not blocks:
        return ""
    return "\n".join(blocks)


# ────────────────────────────────────────────────────────────
# 断点续传
# ────────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    """加载已完成的 unique_id"""
    done = set()
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj["unique_id"])
                except Exception:
                    pass
    return done


def save_checkpoint(unique_id: str):
    """写入检查点"""
    with open(CKPT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"unique_id": unique_id}, ensure_ascii=False) + "\n")


# ────────────────────────────────────────────────────────────
# RAGAs 评测
# ────────────────────────────────────────────────────────────

def run_ragas_evaluation(eval_results: list) -> dict:
    """
    运行 RAGAs 评测（context_recall + context_precision）。
    逻辑与 final_score.py 保持一致。
    """
    from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
    from ragas import evaluate, EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI

    api_key    = os.environ["DOUBAO_API_KEY"]
    model_name = os.environ["DOUBAO_MODEL_NAME"]
    base_url   = os.environ["DOUBAO_BASE_URL"]

    llm           = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.01)
    evaluator_llm = LangchainLLmWrapper(llm)

    NO_ANSWER_SET = {"无答案", "没有答案", "无", "-", ""}
    ragas_data = []
    for item in eval_results:
        response  = item["pred_answer"].strip()
        reference = item["gold_answer"].strip()
        context   = item["context"].strip()
        if not response or not reference or not context:
            continue
        if response in NO_ANSWER_SET or reference in NO_ANSWER_SET:
            continue
        ragas_data.append({
            "user_input":         item["question"],
            "retrieved_contexts": [context],
            "response":           response,
            "reference":          reference,
        })

    print(f"\n[INFO] RAGAs 有效样本：{len(ragas_data)} 条")

    if not ragas_data:
        print("[WARN] 无有效 RAGAs 样本，跳过")
        return {}

    dataset      = EvaluationDataset.from_list(ragas_data)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(llm=evaluator_llm),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
        ],
        run_config=RunConfig(timeout=120, max_retries=3, max_wait=60),
    )

    return {
        "context_recall":                       float(ragas_result["context_recall"]),
        "llm_context_precision_with_reference": float(ragas_result["llm_context_precision_with_reference"]),
    }


# ────────────────────────────────────────────────────────────
# 汇总报告
# ────────────────────────────────────────────────────────────

def print_report(eval_results: list, ragas_scores: dict, skip_ragas: bool):
    """打印评测报告"""
    import numpy as np

    # 语义 + 关键词分
    sem_scores = [item["semantic_keyword_score"] for item in eval_results]
    avg_sem    = np.mean(sem_scores)

    # RL 奖励
    rewards      = [item["reward"] for item in eval_results]
    avg_reward   = np.mean(rewards) if rewards else 0.0
    fmt_scores   = [item["reward_detail"]["format_score"]      for item in eval_results]
    ans_scores   = [item["reward_detail"]["answer_score"]      for item in eval_results]
    tool_scores  = [item["reward_detail"]["tool_score"]        for item in eval_results]
    src_scores   = [item["reward_detail"]["source_score"]      for item in eval_results]
    dom_scores   = [item["reward_detail"]["domain_score"]      for item in eval_results]
    exp_scores   = [item["reward_detail"].get("exploration_score", 0) for item in eval_results]

    # 工具调用统计
    local_counts     = [item["search_calls"]["local"]     for item in eval_results]
    web_counts       = [item["search_calls"]["web"]       for item in eval_results]
    read_page_counts = [item["search_calls"]["read_page"] for item in eval_results]
    rounds_counts    = [item["rounds"] for item in eval_results]

    print("\n" + "=" * 70)
    print("📊 RL 模型评测报告")
    print("=" * 70)
    print(f"  评测数据: {len(eval_results)} 条")
    print(f"  评测模型: RL (Search-R1 + WebWalker)")
    print("=" * 70)

    # ── 传统指标对比 ──
    print("\n  📌 传统评测指标（与 baseline 对比）：")
    print("  " + "-" * 56)
    print(f"  {'指标':<30s} {'Baseline':>10s} {'RL模型':>10s}")
    print("  " + "-" * 56)
    print(f"  {'语义相似度+关键词加权':<30s} {BASELINE_SCORES['semantic_keyword_score']:>10.4f} {avg_sem:>10.4f}")

    if not skip_ragas and ragas_scores:
        print(f"  {'RAGAs 上下文召回率':<30s} {BASELINE_SCORES['context_recall']:>10.4f} {ragas_scores['context_recall']:>10.4f}")
        print(f"  {'RAGAs 上下文精确率':<30s} {BASELINE_SCORES['llm_context_precision_with_reference']:>10.4f} {ragas_scores['llm_context_precision_with_reference']:>10.4f}")
    else:
        print(f"  {'RAGAs 上下文召回率':<30s} {'(跳过)':>10s} {'(跳过)':>10s}")
        print(f"  {'RAGAs 上下文精确率':<30s} {'(跳过)':>10s} {'(跳过)':>10s}")
    print("  " + "-" * 56)

    # ── RL 特有指标 ──
    print(f"\n  📌 RL 特有指标：")
    print(f"  平均奖励: {avg_reward:.4f} / 1.00")
    print(f"    格式完整性: {np.mean(fmt_scores):.4f} / 0.15")
    print(f"    答案质量:   {np.mean(ans_scores):.4f} / 0.30")
    print(f"    工具合理性: {np.mean(tool_scores):.4f} / 0.15")
    print(f"    来源标注:   {np.mean(src_scores):.4f} / 0.10")
    print(f"    领域合规:   {np.mean(dom_scores):.4f} / 0.15")
    print(f"    探索深度:   {np.mean(exp_scores):.4f} / 0.15")

    # ── 工具调用统计 ──
    print(f"\n  📌 工具调用统计：")
    print(f"    平均轮数: {np.mean(rounds_counts):.1f} | "
          f"local: {np.mean(local_counts):.1f} | "
          f"web: {np.mean(web_counts):.1f} | "
          f"read_page: {np.mean(read_page_counts):.1f}")

    # ── 分布 ──
    brackets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for s in sem_scores:
        if s < 0.2:   brackets["0.0-0.2"] += 1
        elif s < 0.4: brackets["0.2-0.4"] += 1
        elif s < 0.6: brackets["0.4-0.6"] += 1
        elif s < 0.8: brackets["0.6-0.8"] += 1
        else:         brackets["0.8-1.0"] += 1

    print(f"\n  📌 语义+关键词分数分布：")
    for bracket, count in brackets.items():
        pct = count / max(len(sem_scores), 1) * 100
        bar = "█" * int(pct / 2)
        print(f"    {bracket}: {count:>4d} ({pct:>5.1f}%) {bar}")

    print("=" * 70)
    print(f"  结果已保存到 {OUTPUT_PATH}")
    print("=" * 70)


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL 模型批量评测")
    parser.add_argument(
        "--vllm-url", type=str,
        default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="vLLM 服务地址",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型名称（默认使用 constant.py 配置）",
    )
    parser.add_argument(
        "--questions", type=str, default=DEFAULT_QUESTIONS,
        help="评测问题文件路径（默认 test_qa_pair_verify.json）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只处理前 5 条，验证流程",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="断点续传，跳过已完成的样本",
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="跳过 RAGAs 评测（省 API 费用）",
    )
    args = parser.parse_args()

    model_name = args.model or qwen3_8b_tune_model_name

    # ── 加载问题 ──────────────────────────────────────────
    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)

    if args.dry_run:
        questions = questions[:5]
        print(f"[DRY-RUN] 仅处理前 {len(questions)} 条")

    # 断点续传
    done_ids = load_checkpoint() if args.resume else set()
    if done_ids:
        before = len(questions)
        questions = [q for q in questions if q.get("unique_id", "") not in done_ids]
        print(f"[INFO] 断点续传：跳过 {before - len(questions)} 条已完成样本")

    print(f"[INFO] 待评测：{len(questions)} 条")

    # ── 初始化组件 ────────────────────────────────────────
    print("[INFO] 加载 text2vec 模型...")
    sim_model = SentenceModel(model_name_or_path=text2vec_model_path)

    print("[INFO] 初始化 RL 推理环境...")
    llm_client = OpenAI(api_key="EMPTY", base_url=args.vllm_url)
    env = RetrievalEnvironment()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # ── 逐条推理 + 评分 ──────────────────────────────────
    eval_results = []
    file_lock = threading.Lock()

    for item in tqdm(questions, desc="评测进度"):
        question = item["question"].strip()
        gold     = item["answer"].strip()
        keywords = item.get("keywords", [])
        uid      = item.get("unique_id", "")

        try:
            # Step 1: RL 推理
            result = run_rl_inference(
                question   = question,
                llm_client = llm_client,
                env        = env,
                model_name = model_name,
                verbose    = False,
            )

            # Step 2: 提取上下文
            context = extract_context_from_trajectory(result["trajectory"])

            # Step 3: 语义 + 关键词评分
            sem_score = score_one(gold, result["answer"], keywords, sim_model)

            # Step 4: 组装结果
            eval_item = {
                "unique_id":              uid,
                "question":               question,
                "gold_answer":            gold,
                "pred_answer":            result["answer"],
                "context":                context,
                "keywords":               keywords,
                "semantic_keyword_score": sem_score,
                "reward":                 result["reward"],
                "reward_detail":          result["reward_detail"],
                "rounds":                 result["rounds"],
                "search_calls":           result["search_calls"],
            }
            eval_results.append(eval_item)

            # 检查点
            with file_lock:
                save_checkpoint(uid)

        except Exception as e:
            print(f"\n[ERROR] 处理失败 ({uid}): {e}")
            continue

    if not eval_results:
        print("[ERROR] 无有效评测结果")
        return

    # ── RAGAs 评测（可选）────────────────────────────────
    ragas_scores = {}
    if not args.skip_ragas:
        print("\n[INFO] 开始 RAGAs 评测...")
        try:
            ragas_scores = run_ragas_evaluation(eval_results)
        except Exception as e:
            print(f"[WARN] RAGAs 评测失败: {e}")
            print("[INFO] 将跳过 RAGAs 指标")

    # ── 保存结果 ──────────────────────────────────────────
    output = {
        "meta": {
            "total":      len(eval_results),
            "model":      model_name,
            "skip_ragas": args.skip_ragas,
        },
        "results":      eval_results,
        "ragas_scores":  ragas_scores,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── 打印报告 ──────────────────────────────────────────
    print_report(eval_results, ragas_scores, args.skip_ragas)


if __name__ == "__main__":
    main()
