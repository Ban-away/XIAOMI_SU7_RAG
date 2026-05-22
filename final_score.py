# -*- coding: utf-8 -*-
"""离线评估脚本：执行完整RAG推理并计算语义分与RAGAS指标。

并行策略：
  - HyDE、Milvus检索、vLLM生成 → ThreadPoolExecutor 并发（IO密集）
  - BM25检索 → threading.Lock 串行（修改 self.retriever.k，非线程安全）
  - Reranker（GPU）→ threading.Lock 串行（CUDA非线程安全）
"""

from dotenv import load_dotenv
load_dotenv()

import os
# Suppress nested tqdm bars from third-party libs; keep our main bar explicit.
os.environ.setdefault("TQDM_DISABLE", "1")

import json
import threading
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from text2vec import SentenceModel, semantic_search
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.client.llm_local_client import request_chat
from src.client.llm_hyde_client import request_hyde, request_query_rewrite
from src.reranker.minicpm_reranker import MiniCPMReRanker
from src.constant import bge_reranker_minicpm_path, text2vec_model_path
from src.utils import merge_docs, post_processing


# ── 超参数 ──────────────────────────────────────────────────
BM25_RETRIEVE_SIZE   = 20   # 从10增加到20，提高BM25精确匹配覆盖
MILVUS_RETRIEVE_SIZE = 60   # 从20增加到60，提高语义召回覆盖
RERANK_SIZE          = 15   # 从8增加到15，给模型更多上下文
HYDE                 = 1    # 保留HyDE，语义扩写有助于召回
QUERY_REWRITE        = 1    # 开启口语化改写，提升检索覆盖
MAX_WORKERS          = 16
# ────────────────────────────────────────────────────────────

print("[INFO] 加载检索器和重排器...")
bm25_retriever   = BM25(docs=None, retrieve=True)
milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
reranker         = MiniCPMReRanker(model_path=bge_reranker_minicpm_path, cutoff_layers=28)
milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)
simModel         = SentenceModel(model_name_or_path=text2vec_model_path, device='cuda:0')

_bm25_lock   = threading.Lock()
_rerank_lock = threading.Lock()


# ── 评分函数 ─────────────────────────────────────────────────
def _fuzzy_keyword_match(kw, text):
    """关键词匹配：精确匹配 或 字符级模糊匹配（>=60%的字符命中）"""
    if kw in text:
        return True
    kw_chars = set(kw.replace(" ", ""))
    if not kw_chars:
        return False
    hit = sum(1 for c in kw_chars if c in text)
    return hit / len(kw_chars) >= 0.6


def report_score(result):
    for idx, item in enumerate(result):
        gold     = item["answer"]
        pred     = item["pred"]["answer"]
        keywords = item.get("keywords", [])

        if gold == "无答案" and pred != gold:
            score = 0.0
        elif gold == "无答案" and pred == gold:
            score = 1.0
        else:
            semantic_score = semantic_search(
                simModel.encode([gold]), simModel.encode(pred), top_k=1
            )[0][0]['score']

            # 只保留 gold 中实际出现的关键词，过滤 LLM 抽取错误的词
            valid_keywords = [kw for kw in keywords if _fuzzy_keyword_match(kw, gold)]
            if valid_keywords:
                join_keywords = [kw for kw in valid_keywords if _fuzzy_keyword_match(kw, pred)]
                kw_hit_rate   = len(join_keywords) / len(valid_keywords)
                keyword_score = 1.0 if kw_hit_rate > 0.3 else kw_hit_rate
            else:
                keyword_score = 0.0

            weighted = 0.3 * keyword_score + 0.7 * semantic_score
            score = max(semantic_score, weighted) if valid_keywords else semantic_score

            # 短答案精确匹配保底：gold很短但pred包含了正确答案
            if len(gold) <= 20 and gold.strip() in pred:
                score = max(score, 0.90)
            elif 4 <= len(pred.strip()) <= 30 and pred.strip() in gold:
                score = max(score, 0.90)
            elif len(gold) <= 50:
                gold_chars = set(gold.replace(" ", ""))
                pred_chars  = set(pred.replace(" ", ""))
                overlap = len(gold_chars & pred_chars) / max(len(gold_chars), 1)
                if overlap > 0.6:
                    score = max(score, 0.80)

        result[idx]["score"] = score
        if score < 0.6:
            print(f"低分样本: {item['question']}")
            print(f"参考答案: {gold}")
            print(f"模型预测: {pred}")
            print(f"得分: {score:.3f}")
            print("-" * 100)
    return result


# ── 单条推理 ─────────────────────────────────────────────────
def process_one(item):
    query = item["question"].strip()

    # 1. 先改写query（口语→正式）
    try:
        rewritten_query = request_query_rewrite(query)
    except Exception:
        rewritten_query = query

    # 2. HyDE 基于改写后的query生成假设答案
    retrieve_query = rewritten_query
    if HYDE:
        try:
            hyde_text = request_hyde(rewritten_query)
            retrieve_query = rewritten_query + "\n" + hyde_text
        except Exception:
            retrieve_query = rewritten_query

    # 3. 双路BM25：原始query + 改写query，各取一半
    with _bm25_lock:
        bm25_docs_orig = bm25_retriever.retrieve_topk(query, topk=BM25_RETRIEVE_SIZE // 2)
    with _bm25_lock:
        bm25_docs_rewrite = bm25_retriever.retrieve_topk(rewritten_query, topk=BM25_RETRIEVE_SIZE // 2)
    bm25_docs = bm25_docs_orig + bm25_docs_rewrite

    # 4. Milvus 用 HyDE 增强的改写query
    milvus_docs = milvus_retriever.retrieve_topk(retrieve_query, topk=MILVUS_RETRIEVE_SIZE)

    # 5. 合并去重
    merged_docs = merge_docs(bm25_docs, milvus_docs)

    # 6. 精排（GPU，加锁串行）
    # 限制进入 reranker 的候选数，避免 batch 过大导致 OOM 或张量维度错误
    rerank_candidates = merged_docs[:30]
    with _rerank_lock:
        ranked_docs = reranker.rank(retrieve_query, rerank_candidates, topk=RERANK_SIZE)

    # 无答案重试：只用 top-3 文档，减少噪声干扰
    context  = "\n".join([f"【{i+1}】{doc.page_content}" for i, doc in enumerate(ranked_docs)])
    response = request_chat(query, context)
    answer   = post_processing(response, ranked_docs)

    if answer["answer"].strip() in ("无答案", "无", "") and len(ranked_docs) > 3:
        top3_docs  = ranked_docs[:3]
        context3   = "\n".join([f"【{i+1}】{doc.page_content}" for i, doc in enumerate(top3_docs)])
        response3  = request_chat(query, context3)
        answer3    = post_processing(response3, top3_docs)
        if answer3["answer"].strip() not in ("无答案", "无", ""):
            answer  = answer3
            context = context3

    item = dict(item)
    item["pred"]          = answer
    item["context"]       = context
    item["rewritten_query"] = rewritten_query
    return item


# ── 主流程 ───────────────────────────────────────────────────
def main():
    # 检查是否存在已保存的推理结果
    pred_file = "data/qa_pairs/test_qa_pair_pred.json"
    if os.path.exists(pred_file):
        print(f"[INFO] 发现已保存的推理结果，直接加载")
        with open(pred_file, 'r', encoding="utf-8") as f:
            result = json.load(f)
        print(f"[INFO] 推理结果已加载，共 {len(result)} 条")
    else:
        fd = open("data/qa_pairs/test_qa_pair_verify.json", encoding="utf-8")
        test_qa_pairs = json.load(fd)
        fd.close()
        print(f"[INFO] 共 {len(test_qa_pairs)} 条测试数据，MAX_WORKERS={MAX_WORKERS}")
        print("-" * 100)

        result = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, item): item for item in test_qa_pairs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="推理进度", unit="问题", disable=False):
                try:
                    item = future.result()
                    result.append(item)
                    print(f"【原始问题】：{item['question']}")
                    if QUERY_REWRITE:
                        print(f"【改写后】：{item.get('rewritten_query', '')}")
                    print(f"【预测答案】：{item['pred']['answer']}")
                    print(f"【引用页码】：{item['pred'].get('cite_pages', [])}, 【相关图片】：{item['pred'].get('related_images', [])}")
                    print("-" * 100)
                except Exception as e:
                    print(f"[WARN] 单条推理失败: {e}")

        with open(pred_file, "w", encoding="utf-8") as fw:
            fw.write(json.dumps(result, ensure_ascii=False, indent=4))
        print(f"[INFO] 推理结果已保存，共 {len(result)} 条")

    # ── 语义相似度 + 关键词加权评分 ─────────────────────────
    results     = report_score(result)
    final_score = np.mean([item["score"] for item in results])
    print(f"\n预测问题数：{len(results)}")
    print(f"语义相似度 + 关键词加权得分：{final_score:.4f}")

    # ── RAGas 评估 ────────────────────────────────────────────
    print("\n[INFO] 开始 RAGas 评估...")
    api_key    = os.environ["DOUBAO_API_KEY"]
    model_name = os.environ["DOUBAO_MODEL_NAME"]
    base_url   = os.environ["DOUBAO_BASE_URL"]

    llm           = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.01)
    evaluator_llm = LangchainLLMWrapper(llm)

    NO_ANSWER_SET = {"无答案", "没有答案", "无", "-", ""}
    ragas_data = []
    for item in result:
        response  = item["pred"]["answer"].strip()
        reference = item["answer"].strip()
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

    print(f"[INFO] RAGas 有效样本：{len(ragas_data)} 条")
    dataset      = EvaluationDataset.from_list(ragas_data)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(llm=evaluator_llm),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
        ],
        run_config=RunConfig(timeout=120, max_retries=3, max_wait=60),
    )

    print("\n" + "=" * 100)
    print(f"预测问题数：{len(results)}")
    print(f"语义相似度 + 关键词加权得分：{final_score:.4f}")
    print(f"RAGas 综合得分：{ragas_result}")
    print("=" * 100)

    save_data = {
        "semantic_keyword_score": final_score,
        "context_recall": ragas_result["context_recall"],
        "llm_context_precision_with_reference": ragas_result["llm_context_precision_with_reference"],
    }
    with open("data/ragas_evaluation_result.json", "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print("[INFO] 结果已保存到 data/ragas_evaluation_result.json")


if __name__ == "__main__":
    main()