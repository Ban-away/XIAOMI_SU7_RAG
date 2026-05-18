# -*- coding: utf-8 -*-
"""离线评估脚本：执行完整RAG推理并计算语义分与RAGAS指标。

并行策略：
  - 检索（BM25/Milvus）、HyDE、Query改写、vLLM生成 → ThreadPoolExecutor 并发（IO密集）
  - Reranker（GPU）→ threading.Lock 串行保护（CUDA非线程安全）
"""

from dotenv import load_dotenv
load_dotenv()

import os
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

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.client.llm_local_client import request_chat
from src.client.llm_hyde_client import request_hyde, request_query_rewrite
from src.reranker.minicpm_reranker import MiniCPMReRanker
from src.constant import bge_reranker_minicpm_path, text2vec_model_path
from src.utils import merge_docs, post_processing


# ── 超参数 ──────────────────────────────────────────────────
BM25_RETRIEVE_SIZE   = 10
MILVUS_RETRIEVE_SIZE = 20
RERANK_SIZE          = 8
HYDE                 = 1
QUERY_REWRITE        = 1
MAX_WORKERS          = 5   # 并发线程数（IO密集部分）
# ────────────────────────────────────────────────────────────

# 预热检索器、重排器、向量评估模型
print("[INFO] 加载检索器和重排器...")
bm25_retriever    = BM25(docs=None, retrieve=True)
milvus_retriever  = MilvusRetriever(docs=None, retrieve=True)
reranker          = MiniCPMReRanker(model_path=bge_reranker_minicpm_path, cutoff_layers=28)
milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)
simModel          = SentenceModel(model_name_or_path=text2vec_model_path, device='cuda:0')

# GPU Reranker 不是线程安全的，用锁保护
_rerank_lock = threading.Lock()


# ── 评分函数 ─────────────────────────────────────────────────
def calc_jaccard(list_a, list_b, threshold=0.3):
    size_c = len([i for i in list_a if i in list_b])
    return 1 if size_c / (len(list_b) + 1e-6) > threshold else 0


def report_score(result):
    """计算语义相似度与关键词加权得分。"""
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
            join_keywords  = [w for w in keywords if w in pred]
            keyword_score  = calc_jaccard(join_keywords, keywords)
            score = semantic_score if not keywords else (
                0.15 * keyword_score + 0.85 * semantic_score
            )

        result[idx]["score"] = score
        print("\n【打印低分样本：】\n")
        if score < 0.6:
            print(f"低分样本: {item['question']}")
            print(f"参考回答: {gold}")
            print(f"模型回答: {pred}")
            print(f"得分: {score:.3f}")
            print("-" * 100)
    return result


# ── 单条推理（在线程池中运行）────────────────────────────────
def process_one(item):
    """对单条测试数据执行完整推理链路，返回带 pred 字段的 item。"""
    query = item["question"].strip()

    # 1. Query 纠错改写（API，可并发）
    rewritten_query = request_query_rewrite(query) if QUERY_REWRITE else query

    # 2. HyDE 扩写（API，可并发）
    retrieve_query = rewritten_query
    if HYDE:
        hyde_text      = request_hyde(rewritten_query)
        retrieve_query = rewritten_query + "\n" + hyde_text

    # 3. 检索（BM25 CPU + Milvus 网络，可并发）
    bm25_docs   = bm25_retriever.retrieve_topk(retrieve_query, topk=BM25_RETRIEVE_SIZE)
    milvus_docs = milvus_retriever.retrieve_topk(retrieve_query, topk=MILVUS_RETRIEVE_SIZE)
    merged_docs = merge_docs(bm25_docs, milvus_docs)

    # 4. 精排（GPU，串行保护）
    with _rerank_lock:
        ranked_docs = reranker.rank(retrieve_query, merged_docs, topk=RERANK_SIZE)

    # 5. 生成（vLLM 网络 API，可并发）
    context  = "\n".join([f"【{i+1}】{doc.page_content}" for i, doc in enumerate(ranked_docs)])
    response = request_chat(query, context)
    answer   = post_processing(response, ranked_docs)

    item = dict(item)   # 避免修改原始数据
    item["pred"]          = answer
    item["context"]       = context
    item["rewritten_query"] = retrieve_query  # 保存改写后的查询
    return item


# ── 主流程 ───────────────────────────────────────────────────
def main(): 
    # 检查是否存在已保存的推理结果 
    pred_file = "data/qa_pairs/test_qa_pair_pred.json" 
    if os.path.exists(pred_file): 
        print(f"[INFO] 发现已保存的推理结果，直接加载") 
        with open(pred_file, 'r') as f: 
            result = json.load(f) 
        print(f"[INFO] 推理结果已加载，共 {len(result)} 条") 
    else: 
        fd = open("data/qa_pairs/test_qa_pair_verify.json") 
        test_qa_pairs = json.load(fd) 
        fd.close() 
        print(f"[INFO] 共 {len(test_qa_pairs)} 条测试数据，MAX_WORKERS={MAX_WORKERS}") 
        print("-" * 100) 
 
 
        result = [] 
        # ThreadPoolExecutor 并发处理：IO密集部分（检索+API）并发，GPU部分锁串行 
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor: 
            futures = {executor.submit(process_one, item): item for item in test_qa_pairs} 
            for future in tqdm(as_completed(futures), total=len(futures), 
                               desc="推理进度", unit="问题"): 
                try: 
                    item = future.result() 
                    result.append(item) 
                    print(f"【原始问题】：{item['question']}") 
                    if QUERY_REWRITE: 
                        print(f"【改写后】：{item.get('rewritten_query', '')}") 
                    print(f"【答案】：{item['pred']['answer']}") 
                    print(f"【引用页码】：{item['pred'].get('cite_pages', [])}, 【相关图片】：{item['pred'].get('related_images', [])}") 
                    print("-" * 100) 
                except Exception as e: 
                    print(f"[WARN] 单条推理失败: {e}") 
 
 
        # 保存推理结果 
        with open(pred_file, "w") as fw: 
            fw.write(json.dumps(result, ensure_ascii=False, indent=4)) 
        print(f"[INFO] 推理结果已保存，共 {len(result)} 条")

    # ── 语义相似度 + 关键词加权评分 ─────────────────────────
    results = report_score(result)
    final_score = np.mean([item["score"] for item in results])
    print(f"\n预测问题数：{len(results)}")
    print(f"语义相似度 + 关键词加权得分：{final_score:.4f}")

    # ── RAGas 评估 ────────────────────────────────────────────
    print("\n[INFO] 开始 RAGas 评估...")
    api_key = os.environ["DOUBAO_API_KEY"]
    model_name = os.environ["DOUBAO_MODEL_NAME"]
    base_url = os.environ["DOUBAO_BASE_URL"]

    llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.01, model_kwargs={ "extra_body": { "system": "You are a helpful assistant. Always respond in English with exact JSON format as instructed. Do not add extra fields." } })
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
    dataset = EvaluationDataset.from_list(ragas_data)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(llm=evaluator_llm),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
        ],
    )

    # ── 汇总输出 ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"预测问题数：{len(results)}")
    print(f"语义相似度 + 关键词加权得分：{final_score:.4f}")
    print(f"RAGas 综合得分：{ragas_result}")
    print("=" * 100)

    # 保存评估结果
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