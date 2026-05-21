# -*- coding: utf-8 -*-
"""离线评估脚本：执行完整RAG推理并计算语义分与RAGAS指标。

并行策略：
  - 检索（BM25/Milvus）、HyDE、Query改写、vLLM生成 → ThreadPoolExecutor 并发（IO密集）
  - Reranker（GPU）→ threading.Lock 串行保护（CUDA非线程安全）
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import time
import threading
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from text2vec import SentenceModel, semantic_search
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.client.llm_local_client import request_chat
from src.client.llm_hyde_client import request_hyde, request_query_rewrite
from src.reranker.minicpm_reranker import MiniCPMReRanker
from src.constant import bge_reranker_minicpm_path, text2vec_model_path
from src.utils import merge_docs, post_processing


# ── 超参数 ──────────────────────────────────────────────────
BM25_RETRIEVE_SIZE   = 20
MILVUS_RETRIEVE_SIZE = 40
RERANK_SIZE          = 12
HYDE                 = 1
QUERY_REWRITE        = 0   # 关闭：避免型号/关键词被改写后检索丢失
# 并发线程数（考虑到 vLLM 已占用大量显存，设置较小值避免 OOM）
MAX_WORKERS          = 4
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
                kw_hit_rate = len(join_keywords) / len(valid_keywords)
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


# ── 单条推理（在线程池中运行）────────────────────────────────
def process_one(item):
    """对单条测试数据执行完整推理链路，返回带 pred 字段的 item。"""
    query = item["question"].strip()
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
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

            # 6. 无答案重试：只用 top-3 最相关文档，减少噪声干扰
            if answer["answer"].strip() in ("无答案", "无", "") and len(ranked_docs) > 3:
                top3_docs  = ranked_docs[:3]
                context3   = "\n".join([f"【{i+1}】{doc.page_content}" for i, doc in enumerate(top3_docs)])
                response3  = request_chat(query, context3)
                answer3    = post_processing(response3, top3_docs)
                if answer3["answer"].strip() not in ("无答案", "无", ""):
                    answer = answer3

            item = dict(item)   # 避免修改原始数据
            item["pred"]          = answer
            item["context"]       = context
            item["rewritten_query"] = retrieve_query  # 保存改写后的查询
            return item
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and attempt < max_retries - 1:
                print(f"[WARN] 显存不足，第 {attempt+1}/{max_retries} 次重试...")
                torch.cuda.empty_cache()
                time.sleep(2)
                continue
            raise


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
        # ThreadPoolExecutor 并发处理：IO密集部分（检索+API）并发，GPU部分锁串行 
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, item): item for item in test_qa_pairs}
            pbar = tqdm(as_completed(futures), total=len(futures),
                        desc="推理进度", unit="问题")
            for future in pbar:
                try:
                    item = future.result()
                    result.append(item)
                    pbar.write(f"【原始问题】：{item['question']}")
                    if QUERY_REWRITE:
                        pbar.write(f"【改写后】：{item.get('rewritten_query', '')}")
                    pbar.write(f"【预测答案】：{item['pred']['answer']}")
                    pbar.write(f"【引用页码】：{item['pred'].get('cite_pages', [])}, 【相关图片】：{item['pred'].get('related_images', [])}")
                    pbar.write("-" * 100)
                except Exception as e:
                    pbar.write(f"[WARN] 单条推理失败: {e}") 
 
 
        # 保存推理结果 
        with open(pred_file, "w", encoding="utf-8") as fw:
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

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.01,
        max_tokens=4096,
    )
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
        # 按【N】拆分为独立文档，让 RAGas 正确评估 context_precision
        docs = re.split(r'(?=【\d+】)', context)
        docs = [re.sub(r'^【\d+】', '', d).strip() for d in docs if d.strip()]
        if not docs:
            docs = [context]
        ragas_data.append({
            "user_input":         item["question"],
            "retrieved_contexts": docs,
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
        run_config=RunConfig(
            timeout=100,
            max_retries=3,
            max_wait=50,
            max_workers=25,
        ),
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
    with open("data/ragas_evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print("[INFO] 结果已保存到 data/ragas_evaluation_result.json")


if __name__ == "__main__":
    main()