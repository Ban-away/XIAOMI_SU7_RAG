# -*- coding: utf-8 -*-
"""离线评估脚本：执行完整RAG推理并计算语义分与RAGAS指标。"""

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

import os
import pickle
import time
import json
import sys
import re
import numpy as np
from text2vec import SentenceModel, semantic_search, Similarity
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from openai import OpenAI
from tqdm import tqdm


from src.retriever.bm25_retriever import BM25
from src.retriever.tfidf_retriever import TFIDF
from src.retriever.faiss_retriever import FaissRetriever
from src.retriever.milvus_retriever import MilvusRetriever 
from src.client.llm_local_client import request_chat
from src.client.llm_hyde_client import request_hyde, request_query_rewrite
from src.reranker.bge_m3_reranker import BGEM3ReRanker
from src.constant import bge_reranker_minicpm_path
from src.constant import qwen3_reranker_model_path 
from src.constant import text2vec_model_path 
from src.utils import merge_docs, post_processing


# 预热检索器、重排器、向量模型
bm25_retriever = BM25(docs=None, retrieve=True)
milvus_retriever = MilvusRetriever(docs=None, retrieve=True) 
bge_minicpm_reranker = BGEM3ReRanker(model_path=bge_reranker_minicpm_path)
milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)
simModel = SentenceModel(model_name_or_path=text2vec_model_path, device='cuda:0')

BM25_RETRIEVE_SIZE = 10
MILVUS_RETRIEVE_SIZE = 20
RERANK_SIZE = 8  # 调大重排数量，给LLM更多上下文
HYDE = 1
QUERY_REWRITE = 1  # 开启 Query 纠错改写


def calc_jaccard(list_a, list_b, threshold=0.3):
    size_a, size_b = len(list_a), len(list_b)
    list_c = [i for i in list_a if i in list_b]
    size_c = len(list_c)
    score = size_c / (size_b + 1e-6)
    if score > threshold:
        return 1
    else:
        return 0


def report_score(result):
    """计算语义相似度与关键词加权得分。"""
    idx = 0
    for item in result:
        question = item["question"]
        keywords = item["keywords"]
        gold = item["answer"]
        pred = item["pred"]["answer"]
        if gold == "无答案" and pred != gold:
            score = 0.0
        elif gold == "无答案" and pred == gold:
            score = 1.0
        else:
            semantic_score = semantic_search(simModel.encode([gold]), simModel.encode(pred), top_k=1)[0][0]['score']
            join_keywords = [word for word in keywords if word in pred]
            keyword_score = calc_jaccard(join_keywords, keywords)
            if not keywords:
                score = semantic_score
            else:
                score = 0.2 * keyword_score + 0.8 * semantic_score
        result[idx]["score"] = score
        idx += 1
        if score < 0.6:
            print(f"预测: {question}, 得分: {score}")

    return result



fd = open("data/qa_pairs/test_qa_pair_verify.json")
test_qa_pairs = json.load(fd)
result = []
# 执行整条推理链路并记录中间结果
print(f"开始推理，共 {len(test_qa_pairs)} 个问题...")
print("-" * 100)

# 创建进度条，固定在底部位置
total_count = len(test_qa_pairs)
with tqdm(total=total_count, desc="推理进度", unit="问题", position=0, leave=True) as pbar:
    for idx, item in enumerate(test_qa_pairs):
        query = item["question"].strip()
        
        # Query 纠错改写：在检索前用LLM对query做纠错和扩写
        if QUERY_REWRITE:
            rewritten_query = request_query_rewrite(query)
            retrieve_query = rewritten_query
        else:
            retrieve_query = query
        
        if HYDE:
            hyde_query = request_hyde(retrieve_query) 
            hyde_query = retrieve_query + "\n" + hyde_query 
            bm25_docs = bm25_retriever.retrieve_topk(hyde_query, topk=BM25_RETRIEVE_SIZE)
            milvus_docs = milvus_retriever.retrieve_topk(hyde_query, topk=MILVUS_RETRIEVE_SIZE)
        else:
            bm25_docs = bm25_retriever.retrieve_topk(retrieve_query, topk=BM25_RETRIEVE_SIZE)
            milvus_docs = milvus_retriever.retrieve_topk(retrieve_query, topk=MILVUS_RETRIEVE_SIZE)
        merged_docs = merge_docs(bm25_docs, milvus_docs)
        ranked_docs = bge_minicpm_reranker.rank(query, merged_docs, topk=RERANK_SIZE)
        context = "\n".join([str(i+1) + "." + doc.page_content for i, doc in enumerate(ranked_docs)])
        response = request_chat(query, context)
        answer = post_processing(response, ranked_docs)
        
        # 打印结果前刷新进度条
        pbar.refresh()
        print(f"原始问题：{query}")
        if QUERY_REWRITE:
            print(f"改写后：{retrieve_query}")
        print(f"答案：{answer['answer']}")
        print(f"cite_pages: {answer.get('cite_pages', [])}, related_images: {answer.get('related_images', [])}")
        print("-" * 100)
        
        item["pred"] = answer
        item["context"] = context
        result.append(item)
        
        # 更新进度条
        pbar.update(1)

with open("data/qa_pairs/test_qa_pair_pred.json", "w") as fw:
    fw.write(json.dumps(result, ensure_ascii=False, indent=4))


with open("data/qa_pairs/test_qa_pair_pred.json") as fw:
    result = json.load(fw) 

results = report_score(result)
final_score = np.mean([item["score"] for item in results])
print("\n")
print(f"预测问题数：{len(results)}, 语义相似度+关键词加权得分：{final_score}")


# RAGAS 扩展评估：上下文召回率与精确率

llm = ChatOpenAI(model=os.environ["DOUBAO_MODEL_NAME"], api_key=os.environ["DOUBAO_API_KEY"], base_url=os.environ["DOUBAO_BASE_URL"])

print("开始做RAGas评估...")
dataset = []
for g in result:
    query = g["question"] # 输入问题
    reference = g["answer"] # 参考答案
    response = g["pred"]["answer"] #生成的答案
    context = [g["context"]] # 上下文
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts": context,
            "response":response,
            "reference":reference
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()],llm=evaluator_llm)

# 系统输出得分
print("\n")
print("="*100)
print(f"预测问题数：{len(results)}, 语义相似度+关键词加权得分：{final_score}")
print(f"预测问题数：{len(results)}, LLM+RAGas综合得分：{result}")
print("="*100)