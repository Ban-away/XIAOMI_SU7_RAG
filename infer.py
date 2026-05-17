# -*- coding: utf-8 -*-
"""在线问答主流程：召回 -> 去重 -> 重排 -> 生成 -> 后处理。"""

import os
import pickle
import time
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

from src.retriever.bm25_retriever import BM25
from src.retriever.tfidf_retriever import TFIDF
from src.retriever.faiss_retriever import FaissRetriever
from src.retriever.milvus_retriever import MilvusRetriever 
from src.client.llm_local_client import request_chat
from src.client.llm_hyde_client import request_hyde, request_query_rewrite
from src.reranker.bge_m3_reranker import BGEM3ReRanker 
from src.constant import bge_reranker_tuned_model_path
from src.utils import merge_docs, post_processing

# 预热：提前加载检索与重排模型，降低首问延迟
bm25_retriever = BM25(docs=None, retrieve=True)
milvus_retriever = MilvusRetriever(docs=None, retrieve=True) 
bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)
milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)

# 配置参数
BM25_RETRIEVE_SIZE = 10
MILVUS_RETRIEVE_SIZE = 20
RERANK_SIZE = 8  # 调大重排数量，给LLM更多上下文
HYDE = 1
QUERY_REWRITE = 1


while True:
    # 接收用户问题
    query = input("输入—>")
    
    # Query 纠错改写：在检索前用LLM对query做纠错和扩写
    if QUERY_REWRITE:
        rewritten_query = request_query_rewrite(query)
        print(f"\n原始问题: {query}")
        print(f"改写后: {rewritten_query}")
        retrieve_query = rewritten_query
    else:
        retrieve_query = query

    # BM25 关键词检索召回
    t1 = time.time()
    bm25_docs = bm25_retriever.retrieve_topk(retrieve_query, topk=BM25_RETRIEVE_SIZE)
    print("BM25召回样例:")
    print(bm25_docs)
    print("="*100)
    t2 = time.time()


    # Milvus 混合召回（Dense + Sparse）
    milvus_docs = milvus_retriever.retrieve_topk(retrieve_query, topk=MILVUS_RETRIEVE_SIZE)
    print("BGE-M3召回样例:")
    print(milvus_docs)
    print("="*100)
    t3 = time.time()


    # 去重并对齐父块，避免上下文冗余
    merged_docs = merge_docs(bm25_docs, milvus_docs)
    print(merged_docs)
    print("="*100)


    # 重排：从候选中选出最相关的 TopK 文档
    ranked_docs = bge_m3_reranker.rank(retrieve_query, merged_docs, topk=RERANK_SIZE)
    print(ranked_docs)
    print("="*100)


    # 生成答案（流式输出）
    context = "\n".join(["【" + str(idx+1) + "】" + doc.page_content for idx, doc in enumerate(ranked_docs)])
    res_handler = request_chat(query, context, stream=True)  # 生成时仍用原始问题
    response = ""
    for r in res_handler:
        uttr = r.choices[0].delta.content
        # 处理流式响应中的 None 值（chunk 结束时 content 为 None）
        if uttr is not None:
            response += uttr 
            print(uttr, end='')
    print("\n" + "="*100)

    # 后处理：抽取引用页码及相关图片
    answer = post_processing(response, ranked_docs)
    print("\n答案—>", answer)