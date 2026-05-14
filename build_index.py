# -*- coding: utf-8 -*-
"""离线建库入口：PDF解析、文本清洗、分层切分、检索索引构建。"""

import os
import pickle
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

from src.parser.pdf_parse import load_pdf, texts_split
from src.retriever.bm25_retriever import BM25
from src.retriever.tfidf_retriever import TFIDF
from src.retriever.faiss_retriever import FaissRetriever
from src.retriever.milvus_retriever import MilvusRetriever 
from src.constant import raw_docs_path, clean_docs_path, split_docs_path
from src.client.llm_clean_client import request_llm_clean


# 1) 解析 PDF 原始页面文档（若本地已有缓存则直接复用）
if not os.path.exists(raw_docs_path):
    raw_docs = load_pdf()
    print("文档page数:", len(raw_docs))
    pickle.dump(raw_docs, open(raw_docs_path, "wb"))
else:
    raw_docs = pickle.load(open(raw_docs_path, "rb"))
    print("加载文档page数:", len(raw_docs))

# 2) 文本清洗整理（若已有 clean 缓存则直接加载）
if not os.path.exists(clean_docs_path):
    clean_docs = request_llm_clean(raw_docs)
    print("清洗后文档page数:", len(clean_docs))
    pickle.dump(clean_docs, open(clean_docs_path, "wb"))
else:
    clean_docs = pickle.load(open(clean_docs_path, "rb"))
    print("加载清洗文档page数:", len(clean_docs))

# 3) 文档切分（语义父块 + 递归子块）
if not os.path.exists(split_docs_path):
    split_docs = texts_split(clean_docs)
    print("解析后文档总数:", len(split_docs))
    pickle.dump(split_docs, open(split_docs_path, "wb"))

else:
    split_docs = pickle.load(open(split_docs_path, "rb"))
    print("加载解析文档总数:", len(split_docs))


# 4) 构建并写入检索索引（BM25 + Milvus Hybrid）
bm25_retriever = BM25(split_docs) 
candidate_docs = bm25_retriever.retrieve_topk("介绍一下离车后自动上锁功能", topk=3)
print("BM25召回样例:")
print(candidate_docs)
print("="*100)

milvus_retriever = MilvusRetriever(split_docs) 
candidate_docs = milvus_retriever.retrieve_topk("介绍一下离车后自动上锁功能", topk=3)
print("BGE-Large-zh-v1.5 + SPLADEv2召回样例:")
print(candidate_docs)
