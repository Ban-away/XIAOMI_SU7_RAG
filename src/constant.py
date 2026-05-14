# -*- coding: utf-8 -*-
"""
全局常量配置文件。

这个文件集中维护项目运行所需的路径与模型位置，避免在业务代码中硬编码。
所有脚本（build_index / infer / final_score / 训练数据构造）都通过这里读取路径。
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

# =========================
# 1) 项目根目录
# =========================
# 支持三种配置方式（优先级从高到低）：
# 1. 环境变量 RAG_BASE_DIR（推荐用于跨平台部署）
# 2. 环境变量 XIAOMI_RAG_HOME（备选）
# 3. 硬编码默认值（Linux 训练环境）
base_dir = os.getenv(
    "RAG_BASE_DIR",
    os.getenv(
        "XIAOMI_RAG_HOME",
        "/root/autodl-tmp/XIAOMI_SU7_RAG/"  # Linux 默认路径
    )
)

# 确保路径以分隔符结尾
if not base_dir.endswith(os.sep):
    base_dir = base_dir + os.sep

# =========================
# 2) 原始数据与中间产物路径
# =========================
# 原始手册 PDF
pdf_path = base_dir + "data/Xiaomi_SU7_Manual.pdf"
# 用于检索器本地调试的小样本文本
test_doc_path = base_dir + "data/test_docs.txt"
# 中文停用词表（BM25 分词过滤会使用）
stopwords_path = base_dir + "data/stopwords.txt"
# 从 PDF 中抽取图片后的保存目录
image_save_dir = base_dir + "data/saved_images"
# PDF 原始解析结果（Document 列表）缓存
raw_docs_path = base_dir + "data/processed_docs/raw_docs.pkl"
# 文本清洗后的文档缓存
clean_docs_path = base_dir + "data/processed_docs/clean_docs.pkl"
# 语义切分 + 子块切分后的最终文档缓存
split_docs_path = base_dir + "data/processed_docs/split_docs.pkl"

# =========================
# 3) 索引持久化路径
# =========================
# BM25 检索器序列化文件
bm25_pickle_path = base_dir + "data/saved_index/bm25retriever.pkl"
# TF-IDF 检索器序列化文件
tfidf_pickle_path = base_dir + "data/saved_index/tfidfretriever.pkl"
# Milvus Lite 本地数据库文件
milvus_db_path = base_dir + "data/saved_index/milvus.db"
# FAISS（BCE embedding）索引目录
faiss_db_path = base_dir + "data/saved_index/faiss.db"
# FAISS（Qwen3 embedding）索引目录
faiss_qwen_db_path = base_dir + "data/saved_index/faiss_qwen.db"

# =========================
# 4) 模型路径（按功能分类）
# =========================
# 语义切分服务 embedding 模型
m3e_small_model_path = base_dir + "models/moka-ai/m3e-small"

# 检索阶段向量模型
bge_m3_model_path = base_dir + "models/BAAI/bge-m3"
bge_large_zh_v1_5_model_path = base_dir + "models/BAAI/bge-large-zh-v1.5"
splade_v2_model_path = base_dir + "models/naver/splade-cocondenser-ensembledistil"
bce_model_path = base_dir + "models/maidalun/bce-embedding-base_v1"
qwen3_embedding_model_path = base_dir + "models/Qwen3-Embedding-0.6B"

# 重排模型
qwen3_reranker_model_path = base_dir + "models/Qwen3-Reranker-0.6B"
qwen3_4b_reranker_model_path = base_dir + "models/Qwen3-Reranker-4B"
bge_reranker_model_path = base_dir + "models/BAAI/bge-reranker-v2-m3"
bge_reranker_tuned_model_path = base_dir + "RAG-Retrieval/rag_retrieval/train/reranker/output/bert/runs/checkpoints/checkpoint_0/"
bge_reranker_minicpm_path = base_dir + "models/bge-reranker-v2-minicpm-layerwise"
jina_reranker_v2_model_path = base_dir + "models/jinaai/jina-reranker-v2-base-multilingual"

# 评估向量模型
text2vec_model_path = base_dir + "models/text2vec-base-chinese"

# 本地生成模型（vLLM 侧 model 字段）
qwen3_8b_tune_model_name = "LLaMA-Factory-main/output/qwen3_lora_sft_int4"
