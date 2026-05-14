# XIAOMI_SU7_RAG

面向**小米 SU7 用户手册问答**的 RAG 项目，覆盖文档解析、知识入库、混合检索、重排、答案生成、评估与训练数据构造。

## 1. 技术与模型总览（完整清单）

### 1.1 技术栈

| 类别 | 技术/库 | 在项目中的作用 |
|---|---|---|
| 语言与运行时 | Python | 全部脚本与服务实现 |
| 文档解析 | pdfplumber、PyMuPDF(fitz) | 提取手册文本、图片、版面区域信息 |
| 文本切分 | LangChain `RecursiveCharacterTextSplitter`、tiktoken | 子块切分与 token 级长度控制 |
| 语义切分服务 | FastAPI、Uvicorn、sentence-transformers、scikit-learn | 语义聚类分组（`src/server/semantic_chunk.py`） |
| 存储 | MongoDB + pymongo | 存储父/子分片及元数据（页码、图片、父子关系） |
| 关键词检索 | LangChain BM25Retriever、TFIDFRetriever、jieba | 稀疏检索通道 |
| 向量/混合检索 | Milvus Lite + pymilvus、transformers、torch | Dense+Sparse 混合召回 |
| 向量库备选 | FAISS（langchain_community.vectorstores） | 备选向量检索方案 |
| 重排 | transformers / vLLM | Cross-Encoder 与 LLM-based 重排 |
| 生成调用协议 | OpenAI Compatible API（openai SDK） | 云端模型与本地 vLLM 一致调用 |
| 评估 | text2vec、ragas、langchain-openai | 语义分、关键词加权分、RAGAS 指标 |
| 数据处理 | numpy、pandas、json | 训练与评测数据构建 |

### 1.2 模型清单（按用途）

> 以下模型来自 `src/constant.py` 与实际调用代码，包含主流程与备选流程。

| 用途 | 模型 | 代码位置 | 当前使用情况 |
|---|---|---|---|
| 语义切分向量 | `AI-ModelScope/m3e-small` | `constant.m3e_small_model_path` | **启用**（语义切分服务） |
| 混合检索 Dense 向量 | `BAAI/bge-large-zh-v1.5` | `constant.bge_large_zh_v1_5_model_path` | **启用**（Milvus） |
| 混合检索 Sparse 向量 | `naver/splade-cocondenser-ensembledistil` | `constant.splade_v2_model_path` | **启用**（Milvus） |
| 向量检索备选 | `maidalun/bce-embedding-base_v1` | `constant.bce_model_path` | 备选（FAISS） |
| 向量检索备选 | `Qwen3-Embedding-0.6B` | `constant.qwen3_embedding_model_path` | 备选（`qwen3_retriever.py`） |
| 在线问答重排 | `bge-reranker-v2-m3` 微调版 | `constant.bge_reranker_tuned_model_path` | **启用**（`infer.py`） |
| 评测重排 | `jina-reranker-v2-base-multilingual` | `constant.jina_reranker_v2_model_path` | **启用**（`final_score.py`） |
| 重排备选 | `Qwen3-Reranker-0.6B` | `constant.qwen3_reranker_model_path` | 备选 |
| 重排备选 | `Qwen3-Reranker-4B` | `constant.qwen3_4b_reranker_model_path` | 备选 |
| 重排备选 | `bge-reranker-v2-minicpm-layerwise` | `constant.bge_reranker_minicpm_path` | 备选 |
| 语义评估向量 | `text2vec-base-chinese` | `constant.text2vec_model_path` | **启用**（`final_score.py`） |
| 生成模型（本地） | `LLaMA-Factory-main/output/qwen3_lora_sft_int4` | `constant.qwen3_8b_tune_model_name` | **启用**（`llm_local_client.py`） |
| 生成模型（云端） | `DOUBAO_MODEL_NAME`（环境变量） | `llm_chat_client.py` 等 | **可启用** |

## 2. 项目定位与目标

- 构建小米 SU7 手册问答系统，输出可追溯引用答案
- 在召回阶段融合关键词检索与语义检索，提高覆盖率
- 在重排阶段提升上下文相关性，降低幻觉风险
- 提供评估与训练数据生成脚本，支持持续迭代

## 3. 项目结构

```text
XIAOMI_SU7_RAG/
├─ build_index.py
├─ infer.py
├─ final_score.py
├─ generate_sft_data.py
├─ requirements.txt
├─ src/
│  ├─ constant.py
│  ├─ utils.py
│  ├─ parser/
│  │  ├─ pdf_parse.py
│  │  └─ image_handler.py
│  ├─ retriever/
│  │  ├─ bm25_retriever.py
│  │  ├─ milvus_retriever.py
│  │  ├─ tfidf_retriever.py
│  │  ├─ faiss_retriever.py
│  │  └─ qwen3_retriever.py
│  ├─ reranker/
│  │  ├─ bge_m3_reranker.py
│  │  ├─ jina_reranker_v2.py
│  │  ├─ qwen3_reranker.py
│  │  └─ qwen3_reranker_vllm.py
│  ├─ client/
│  │  ├─ llm_local_client.py
│  │  ├─ llm_chat_client.py
│  │  ├─ llm_hyde_client.py
│  │  ├─ llm_clean_client.py
│  │  ├─ semantic_chunk_client.py
│  │  └─ mongodb_config.py
│  ├─ server/
│  │  └─ semantic_chunk.py
│  ├─ fields/
│  │  ├─ manual_info_mongo.py
│  │  └─ manual_images.py
│  └─ gen_qa/
│     └─ run.py
└─ data/
   ├─ Xiaomi_SU7_Manual.pdf
   ├─ processed_docs/
   ├─ saved_index/
   ├─ qa_pairs/
   ├─ summary_data/
   ├─ rerank_data/
   └─ saved_images/
```

## 4. 端到端流程（准确对应代码）

### 4.1 离线建库：`build_index.py`

1. **PDF 解析**：`load_pdf()` 使用 pdfplumber 提取文本，使用 fitz 提取图片与位置信息  
2. **文档清洗（可选）**：`request_llm_clean()` 对页面文本做规整  
3. **分层切分**：`texts_split()` 先调语义切分服务，再进行递归子块切分  
4. **Mongo 入库**：父块和子块都写入 `manual_text` 集合  
5. **索引构建**：
   - BM25 持久化到 `data/saved_index/bm25retriever.pkl`
   - Milvus 持久化到 `data/saved_index/milvus.db`

### 4.2 在线问答：`infer.py`

1. 加载 BM25、Milvus、重排器（BGEM3 微调权重）  
2. 输入问题后执行双路召回（BM25 + Milvus Hybrid）  
3. `merge_docs` 去重并回溯父块  
4. 重排取 TopK  
5. `llm_local_client.request_chat()` 生成带引用编号答案  
6. `post_processing()` 解析引用页码与图片信息

### 4.3 离线评估：`final_score.py`

1. 使用测试集执行完整 RAG 推理  
2. 计算语义相似度 + 关键词加权分  
3. 使用 RAGAS 计算上下文召回/精确率指标

### 4.4 训练数据构建

- `src/gen_qa/run.py`：从文档生成 QA、问题扩写、关键词抽取、负样本拼接  
- `generate_sft_data.py`：生成总结训练集与重排训练/验证/测试集

## 5. 运行要求

### 5.1 基础环境

- Python 3.10+（建议）
- MongoDB（默认 `localhost:27017`）
- CUDA GPU（推荐，CPU 可运行但速度显著下降）

### 5.2 必需环境变量

```bash
DOUBAO_API_KEY=...
DOUBAO_BASE_URL=...
DOUBAO_MODEL_NAME=...
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB_NAME=mydatabase
MONGO_USERNAME=
MONGO_PASSWORD=
MONGO_AUTH_SOURCE=admin
```

### 5.3 路径配置

`src/constant.py` 当前默认 `base_dir = "/root/autodl-tmp/RAG/"`。  
在本地运行前请改为你的实际项目根目录。

## 6. 快速运行

```bash
pip install -r requirements.txt
python src/server/semantic_chunk.py
python build_index.py
python infer.py
```

如需离线评测：

```bash
python final_score.py
```

如需本地 LLM 推理，请提前启动 vLLM（与 `llm_local_client.py` 一致）：

```bash
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --max-model-len 8192
```

## 7. 可重建数据目录

下列目录均可由脚本重新生成：

- `data/processed_docs/`
- `data/saved_index/`
- `data/qa_pairs/`
- `data/summary_data/`
- `data/rerank_data/`
- `data/saved_images/`

## 8. 说明

- 项目提示词已统一为小米 SU7 场景文案。
- 重排与生成模型体积较大，首次加载耗时和显存占用较高。
