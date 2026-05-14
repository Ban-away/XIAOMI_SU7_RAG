# XIAOMI_SU7_RAG

基于小米 SU7 用户手册的 RAG 问答项目，覆盖**文档解析、语义切分、索引构建、检索重排、答案生成、离线评估、训练数据构造**全流程。

---

## 1. 技术与模型

### 1.1 技术栈

| 模块 | 使用技术/框架 | 代码位置 |
|---|---|---|
| 文档解析 | `pdfplumber` + `PyMuPDF(fitz)` | `src\parser\pdf_parse.py`、`src\parser\image_handler.py` |
| 文本切分 | `langchain_text_splitters.RecursiveCharacterTextSplitter` + `tiktoken` | `src\parser\pdf_parse.py` |
| 语义切分服务 | `FastAPI` + `sentence-transformers` + `scikit-learn` | `src\server\semantic_chunk.py` |
| 存储 | `MongoDB` + `pymongo` | `src\client\mongodb_config.py` |
| 稀疏检索 | BM25 / TF-IDF (`langchain_community`) + `jieba` | `src\retriever\bm25_retriever.py`、`tfidf_retriever.py` |
| 混合检索 | `Milvus Lite` + `pymilvus` + `transformers` + `torch` | `src\retriever\milvus_retriever.py` |
| 向量检索备选 | `FAISS` | `src\retriever\faiss_retriever.py`、`qwen3_retriever.py` |
| 重排 | BGE / Jina / Qwen3（`transformers`、`vllm`） | `src\reranker\*.py` |
| 生成 | OpenAI 兼容协议（本地 vLLM / 云端 API） | `src\client\llm_local_client.py`、`llm_chat_client.py` |
| 评估 | `text2vec` + `ragas` + `langchain-openai` | `final_score.py` |

### 1.2 模型清单（与代码一致）

| 用途 | 模型 | 配置键（`src\constant.py`） | 实际调用位置 |
|---|---|---|---|
| 语义切分向量 | `AI-ModelScope/m3e-small` | `m3e_small_model_path` | `src\server\semantic_chunk.py` |
| Dense 检索向量 | `BAAI/bge-large-zh-v1.5` | `bge_large_zh_v1_5_model_path` | `src\retriever\milvus_retriever.py` |
| Sparse 检索向量 | `naver/splade-cocondenser-ensembledistil` | `splade_v2_model_path` | `src\retriever\milvus_retriever.py` |
| 向量检索备选 | `maidalun/bce-embedding-base_v1` | `bce_model_path` | `src\retriever\faiss_retriever.py` |
| 向量检索备选 | `Qwen3-Embedding-0.6B` | `qwen3_embedding_model_path` | `src\retriever\qwen3_retriever.py` |
| 在线重排 | `bge-reranker-v2-m3` 微调版 | `bge_reranker_tuned_model_path` | `infer.py` |
| 评估重排 | `jina-reranker-v2-base-multilingual` | `jina_reranker_v2_model_path` | `final_score.py` |
| 重排备选 | `Qwen3-Reranker-0.6B` / `Qwen3-Reranker-4B` | `qwen3_reranker_model_path` / `qwen3_4b_reranker_model_path` | `src\reranker\qwen3_*.py` |
| 语义评估向量 | `text2vec-base-chinese` | `text2vec_model_path` | `final_score.py` |
| 本地生成模型 | `LLaMA-Factory-main/output/qwen3_lora_sft_int4` | `qwen3_8b_tune_model_name` | `src\client\llm_local_client.py` |
| 云端生成模型 | `DOUBAO_MODEL_NAME`（环境变量） | - | `src\client\llm_chat_client.py` |

---

## 2. 本地项目结构（完整）

```text
XIAOMI_SU7_RAG/
├─ README.md
├─ requirements.txt                              # 项目依赖
├─ config.ini                                    # 启动示例与环境变量模板
├─ build_index.py                                # 离线建库入口
├─ infer.py                                      # 在线问答入口
├─ final_score.py                                # 离线评估入口
├─ generate_sft_data.py                          # summary/rerank 数据构造
├─ src/                                          # 主业务代码
│  ├─ constant.py                                # 全局路径/模型配置
│  ├─ utils.py                                   # merge_docs/post_processing
│  ├─ parser/                                    # PDF 解析与图片处理
│  ├─ client/                                    # LLM、Mongo、语义切分客户端
│  ├─ server/                                    # 语义切分服务
│  ├─ retriever/                                 # BM25/TFIDF/FAISS/Milvus 检索
│  ├─ reranker/                                  # 多种重排器
│  ├─ fields/                                    # Pydantic 数据结构
│  └─ gen_qa/                                    # QA 生成与扩写
├─ data/                                         # 数据与中间产物
│  ├─ Xiaomi_SU7_Manual.pdf                      # 原始手册
│  ├─ stopwords.txt                              # BM25 停用词
│  ├─ processed_docs/                            # raw/clean/split pickle 产物
│  ├─ saved_index/                               # BM25/TFIDF/FAISS/Milvus 索引产物
│  ├─ qa_pairs/                                  # QA 与评测数据
│  ├─ summary_data/                              # summary 训练数据
│  ├─ rerank_data/                               # rerank 训练/验证/测试数据
│  ├─ saved_images/                              # PDF 抽取图片
│  ├─ mongodb/                                   # 本地 Mongo 数据与日志目录
│  │  ├─ data/
│  │  └─ log/
│  └─ ut/                                        # 小样本/测试文本
├─ log/                                          # 运行日志（语义服务、vLLM 等）
├─ models/                                       # 本地模型缓存/下载目录
├─ mongodb-7.0.20/                               # 本地 mongod 二进制
├─ LLaMA-Factory-main/                           # 训练/量化/部署相关子项目
└─ RAG-Retrieval/                                # 检索训练与实验子项目
```

### 2.1 Git 跟踪状态补充（按当前仓库）

- **已跟踪（在 Git 中）**：
  - `src/`、入口脚本、`README.md`、`requirements.txt`
  - `LLaMA-Factory-main/`（子项目代码与脚本）
  - `RAG-Retrieval/`（部分代码）
- **本地生成或本地资源（未跟踪）**：
  - `log/`
  - `models/`
  - `mongodb-7.0.20/`
  - `data\mongodb\`
  - `data\processed_docs\`、`data\saved_index\`、`data\qa_pairs\`、`data\summary_data\`、`data\rerank_data\`、`data\saved_images\`

---

## 3. 端到端流程（文件 + 函数 + 工具 + 产物）

### 3.1 环境与服务准备

| 步骤 | 执行位置 | 使用工具/框架 | 产物/目录变化 |
|---|---|---|---|
| 配置 API / Mongo 环境变量 | `config.ini`（示例） | Shell 环境变量 | 不产生文件 |
| 启动语义切分服务 | `python src\server\semantic_chunk.py` | FastAPI + Uvicorn + SentenceTransformer | 日志写入 `log\semantic_chunk.log`（若按示例启动） |
| 启动本地生成模型 | `vllm serve ...`（`config.ini` / `LLaMA-Factory-main\vllm_serve.sh`） | vLLM | 日志写入 `log\qwen3-7b.log`（按示例） |
| 启动 MongoDB | `mongodb-7.0.20\bin\mongod` | MongoDB 7.0 | 数据文件写入 `data\mongodb\data\`，日志在 `data\mongodb\log\` |

> 代码中**不会自动创建这些目录**，需要提前准备目录结构。

### 3.2 离线建库（`build_index.py`）

| 子步骤 | 具体函数（文件） | 用到的工具/框架 | 输入 | 输出/落盘 |
|---|---|---|---|---|
| 解析 PDF 文本 + 图片 | `load_pdf()`（`src\parser\pdf_parse.py`） -> `handle_image()`（`src\parser\image_handler.py`） | pdfplumber + PyMuPDF | `data\Xiaomi_SU7_Manual.pdf` | `raw_docs`（内存），随后落盘 `data\processed_docs\raw_docs.pkl` |
| 文本清洗（可选） | `request_llm_clean()`（`src\client\llm_clean_client.py`） | OpenAI SDK（云端）+ 并发线程池 | `raw_docs.pkl` | `clean_docs.pkl` |
| 语义切分 + 子块切分 | `texts_split()`（`src\parser\pdf_parse.py`） -> `request_semantic_chunk()`（`src\client\semantic_chunk_client.py`） -> `/v1/semantic-chunks`（`src\server\semantic_chunk.py:create_chat_completion`） | FastAPI + sklearn 聚类 + LangChain splitter | `clean_docs.pkl` | `split_docs.pkl` |
| 文档入 Mongo | `save_2_mongo()`（`src\parser\pdf_parse.py`） | pymongo | 父/子文档块 | `manual_text` 集合更新（`data\mongodb\data\`） |
| 构建 BM25 索引 | `BM25(...).get_BM25_retriever()`（`src\retriever\bm25_retriever.py`） | BM25Retriever + pickle | `split_docs.pkl` | `data\saved_index\bm25retriever.pkl` |
| 构建 Milvus 混合索引 | `MilvusRetriever.save_vectorstore()`（`src\retriever\milvus_retriever.py`） | pymilvus + transformers + torch | `split_docs.pkl` + 模型目录 | `data\saved_index\milvus.db` |

### 3.3 在线问答（`infer.py`）

| 子步骤 | 具体函数（文件） | 工具/框架 | 结果 |
|---|---|---|---|
| 加载检索器与重排器 | `BM25(...)`、`MilvusRetriever(...)`、`BGEM3ReRanker(...)` | BM25 / Milvus / transformers | 模型预热完成 |
| BM25 召回 | `retrieve_topk()`（`src\retriever\bm25_retriever.py`） | jieba + BM25 | 候选文档 A |
| 混合召回 | `retrieve_topk()`（`src\retriever\milvus_retriever.py`） | BGE + SPLADE + Milvus WeightedRanker | 候选文档 B |
| 合并去重与父块回溯 | `merge_docs()`（`src\utils.py`） | pymongo 回表 | 合并候选 |
| 精排 | `rank()`（`src\reranker\bge_m3_reranker.py`） | Cross-Encoder | TopK 上下文 |
| 答案生成 | `request_chat()`（`src\client\llm_local_client.py`） | vLLM(OpenAI兼容) | 流式答案文本 |
| 引用页码/图片回填 | `post_processing()`（`src\utils.py`） | 正则 + metadata | `answer`、`cite_pages`、`related_images` |

### 3.4 离线评估（`final_score.py`）

| 子步骤 | 具体函数（文件） | 工具/框架 | 输入/输出 |
|---|---|---|---|
| 批量推理 | 主循环（`final_score.py`） | 同检索/重排/生成链路 | 输入 `data\qa_pairs\test_qa_pair_verify.json` |
| 结果落盘 | `json.dump`（`final_score.py`） | Python json | 输出 `data\qa_pairs\test_qa_pair_pred.json` |
| 语义+关键词评分 | `report_score()`（`final_score.py`） | text2vec + 自定义 Jaccard | 输出均分日志 |
| RAGAS 指标 | `evaluate(...)`（`final_score.py`） | ragas + langchain-openai | 输出上下文召回/精确率日志 |

### 3.5 训练数据构造

| 脚本 | 关键函数 | 作用 | 产物 |
|---|---|---|---|
| `src\gen_qa\run.py` | `gen_qa()` / `chat()` | 生成 QA、问题扩写、关键词抽取、负样本拼接 | `data\qa_pairs\*.json` |
| `generate_sft_data.py` | 主流程 | 生成 summary/rerank 训练、验证、测试集 | `data\summary_data\*.json`、`data\rerank_data\*.json` |

---

## 4. 快速运行

```bash
pip install -r requirements.txt
python src/server/semantic_chunk.py
python build_index.py
python infer.py
```

离线评估：

```bash
python final_score.py
```

---

## 5. 配置要点

### 5.1 环境变量

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

### 5.2 路径

`src\constant.py` 默认：

```python
base_dir = "/root/autodl-tmp/RAG/"
```

本地运行前请改为实际路径。

---

## 6. 已知限制与说明

- 目录默认需提前创建，脚本本身不负责 `mkdir`。
- 模型较大，首次加载耗时较高。
- `LLaMA-Factory-main` 与 `RAG-Retrieval` 是配套子项目，当前主问答链路核心运行在 `src\` 下。

