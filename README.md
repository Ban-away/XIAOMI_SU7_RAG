# XIAOMI_SU7_RAG

基于小米 SU7 用户手册构建的 RAG（Retrieval-Augmented Generation）问答项目，覆盖文档解析、索引构建、检索、重排、答案生成、评估与训练数据生成全流程。

## 1. 项目目标

- 将《Xiaomi SU7 用户手册》转为可检索知识库
- 对用户问题进行多路召回与重排，提高答案相关性
- 基于本地或云端 LLM 生成带引用的答案
- 支持离线评测（语义得分 + RAGAS）与训练数据构造（SFT / Rerank）

## 2. 核心能力

- **PDF 解析与图文信息提取**：文本、页码、图片及标题关联信息一并入库
- **分层切分**：语义切分（服务端）+ 递归字符切分（子块）
- **混合检索**：
  - BM25 稀疏检索（`src/retriever/bm25_retriever.py`）
  - Milvus Hybrid 检索：Dense（BGE-Large-zh-v1.5）+ Sparse（SPLADEv2）
- **重排**：
  - 在线推理默认脚本使用 `BGEM3ReRanker`（`infer.py`）
  - 评估脚本使用 `JinaRerankerV2`（`final_score.py`）
- **答案后处理**：解析引用编号，返回引用页码与相关图片元信息

## 3. 项目结构

```text
XIAOMI_SU7_RAG/
├─ build_index.py                 # 离线：解析文档、切分、建索引（BM25 + Milvus）
├─ infer.py                       # 在线：交互式问答主流程
├─ final_score.py                 # 离线评测（语义得分 + RAGAS）
├─ generate_sft_data.py           # 构造总结/SFT与重排训练数据
├─ requirements.txt
├─ src/
│  ├─ constant.py                 # 全局路径与模型常量
│  ├─ utils.py                    # merge_docs / post_processing
│  ├─ parser/
│  │  ├─ pdf_parse.py             # PDF文本解析、语义切分、Mongo写入
│  │  └─ image_handler.py         # 图片提取与标题关联
│  ├─ retriever/
│  │  ├─ bm25_retriever.py
│  │  ├─ milvus_retriever.py      # Dense + Sparse Hybrid Search
│  │  ├─ tfidf_retriever.py
│  │  ├─ faiss_retriever.py
│  │  └─ qwen3_retriever.py
│  ├─ reranker/
│  │  ├─ bge_m3_reranker.py
│  │  ├─ jina_reranker_v2.py
│  │  ├─ qwen3_reranker.py
│  │  └─ qwen3_reranker_vllm.py
│  ├─ client/
│  │  ├─ llm_local_client.py      # 本地 vLLM OpenAI 兼容接口
│  │  ├─ llm_chat_client.py       # 云端 OpenAI 兼容接口
│  │  ├─ llm_hyde_client.py       # HyDE 查询扩展
│  │  ├─ llm_clean_client.py      # 文档清洗
│  │  ├─ semantic_chunk_client.py # 调用语义切分服务
│  │  └─ mongodb_config.py
│  ├─ server/
│  │  └─ semantic_chunk.py        # FastAPI 语义切分服务
│  ├─ fields/
│  │  ├─ manual_info_mongo.py
│  │  └─ manual_images.py
│  └─ gen_qa/
│     └─ run.py                   # QA 生成/扩展/关键词抽取
└─ data/
   ├─ Xiaomi_SU7_Manual.pdf       # 原始手册
   ├─ processed_docs/             # 解析/切分产物（可重建）
   ├─ saved_index/                # 索引产物（可重建）
   ├─ qa_pairs/                   # QA相关数据（可重建）
   ├─ summary_data/               # SFT总结数据（可重建）
   ├─ rerank_data/                # 重排训练数据（可重建）
   └─ saved_images/               # 提取图片（可重建）
```

## 4. 端到端流程

### 4.1 离线建库流程（`build_index.py`）

1. 读取 PDF（`pdf_parse.load_pdf`）  
2. 可选文档清洗（`llm_clean_client.request_llm_clean`）  
3. 文本切分（`texts_split`）：
   - 调用语义切分服务（`src/server/semantic_chunk.py`）
   - 父块 + 子块写入 MongoDB（`manual_text`）  
4. 构建并持久化索引：
   - BM25：`data/saved_index/bm25retriever.pkl`
   - Milvus：`data/saved_index/milvus.db`

### 4.2 在线问答流程（`infer.py`）

1. 初始化检索器与重排器  
2. 用户输入问题  
3. 双路召回：
   - BM25 TopK
   - Milvus Hybrid TopK（Dense + Sparse）  
4. `merge_docs` 去重并回溯父块  
5. Reranker 精排 TopK  
6. LLM 根据上下文生成答案（带引用编号）  
7. `post_processing` 输出结构化结果：答案、引用页码、相关图片

### 4.3 评测流程（`final_score.py`）

- 读取测试 QA 集，执行检索 + 重排 + 生成
- 计算语义相似度 + 关键词加权得分
- 使用 RAGAS 计算上下文精确率/召回率

### 4.4 训练数据流程

- `src/gen_qa/run.py`：从文档生成 QA、扩写问法、抽取关键词
- `generate_sft_data.py`：构建 `summary_data` 与 `rerank_data`

## 5. 运行环境与依赖

### 5.1 基础依赖

- Python 3.10+（建议）
- MongoDB（默认 `localhost:27017`）
- CUDA 环境（多数检索/重排模型依赖 GPU，CPU 可运行但显著变慢）
- 关键 Python 依赖见 `requirements.txt`（已包含 `pdfplumber`）

### 5.2 环境变量

云端模型接口（`llm_chat_client.py` / `llm_clean_client.py` / `llm_hyde_client.py`）依赖：

```bash
DOUBAO_API_KEY=...
DOUBAO_BASE_URL=...
DOUBAO_MODEL_NAME=...
```

Mongo 可通过以下变量覆盖默认连接：

```bash
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB_NAME=mydatabase
MONGO_USERNAME=
MONGO_PASSWORD=
MONGO_AUTH_SOURCE=admin
```

## 6. 快速开始

> 注意：`src/constant.py` 当前默认路径为 Linux 风格（`/root/autodl-tmp/RAG/`）。在本地运行前请先按你的实际目录修改 `base_dir`。

### 6.1 安装依赖

```bash
pip install -r requirements.txt
```

### 6.2 启动服务

1. 启动 MongoDB  
2. 启动语义切分服务：

```bash
python src/server/semantic_chunk.py
```

3. 若使用本地生成模型，启动 vLLM（与 `src/client/llm_local_client.py` 对齐）：

```bash
vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --max-model-len 8192
```

### 6.3 建索引

```bash
python build_index.py
```

### 6.4 交互式问答

```bash
python infer.py
```

### 6.5 离线评测

```bash
python final_score.py
```

## 7. 可重建数据说明

以下目录内容可通过代码重新生成：

- `data/processed_docs/`
- `data/saved_index/`
- `data/qa_pairs/`
- `data/summary_data/`
- `data/rerank_data/`
- `data/saved_images/`

建议仅保留原始数据与配置，产物按需重建，避免陈旧索引影响效果。

## 8. 已知事项

- 项目部分 prompt 文案仍保留早期模板中的车型文字（如 Model 3），不影响主流程运行，但建议统一为 SU7 场景文案以提升一致性。
- 检索与重排模型较大，首次加载时间较长，显存占用高。
