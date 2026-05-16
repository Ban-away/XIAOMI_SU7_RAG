# 🚗 XIAOMI_SU7_RAG

> 基于小米 SU7 用户手册的**完整 RAG 问答系统**  
> 覆盖文档解析、语义切分、索引构建、检索重排、答案生成、离线评估、训练数据构造全流程

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![vLLM](https://img.shields.io/badge/vLLM-Supported-green?logo=lightning&logoColor=white)](https://github.com/vllm-project/vllm)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-13aa52?logo=mongodb&logoColor=white)](https://www.mongodb.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📑 目录

- [🔧 技术与模型](#-技术与模型)
- [📦 项目结构](#-项目结构)
- [🔄 端到端流程](#-端到端流程)
- [⚡ 快速运行](#-快速运行)
- [⚙️ 配置要点](#️-配置要点)
- [⚠️ 已知限制](#️-已知限制)

---

## 🔧 技术与模型

### 📋 技术栈概览

<table>
<tr>
<td width="25%"><b>🔍 检索</b></td>
<td width="75%">
BM25 / TF-IDF / FAISS / Milvus 混合检索  
<code>langchain_community</code> + <code>pymilvus</code> + <code>torch</code>
</td>
</tr>
<tr>
<td><b>📄 解析</b></td>
<td>
PDF 文本 + 图片抽取  
<code>pdfplumber</code> + <code>PyMuPDF</code> + <code>tiktoken</code>
</td>
</tr>
<tr>
<td><b>✂️ 切分</b></td>
<td>
递归切分 + 语义聚类  
<code>langchain_text_splitters</code> + <code>FastAPI</code> + <code>sentence-transformers</code>
</td>
</tr>
<tr>
<td><b>🗄️ 存储</b></td>
<td>
结构化元数据 + 向量  
<code>MongoDB</code> + <code>Milvus Lite</code>
</td>
</tr>
<tr>
<td><b>📊 重排</b></td>
<td>
跨编码器精排  
<code>BGE</code> / <code>Jina</code> / <code>Qwen3</code>
</td>
</tr>
<tr>
<td><b>🤖 生成</b></td>
<td>
本地推理 + 云端 API  
<code>vLLM</code> (OpenAI 协议) + Doubao
</td>
</tr>
<tr>
<td><b>📈 评估</b></td>
<td>
多维评分  
<code>ragas</code> + <code>text2vec</code> + 自定义指标
</td>
</tr>
</table>

### 🤖 模型清单

| 用途 | 模型 | 配置键 | 调用位置 |
|:---:|:---:|:---:|:---:|
| 🔤 语义切分 | `moka-ai/m3e-small` | `m3e_small_model_path` | `src\server\semantic_chunk.py` |
| 📚 Dense检索 | `BAAI/bge-large-zh-v1.5` | `bge_large_zh_v1_5_model_path` | `src\retriever\milvus_retriever.py` |
| 🎯 Sparse检索 | `naver/splade-cocondenser-ensembledistil` | `splade_v2_model_path` | `src\retriever\milvus_retriever.py` |
| 🔎 向量备选 | `Qwen3-Embedding-0.6B` | `qwen3_embedding_model_path` | `src\retriever\qwen3_retriever.py` |
| ⭐ 在线重排 | `bge-reranker-v2-m3` | `bge_reranker_tuned_model_path` | `infer.py` |
| ✅ 评估重排 | `jina-reranker-v2-base-multilingual` | `jina_reranker_v2_model_path` | `final_score.py` |
| 💬 生成模型 | `Qwen3-8B-Instruct (SFT)` | `qwen3_8b_tune_model_name` | `src\client\llm_local_client.py` |
| ☁️ 云端生成 | `Doubao` | `DOUBAO_MODEL_NAME` | `src\client\llm_chat_client.py` |

#### ☁️ 调用远程 API 的步骤

1. **PDF 文本清洗**（`build_index.py`）  
   `llm_clean_client.py`  
   → 调用豆包 API（`DOUBAO_API_KEY`）  
   → 把解析出的 PDF 原始文本整理成通顺的 Markdown 格式

2. **生成 QA 训练数据**（`src/gen_qa/run.py`）  
   `llm_chat_client.py`  
   → 调用豆包 API  
   → 生成问题、泛化问题、抽关键词、QA 质量打分

3. **HyDE 假设文档扩写**（`final_score.py`，可选，默认 `HYDE=0` 关闭）  
   `llm_hyde_client.py`  
   → 调用豆包 API  
   → 把 query 扩写成假设答案，增强检索效果

4. **RAGas 评估**（`final_score.py`）  
   `ragas` + `langchain-openai`  
   → 调用豆包 API  
   → 用 LLM 评估上下文召回率与精确率

---

#### 🖥️ 本地模型完成的步骤

1. **语义切分**（`src/server/semantic_chunk.py`）  
   模型：`m3e-small`  
   作用：把句子向量化，聚类后语义感知切分

2. **向量化建索引**（`src/retriever/milvus_retriever.py`）  
   模型：`BGE-Large-zh-v1.5`（Dense）+ `SPLADEv2`（Sparse）  
   作用：把文档编码为稠密 + 稀疏向量存入 Milvus

3. **检索召回**（`infer.py`）  
   模型：`BGE-Large-zh-v1.5` + `SPLADEv2`  
   作用：把 query 编码，做混合向量检索 Top-K

4. **重排序**（`infer.py` → `bge_m3_reranker.py`）  
   模型：`Jina-Reranker-v2`  
   作用：Cross-Encoder 精排，筛出最终上下文

5. **最终答案生成**（`src/client/llm_local_client.py`）  
   模型：`Qwen3-8B`（SFT 微调，本地 vLLM 部署）  
   作用：根据召回文档生成答案

6. **评测相似度**（`final_score.py` → `report_score()`）  
   模型：`text2vec-base-chinese`  
   作用：计算预测答案与标准答案的语义相似度

---

#### 🗺️ 全流程一览

```
阶段一：建索引（一次性）
PDF → [☁️ 豆包API 清洗] → [🖥️ m3e-small 语义切分] → [🖥️ BGE-Large 向量化] → Milvus + MongoDB

阶段二：生成训练数据（一次性）
文档 → [☁️ 豆包API 生成QA] → [☁️ 豆包API 泛化问题] → 训练集 / 测试集

阶段三：模型训练（一次性）
训练集 → LLaMA-Factory 微调 Qwen3-8B → 本地模型

阶段四：日常推理（每次问答）
query → [🖥️ BGE 召回] → [🖥️ Jina 精排] → [🖥️ Qwen3-8B 生成答案]
          ↑本地               ↑本地                ↑本地
全程不需要调用任何远程 API

阶段五：离线评估（按需）
预测结果 → [🖥️ text2vec 相似度] → [☁️ 豆包API RAGas] → 综合得分
              ↑本地                    ↑远程API
```
---

## 📦 项目结构

```
XIAOMI_SU7_RAG/
├─ README.md                                        # 本文档
├─ .env.example                                     # 环境变量模板（复制后按需加载）
├─ requirements.txt                                 # 依赖清单
├─ config.ini                                       # 环境变量模板
│
├─ 📂 入口脚本
│  ├─ build_index.py          # 离线建库：解析 → 切分 → 向量 → 存储
│  ├─ infer.py               # 在线问答：检索 → 重排 → 生成
│  ├─ final_score.py         # 离线评估：批量推理 + 多维评分
│  └─ generate_sft_data.py   # 数据构造：QA/Summary/Rerank 数据集
│
├─ 📂 src/  核心业务代码
│  ├─ constant.py            # 全局路径 & 模型配置
│  ├─ utils.py              # 文档合并 & 后处理工具
│  │
│  ├─ 📂 parser/            # PDF 解析与处理
│  │  ├─ pdf_parse.py       # PDF 文本/表格/布局抽取 (pdfplumber)
│  │  └─ image_handler.py   # 图片检测/抽取/存储 (PyMuPDF)
│  │
│  ├─ 📂 client/            # 模型客户端
│  │  ├─ llm_chat_client.py         # 云端 API (Doubao)
│  │  ├─ llm_local_client.py        # 本地推理 (vLLM)
│  │  ├─ llm_clean_client.py        # 文本清洗
│  │  ├─ llm_hyde_client.py         # HyDE 问题扩写
│  │  ├─ semantic_chunk_client.py   # 语义切分调用
│  │  └─ mongodb_config.py          # Mongo 连接配置
│  │
│  ├─ 📂 server/            # 后台服务
│  │  └─ semantic_chunk.py  # FastAPI 语义聚类服务
│  │
│  ├─ 📂 retriever/         # 检索器模块
│  │  ├─ bm25_retriever.py       # 稀疏检索 (BM25)
│  │  ├─ tfidf_retriever.py      # 稀疏检索 (TF-IDF)
│  │  ├─ faiss_retriever.py      # 密集向量 (FAISS)
│  │  ├─ qwen3_retriever.py      # Qwen3 向量
│  │  ├─ milvus_retriever.py     # 混合检索 (Dense+Sparse)
│  │  └─ retriever.py            # 检索器基类
│  │
│  ├─ 📂 reranker/         # 重排器模块
│  │  ├─ bge_m3_reranker.py        # BGE 跨编码器
│  │  ├─ jina_reranker_v2.py       # Jina 跨编码器
│  │  ├─ qwen3_reranker.py         # Qwen3 轻量重排
│  │  └─ qwen3_reranker_vllm.py    # Qwen3 vLLM 多卡重排
│  │
│  ├─ 📂 fields/           # 数据结构 (Pydantic)
│  │
│  └─ 📂 gen_qa/           # QA 与训练数据生成
│     └─ run.py            # QA 生成 & 问题扩写
│
├─ 📂 data/  数据与产物
│  ├─ Xiaomi_SU7_Manual.pdf                  # 原始手册 (PDF 源)
│  ├─ stopwords.txt                          # BM25 停用词表
│  │
│  ├─ 📂 processed_docs/                     # 处理后文档 (Pickle)
│  │  ├─ raw_docs.pkl           # 原始文本
│  │  ├─ clean_docs.pkl         # 清洗后文本 (可选)
│  │  └─ split_docs.pkl         # 切分后文档块
│  │
│  ├─ 📂 saved_index/                        # 索引产物
│  │  ├─ bm25retriever.pkl      # BM25 索引
│  │  ├─ milvus.db/             # Milvus 混合索引 (SQLite)
│  │  └─ faiss*.bin             # FAISS 向量索引 (可选)
│  │
│  ├─ 📂 qa_pairs/                           # QA 与评测数据
│  │  ├─ qa_pairs_*.json        # 生成的 QA 对
│  │  ├─ test_qa_pair_verify.json            # 评估输入
│  │  └─ test_qa_pair_pred.json              # 评估输出
│  │
│  ├─ 📂 summary_data/                       # Summary 训练数据
│  │  ├─ train.json / val.json / test.json
│  │
│  ├─ 📂 rerank_data/                        # Rerank 训练数据
│  │  ├─ train.json / val.json / test.json
│  │
│  ├─ 📂 saved_images/                       # PDF 抽取图片
│  │  └─ page_*.png / figure_*.png
│  │
│  ├─ 📂 mongodb/                            # 本地 Mongo 数据
│  │  ├─ data/                 # 数据文件
│  │  └─ log/                  # Mongo 日志
│  │
│  └─ 📂 ut/                                 # 单元测试文本
│
├─ 📂 log/                                       # 运行日志
│  ├─ semantic_chunk.log        # 语义切分服务日志
│  └─ qwen3-7b.log              # vLLM 推理日志
│
├─ 📂 models/                                    # 本地模型缓存
│  ├─ m3e-small/
│  ├─ bge-large-zh-v1.5/
│  └─ ... (其他下载模型)
│
├─ 📂 mongodb-7.0.20/                            # Mongo 服务器二进制
│  ├─ bin/ mongod
│  └─ ... (MongoDB 核心文件)
│
├─ 📂 LLaMA-Factory-main/                        # 训练框架 (子项目)
│  ├─ vllm_serve.sh             # vLLM 启动脚本
│  ├─ output/
│  │  └─ qwen3_lora_sft_int4/   # 微调模型产物
│  └─ ... (完整训练工具链)
│
├─ 📂 RAG-Retrieval/                             # 检索实验框架 (子项目)
│  └─ ... (检索模块训练与评估)
│
└─ 📂 deploy/                                    # 部署脚本
   ├─ auto_vllm_server.py       # 自动识别单/多卡启动脚本
   └─ download_models.py        # 一键下载项目公开模型（core/all）
```

#### 📌 Git 跟踪状态说明

| 目录 | 状态 | 说明 |
|:---:|:---:|:---:|
| `src/` | ✅ 已跟踪 | 核心业务代码 |
| `LLaMA-Factory-main/` | ✅ 已跟踪 | 训练/量化框架 |
| `RAG-Retrieval/` | ✅ 部分跟踪 | 检索训练代码 |
| `data/processed_docs/` | ❌ 未跟踪 | 动态生成 |
| `data/saved_index/` | ❌ 未跟踪 | 动态生成 |
| `data/qa_pairs/` | ❌ 未跟踪 | 动态生成 |
| `log/` | ❌ 未跟踪 | 运行日志 |
| `models/` | ❌ 未跟踪 | 本地模型缓存 |
| `mongodb-7.0.20/` | ❌ 未跟踪 | 本地服务二进制 |

---

## 🔄 端到端流程


### 📍 第 0 步：环境与服务准备

| 步骤 | 命令（示例） | 说明 |
|:---:|:---|:---|
| **1️⃣ 安装 Python 依赖** | `pip install -r requirements.txt` | 安装项目运行所需的 Python 包 |
| **2️⃣ 准备并导出微调模型（必需）** | 见下方“生成 Qwen3 SFT 与 Int4” | 训练/合并/导出得到 `LLaMA-Factory-main/output/qwen3_lora_sft` 和量化后的 `..._int4` |
| **3️⃣ 启动语义切分服务** | `python src/server/semantic_chunk.py` | FastAPI + Uvicorn，用于语义切分 API |
| **4️⃣ 启动 vLLM 推理服务** | `python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000` | vLLM（本地推理，require: output 下存在量化模型） |
| **5️⃣ 启动 MongoDB（可用系统安装或捆绑二进制）** | 参见下方“MongoDB 启动示例” | 官方安装或使用仓库内的 `mongodb-7.0.20` 二进制 |

> ⚠️ **提前创建目录**：`data/processed_docs`、`data/saved_index`、`data/qa_pairs` 等，脚本默认不会创建这些目录。

---

### 📍 第 1 步：离线建库（`build_index.py`）

| 子步骤 | 函数 | 工具 | 输入 | 产物 |
|:---|:---|:---|:---|:---|
| **1.1 PDF 解析** | `load_pdf()`<br/>`src\parser\pdf_parse.py` | pdfplumber + PyMuPDF | `data\Xiaomi_SU7_Manual.pdf` | 文本块 + 图片 |
| **1.2 可选清洗** | `request_llm_clean()`<br/>`src\client\llm_clean_client.py` | OpenAI SDK | 原始文本 | `data\processed_docs\clean_docs.pkl` |
| **1.3 语义切分** | `request_semantic_chunk()`<br/>`src\client\semantic_chunk_client.py`<br/>→ FastAPI `/v1/semantic-chunks` | FastAPI + sklearn 聚类 | 文本块 | `data\processed_docs\split_docs.pkl` |
| **1.4 入 MongoDB** | `save_2_mongo()`<br/>`src\parser\pdf_parse.py` | pymongo | 切分块 + 元数据 | `data\mongodb\data\manual_text` 集合 |
| **1.5 BM25 索引** | `BM25(...).get_BM25_retriever()`<br/>`src\retriever\bm25_retriever.py` | langchain_community | 文本块 + jieba | `data\saved_index\bm25retriever.pkl` |
| **1.6 Milvus 混合索引** | `MilvusRetriever.save_vectorstore()`<br/>`src\retriever\milvus_retriever.py` | pymilvus + transformers | 文本块 + 向量模型 | `data\saved_index\milvus.db` |

---

### 📍 第 2 步：在线问答（`infer.py`）

| 步骤 | 函数 | 工具 | 处理 |
|:---|:---|:---|:---|
| **2.1 加载模型** | 构造 BM25 / Milvus / BGEM3 实例 | pickle + pymilvus | 模型预热 |
| **2.2 BM25 召回** | `retrieve_topk()`<br/>`src\retriever\bm25_retriever.py` | jieba + BM25 | Top-K 候选文档 |
| **2.3 混合召回** | `retrieve_topk()`<br/>`src\retriever\milvus_retriever.py` | BGE Dense + SPLADE Sparse | Top-K 候选文档 |
| **2.4 合并去重** | `merge_docs()`<br/>`src\utils.py` | pymongo 回表 | 去重后候选 |
| **2.5 精排** | `rank()`<br/>`src\reranker\bge_m3_reranker.py` | Cross-Encoder | 最终 Top-K 上下文 |
| **2.6 答案生成** | `request_chat()`<br/>`src\client\llm_local_client.py` | vLLM (OpenAI 协议) | 流式答案文本 |
| **2.7 后处理** | `post_processing()`<br/>`src\utils.py` | 正则 + metadata | `answer` + `cite_pages` + `related_images` |

---

### 📍 第 3 步：离线评估（`final_score.py`）

| 步骤 | 函数 | 工具 | 输入/输出 |
|:---|:---|:---|:---|
| **3.1 批量推理** | 主循环 | 同检索/重排/生成链路 | 输入：`data\qa_pairs\test_qa_pair_verify.json` |
| **3.2 结果保存** | `json.dump()` | Python json | 输出：`data\qa_pairs\test_qa_pair_pred.json` |
| **3.3 语义评分** | `report_score()`<br/>`final_score.py` | text2vec + Jaccard | 日志：平均评分 |
| **3.4 RAGAS 指标** | `evaluate()`<br/>`final_score.py` | ragas + langchain-openai | 日志：上下文召回/精确率 |

---

### 📍 第 4 步：数据构造

| 脚本 | 函数 | 作用 | 产物 |
|:---|:---|:---|:---|
| `src\gen_qa\run.py` | `gen_qa()` / `chat()` | QA 生成 + 问题扩写 + 关键词抽取 | `data\qa_pairs\qa_pairs_*.json` |
| `generate_sft_data.py` | 主流程 | Summary & Rerank 数据集构造 | `data\summary_data\*.json`<br/>`data\rerank_data\*.json` |

---

## ⚙️ 配置要点

### 📌 环境变量

创建 `.env` 或在 Shell 中设置：

```bash
# Doubao API 配置 (云端)
export DOUBAO_API_KEY=sk-xxx
export DOUBAO_BASE_URL=https://api.doubao.com/v1
export DOUBAO_MODEL_NAME=doubao-pro-4k

# MongoDB 配置
export MONGO_HOST=localhost
export MONGO_PORT=27017
export MONGO_DB_NAME=mydatabase
export MONGO_USERNAME=
export MONGO_PASSWORD=
export MONGO_AUTH_SOURCE=admin
```

可直接参考 `.env.example` 中的完整变量模板。

### 🗂️ 核心路径配置

编辑 `src\constant.py` 中的路径（当前默认为 Linux 路径）：

```python
# ❌ Linux (Azure autodl-tmp 环境)
base_dir = "/root/autodl-tmp/XIAOMI_SU7_RAG/"

# ✅ Windows (本地开发)
base_dir = "D:\\Development\\Exercise\\0_personal_project\\XIAOMI_SU7_RAG\\"

# ✅ Linux (本地开发)
base_dir = "/home/user/XIAOMI_SU7_RAG/"
```

---

## ⚡ 快速运行

### 📥 环境准备

```bash
# 1. 克隆项目
git clone <repo-url>
cd XIAOMI_SU7_RAG

# 2. 安装依赖
pip install -r requirements.txt
# 或切换成国内阿里云加速源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip install av

# 3. 下载主流程所需模型（默认 core 预设）
# 设置 HF 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
python deploy/download_models.py

# 4. 创建必要目录结构
mkdir -p data/{processed_docs,saved_index,qa_pairs,summary_data,rerank_data,saved_images,mongodb/{data,log}}
mkdir -p log models
```

### 生成 Qwen3 SFT 与 Int4

项目期望在 `LLaMA-Factory-main/output/` 下存在导出模型（例如 `qwen3_lora_sft`）以及量化后模型 `qwen3_lora_sft_int4`。示例流程：

1. 安装 LLaMA-Factory 依赖

```bash
cd LLaMA-Factory-main
pip install -r requirements.txt
pip install -e .
```

2. 先跑 LoRA 训练，生成 `saves/qwen3-8b/lora/sft/`

```bash
cd LLaMA-Factory-main
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

3. 放好基础模型，并确认 YAML 里的 `model_name_or_path` 指向它

```bash
cd LLaMA-Factory-main
ls /root/autodl-tmp/XIAOMI_SU7_RAG/models/Qwen3-8B/
sed -n '1,20p' examples/train_lora/qwen3_lora_sft.yaml
```

4. 导出合并模型，生成 `output/qwen3_lora_sft/`

```bash
cd LLaMA-Factory-main
bash export.sh
```

5. 安装并启动 MongoDB（生成数据前必需）
```bash
# 下载 MongoDB 7.0.20
wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-7.0.20.tgz

# 解压并移动到安装目录
tar -zxvf mongodb-linux-x86_64-ubuntu2204-7.0.20.tgz
mv mongodb-linux-x86_64-ubuntu2204-7.0.20 /usr/local/mongodb

# 创建数据和日志目录
mkdir -p /data/db
mkdir -p /var/log/mongodb

# 启动 MongoDB（后台运行）
/usr/local/mongodb/bin/mongod --dbpath /data/db --logpath /var/log/mongodb/mongod.log --bind_ip_all --fork
```

6. 生成小米 SU7 的数据（QA、训练集、测试集）
```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG
# 先生成或加载 split_docs.pkl（run build_index.py）
# 注意：首次运行可能需要修复 setuptools 版本
pip install --force-reinstall setuptools==69.0.0
python build_index.py

# 生成所有 QA 数据
# 默认模式：跳过已存在的文件，不会覆盖
python generate_all_data.py

# 可选参数：
# --force    : 强制重新生成所有文件（覆盖已存在的）
# --skip-expand : 跳过扩展 QA 生成（加快速度）
python generate_all_data.py --skip-expand  # 快速模式
python generate_all_data.py --force        # 强制覆盖模式
```

7. 生成 `summary_train.json`、`summary_test.json`
```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG
# 根据 QA 生成 summary/rerank 数据
python generate_sft_data.py

# 复制到 LLaMA-Factory 目录
cp data/summary_data/train.json LLaMA-Factory-main/data/summary_train.json
cp data/summary_data/test.json LLaMA-Factory-main/data/summary_test.json
```

8. 生成量化模型（`output/qwen3_lora_sft_int4/`）
```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG/LLaMA-Factory-main
pip install llmcompressor
python awq_quant.py
ls -l output/qwen3_lora_sft_int4
```

说明：
- 使用 llmcompressor（vLLM 维护）进行 AWQ 量化
- 自动使用训练数据作为校准集
- 输出量化模型到 `output/qwen3_lora_sft_int4/`

注意：量化前需先完成第 4 步，确保 `output/qwen3_lora_sft/` 已存在。

9. 启动 vLLM 推理服务（新终端）
```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG
python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000 --quantization none
```

说明：
- 使用预量化的 AWQ 模型
- `--quantization none` 表示不使用额外量化

10. 生成 `summary_test_pred.json`
```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG/LLaMA-Factory-main
python predict.py
```

11. 校验 summary 文件
```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG/LLaMA-Factory-main
ls -l data/summary_train.json data/summary_test.json data/summary_test_pred.json
```

### 📊 数据文件说明

#### 项目数据目录 (`data/`)

| 文件路径 | 作用 | 记录数 | 生成时机 |
|:---|:---|:---|:---|
| `data/qa_pairs/qa_pair.json` | 原始 QA 对（每个语义块生成5个问题） | 2,616 | Step 1 |
| `data/qa_pairs/expand_qa_pair.json` | 扩展 QA 对（每个问题生成5个同义问法） | 12,288 | Step 2 |
| `data/qa_pairs/train_qa_pair.json` | 训练集（90%数据） | 21,727 | Step 3 |
| `data/qa_pairs/test_qa_pair.json` | 测试集（10%数据，含关键词） | 2,399 | Step 3 |
| `data/qa_pairs/test_keywords_pair.json` | 测试集答案关键词标注 | 2,399 | Step 4 |
| `data/qa_pairs/test_qa_pair_verify.json` | 评估输入文件 | 2,399 | Step 5 |
| `data/qa_pairs/train_data.json` | SFT 训练数据（含检索上下文） | 21,727 | Step 6 |
| `data/summary_data/train.json` | 摘要训练集 | 19,996 | `generate_sft_data.py` |
| `data/summary_data/test.json` | 摘要测试集 | 1,731 | `generate_sft_data.py` |
| `data/rerank_data/train.json` | 重排训练集 | 48,999 | `generate_sft_data.py` |
| `data/rerank_data/dev.json` | 重排验证集 | 1,000 | `generate_sft_data.py` |
| `data/rerank_data/test.json` | 重排测试集 | 1,234 | `generate_sft_data.py` |

#### LLaMA-Factory 训练数据 (`LLaMA-Factory-main/data/`)

| 文件路径 | 作用 | 来源 |
|:---|:---|:---|
| `LLaMA-Factory-main/data/summary_train.json` | 摘要训练数据（用于 LLaMA-Factory 训练） | 复制自 `data/summary_data/train.json` |
| `LLaMA-Factory-main/data/summary_test.json` | 摘要测试数据（用于 LLaMA-Factory 评估） | 复制自 `data/summary_data/test.json` |
| `LLaMA-Factory-main/data/summary_test_pred.json` | 摘要测试预测结果（模型推理输出） | 运行 `predict.py` 生成 |
| `LLaMA-Factory-main/data/rerank_train.json` | 重排训练数据（用于交叉熵损失训练） | 复制自 `data/rerank_data/train.json` |
| `LLaMA-Factory-main/data/rerank_dev.json` | 重排验证数据（用于训练验证） | 复制自 `data/rerank_data/dev.json` |
| `LLaMA-Factory-main/data/rerank_test.json` | 重排测试数据（用于离线评估） | 复制自 `data/rerank_data/test.json` |

### 🚀 启动在线服务

```bash
# 终端 1：启动 MongoDB（必需，需先安装）
/usr/local/mongodb/bin/mongod --dbpath /data/db --logpath /var/log/mongodb/mongod.log --bind_ip_all --fork

# 终端 2：启动语义切分服务
python src/server/semantic_chunk.py

# 终端 3：启动 vLLM（自动识别单卡/多卡）
python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000

# 终端 4：构建索引（如果尚未构建）
pip install --force-reinstall setuptools==69.0.0
python build_index.py

# 终端 5：在线问答
python infer.py
```

#### vLLM 启动参数说明

```bash
python deploy/auto_vllm_server.py \
  --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75

# 检测到 1 张 GPU   → 不设置 --tensor-parallel-size (单卡模式)
# 检测到多张 GPU    → 自动添加 --tensor-parallel-size=<GPU数量> (多卡张量并行)
```

支持透传其他 vLLM 参数：

```bash
python deploy/auto_vllm_server.py \
  --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 \
  -- --max-num-seqs 16 --enforce-eager
```

### 📊 离线评估

```bash
python final_score.py
```

评估结果将输出到控制台，包括 BLEU 分数、ROUGE 分数和关键词匹配准确率。

---

## ⚠️ 已知限制

| 限制 | 说明 | 解决方案 |
|:---:|:---|:---|
| 📁 目录创建 | 脚本不自动创建 `mkdir` | 需提前手动创建必要目录结构 |
| 🤖 模型加载 | 首次加载耗时较长 (可达 5-10 分钟) | 模型会缓存到 `models/` 目录 |
| 🔑 环境变量 | 需设置 API Key 与 Mongo 连接 | 参考 **配置要点** 章节 |
| 🛣️ 路径耦合 | `src\constant.py` 中硬编码路径 | 本地运行前需改为实际路径 |
| 📦 子项目 | `LLaMA-Factory-main` 与 `RAG-Retrieval` 较大 | 可独立配置或跳过 |

---

## 📚 核心模块说明

<details>
<summary><b>🔍 检索模块 (Retriever)</b></summary>

- **BM25 稀疏检索**：快速粗排，基于词频
- **Milvus 混合检索**：Dense (BGE) + Sparse (SPLADE)，适合中文
- **FAISS 密集向量**：备选方案，支持多种向量编码
- 所有检索器实现统一接口 `retrieve_topk(query, top_k) → List[Doc]`

</details>

<details>
<summary><b>📊 重排模块 (Reranker)</b></summary>

- **BGE-M3 跨编码器**：在线使用，精准重排
- **Jina Reranker V2**：评估使用，支持多语言
- **Qwen3 轻量重排**：更快速，可选多卡 vLLM
- 所有重排器实现统一接口 `rank(query, docs, top_k) → List[RankedDoc]`

</details>

<details>
<summary><b>🤖 生成模块 (LLM Client)</b></summary>

- **本地推理** (`llm_local_client.py`)：vLLM 推理服务 (OpenAI 兼容)
- **云端 API** (`llm_chat_client.py`)：Doubao 等云端模型
- **问题扩写** (`llm_hyde_client.py`)：HyDE 提升召回
- **文本清洗** (`llm_clean_client.py`)：可选预处理

</details>

---

## 📄 许可证

MIT License - 详见 LICENSE 文件

---

<p align="center">
  <b>🚗 Smart EV, Strong RAG System 🚗</b>  
  <br/>
  <em>为小米 SU7 量身定制的高效 RAG 问答系统</em>
</p>