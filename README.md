# 🚗 XIAOMI_SU7_RAG

> 基于小米 SU7 用户手册的**完整 RAG 问答系统**  
> 覆盖文档解析、语义切分、索引构建、检索重排、答案生成、离线评估、RL训练全流程  
> Search-R1 强化学习 + WebWalker 垂直搜索：模型自主决定何时检索、检索什么、是否深度阅读页面

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
- [⚙️ 配置要点](#️-配置要点)
- [⚡ 快速运行](#-快速运行)
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
<code>BGE-Reranker-v2-MiniCPM</code> / <code>Qwen3</code>
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
多维评分 + 性能对比<br/>
<code>ragas</code> + <code>text2vec</code> + 自定义指标<br/>
<span style="color:green">语义相似度 + 关键词加权得分：0.8965</span>
</td>
</tr>
<tr>
<td><b>🧠 RL 训练</b></td>
<td>
TRL GRPOTrainer + PEFT LoRA
<code>trl</code> + <code>peft</code> + 自定义 6 维奖励函数
</td>
</tr>
<tr>
<td><b>⚡ 性能</b></td>
<td>
TTFT 均值：52 ms<br/>
单卡吞吐率：669 token/s<br/>
8卡吞吐率：~4,550 token/s<br/>
<code>vLLM</code> + <code>AWQ INT4</code> 量化（提升 43.8%）
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
| ⭐ 在线重排 | `bge-reranker-v2-minicpm-layerwise` | `bge_reranker_minicpm_path` | `infer.py` |
| ✅ 评估重排 | `bge-reranker-v2-minicpm-layerwise` | `bge_reranker_minicpm_path` | `final_score.py` |
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

3. **HyDE 假设文档扩写**（`final_score.py`，可选，默认 `HYDE=1` 开启）  
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

4. **重排序**（`infer.py` → `minicpm_reranker.py`）  
   模型：`BGE-Reranker-v2-MiniCPM-Layerwise`  
   作用：Layerwise Cross-Encoder 精排，筛出最终上下文

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
query → [🖥️ BGE 召回] → [🖥️ MiniCPM 精排] → [🖥️ Qwen3-8B 生成答案]
          ↑本地                ↑本地                  ↑本地
全程不需要调用任何远程 API

阶段五：离线评估（按需）
预测结果 → [🖥️ text2vec 相似度] → [☁️ 豆包API RAGas] → 综合得分
              ↑本地                    ↑远程API

阶段六：RL 强化学习（Search-R1 + WebWalker 垂直搜索，进阶）
问题库 → [🖥️ 本地+网络检索] → [☁️ 豆包API 生成轨迹] → SFT warm-up → GRPO 强化学习（TRL + PEFT）
                                    ↑边生成边检索                    ↑6维自定义奖励函数
模型自主决定何时检索、检索什么，还可对搜索结果页面深度阅读（垂直探索），而非固定管线检索→生成
训练数据 = 网络兜底轨迹（~79条）+ 本地可答轨迹（全量 QA 对，去重后数千条）
```
---

## 📦 项目结构

```
XIAOMI_SU7_RAG/
├─ README.md                                        # 本文档
├─ .env.example                                     # 环境变量模板（复制后按需加载）
├─ requirements.txt                                 # 依赖清单
├─ config.ini                                       # 环境变量模板
├─ Dockerfile                                       # 主应用容器镜像
├─ docker-compose.yml                               # 多服务编排（MongoDB + vLLM + 应用）
│
├─ 📂 入口脚本
│  ├─ build_index.py            # 离线建库：解析 → 切分 → 向量 → 存储
│  ├─ infer.py                  # 在线问答：检索 → 重排 → 生成
│  ├─ final_score.py            # 离线评估：批量推理 + 多维评分
│  ├─ generate_all_data.py      # 全量数据生成：QA / 扩展 / 训练集 / 测试集
│  ├─ generate_sft_data.py      # 数据构造：Summary / Rerank 数据集
│  ├─ evaluate_parse_quality.py # 文档解析质量评估报告
│  └─ check_training_data.py    # 训练数据质量检查
│
├─ 📂 src/  核心业务代码
│  ├─ constant.py            # 全局路径 & 模型配置
│  ├─ utils.py              # 文档合并 & 后处理工具
│  │
│  ├─ 📂 parser/            # PDF 解析与处理
│  │  ├─ pdf_parse.py       # PDF 文本/表格/布局抽取 (pdfplumber)
│  │  ├─ image_handler.py   # 图片检测/抽取/存储 (PyMuPDF)
│  │  └─ parse_evaluator.py # 文档解析质量评估工具
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
│  │  ├─ minicpm_reranker.py       # MiniCPM 跨编码器（默认）
│  │  ├─ qwen3_reranker.py         # Qwen3 轻量重排
│  │  └─ qwen3_reranker_vllm.py    # Qwen3 vLLM 多卡重排
│  │
│  ├─ 📂 fields/           # 数据结构 (Pydantic)
│  │
│  ├─ 📂 rl/               # RL 强化学习模块（Search-R1 + WebWalker 垂直搜索）
│  │  ├─ __init__.py              # 模块说明
│  │  ├─ web_reader.py           # 网页内容抓取器（垂直搜索基础设施）
│  │  ├─ data_builder.py         # 网络兜底轨迹生成器（含多跳页面阅读）
│  │  ├─ build_local_trajectories.py  # 本地可答轨迹生成器（BM25 召回）
│  │  ├─ format_converter.py     # 训练数据格式转换器（含自动修复）
│  │  ├─ reward_model.py         # 6 维度奖励函数（兼容 TRL 接口）
│  │  ├─ environment.py          # 工具调用路由环境（local/web/read_page）
│  │  ├─ train_grpo.py           # GRPO 训练入口（TRL + PEFT，非 LLaMA-Factory）
│  │  ├─ batch_eval.py           # RL 模型批量评测（与 baseline 对比）
│  │  └─ infer_rl.py             # RL 增强推理（边生成边检索 + 深度阅读）
│  │
│  └─ 📂 gen_qa/           # QA 与训练数据生成
│     ├─ run.py            # QA 生成 & 问题扩写
│     └─ qa_filter.py      # QA 质量过滤工具
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
│  ├─ 📂 rl_data/                            # RL 强化学习数据
│  │  ├─ web_fallback_questions.json         # 网络兜底问题库（79条/10类）
│  │  ├─ web_fallback_trajectories.json      # 网络兜底原始轨迹
│  │  ├─ web_fallback_trajectories_sft.json  # 网络兜底 SFT 格式
│  │  ├─ web_fallback_trajectories_grpo.jsonl # 网络兜底 GRPO 格式
│  │  ├─ local_trajectories.json             # 本地可答轨迹数据（~100条）
│  │  ├─ local_trajectories_sft.json         # 本地可答 SFT 格式
│  │  ├─ local_trajectories_grpo.jsonl       # 本地可答 GRPO 格式
│  │  ├─ combined_trajectories_sft.json      # 合并 SFT 格式（网络+本地）
│  │  ├─ combined_sft_train.json             # 合并 SFT 训练集（80% 拆分）
│  │  ├─ combined_sft_eval.json              # 合并 SFT 评估集（20% 拆分）
│  │  ├─ combined_trajectories_grpo.jsonl    # 合并 GRPO 格式（网络+本地，训练用）
│  │  ├─ rl_eval_results.json               # RL 模型评测结果
│  │  └─ rl_eval_ckpt.jsonl                 # 评测断点续传检查点
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
├─ 📂 deploy/                                    # 部署脚本
│  ├─ auto_vllm_server.py       # 自动识别单/多卡启动脚本
│  ├─ download_models.py        # 一键下载项目公开模型（core/all）
│  └─ baseline_gpt4o.py         # 基线对比测试
│
└─ 📂 configs/                                   # RL 训练配置
   ├─ qwen3_lora_rl_sft.yaml    # SFT warm-up 配置
   └─ qwen3_lora_grpo.yaml     # GRPO 强化学习配置
```

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
| **2.4 WRRF 粗排** | `merge_docs()`<br/>`src\utils.py` | WRRF (Weighted Reciprocal Rank Fusion) | 加权融合 BM25 + Milvus 结果 |
| **2.5 MiniCPM 精排** | `rank()`<br/>`src\reranker\minicpm_reranker.py` | Cross-Encoder | 最终 Top-K 上下文 |
| **2.6 答案生成** | `request_chat()`<br/>`src\client\llm_local_client.py` | vLLM (OpenAI 协议) | 流式答案文本 |
| **2.7 后处理** | `post_processing()`<br/>`src\utils.py` | 正则 + metadata | `answer` + `cite_pages` + `related_images` |

---

### 📍 第 3 步：离线评估（`final_score.py`）

| 步骤 | 函数 | 工具 | 输入/输出 |
|:---|:---|:---|:---|
| **3.1 批量推理** | 主循环 | WRRF 粗排 + MiniCPM 精排 + vLLM 生成 | 输入：`data\qa_pairs\test_qa_pair_verify.json` |
| **3.2 无答案重试** | `process_one()`<br/>`final_score.py` | top-3 文档重新生成 | 减少"无答案"误判 |
| **3.3 结果保存** | `json.dump()` | Python json | 输出：`data\qa_pairs\test_qa_pair_pred.json` |
| **3.4 语义评分** | `report_score()`<br/>`final_score.py` | text2vec + 关键词加权 | 日志：平均评分 |
| **3.5 RAGAS 指标** | `evaluate()`<br/>`final_score.py` | ragas + langchain-openai | 日志：上下文召回/精确率 |

---

### 📍 第 4 步：数据构造

| 脚本 | 函数 | 作用 | 产物 |
|:---|:---|:---|:---|
| `src\gen_qa\run.py` | `gen_qa()` / `chat()` | QA 生成 + 问题扩写 + 关键词抽取 | `data\qa_pairs\qa_pairs_*.json` |
| `generate_sft_data.py` | 主流程 | Summary & Rerank 数据集构造 | `data\summary_data\*.json`<br/>`data\rerank_data\*.json` |

---

## ⚙️ 配置要点

### 📌 环境变量

复制 `.env.example` 为 `.env` 并填入实际值：

```bash
cp .env.example .env
# 编辑 .env 填入你的实际配置
```

或直接在 Shell 中设置：

```bash
# ── 🔴 必需配置 ──

# 豆包 LLM API 配置 (云端生成模型)
# 获取地址: https://console.volcengine.com/ark/
export DOUBAO_API_KEY="sk-your-api-key-here"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export DOUBAO_MODEL_NAME="ep-20250206xxxxx"  # 替换为你的部署 ID

# 项目根目录（推荐设置，避免手动改代码）
# Windows 用户必须设置
export RAG_BASE_DIR="/root/autodl-tmp/XIAOMI_SU7_RAG/"
# 或使用备选变量：
# export XIAOMI_RAG_HOME="/root/autodl-tmp/XIAOMI_SU7_RAG/"

# ── 🟡 可选配置（有默认值，可按需覆盖）──

# SerpAPI 网络搜索配置 (RL 网络兜底检索使用)
# 获取地址: https://serpapi.com/manage-api-key (免费额度 100 次/月)
# 未设置时自动降级为豆包 LLM 模拟搜索
export SERPAPI_KEY="your-serpapi-key-here"

# MongoDB 配置 (默认本地 localhost:27017)
export MONGO_HOST="localhost"
export MONGO_PORT="27017"
export MONGO_DB_NAME="mydatabase"
export MONGO_USERNAME=""
export MONGO_PASSWORD=""
export MONGO_AUTH_SOURCE="admin"

# vLLM 服务配置 (默认 http://localhost:8000/v1)
export VLLM_BASE_URL="http://localhost:8000/v1"

# 语义切分服务配置 (默认 http://localhost:6000/v1/semantic-chunks)
export SEMANTIC_CHUNK_URL="http://localhost:6000/v1/semantic-chunks"
```

> 完整变量说明参见 `.env.example`。

### 🗂️ 核心路径配置

`src/constant.py` 会自动从环境变量读取项目根目录，**无需手动编辑代码**：

```python
# 优先级：RAG_BASE_DIR > XIAOMI_RAG_HOME > 硬编码默认值
# 推荐通过 .env 或环境变量设置：
export RAG_BASE_DIR="/root/autodl-tmp/XIAOMI_SU7_RAG/"

# Windows 示例：
# $env:RAG_BASE_DIR="D:\Development\Exercise\0_personal_project\XIAOMI_SU7_RAG\"
```

> ⚠️ 若未设置环境变量，将 fallback 到硬编码路径 `/root/autodl-tmp/XIAOMI_SU7_RAG/`。

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
mkdir -p data/{processed_docs,saved_index,qa_pairs,summary_data,rerank_data,rl_data,saved_images,mongodb/{data,log}}
mkdir -p log models configs
```

### 生成 Qwen3 SFT 与 Int4

项目期望在 `LLaMA-Factory-main/output/` 下存在导出模型（例如 `qwen3_lora_sft`）以及量化后模型 `qwen3_lora_sft_int4`。示例流程：

1. 安装 LLaMA-Factory 依赖

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG/LLaMA-Factory-main
pip install -r requirements.txt
pip install -e .
```

2. 安装并启动 MongoDB（生成数据前必需）

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

3. 生成小米 SU7 的数据（QA、训练集、测试集）

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG

# 启动语义切分服务（build_index.py 依赖此服务）
python src/server/semantic_chunk.py

# 先生成或加载 split_docs.pkl（run build_index.py）
# 注意：首次运行可能需要修复 setuptools 版本
pip install --force-reinstall setuptools==69.0.0
python build_index.py

# 评估文档解析质量（可选）
# 生成解析准确率报告，包含文本保留率、切分质量等指标
python evaluate_parse_quality.py

# 生成所有 QA 数据
# 默认模式：跳过已存在的文件，不会覆盖
python generate_all_data.py

# 可选参数：
# --force    : 强制重新生成所有文件（覆盖已存在的）
# --skip-expand : 跳过扩展 QA 生成（加快速度）
python generate_all_data.py --skip-expand  # 快速模式
python generate_all_data.py --force        # 强制覆盖模式
```

4. 生成 `summary_train.json`、`summary_test.json`

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG
# 根据 QA 生成 summary/rerank 数据
python generate_sft_data.py

# 复制到 LLaMA-Factory 目录
cp data/summary_data/train.json LLaMA-Factory-main/data/summary_train.json
cp data/summary_data/test.json LLaMA-Factory-main/data/summary_test.json
```

5. LoRA 训练，生成 `saves/qwen3-8b/lora/sft/`

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG/LLaMA-Factory-main
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

**训练结果记录：**

| 指标 | 值 |
|------|------|
| 训练轮数 | 3.0 epoch |
| 训练损失 | 0.3644 |
| 评估损失 | 0.1533 |
| 评估样本数 | 1000 条 |
| 训练耗时 | 49分41秒 |

> **说明**：评估损失（0.1533）低于训练损失（0.3644），表明模型未过拟合，训练效果良好。

6. 导出合并模型，生成 `output/qwen3_lora_sft/`

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG/LLaMA-Factory-main
bash export.sh
```

7. 生成量化模型（`output/qwen3_lora_sft_int4/`）

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG/LLaMA-Factory-main
python awq_quant.py
ls -l output/qwen3_lora_sft_int4
```

注意：量化前需先完成第 6 步，确保 `output/qwen3_lora_sft/` 已存在。

8. 启动 vLLM 推理服务（新终端）

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG
python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000
```

9. 生成 `summary_test_pred.json`（含 API 评估）

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG/LLaMA-Factory-main
python predict.py
```

**评估方式：** 使用豆包 API 进行 RAG 评估（需配置 `DOUBAO_MODEL_NAME`、`DOUBAO_API_KEY`、`DOUBAO_BASE_URL` 环境变量）

**预测与评估输出：**

```bash
# predict.py 执行后生成：
# - data/summary_test_pred.json：模型预测结果（含 response 字段）
# - data/ragas_evaluation_result.json：RAGas 评估指标

# 评估结果示例：
# {'context_recall': 0.8223, 'llm_context_precision_with_reference': 0.9875}
```

**10. 执行完整离线评估（可选）**

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG
python final_score.py
```

> **说明**：`final_score.py` 会进行更全面的评估，包括语义相似度评分、关键词加权评分等，输出综合得分。

**11. 校验 summary 文件**

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG/LLaMA-Factory-main
ls -l data/summary_train.json data/summary_test.json data/summary_test_pred.json
```

### 🧠 Search-R1 强化学习训练（进阶，在以上步骤完成后）

> **前置条件**：步骤 1-7 已完成，`output/qwen3_lora_sft_int4` 已存在，检索索引和 MongoDB 已就绪。

12. 生成 RL 训练轨迹

```bash
cd /root/autodl-tmp/XIAOMI_SU7_RAG

# 读取 66 条网络兜底问题 → 本地检索 + 网络搜索 + 自动页面抓取 → LLM 生成完整轨迹
# 轨迹包含 <read_page> 垂直搜索周期
python src/rl/data_builder.py                # 全量运行
python src/rl/data_builder.py --dry-run      # 先跑 5 条验证
python src/rl/data_builder.py --resume       # 断点续传
```

13. 格式转换（轨迹 → SFT / GRPO 格式）

```bash
python src/rl/format_converter.py
```

14. 生成本地可答轨迹 + 合并训练数据

```bash
# 自动加载 test_qa_pair_verify.json + train_qa_pair.json，去重后全量生成
# 走 BM25 检索生成只有 <search_local> 的轨迹
# 自动与网络兜底轨迹合并为 combined_trajectories_*.jsonl
python src/rl/build_local_trajectories.py

# 可选参数：
python src/rl/build_local_trajectories.py --dry-run       # 先跑 5 条验证
python src/rl/build_local_trajectories.py --sample 200     # 限制采样数量（默认 0=全量）
```

15. 注册数据集 + SFT warm-up + GRPO 训练

```bash
# SFT warm-up 使用 LLaMA-Factory（步骤 5 相同的框架）
# GRPO 训练使用 TRL GRPOTrainer + PEFT LoRA（不依赖 LLaMA-Factory GRPO）

# 一键全流程：数据注册 → SFT warm-up → GRPO → 导出
python src/rl/train_grpo.py --stage all

# 或分步执行：
python src/rl/train_grpo.py --stage data     # 数据准备 + 本地轨迹生成 + 注册
python src/rl/train_grpo.py --stage sft      # SFT warm-up（LLaMA-Factory + QLoRA 4-bit 量化）
python src/rl/train_grpo.py --stage grpo     # GRPO 强化学习（TRL + PEFT，采样 300 条子集）
python src/rl/train_grpo.py --stage export   # 导出合并模型（链式：base→SFT→GRPO，保留全部学习成果）
```

> **SFT warm-up 技术细节**：
> - 使用 LLaMA-Factory + QLoRA 4-bit 量化（BitsAndBytes），单卡 A100 40GB 即可训练
> - 采用合并数据集（全量 QA 对，含网络兜底 + 本地可答轨迹，去重后数千条）
> - 按 `data_source` 分层 80/20 拆分训练集和评估集，确保两类轨迹均有代表
> - 目的：让模型学会 `<search_local>`/`<search_web>`/`<read_page>`/`<answer>` 标签格式
> - 配置文件：`configs/qwen3_lora_rl_sft.yaml`，QLoRA 4-bit，LoRA rank=16，5 epoch
> - `<information>` 内容不参与 SFT 训练目标（模型只需学习工具调用决策和答案生成）
>
> **GRPO 训练技术细节**：
> - 使用 `trl` 库的 `GRPOTrainer`（需 `trl >= 0.14`）+ `peft` 的 `LoRA`
> - 训练数据：从合并数据集中随机采样 300 条子集（`max_train_samples` 可调）
> - 加载 Qwen3-8B 基座 → 合并 SFT 适配器 → 应用新 LoRA → GRPO 训练
> - 6 维自定义奖励函数通过 `reward_funcs` 参数注入
> - 训练配置（`GRPO_HYPERPARAMS` 字典）定义在 `src/rl/train_grpo.py` 中
> - GRPO 训练使用 HuggingFace `generate()`，不需要 vLLM
> - 关键超参数：`beta=0.01`（KL 惩罚），`lr=1e-5`，3 epoch，`num_generations=4`，`max_completion_length=768`

16. RL 增强推理（边生成边检索 + 深度页面阅读）

```bash
# 启动 vLLM（使用 RL 训练后的模型）
python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_rl --port 8000

# 另一个终端：启动交互式问答
python src/rl/infer_rl.py --show-reward --show-trajectory
```

**交互示例（含垂直搜索）：**
```
🧑 用户 ➜ 小米SU7最新的OTA版本更新了什么功能？

🤖 助手 ➜ <search_local>OTA更新 功能</search_local>
<information>[1]【第45页】小米SU7支持OTA远程升级...
[提示：本地知识库相关性较低（0.18），如需更准确信息可调用网络搜索]
<search_web>小米SU7 最新OTA版本 2025 更新内容</search_web>
<information>【小米SU7 OTA v2.4.0 发布公告】新增城市领航辅助、HUD自定义显示等12项更新
网址：https://www.xiaomi.com/ev/su7/ota
【车主社区】OTA v2.4.0 详细体验报告
网址：https://www.autohome.com.cn/news/202501/su7-ota</information>
<read_page>https://www.xiaomi.com/ev/su7/ota</read_page>
<information>[页面来源：www.xiaomi.com]
小米SU7 OTA v2.4.0 正式发布，本次更新包含12项功能升级：
1. 城市领航辅助（City NOA）正式上线
2. HUD抬头显示新增自定义模式
3. 语音助手升级，支持多轮对话
4. 座椅记忆功能优化...</information>
<answer>根据小米官方页面信息，小米SU7最新的OTA v2.4.0更新了以下主要功能：
1. 城市领航辅助（City NOA）正式上线
2. HUD抬头显示新增自定义模式
3. 语音助手升级，支持多轮对话
4. 座椅记忆功能优化
本次更新共包含12项功能升级。
（以上信息来源于www.xiaomi.com，请以小米官方最新公告为准）</answer>
```

17. RL 模型评测（与 baseline 对比）

```bash
# RL 模型用同一套评测指标与传统 RAG 对比效果
# 前提：vLLM 已启动（RL 模型），检索环境就绪

# 完整评测（676条手册问答，含 RAGAs）
python src/rl/batch_eval.py --vllm-url http://localhost:8000/v1

# 快速验证
python src/rl/batch_eval.py --dry-run

# 跳过 RAGAs（省 API 费用，只算语义+关键词分）
python src/rl/batch_eval.py --skip-ragas

# 断点续传
python src/rl/batch_eval.py --resume
```

**评测原理：** 对 `test_qa_pair_verify.json`（676条）逐条运行 RL 推理，从轨迹中提取 `<answer>` 作为预测答案、`<information>` 拼接为检索上下文，复用 `final_score.py` 相同的评分逻辑（text2vec 语义相似度 + 关键词加权 + RAGAs），与 baseline 指标并列对比。额外输出 RL 6 维奖励平均分和工具调用统计。

**预期输出：**
```
📌 传统评测指标（与 baseline 对比）：
  ──────────────────────────────────────────────────────────
  指标                          Baseline   RL模型
  ──────────────────────────────────────────────────────────
  语义相似度+关键词加权           0.8965    0.xxxx
  RAGAs 上下文召回率              0.9386    0.xxxx
  RAGAs 上下文精确率              0.9488    0.xxxx
  ──────────────────────────────────────────────────────────

  📌 RL 特有指标：
  平均奖励: 0.xxxx / 1.00
    格式完整性: 0.xxxx / 0.05  答案质量: 0.xxxx / 0.40
    工具合理性: 0.xxxx / 0.15  来源标注: 0.xxxx / 0.10
    领域合规:   0.xxxx / 0.15  探索深度: 0.xxxx / 0.15

  📌 工具调用统计：
  平均轮数: x.x | local: x.x | web: x.x | read_page: x.x
```

#### 项目数据目录 (`data/`)

| 文件路径 | 作用 | 记录数 | 生成时机 |
|:---|:---|:---|:---|
| `data/qa_pairs/qa_pair.json` | 原始 QA 对（每个语义块生成） | 823 | Step 3 (`generate_all_data.py`) |
| `data/qa_pairs/expand_qa_pair.json` | 扩展 QA 对（每个问题生成5个同义问法） | 3,864 | Step 3 (`generate_all_data.py`) |
| `data/qa_pairs/train_qa_pair.json` | 训练集（质量审核后） | 21,595 | Step 3 (`generate_all_data.py`) |
| `data/qa_pairs/test_qa_pair.json` | 测试集（质量审核后，含关键词） | 2,325 | Step 3 (`generate_all_data.py`) |
| `data/qa_pairs/test_keywords_pair.json` | 测试集答案关键词标注（去重后） | 1,754 | Step 3 (`generate_all_data.py`) |
| `data/qa_pairs/test_qa_pair_verify.json` | 评估输入文件 | 2,325 | Step 3 (`generate_all_data.py`) |
| `data/qa_pairs/train_data.json` | SFT 训练数据（含检索上下文） | 21,595 | Step 3 (`generate_all_data.py`) |
| `data/summary_data/train.json` | 摘要训练集 | 19,878 | Step 4 (`generate_sft_data.py`) |
| `data/summary_data/test.json` | 摘要测试集 | 1,717 | Step 4 (`generate_sft_data.py`) |
| `data/rerank_data/train.json` | 重排训练集 | 40,849 | Step 4 (`generate_sft_data.py`) |
| `data/rerank_data/dev.json` | 重排验证集 | - | Step 4 (`generate_sft_data.py`) |
| `data/rerank_data/test.json` | 重排测试集 | 936 | Step 4 (`generate_sft_data.py`) |
| `data/rl_data/web_fallback_questions.json` | RL 网络兜底问题库（10 个分类） | 66 | 预置 |
| `data/rl_data/web_fallback_trajectories.json` | RL 原始轨迹数据 | - | Step 12 (`data_builder.py`) |
| `data/rl_data/web_fallback_trajectories_sft.json` | RL SFT 格式训练数据 | - | Step 13 (`format_converter.py`) |
| `data/rl_data/web_fallback_trajectories_grpo.jsonl` | RL GRPO 格式训练数据 | - | Step 13 (`format_converter.py`) |
| `data/rl_data/local_trajectories.json` | 本地可答原始轨迹 | 全量 | Step 14 (`build_local_trajectories.py`) |
| `data/rl_data/local_trajectories_grpo.jsonl` | 本地可答 GRPO 格式 | 全量 | Step 14 (`build_local_trajectories.py`) |
| `data/rl_data/local_trajectories_sft.json` | 本地可答 SFT 格式 | 全量 | Step 14 (`build_local_trajectories.py`) |
| `data/rl_data/combined_trajectories_grpo.jsonl` | 合并 GRPO 格式（网络+本地，训练用） | 全量 | Step 14 (`build_local_trajectories.py`) |
| `data/rl_data/combined_trajectories_sft.json` | 合并 SFT 格式（网络+本地） | 全量 | Step 14 (`build_local_trajectories.py`) |

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

> 提供两种方式，二选一。

#### 方式一：Docker Compose（推荐）

```bash
# 确保已有 .env 文件（参考 .env.example）
cp .env.example .env
# 编辑 .env 填入 DOUBAO_API_KEY 等配置

# 一键启动所有服务（MongoDB + vLLM + 主应用）
docker-compose up -d

# 进入主应用容器进行交互式问答
docker exec -it xiaomi_rag_app python infer.py

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f vllm
docker-compose logs -f app

# 停止所有服务
docker-compose down
```

> ⚠️ **注意**：`semantic-chunk` 服务仅在**建索引阶段**（`build_index.py`）需要，日常推理无需启动。  
> 建索引时单独启动：`docker-compose up -d semantic-chunk`

#### 文档解析质量评估

```bash
# 在 build_index.py 执行完成后，评估文档解析准确率
python evaluate_parse_quality.py

# 输出示例：
# ============================================================
# 📄 文档解析质量评估报告
# ============================================================
# 【文本解析质量】
#   ├─ 文本保留率: 100.00%
#   ├─ 空白行比例: 11.14%
#   ├─ 标题保留率: 100.00%
#   ├─ 列表保留率: 100.00%
#   ├─ 异常字符率: 0.00%
#   └─ 综合评分: 98.89%
# 【文档切分质量】
#   ├─ 切分数量: 999
#   ├─ 平均长度: 182 字符
#   ├─ 长度标准差: 92
#   ├─ 上下文相关性: 32.07%
#   └─ 切分质量评分: 80.30%
# ============================================================
# 📊 最终解析准确率: 96.10%
# ============================================================
```

#### 基线对比测试

支持两种模型进行对比测试，可根据需求选择：

```bash
# 1. 使用本地模型（推荐，完全免费，无需 API）
# 需先启动 vLLM 服务：python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000
python deploy/baseline_gpt4o.py --model local

# 2. 使用 OpenAI API（GPT-4o 等）
# 需配置环境变量：export OPENAI_API_KEY=sk-xxx
python deploy/baseline_gpt4o.py --model openai

# 输出示例：
# ============================================================
# 📊 对比结果
# ============================================================
# 本地 Qwen3-8B 得分：0.8910
# 本系统得分：         0.8965
# 提升幅度：           +0.6%
# ============================================================
```

| 模型选项 | 成本 | 需要条件 |
|---------|------|----------|
| `--model local` | **免费** | 启动 vLLM 服务 |
| `--model openai` | 较贵 (~$0.01/千token) | OPENAI_API_KEY |

#### vLLM 性能压测

```bash
# 先启动 vLLM 服务（以非量化模型为例）
python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft --port 8000

# 第一次运行（非量化模型）
python deploy/benchmark.py

# 输出示例：
# ============================================================
# 📊 性能测试结果
# ============================================================
# 模型：qwen3_lora_sft
# TTFT 均值：56 ms
# TTFT P95：43 ms
# 吞吐率：465 token/s
# ============================================================

# 换为 INT4 量化模型，重启 vLLM
python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000

# 第二次运行（量化模型），自动对比两次结果
python deploy/benchmark.py

# 输出示例：
# ============================================================
# 📊 性能测试结果
# ============================================================
# 模型：qwen3_lora_sft_int4
# TTFT 均值：52 ms
# TTFT P95：33 ms
# 吞吐率：669 token/s
# ============================================================
# 对比上次结果（qwen3_lora_sft）：
#   TTFT：56 ms → 52 ms  (+7.0%)
#   吞吐率：465 → 669 token/s  (+43.8%)
# ============================================================
```

**多卡部署与扩展效率**

```bash
# 多卡自动检测（支持张量并行）
python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000

# 手动指定张量并行数（如 8 卡）
python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000 -- --tensor-parallel-size 8
```

| 配置 | 吞吐量 | 扩展效率 |
|------|--------|----------|
| 单卡（非量化） | 465 token/s | - |
| 单卡（INT4） | 669 token/s | +43.8% |
| 8卡（INT4，85%效率） | ~4,550 token/s | 6.8x |

> **说明**：多卡扩展效率通常在 70%-90% 之间，取决于 GPU 型号和网络带宽（NVLink > PCIe）。

---

#### 方式二：手动逐服务启动

**方式 A：使用 Docker（推荐）**
```bash
# 终端 1：启动 MongoDB（用 Docker）
docker run -d --name mongodb -p 27017:27017 \
  -v $(pwd)/data/mongodb/data:/data/db mongo:7.0

# 终端 2：启动 vLLM（自动识别单卡/多卡）
python deploy/auto_vllm_server.py \
  --model /root/autodl-tmp/XIAOMI_SU7_RAG/LLaMA-Factory-main/output/qwen3_lora_sft_int4 \
  --port 8000

# 终端 3：在线问答
python infer.py
```

**方式 B：本地安装 MongoDB**
```bash
# 终端 1：启动 MongoDB（必需，需先安装）
/usr/local/mongodb/bin/mongod --dbpath /data/db --logpath /var/log/mongodb/mongod.log --bind_ip_all --fork

# 终端 2：启动 vLLM（自动识别单卡/多卡）
python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000

# 终端 3：在线问答
python infer.py
```

> ⚠️ vLLM 启动时模型路径推荐使用**绝对路径**，否则可能被误认为 HuggingFace repo id。

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

**传统 RAG 评测：**

```bash
python final_score.py
```

**RL 模型评测（与 baseline 同指标对比）：**

```bash
python src/rl/batch_eval.py --vllm-url http://localhost:8000/v1
```

**评分机制（两种评测共用）：**

- **语义相似度**（`text2vec-base-chinese`）：计算预测答案与标准答案的向量余弦相似度
- **关键词加权**：提取标准答案中的关键词，检查预测答案是否命中，关键词只加分不扣分
- **综合评分**：`max(semantic_score, 0.3 × keyword_score + 0.7 × semantic_score)`
- **短答案补偿**：短答案精确匹配和字符重叠率保底机制
- **无答案重试**：首次生成"无答案"时，用 top-3 文档重新生成，减少误判

**推理参数：**

| 参数 | 值 | 说明 |
|:---|:---|:---|
| `BM25_RETRIEVE_SIZE` | 20 | BM25 稀疏检索召回数 |
| `MILVUS_RETRIEVE_SIZE` | 40 | Milvus 混合检索召回数 |
| `RERANK_SIZE` | 12 | MiniCPM 精排后保留文档数 |
| `HYDE` | 1 | HyDE 扩写增强检索 |
| `MAX_WORKERS` | 4 | 并发线程数 |

**最新评估结果：**

| 指标 | 得分 |
|:---|:---|
| 语义相似度 + 关键词加权得分 | **0.8965** |
| RAGas context_recall | 0.9386 |
| RAGas llm_context_precision_with_reference | 0.9488 |

评估结果保存到 `data/ragas_evaluation_result.json`。

---

## ⚠️ 已知限制

| 限制 | 说明 | 解决方案 |
|:---:|:---|:---|
| 📦 TRL 版本 | GRPO 训练需要 `trl >= 0.14`，项目自带 `trl==0.9.6` | `pip install "trl>=0.14" --upgrade` |
| 📦 transformers 兼容性 | 升级 TRL 后可能拉高 transformers 版本，与旧版 vllm/FlagEmbedding 冲突 | GRPO 训练脚本已自动屏蔽 vllm 导入；重排模型仅在推理/评测时使用 |
| 📁 目录创建 | 部分脚本不自动创建输出目录 | 提前 `mkdir -p data/{processed_docs,saved_index,qa_pairs,summary_data,rerank_data,rl_data,saved_images,mongodb/{data,log}}` |
| 🔑 网络搜索 API | RL 网络兜底依赖 SerpAPI（免费额度有限） | 未设置时自动降级为豆包 LLM 模拟搜索；也可配置 `BING_SEARCH_KEY` 作为备选后端 |
| 🖥️ GPU 要求 | vLLM 推理 + 重排模型 + GRPO 训练需要 GPU | 推理建议单卡 ≥ 16GB；SFT warm-up 使用 QLoRA 4-bit 量化，单卡 A100 40GB 即可；GRPO 训练 `num_generations=4` 约需 30-40GB |

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

- **BGE-Reranker-v2-MiniCPM-Layerwise**：在线推理与离线评估均使用，基于 MiniCPM 的 Layerwise 轻量高性能重排模型，可通过 `cutoff_layers` 控制精度/速度平衡
- **BGE-M3 跨编码器**：备选方案，精准重排
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

<details>
<summary><b>🧠 RL 强化学习模块 (Search-R1 + WebWalker 垂直搜索)</b></summary>

### 概述

传统 RAG 采用固定管线（检索→重排→生成），模型被动接收上下文。RL 模块升级为 **Search-R1 范式**：模型自主决定何时检索、检索什么，实现 **边生成边检索** 的智能工具调用。在此基础上，引入 **WebWalker 垂直搜索**能力：模型可以点击搜索结果中的链接，深入阅读页面内容，像人一样一层层钻进去获取详细信息。

### 核心机制

```
传统 RAG（infer.py）：
  query → [固定检索] → [固定重排] → [生成答案]

Search-R1 + WebWalker（infer_rl.py）：
  query → model 生成 "<search_local>关键词"
        → 系统拦截，执行本地检索，注入 <information>
        → model 继续生成 "<search_web>关键词"
        → 系统拦截，执行网络搜索，注入 <information>（含"网址："字段）
        → model 生成 "<read_page>URL"            ← 垂直搜索：深度阅读
        → 系统拦截，抓取页面内容，注入 <information>
        → model 生成 "<answer>最终答案"
```

### 模块结构

| 文件 | 功能 |
|:---|:---|
| `src/rl/web_reader.py` | 网页内容抓取器（HTML→纯文本，超时/大小/类型防护） |
| `src/rl/data_builder.py` | 网络兜底轨迹生成器（含多跳页面阅读，自动提取URL并抓取） |
| `src/rl/build_local_trajectories.py` | 本地可答轨迹生成器（BM25 召回，教模型"本地够用就停"） |
| `src/rl/format_converter.py` | 格式转换器（轨迹 → SFT / GRPO / ShareGPT 多格式，含自动修复截断标签） |
| `src/rl/reward_model.py` | 6 维度奖励函数（格式 0.05 + 答案 0.40 + 工具 0.15 + 来源 0.10 + 领域 0.15 + **探索深度 0.15**），兼容 TRL 接口 |
| `src/rl/environment.py` | 工具调用路由环境（拦截 `<search_local>`/`<search_web>`/`<read_page>` → 执行检索 → 注入结果） |
| `src/rl/train_grpo.py` | GRPO 训练入口（数据准备 → SFT warm-up → **TRL GRPOTrainer + PEFT LoRA** → 导出） |
| `src/rl/infer_rl.py` | RL 增强推理（边生成边检索 + 深度页面阅读，含跳数和总调用限制） |

### 奖励函数（6 维度，总分 1.0）

| 维度 | 权重 | 说明 |
|:---|:---:|:---|
| 格式完整性 | 0.05 | 标签齐全、正确闭合（SFT 已教格式，GRPO 弱化此维度） |
| 答案质量 | 0.40 | 回答准确性、信息量、基于检索信息的 groundedness |
| 工具合理性 | 0.15 | 检索关键词精准、调用顺序合理 |
| 来源标注 | 0.10 | 网络信息正确注明来源 |
| 领域合规 | 0.15 | 正确拒答非SU7问题 |
| **探索深度** | **0.15** | **本地充足即停 / 网络搜索有效利用 read_page** |

探索深度评分逻辑：
- **Local-only**：本地信息充分 + 答案充实 → 0.10；本地信息不充分 → 0.02
- **Web 轨迹**：有web无read_page得0.03（表面搜索）；有read_page且URL有效+0.05；内容被answer引用+0.04；多次read_page（≤2）+0.03；顺序错误-0.03

### 训练流程

```bash
# 1. 生成网络兜底轨迹（本地检索 + 网络搜索 + 自动页面抓取）
python src/rl/data_builder.py

# 2. 生成本地可答轨迹（BM25 召回，教模型不联网，默认加载全部 QA 对）
python src/rl/build_local_trajectories.py

# 3. 格式转换（轨迹 → SFT/GRPO 格式，自动修复 + 合并）
python src/rl/format_converter.py

# 4. 一键训练（SFT warm-up + GRPO 强化学习）
#    SFT warm-up 使用 LLaMA-Factory
#    GRPO 使用 TRL GRPOTrainer + PEFT LoRA
python src/rl/train_grpo.py --stage all

# 5. RL 增强推理（边生成边检索 + 深度阅读）
python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_rl --port 8000
python src/rl/infer_rl.py --show-reward --show-trajectory
```

> **依赖要求**：GRPO 训练需要 `trl >= 0.14` + `peft` + `bitsandbytes`。SFT warm-up 和导出合并仍使用 LLaMA-Factory。SFT warm-up 使用合并数据集（全量 QA 对，按 data_source 分层 80/20 拆分训练/评估集），采用 QLoRA 4-bit 量化降低显存需求；GRPO 从合并数据集中随机采样 300 条子集训练，兼顾"何时联网"和"何时不联网"两种场景。

### 训练数据

**训练数据由两类轨迹合并而成：**

| 数据类型 | 数量 | 轨迹特点 | 训练目标 |
|:---|:---:|:---|:---|
| 网络兜底轨迹 | ~79 条 | `<search_local>` + `<search_web>` + `<read_page>` | 学会何时联网 + 深度阅读 |
| 本地可答轨迹 | 全量 QA（去重后数千条） | 只有 `<search_local>` + `<answer>` | 学会本地够用时不要联网搜索 |

| 分类 | 数量 | 示例问题 |
|:---|:---:|:---|
| OTA软件更新 | 13 | 小米SU7目前最新的OTA版本号是多少？ |
| 竞品对比信息 | 8 | 小米SU7和特斯拉Model 3相比哪个更值得买？ |
| 官方新闻动态 | 10 | 小米汽车最近有什么重大新闻？ |
| 召回与技术服务公告 | 7 | 小米SU7有没有发过召回公告？ |
| 价格与购车优惠 | 7 | 小米SU7现在的官方售价是多少？ |
| 充电网络与基础设施 | 7 | 小米汽车目前在全国建了多少个充电桩？ |
| 车主真实反馈 | 7 | 小米SU7车主普遍反映有哪些槽点？ |
| 新能源政策 | 6 | 2025年买小米SU7还能享受哪些补贴？ |
| 保险与金融 | 6 | 小米SU7第一年保险大概多少钱？ |
| 维护保养费用 | 5 | 小米SU7每年保养大概要花多少钱？ |

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