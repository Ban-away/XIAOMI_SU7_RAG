# -*- coding: utf-8 -*-
"""训练数据构建脚本：从 train_data 生成 summary/rerank 数据集。"""

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

import os
import pickle
import time
import json
import re
import random
import concurrent.futures
from tqdm import tqdm
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever 
from src.client.llm_chat_client import request_chat
from src.reranker.bge_m3_reranker import BGEM3ReRanker 
# from src.reranker.qwen3_reranker_vllm import Qwen3ReRankervLLM 
from src.constant import bge_reranker_model_path
# from src.constant import qwen3_4b_reranker_model_path
from src.utils import merge_docs, post_processing

random.seed(42)


LLM_CHAT_PROMPT = """
### 信息
{context}

### 任务
你是小米 SU7 车型的用户手册问答系统，你具备{{信息}}中的知识。
请回答问题"{query}"，答案需要精准，语句通顺，并严格按照以下格式输出

{{答案}}【{{引用编号1}},{{引用编号2}},...】
如果无法从中得到答案，请说 "无答案" ，不允许在答案中添加编造成分。
"""


# 批量处理配置（移到外面，确保总是被定义）
BATCH_SIZE = 700  # 每批处理的样本数
MAX_WORKERS = 50  # 并发数（增加以提高处理速度）

# ==================== 自动生成 train_data.json ====================
if not os.path.exists("data/qa_pairs/train_data.json"):
    print("[INFO] train_data.json 不存在，开始生成...")
    
    # 加载检索器和重排器（使用轻量级 BGE-M3 重排器，避免 vLLM 多进程问题）
    bm25_retriever = BM25(docs=None, retrieve=True)
    milvus_retriever = MilvusRetriever(docs=None, retrieve=True) 
    bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_model_path)
    
    # 预热模型
    milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)
    
    # 读取 train_qa_pair.json
    fd = open("data/qa_pairs/train_qa_pair.json")
    train_qa_pairs = json.load(fd)  # 修正变量名
    fd.close()
    
    # 生成 train_data.json（批量并发处理）
    output_path = "data/qa_pairs/train_data.json"

    def process_item(item):
        """处理单个样本"""
        try:
            query = item["question"].strip()
            
            # 检索
            bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
            milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
            merged_docs = merge_docs(bm25_docs, milvus_docs)
            
            # 重排
            ranked_docs = bge_m3_reranker.rank(query, merged_docs, topk=5)
            
            # 生成答案
            context = "\n".join([str(idx+1) + "." + doc.page_content for idx, doc in enumerate(ranked_docs)])
            response = request_chat(query, context)
            
            # 返回结果
            info = {
                "query": query,
                "context": [doc.page_content for doc in ranked_docs],
                "response": response,
                "merged_docs": [doc.page_content for doc in merged_docs]
            }
            return info
        except Exception as e:
            print(f"⚠️ 处理失败: {e}")
            return None

    # 分批并发处理（正确缩进）
    with open(output_path, "w", encoding="utf-8") as f:
        total_batches = (len(train_qa_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
        wait_time = 5  # 批次之间等待时间（秒）
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(train_qa_pairs))
            batch_items = train_qa_pairs[start_idx:end_idx]
            
            print(f"\n🚀 处理批次 {batch_idx+1}/{total_batches} ({len(batch_items)} 条)")
            
            # 使用线程池并发处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_item, item): item for item in batch_items}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()  # 立即写入，防止内存累积
            
            # 批次之间等待（最后一批不需要等待）
            if batch_idx < total_batches - 1:
                print(f"⏳ 等待 {wait_time} 秒...")
                time.sleep(wait_time)
    print("[INFO] train_data.json 生成完成")


MAX_INPUT_SIZE = 4096
RERANK_DEV_SIZE = 1000
TEST_RATE = 0.08

print("\n" + "="*60)
print("Step 6: 生成 SFT 训练数据")
print("="*60)
print(f"📝 训练数据路径: data/qa_pairs/train_data.json")
print(f"📝 摘要训练集: ./data/summary_data/train.json")
print(f"📝 摘要测试集: ./data/summary_data/test.json")
print(f"📝 重排训练集: ./data/rerank_data/train.json")
print(f"📝 重排验证集: ./data/rerank_data/dev.json")
print(f"📝 重排测试集: ./data/rerank_data/test.json")

# 训练/测试输出文件句柄
summary_train_handler = open("./data/summary_data/train.json", "w")
summary_test_handler = open("./data/summary_data/test.json", "w")
rerank_train_handler = open("./data/rerank_data/train.json", "w")
rerank_dev_handler = open("./data/rerank_data/dev.json", "w")
rerank_test_handler = open("./data/rerank_data/test.json", "w")

summary_train = []
summary_test = []
rerank_train = []
rerank_test = []

# 读取上游生成的 train_data，拆解引用并构建监督样本
def process_train_data(line):
    """处理单行 train_data"""
    info = json.loads(line)
    response = info["response"]
    all_cites = re.findall("[【](.*?)[】]", response)
    cites = []
    for cite in all_cites:
        cite = re.sub("[{} 【】]", "", cite)
        cite = cite.replace(",", "，")
        cite = [int(k) for k in cite.split("，") if k.isdigit()]
        cites.extend(cite)
    cites = sorted(list(set(cites)))
    cites = ",".join([str(c) for c in cites])
    answer = re.sub("[【](.*?)[】]", "", response)
    answer = re.sub("[{}【】]", "", answer)
    if cites:
        format_answer = answer + f"【{cites}】"
    else:
        format_answer = "无答案"
    context = "\n".join([str(idx+1) + "." + doc for idx, doc in enumerate(info["context"])])
    if len(context) > MAX_INPUT_SIZE:
        context = context[:MAX_INPUT_SIZE]

    query = info["query"].strip()
    instruction = LLM_CHAT_PROMPT.format(query=query, context=context)
    item = {
        "query": query,
        "context": context,
        "instruction": instruction,
        "input": "",
        "output": format_answer
    }
    neg_docs = [doc for doc in info["merged_docs"] if doc not in info["context"]]
    
    result = {
        "item": item,
        "query": query,
        "format_answer": format_answer,
        "neg_docs": neg_docs,
        "context": info["context"],
        "merged_docs": info["merged_docs"]
    }
    return result

# 批量并发处理 train_data
fd = open("data/qa_pairs/train_data.json")
lines = fd.readlines()
fd.close()

print(f"\n🚀 处理 train_data ({len(lines)} 条)...")
processed_results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_train_data, line): line for line in lines}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        processed_results.append(future.result())

# 构建训练/测试集
for result in processed_results:
    item = result["item"]
    query = result["query"]
    format_answer = result["format_answer"]
    neg_docs = result["neg_docs"]
    context = result["context"]
    merged_docs = result["merged_docs"]
    
    # 按固定比例切分 summary 训练/测试集
    if random.random() < TEST_RATE:
        summary_test.append(item)

        if format_answer != "无答案":
            content_list = [context[0], random.choice(context[-2:])]
            if neg_docs:
                content_list.append(random.choice(neg_docs))
            rerank_test.append({"query": query, "content": content_list})

    else:
        summary_train.append(item)

        # 构建重排训练样本：正样本(2) / 次相关(1) / 负样本(0)
        if format_answer != "无答案":
            positive = context[0]
            middle = random.choice(context[-2:])
            rerank_train.append({"query": query, "content": positive, "label": 2})
            rerank_train.append({"query": query, "content": middle, "label": 1})
            if neg_docs:
                negative = random.choice(neg_docs)
                rerank_train.append({"query": query, "content": negative, "label": 0})
        else:
            negative = random.choice(merged_docs)
            rerank_train.append({"query": query, "content": negative, "label": 0})

rerank_train = [item for item in rerank_train if len(item["query"]) > 0 and len(item["content"]) > 0]
rerank_dev = rerank_train[-RERANK_DEV_SIZE:]
random.shuffle(rerank_train)
print("Rerank Train size:", len(rerank_train), "Rerank Test size:", len(rerank_test))

for item in rerank_train:
    rerank_train_handler.write(json.dumps(item, ensure_ascii=False) + "\n")
for item in rerank_dev:
    rerank_dev_handler.write(json.dumps(item, ensure_ascii=False) + "\n")
for item in rerank_test:
    rerank_test_handler.write(json.dumps(item, ensure_ascii=False) + "\n")


print("Summary Train size:", len(summary_train), "Summary Test size:", len(summary_test))
summary_train_handler.write(json.dumps(summary_train, ensure_ascii=False, indent=4))
summary_test_handler.write(json.dumps(summary_test, ensure_ascii=False, indent=4))