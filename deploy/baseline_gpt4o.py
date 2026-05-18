# -*- coding: utf-8 -*-
"""
GPT-4o + OpenAI Embeddings 基线对比实验
对比指标：语义相似度 + 关键词加权得分，与本系统结果做比较

运行前准备：
  export OPENAI_API_KEY=sk-xxx
  pip install openai langchain-openai faiss-cpu text2vec
  
运行：
  python deploy/baseline_gpt4o.py
"""

import os
import json
import time
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from text2vec import SentenceModel, semantic_search

load_dotenv()

# 导入常量路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constant import split_docs_path, text2vec_model_path

# ── 配置 ─────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBED_MODEL     = "text-embedding-3-large"
CHAT_MODEL      = "gpt-4o"
TOPK            = 8
MAX_WORKERS     = 5   # GPT-4o 并发数，避免触发 rate limit

# 评估数据路径（使用绝对路径，避免运行目录问题）
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILE       = os.path.join(base_dir, "data/qa_pairs/test_qa_pair_verify.json")
SPLIT_DOCS_FILE = split_docs_path
RESULT_FILE     = os.path.join(base_dir, "data/baseline_gpt4o_result.json")
TEXT2VEC_MODEL  = text2vec_model_path

# prompt 模板（和本系统保持一致，确保公平对比）
GPT4O_PROMPT = """
### 信息
{context}

### 任务
你是小米 SU7 车型的用户手册问答系统。
请回答问题"{query}"，答案需要精准，语句通顺。

如果无法从中得到答案，请说 "无答案"，不允许在答案中添加编造成分。
"""

# ── 初始化 ───────────────────────────────────────────────────
if not OPENAI_API_KEY:
    raise ValueError("请设置 OPENAI_API_KEY 环境变量")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
print(f"[INFO] 使用模型：{CHAT_MODEL} + {EMBED_MODEL}")


def build_faiss_index(split_docs_file):
    """用 OpenAI Embeddings 构建 FAISS 索引"""
    import pickle
    faiss_cache = "data/saved_index/faiss_gpt4o.db"

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    if os.path.exists(faiss_cache):
        print(f"[INFO] 加载已有 FAISS 索引：{faiss_cache}")
        vector_store = FAISS.load_local(
            faiss_cache, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print(f"[INFO] 构建 OpenAI Embeddings FAISS 索引...")
        with open(split_docs_file, "rb") as f:
            split_docs = pickle.load(f)
        print(f"[INFO] 文档数：{len(split_docs)}")
        # 分批构建，避免单次请求过大
        BATCH = 500
        vector_store = None
        for i in tqdm(range(0, len(split_docs), BATCH), desc="构建索引"):
            batch = split_docs[i:i + BATCH]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            time.sleep(0.5)  # 避免 rate limit
        vector_store.save_local(faiss_cache)
        print(f"[INFO] 索引已保存：{faiss_cache}")

    return vector_store


def retrieve(vector_store, query, topk=TOPK):
    """OpenAI Embeddings 检索"""
    docs = vector_store.similarity_search(query, k=topk)
    return docs


def generate(query, context):
    """GPT-4o 生成答案"""
    prompt = GPT4O_PROMPT.format(query=query, context=context)
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.01,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] GPT-4o 调用失败：{e}")
        return "无答案"


def calc_jaccard(list_a, list_b, threshold=0.3):
    size_c = len([i for i in list_a if i in list_b])
    return 1 if size_c / (len(list_b) + 1e-6) > threshold else 0


def main():
    # 1. 加载测试集
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"[INFO] 测试集：{len(test_data)} 条")

    # 2. 构建索引
    vector_store = build_faiss_index(SPLIT_DOCS_FILE)

    # 3. 加载评分模型
    sim_model = SentenceModel(model_name_or_path=TEXT2VEC_MODEL, device="cuda:0")

    # 4. 批量推理
    results = []
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[INFO] 加载已有结果：{len(results)} 条，继续剩余...")
        done_ids = {r["unique_id"] for r in results}
        test_data = [d for d in test_data if d["unique_id"] not in done_ids]

    for item in tqdm(test_data, desc="GPT-4o 推理"):
        query = item["question"]
        docs  = retrieve(vector_store, query)
        context = "\n".join(
            [f"【{i+1}】{doc.page_content}" for i, doc in enumerate(docs)]
        )
        pred = generate(query, context)

        results.append({
            "unique_id": item["unique_id"],
            "question":  query,
            "answer":    item["answer"],
            "keywords":  item.get("keywords", []),
            "pred":      pred,
            "context":   context,
        })

        # 每 50 条保存一次
        if len(results) % 50 == 0:
            with open(RESULT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(0.2)  # 避免 rate limit

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 推理完成，结果已保存：{RESULT_FILE}")

    # 5. 评分（和 final_score.py 完全一致的评分逻辑）
    scores = []
    for item in tqdm(results, desc="评分"):
        gold     = item["answer"]
        pred     = item["pred"]
        keywords = item.get("keywords", [])

        if gold == "无答案" and pred != gold:
            score = 0.0
        elif gold == "无答案" and pred == gold:
            score = 1.0
        else:
            semantic_score = semantic_search(
                sim_model.encode([gold]), sim_model.encode(pred), top_k=1
            )[0][0]["score"]
            join_keywords = [w for w in keywords if w in pred]
            keyword_score = calc_jaccard(join_keywords, keywords)
            score = semantic_score if not keywords else (
                0.2 * keyword_score + 0.8 * semantic_score
            )
        scores.append(score)

    baseline_score = float(np.mean(scores))

    # 6. 对比输出
    print("\n" + "=" * 60)
    print("📊 对比结果")
    print("=" * 60)
    print(f"GPT-4o + OpenAI Embeddings 得分：{baseline_score:.4f}")

    # 读取本系统得分
    our_result_file = "data/ragas_evaluation_result.json"
    our_score = None
    improvement = None
    if os.path.exists(our_result_file):
        with open(our_result_file, "r") as f:
            our_result = json.load(f)
        our_score = our_result.get("semantic_keyword_score", 0)
        improvement = (our_score - baseline_score) / (baseline_score + 1e-9) * 100
        print(f"本系统得分：                     {our_score:.4f}")
        print(f"提升幅度：                       {improvement:+.1f}%")
    print("=" * 60)

    # 保存对比结果
    compare = {
        "baseline_gpt4o_score":  baseline_score,
        "our_score":             our_score,
        "improvement_pct":       improvement,
        "total_samples":         len(results),
    }
    with open("data/comparison_result.json", "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)
    print("[INFO] 对比结果已保存：data/comparison_result.json")


if __name__ == "__main__":
    main()