# -*- coding: utf-8 -*-
"""
基线对比实验
支持多种模型：GPT-4o / 自定义 OpenAI 兼容 API

对比指标：语义相似度 + 关键词加权得分，与本系统结果做比较

运行前准备：
  1. 使用本地模型（完全免费，无需API）：
     python deploy/baseline_gpt4o.py --model local

  2. 使用 OpenAI API（GPT-4o等）：
     export OPENAI_API_KEY=sk-xxx
     python deploy/baseline_gpt4o.py --model openai

运行：
  python deploy/baseline_gpt4o.py --model local    # 本地模型（免费）
  python deploy/baseline_gpt4o.py --model openai   # OpenAI API（GPT-4o等）
"""

import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from text2vec import SentenceModel, semantic_search
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# 导入常量路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constant import split_docs_path, text2vec_model_path, qwen3_8b_tune_model_name


def main():
    # ── 解析命令行参数 ─────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="基线对比实验")
    parser.add_argument("--model", type=str, default="local",
                        choices=["openai", "local"],
                        help="选择对比模型：openai（GPT-4o等）、local（本地模型，免费）")
    args = parser.parse_args()

    # ── 配置 ─────────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TOPK = 8
    RESULT_FILE = os.path.join(BASE_DIR, f"data/baseline_{args.model}_result.json")

    # 根据选择的模型加载配置
    if args.model == "openai":
        API_KEY = os.getenv("OPENAI_API_KEY")
        BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        CHAT_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        print(f"[INFO] 使用模型：{CHAT_MODEL}")
    else:  # local
        API_KEY = None
        BASE_URL = None
        CHAT_MODEL = "local-qwen3"
        print(f"[INFO] 使用模型：本地 Qwen3-8B（无需API）")
    
    # 评估数据路径
    TEST_FILE = os.path.join(BASE_DIR, "data/qa_pairs/test_qa_pair_verify.json")
    SPLIT_DOCS_FILE = split_docs_path
    TEXT2VEC_MODEL = text2vec_model_path

    # prompt 模板（和本系统保持一致，确保公平对比）
    PROMPT = """
### 信息
{context}

### 任务
你是小米 SU7 车型的用户手册问答系统。
请回答问题"{query}"，答案需要精准，语句通顺。

如果无法从中得到答案，请说 "无答案"，不允许在答案中添加编造成分。
"""

    # ── 初始化客户端 ───────────────────────────────────────────────────
    client = None
    if API_KEY and BASE_URL:
        try:
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            # 测试连接
            models = client.models.list()
            print(f"[INFO] API 连接成功")
        except Exception as e:
            print(f"[WARN] API 连接失败: {e}")
            print(f"[INFO] 将回退到本地模型")
            client = None
            args.model = "local"
    
    # 本地 vLLM 客户端
    local_client = None
    if args.model == "local":
        try:
            from openai import OpenAI as LocalOpenAI
            local_client = LocalOpenAI(
                base_url="http://localhost:8000/v1",
                api_key="dummy_key",
            )
            # 测试连接
            models = local_client.models.list()
            print(f"[INFO] 本地 vLLM 连接成功")
        except Exception as e:
            print(f"[ERROR] 无法连接本地 vLLM 服务: {e}")
            print(f"[ERROR] 请先启动 vLLM 服务：python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_sft_int4 --port 8000")
            raise


    def build_faiss_index(split_docs_file):
        """构建 FAISS 索引（使用本地 BGE 模型）"""
        import pickle
        faiss_cache = os.path.join(BASE_DIR, "data/saved_index/faiss_bge.db")
        
        # 使用本地已有的 BGE 模型，避免网络下载
        local_bge_path = os.path.join(BASE_DIR, "models/BAAI/bge-large-zh-v1.5")
        if not os.path.exists(local_bge_path):
            local_bge_path = "BAAI/bge-large-zh-v1.5"  # 回退到 HuggingFace 下载
        
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=local_bge_path,
            model_kwargs={"device": "cuda:0"},
            encode_kwargs={"normalize_embeddings": True}
        )

        if os.path.exists(faiss_cache):
            print(f"[INFO] 加载已有 FAISS 索引：{faiss_cache}")
            try:
                vector_store = FAISS.load_local(
                    faiss_cache, embeddings, allow_dangerous_deserialization=True
                )
                return vector_store
            except Exception as e:
                print(f"[WARN] 加载缓存失败，重新构建: {e}")
        
        print(f"[INFO] 构建 BGE FAISS 索引...")
        with open(split_docs_file, "rb") as f:
            split_docs = pickle.load(f)
        print(f"[INFO] 文档数：{len(split_docs)}")
        
        vector_store = FAISS.from_documents(split_docs, embeddings)
        vector_store.save_local(faiss_cache)
        print(f"[INFO] 索引已保存：{faiss_cache}")

        return vector_store


    def retrieve(vector_store, query, topk=TOPK):
        """检索"""
        docs = vector_store.similarity_search(query, k=topk)
        return docs


    def generate(query, context):
        """生成答案"""
        # 截断上下文防止超出模型最大长度
        MAX_CONTEXT_CHARS = 9000
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]
        prompt = PROMPT.format(query=query, context=context)
        try:
            if args.model == "local" and local_client:
                response = local_client.chat.completions.create(
                    model=qwen3_8b_tune_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.01,
                    extra_body={
                        "top_k": 1,
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                )
            elif client:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.01,
                )
            else:
                return "无答案"
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] 模型调用失败：{e}")
            return "无答案"


    def calc_jaccard(list_a, list_b, threshold=0.3):
        size_c = len([i for i in list_a if i in list_b])
        return 1 if size_c / (len(list_b) + 1e-6) > threshold else 0

    def _fuzzy_keyword_match(kw, text):
        """关键词匹配：精确匹配 或 字符级模糊匹配（>=60%的字符命中）"""
        if kw in text:
            return True
        kw_chars = set(kw.replace(" ", ""))
        if not kw_chars:
            return False
        hit = sum(1 for c in kw_chars if c in text)
        return hit / len(kw_chars) >= 0.6


    # 1. 加载测试集
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"[INFO] 测试集：{len(test_data)} 条")

    # 2. 构建索引
    vector_store = build_faiss_index(SPLIT_DOCS_FILE)

    # 3. 加载评分模型
    sim_model = SentenceModel(model_name_or_path=TEXT2VEC_MODEL, device="cuda:0")

    # 4. 批量推理（分批并行处理）
    results = []
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[INFO] 加载已有结果：{len(results)} 条，继续剩余...")
        done_ids = {r.get("unique_id") for r in results if "unique_id" in r}
        test_data = [d for d in test_data if d["unique_id"] not in done_ids]

    # 分批并行处理配置
    BATCH_SIZE = 700  # 每批处理700条
    MAX_WORKERS = 25  # 并发数
    BASE_WAIT_SECONDS = 3  # 批间等待时间
    
    total_batches = (len(test_data) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📦 分批处理：共 {total_batches} 批，每批 {BATCH_SIZE} 条，并发数 {MAX_WORKERS}")
    
    def process_item(item):
        """处理单个测试样本"""
        query = item["question"]
        docs = retrieve(vector_store, query)
        context = "\n".join(
            [f"【{i+1}】{doc.page_content}" for i, doc in enumerate(docs)]
        )
        pred = generate(query, context)
        return {
            "unique_id": item["unique_id"],
            "question": query,
            "answer": item["answer"],
            "keywords": item.get("keywords", []),
            "pred": pred,
            "context": context,
        }

    model_desc = "本地Qwen3" if args.model == "local" else CHAT_MODEL
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(test_data))
        batch_data = test_data[start_idx:end_idx]
        
        print(f"\n🚀 处理批次 {batch_idx+1}/{total_batches} ({len(batch_data)} 条)")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_item, item): item for item in batch_data}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model_desc} 推理"):
                result = future.result()
                if result:
                    results.append(result)
        
        # 保存当前进度
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 已保存 {len(results)} 条结果")
        
        # 批间等待（最后一批不需要）
        if batch_idx < total_batches - 1:
            print(f"⏳ 等待 {BASE_WAIT_SECONDS} 秒...")
            time.sleep(BASE_WAIT_SECONDS)

    print(f"[INFO] 推理完成，结果已保存：{RESULT_FILE}")

    # 5. 评分（和 final_score.py 完全一致的评分逻辑）
    scores = []
    for item in tqdm(results, desc="评分"):
        gold = item["answer"]
        pred = item["pred"]
        keywords = item.get("keywords", [])

        if gold == "无答案" and pred != gold:
            score = 0.0
        elif gold == "无答案" and pred == gold:
            score = 1.0
        else:
            semantic_score = semantic_search(
                sim_model.encode([gold]), sim_model.encode(pred), top_k=1
            )[0][0]["score"]

            valid_keywords = [kw for kw in keywords if _fuzzy_keyword_match(kw, gold)]
            if valid_keywords:
                join_keywords = [kw for kw in valid_keywords if _fuzzy_keyword_match(kw, pred)]
                kw_hit_rate = len(join_keywords) / len(valid_keywords)
                keyword_score = 1.0 if kw_hit_rate > 0.3 else kw_hit_rate
            else:
                keyword_score = 0.0

            weighted = 0.3 * keyword_score + 0.7 * semantic_score
            score = max(semantic_score, weighted) if valid_keywords else semantic_score

            if len(gold) <= 20 and gold.strip() in pred:
                score = max(score, 0.90)
            elif 4 <= len(pred.strip()) <= 30 and pred.strip() in gold:
                score = max(score, 0.90)
            elif len(gold) <= 50:
                gold_chars = set(gold.replace(" ", ""))
                pred_chars = set(pred.replace(" ", ""))
                overlap = len(gold_chars & pred_chars) / max(len(gold_chars), 1)
                if overlap > 0.6:
                    score = max(score, 0.80)

        scores.append(score)

    baseline_score = float(np.mean(scores))

    # 6. 对比输出
    print("\n" + "=" * 60)
    print("📊 对比结果")
    print("=" * 60)
    model_label = f"本地 Qwen3-8B" if args.model == "local" else f"{CHAT_MODEL}"
    print(f"{model_label} 得分：{baseline_score:.4f}")

    # 读取本系统得分
    our_result_file = os.path.join(BASE_DIR, "data/ragas_evaluation_result.json")
    our_score = None
    improvement = None
    if os.path.exists(our_result_file):
        with open(our_result_file, "r") as f:
            our_result = json.load(f)
        our_score = our_result.get("semantic_keyword_score", our_result.get("context_recall", 0))
        improvement = (our_score - baseline_score) / (baseline_score + 1e-9) * 100
        print(f"本系统得分：         {our_score:.4f}")
        print(f"提升幅度：           {improvement:+.1f}%")
    print("=" * 60)

    # 保存对比结果
    compare = {
        "baseline_model": args.model,
        "baseline_model_name": CHAT_MODEL,
        "baseline_score": baseline_score,
        "our_score": our_score,
        "improvement_pct": improvement,
        "total_samples": len(results),
    }
    comparison_file = os.path.join(BASE_DIR, f"data/comparison_{args.model}_result.json")
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 对比结果已保存：data/comparison_{args.model}_result.json")


if __name__ == "__main__":
    main()