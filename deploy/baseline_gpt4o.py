# -*- coding: utf-8 -*-
"""
基线对比实验
支持多种模型：GPT-4o / 豆包 / 自定义 OpenAI 兼容 API

对比指标：语义相似度 + 关键词加权得分，与本系统结果做比较

运行前准备：
  1. 使用豆包 API（推荐，更便宜）：
     export DOUBAO_API_KEY=sk-xxx
     python deploy/baseline_gpt4o.py --model doubao
  
  2. 使用本地模型（完全免费，无需API）：
     python deploy/baseline_gpt4o.py --model local
     
  3. 使用 OpenAI API（GPT-4o等）：
     export OPENAI_API_KEY=sk-xxx
     python deploy/baseline_gpt4o.py --model openai

运行：
  python deploy/baseline_gpt4o.py --model doubao   # 豆包API（便宜）
  python deploy/baseline_gpt4o.py --model local    # 本地模型（免费）
  python deploy/baseline_gpt4o.py --model openai   # OpenAI API（GPT-4o等）
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from text2vec import SentenceModel, semantic_search

load_dotenv()

# 导入常量路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constant import split_docs_path, text2vec_model_path


def main():
    # ── 解析命令行参数 ─────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="基线对比实验")
    parser.add_argument("--model", type=str, default="local", 
                        choices=["doubao", "openai", "local"],
                        help="选择对比模型：doubao（豆包）、openai（GPT-4o等）、local（本地模型，免费）")
    args = parser.parse_args()
    
    # ── 配置 ─────────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TOPK = 8
    RESULT_FILE = os.path.join(BASE_DIR, f"data/baseline_{args.model}_result.json")
    
    # 根据选择的模型加载配置
    if args.model == "doubao":
        API_KEY = os.getenv("DOUBAO_API_KEY")
        BASE_URL = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        CHAT_MODEL = os.getenv("DOUBAO_MODEL_NAME", "doubao-1-5-lite-32k-250115")
        print(f"[INFO] 使用模型：豆包 {CHAT_MODEL}")
    elif args.model == "openai":
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
        
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
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
        prompt = PROMPT.format(query=query, context=context)
        try:
            if args.model == "local" and local_client:
                response = local_client.chat.completions.create(
                    model="qwen3_lora_sft_int4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.01,
                )
            elif client:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
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

    model_desc = "本地Qwen3" if args.model == "local" else ("豆包" if args.model == "doubao" else CHAT_MODEL)
    for item in tqdm(test_data, desc=f"{model_desc} 推理"):
        query = item["question"]
        docs = retrieve(vector_store, query)
        context = "\n".join(
            [f"【{i+1}】{doc.page_content}" for i, doc in enumerate(docs)]
        )
        pred = generate(query, context)

        results.append({
            "unique_id": item["unique_id"],
            "question": query,
            "answer": item["answer"],
            "keywords": item.get("keywords", []),
            "pred": pred,
            "context": context,
        })

        if len(results) % 50 == 0:
            with open(RESULT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
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
    model_label = f"本地 Qwen3-8B" if args.model == "local" else (f"豆包 {CHAT_MODEL}" if args.model == "doubao" else f"{CHAT_MODEL}")
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
    with open(os.path.join(BASE_DIR, "data/comparison_result.json"), "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)
    print("[INFO] 对比结果已保存：data/comparison_result.json")


if __name__ == "__main__":
    main()