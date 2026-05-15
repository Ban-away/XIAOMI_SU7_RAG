#!/usr/bin/env python3
"""Generate QA pairs and expanded QA from split docs.

Usage:
  python scripts/generate_qa.py

This script expects `data/processed_docs/split_docs.pkl` to exist (run `python build_index.py`).
It writes `data/qa_pairs/qa_pair.json` and `data/qa_pairs/expand_qa_pair.json`.
"""
import os
import sys
import pickle
import json

from langchain_core.documents import Document

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gen_qa import run as gen_module


def main():
    os.makedirs("data/qa_pairs", exist_ok=True)

    split_path = "data/processed_docs/split_docs.pkl"
    if not os.path.exists(split_path):
        raise SystemExit(f"missing split docs: {split_path} — run python build_index.py first")

    with open(split_path, "rb") as fd:
        splitted_docs = pickle.load(fd)

    # Step 1: 生成原始 QA 对
    print("[INFO] Generating QA pairs ->", gen_module.QA_PATH)
    gen_module.gen_qa(splitted_docs, gen_module.CONTEXT_PROMPT_TPL, gen_module.QA_PATH, expand=False)

    # Step 2: 生成扩展 QA 对
    question_docs = []
    idx = 0
    with open(gen_module.QA_PATH, "r", encoding="utf-8") as fd:
        for line in fd:
            info = json.loads(line)
            try:
                resp = json.loads(info.get("raw_resp", "[]"))
            except Exception:
                continue
            for qa in resp:
                question = qa.get("question", "").strip()
                if not question:
                    continue
                question_docs.append(Document(page_content=question, metadata={"unique_id": str(idx)}))
                idx += 1

    print("[INFO] Generating expanded QA ->", gen_module.OUTPUT_PATH)
    gen_module.gen_qa(question_docs, gen_module.GENERALIZE_PROMPT_TPL, gen_module.OUTPUT_PATH, expand=True)

    # Step 3: 合并并生成 train_qa_pair.json 和 test_qa_pair.json
    print("[INFO] Generating train/test splits ->", gen_module.TRAIN_PATH, gen_module.TEST_PATH)
    
    # 加载原始 QA 和扩展 QA
    qa_dict = {}
    with open(gen_module.QA_PATH) as fd:
        for line in fd:
            info = json.loads(line)
            qa_dict[info["unique_id"]] = info
    
    expand_qa_dict = {}
    with open(gen_module.OUTPUT_PATH) as fd:
        for line in fd:
            info = json.loads(line)
            expand_qa_dict[info["unique_id"]] = info

    # 构建扩展问题映射
    expand_qa_pairs = {}
    for unique_id, info in expand_qa_dict.items():
        question = info["unique_id"]
        expand_questions = info["raw_resp"]
        expand_questions = expand_questions.split("\n")
        expand_questions = [re.sub(r'^\d[.. ]', '', item).strip() for item in expand_questions]
        expand_qa_pairs[question] = expand_questions

    # 生成训练集和测试集
    train_qa_pairs = []
    test_qa_pairs = []
    for unique_id, info in qa_dict.items():
        resp = json.loads(info["raw_resp"])
        for qa in resp:
            question = qa["question"].strip()
            answer = qa["answer"].strip()
            if "无法准确" in answer or "未提及" in answer:
                continue
            expand_questions = [question] + expand_qa_pairs.get(question, [])
            for query in expand_questions:
                item = {
                    "unique_id": hashlib.md5(query.encode('utf-8')).hexdigest(),
                    "question": query,
                    "answer": answer
                }
                if random.random() < 0.9:
                    train_qa_pairs.append(item)
                else:
                    test_qa_pairs.append(item)

    # 写入文件
    random.seed(42)
    with open(gen_module.TRAIN_PATH, "w", encoding="utf-8") as fd:
        random.shuffle(train_qa_pairs)
        fd.write(json.dumps(train_qa_pairs, ensure_ascii=False, indent=2))
        print(f"[INFO] 训练集已写入: {gen_module.TRAIN_PATH}, 数量: {len(train_qa_pairs)}")

    random.seed(42)
    with open(gen_module.TEST_PATH, "w", encoding="utf-8") as fd:
        random.shuffle(test_qa_pairs)
        fd.write(json.dumps(test_qa_pairs, ensure_ascii=False, indent=2))
        print(f"[INFO] 测试集已写入: {gen_module.TEST_PATH}, 数量: {len(test_qa_pairs)}")

    print("[DONE] QA generation finished.")


if __name__ == "__main__":
    main()