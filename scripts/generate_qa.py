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

    print("[INFO] Generating QA pairs ->", gen_module.QA_PATH)
    gen_module.gen_qa(splitted_docs, gen_module.CONTEXT_PROMPT_TPL, gen_module.QA_PATH, expand=False)

    # build question docs for generalization
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

    print("[DONE] QA generation finished.")


if __name__ == "__main__":
    main()