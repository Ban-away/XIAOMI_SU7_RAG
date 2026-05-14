# -*- coding: utf-8 -*-

import random
import json
import requests
import os
import pickle
from typing import List

from src.constant import clean_docs_path


URL = os.getenv("SEMANTIC_CHUNK_URL", "http://localhost:6000/v1/semantic-chunks")


def request_semantic_chunk(sentences, group_size):
    """调用语义切分服务，将长文本切分为语义块列表。"""
    headers = {
        "Content-Type":"application/json"
    }
    payload = json.dumps({
        "sentences": sentences,
        "group_size": group_size
    })
    try:
        # 调用本地语义切分服务，避免请求无上限等待
        response = requests.post(
            URL,
            headers=headers,
            data=payload,
            timeout=30
        )
        # 非 2xx 状态直接抛错，避免把错误页当作正常 JSON 解析
        response.raise_for_status()
        res = response.json()
        text = res["chunks"]
    except Exception as e:
        print(f"call reject failed:{e}")
        # 兜底返回列表，避免上游按“可迭代字符串”逐字符误切分
        text = [sentences]
    return text


if __name__ == '__main__':
    data = pickle.load(open("data/processed_docs/clean_docs.pkl", "rb"))
    index = random.sample(range(len(data)), 10)
    for idx in index:
        doc = data[idx].page_content
        res = request_semantic_chunk(doc, 10)
        print("="*100)
        for r in res:
            print(r)
            print("="*100)

