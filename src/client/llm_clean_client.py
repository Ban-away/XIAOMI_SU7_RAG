# -*- coding: utf-8 -*-

import os
import json
import re
import requests
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
from more_itertools import divide
from langchain_core.documents import Document


MAX_WORKERS = 20

# 清洗提示词：要求模型对原始手册文本做语句规整与标题归并
LLM_CLEAN_PROMPT = """
你是一个专业的文档整理助手，负责对汽车用户手册中的内容进行整理和总结。请根据以下要求对文档进行处理：

1. **让句子变得更加通顺**：重新整合句子、段落，去除一些不必要的符号，例如换行符等。
2. **按标题归类整理**：按照文档的语义关系，把属于同一个标题下的文档做归类合并, 记住标题要用markdown的形式加粗，例如###。

请根据以下文档内容进行整理：
{}
整理后的输出：
"""

llm_client = OpenAI(
    api_key=os.environ['DOUBAO_API_KEY'],
    base_url=os.environ['DOUBAO_BASE_URL']
)


def chat(doc, model="ep-20250206092527-ms2qn"):
    """对单个文档块发起一次清洗请求。"""

    # 调用云端聊天模型，返回规整后的正文
    completion = llm_client.chat.completions.create(
        model=os.environ["DOUBAO_MODEL_NAME"],
        messages=[
            {"role": "user", "content": doc}
        ],
        top_p=0,
        temperature=0.001
    )
    result = completion.choices[0].message.content

    return result


def request_llm_clean(docs):
    """
    并发清洗文档列表。

    返回值仍是 Document 列表，并且保留原有 metadata（页码、图片信息等）。
    """
    # 汇总清洗后的文档
    clean_docs = []
    # 用 unique_id 建映射，后续把清洗结果和原 metadata 对齐
    docs_mapping = {doc.metadata['unique_id']: doc for doc in docs}
    # 按并发数切批，避免一次并发过大
    docs_groups = [list(group) for group in divide(MAX_WORKERS, docs)]

    # 逐批并发执行清洗
    for groups in docs_groups:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交每个文档块的清洗任务
            futures = {doc.metadata['unique_id']: executor.submit(chat,
                LLM_CLEAN_PROMPT.format(doc.page_content)) for doc in groups}

            # 回收每个任务结果
            for unique_id in tqdm(futures):
                future = futures[unique_id]
                result = future.result()
                # 空结果直接跳过
                if result is None:
                    continue
                # 重建 Document，并复用原元数据
                clean_docs.append(
                   Document(page_content=result, metadata=docs_mapping[unique_id].metadata) 
                )
    # 返回清洗完成的文档列表
    return clean_docs


if __name__ == "__main__":
    doc = "".join(open("./data/ut/test_docs.txt").readlines())
    res = chat(LLM_CLEAN_PROMPT.format(doc))
    print(res)
