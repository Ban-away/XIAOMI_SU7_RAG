# -*- coding: utf-8 -*-

import os
import json
import re
from openai import OpenAI
from langchain_core.documents import Document


LLM_HYDE_PROMPT = """
你是一位小米 SU7 汽车专家，现在请你结合小米 SU7 车辆和新能源电动汽车相关知识回答下列问题。
请给出用户问题的使用方法，详细分析问题原因，返回有用的内容。
{query}
最终的回答请尽可能的精简, 不超过100字:
"""


LLM_QUERY_REWRITE_PROMPT = """
你是一个智能问题改写助手，专门处理用户关于小米SU7汽车的问题。

请对以下用户问题进行分析和优化：
1. **纠错**: 识别并修正问题中的错别字、语法错误或表达不清的地方
2. **扩写**: 将简短的问题扩展为更完整、更明确的表达，补充必要的上下文
3. **语义增强**: 补充相关的同义词、近义词，帮助后续检索更好地理解用户意图

原始问题：{query}

请输出优化后的问题，保持原意不变，不超过50字：
"""


llm_client = OpenAI(
    api_key=os.environ['DOUBAO_API_KEY'],
    base_url=os.environ['DOUBAO_BASE_URL']
)


def request_hyde(query):
    """生成 HyDE 扩展文本，增强后续检索召回。"""
    prompt = LLM_HYDE_PROMPT.format(query=query) 

    completion = llm_client.chat.completions.create(
        model=os.environ["DOUBAO_MODEL_NAME"],
        messages=[
            {"role": "system", "content": "你是一个有用的人工智能助手."},
            {"role": "user", "content": prompt}
        ],
        top_p=0,
        temperature=0.001
    )
    result = completion.choices[0].message.content

    return result


def request_query_rewrite(query):
    """
    Query纠错改写：在检索前用LLM对query做纠错和扩写。
    
    Args:
        query: 用户原始问题
        
    Returns:
        优化后的问题字符串
    """
    prompt = LLM_QUERY_REWRITE_PROMPT.format(query=query)

    completion = llm_client.chat.completions.create(
        model=os.environ["DOUBAO_MODEL_NAME"],
        messages=[
            {"role": "system", "content": "你是一个专业的问题改写助手，擅长处理汽车相关问题。"},
            {"role": "user", "content": prompt}
        ],
        top_p=0.9,
        temperature=0.1
    )
    result = completion.choices[0].message.content

    return result.strip()



if __name__ == "__main__":
    query = "介绍一下离车后自动上锁功能"
    res = request_hyde(query)
    print("HyDE结果:", res)
    
    query2 = "车窗怎么开"
    res2 = request_query_rewrite(query2)
    print("Query改写结果:", res2)