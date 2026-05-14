# -*- coding: utf-8 -*-


import re
from langchain_core.documents import Document
from src.client.mongodb_config import MongoConfig

manual_collection = MongoConfig.get_collection("manual_text")


def merge_docs(docs1, docs2):
    """
    合并两路召回文档，并按父块去重。

    设计目的：
    1. 召回阶段会同时返回父块与子块，直接拼接会有重复信息。
    2. 如果命中子块（含 parent_id），这里回溯父块，统一上下文粒度。
    """
    # 最终合并结果（去重后）
    merged_docs = []
    # 记录已加入结果集的 unique_id，防止重复
    merged_ids = set()
    # 拼接两路候选文档（例如 BM25 + Milvus）
    candidate_docs = docs1 + docs2

    # 逐条处理候选文档
    for doc in candidate_docs:
        # 如果是子块，metadata 会携带 parent_id
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            # 回表找到父块原文
            parent_mg = manual_collection.find_one({"unique_id": parent_id})
            # 防御：如果 Mongo 数据丢失，直接跳过这条
            if not parent_mg:
                continue
            unique_id = parent_mg["unique_id"]
            # 只保留第一次命中的父块
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                # 重新包装成 LangChain Document，保持下游接口一致
                parent_doc = Document(page_content=parent_mg["page_content"], metadata=parent_mg["metadata"])
                merged_docs.append(parent_doc)
        else:
            # 当前文档本身就是父块（或没有 parent_id），直接按自身 unique_id 去重
            unique_id = doc.metadata.get("unique_id")
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                merged_docs.append(doc)
    return merged_docs




def post_processing(response, docs):
    """
    对模型回答做后处理：
    1) 解析引用编号
    2) 去除引用标记得到纯答案
    3) 根据引用编号回填页码与相关图片
    """
    # 提取回答中的所有引用片段，如【1,2】、【3】
    all_cites = re.findall("[【](.*?)[】]", response) 
    # 聚合引用编号（去重前）
    cites = []

    # 逐段解析引用编号
    for cite in all_cites:
        # 清理无关字符
        cite = re.sub("[{} 【】]", "", cite)
        # 统一中英文逗号，避免 split 漏切
        cite = cite.replace(",", "，")
        # 只保留数字编号
        cite = [int(k) for k in cite.split("，") if k.isdigit()]
        cites.extend(cite)

    # 引用编号去重
    cites = list(set(cites))
    # 去除回答里的【引用】标记，得到可读答案正文
    answer = re.sub("[【](.*?)[】]", "", response)
    answer = re.sub("[{}【】]", "", answer)

    # 收集输出结构中的补充信息
    related_images = []
    pages = []

    # 遍历每个引用编号，提取对应文档的页码与图片信息
    for index in cites:
        # 编号越界时跳过，避免索引错误
        if index > len(docs):
            continue
        # docs 中引用编号从 1 开始，所以访问时减 1
        images = docs[index-1].metadata["images_info"]
        pages.append(docs[index-1].metadata["page"])
        for image in images:
            # 只保留能识别出标题的图片，提高结果可用性
            if image["title"]:
                related_images.append(image)

    # 页码去重并排序，便于前端展示
    pages = sorted(list(set(pages)))

    # 返回结构化结果，供接口层或前端直接消费
    return {
        "answer": answer,
        "cite_pages": pages,
        "related_images": related_images
    }
