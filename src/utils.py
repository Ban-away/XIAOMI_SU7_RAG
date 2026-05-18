# -*- coding: utf-8 -*-


import re
from langchain_core.documents import Document
from src.client.mongodb_config import MongoConfig

manual_collection = MongoConfig.get_collection("manual_text")


def wrrf_fusion(results_list, weights=None, k=60):
    """
    Weighted Reciprocal Rank Fusion (WRRF) 加权倒数排名融合

    公式: score(d) = Σ (w_i / (k + rank_i(d)))
    
    参数:
        results_list: 多个检索器的结果列表，每个元素是一个 Document 列表
        weights: 各检索器的权重列表，长度应与 results_list 相同
        k: 排名衰减常数，通常取 60-100
    
    返回:
        按 WRRF 分数排序后的 Document 列表
    """
    if not results_list or all(not results for results in results_list):
        return []
    
    # 默认权重相等
    if weights is None:
        weights = [1.0] * len(results_list)
    
    # 存储每个文档的分数和原始文档对象
    doc_scores = {}
    doc_map = {}
    
    for idx, results in enumerate(results_list):
        weight = weights[idx]
        for rank, doc in enumerate(results, 1):
            # 获取文档唯一标识
            parent_id = doc.metadata.get("parent_id")
            unique_id = parent_id if parent_id else doc.metadata.get("unique_id", str(id(doc)))
            
            # 计算 WRRF 分数
            if unique_id not in doc_scores:
                doc_scores[unique_id] = 0
            doc_scores[unique_id] += weight / (k + rank)
            
            # 保存原始文档（第一次出现时保存）
            if unique_id not in doc_map:
                doc_map[unique_id] = doc
    
    # 按分数降序排序
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: -doc_scores[x])
    
    # 返回排序后的文档列表
    return [doc_map[uid] for uid in sorted_ids]


def merge_docs(docs1, docs2, use_wrrf=True):
    """
    合并两路召回文档，并按父块去重，支持 WRRF 排序。

    设计目的：
    1. 召回阶段会同时返回父块与子块，直接拼接会有重复信息。
    2. 如果命中子块（含 parent_id），这里回溯父块，统一上下文粒度。
    3. 使用 WRRF 融合提升检索效果。

    参数:
        docs1: 第一路检索结果（如 BM25）
        docs2: 第二路检索结果（如 Milvus）
        use_wrrf: 是否使用 WRRF 排序
    
    返回:
        合并去重后的文档列表
    """
    # 存储去重后的文档及其在原始检索中的排名信息
    merged_ids = set()
    docs1_unique = []  # 去重后的第一路结果
    docs2_unique = []  # 去重后的第二路结果
    
    # 处理第一路检索结果
    for doc in docs1:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            parent_mg = manual_collection.find_one({"unique_id": parent_id})
            if not parent_mg:
                continue
            unique_id = parent_mg["unique_id"]
            if unique_id not in merged_ids:
                merged_ids.add(unique_id)
                parent_doc = Document(page_content=parent_mg["page_content"], metadata=parent_mg["metadata"])
                docs1_unique.append(parent_doc)
        else:
            unique_id = doc.metadata.get("unique_id")
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                docs1_unique.append(doc)
    
    # 重置去重集合，处理第二路
    merged_ids.clear()
    
    for doc in docs2:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            parent_mg = manual_collection.find_one({"unique_id": parent_id})
            if not parent_mg:
                continue
            unique_id = parent_mg["unique_id"]
            if unique_id not in merged_ids:
                merged_ids.add(unique_id)
                parent_doc = Document(page_content=parent_mg["page_content"], metadata=parent_mg["metadata"])
                docs2_unique.append(parent_doc)
        else:
            unique_id = doc.metadata.get("unique_id")
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                docs2_unique.append(doc)
    
    # 使用 WRRF 融合排序
    if use_wrrf:
        # BM25 和 Milvus 的权重可以根据效果调整
        # 这里设置 BM25 权重为 0.4，Milvus 权重为 0.6（向量检索通常更重要）
        return wrrf_fusion([docs1_unique, docs2_unique], weights=[0.4, 0.6], k=60)
    else:
        # 传统方式：简单拼接去重
        final_ids = set()
        final_docs = []
        for doc in docs1_unique + docs2_unique:
            unique_id = doc.metadata.get("unique_id")
            if unique_id and unique_id not in final_ids:
                final_ids.add(unique_id)
                final_docs.append(doc)
        return final_docs




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
        if index > len(docs) or index < 1:
            continue
        # docs 中引用编号从 1 开始，所以访问时减 1
        doc_ref = docs[index-1]
        # 确保元数据中存在 images_info 字段，避免 KeyError
        images = doc_ref.metadata.get("images_info", [])
        page = doc_ref.metadata.get("page")
        if page is not None:
            pages.append(page)
        for image in images:
            # 只保留能识别出标题的图片，提高结果可用性
            if image.get("title"):
                related_images.append(image)

    # 页码去重并排序，便于前端展示
    pages = sorted(list(set(pages)))

    # 返回结构化结果，供接口层或前端直接消费
    return {
        "answer": answer,
        "cite_pages": pages,
        "related_images": related_images
    }