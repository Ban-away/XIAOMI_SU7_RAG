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
            # 获取文档唯一标识（传入的文档已经是父块，无需处理 parent_id）
            unique_id = doc.metadata.get("unique_id", str(id(doc)))

            # 计算 WRRF 分数（同一文档在多路出现时累加）
            if unique_id not in doc_scores:
                doc_scores[unique_id] = 0
            doc_scores[unique_id] += weight / (k + rank)

            # 保存原始文档（第一次出现时保存）
            if unique_id not in doc_map:
                doc_map[unique_id] = doc

    # 按分数降序排序
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: -doc_scores[x])

    return [doc_map[uid] for uid in sorted_ids]


def _fetch_parent_doc(doc):
    """
    如果文档是子块（含 parent_id），回表取父块；否则直接返回原文档。
    返回 (unique_id, Document) 或 None（回表失败时）
    """
    parent_id = doc.metadata.get("parent_id")
    if parent_id:
        parent_mg = manual_collection.find_one({"unique_id": parent_id})
        if not parent_mg:
            return None
        return parent_mg["unique_id"], Document(
            page_content=parent_mg["page_content"],
            metadata=parent_mg["metadata"]
        )
    else:
        unique_id = doc.metadata.get("unique_id")
        if not unique_id:
            return None
        return unique_id, doc


def merge_docs(docs1, docs2, use_wrrf=True):
    """
    合并两路召回文档，并按父块去重，支持 WRRF 排序。

    设计目的：
    1. 召回阶段会同时返回父块与子块，直接拼接会有重复信息。
    2. 如果命中子块（含 parent_id），这里回溯父块，统一上下文粒度。
    3. 使用 WRRF 融合提升检索效果。

    注意：两路分别去重后再传入 WRRF，让 WRRF 感知到同一文档在两路都出现
    （在两路各自的排名中都会被计分，最终分数更高），而不是提前去重。

    参数:
        docs1: 第一路检索结果（如 BM25）
        docs2: 第二路检索结果（如 Milvus）
        use_wrrf: 是否使用 WRRF 排序

    返回:
        合并去重后的文档列表
    """
    def dedup_single_list(docs):
        """对单路结果做父块回表 + 去重，保持排名顺序"""
        seen = set()
        result = []
        for doc in docs:
            ret = _fetch_parent_doc(doc)
            if ret is None:
                continue
            unique_id, parent_doc = ret
            if unique_id not in seen:
                seen.add(unique_id)
                result.append(parent_doc)
        return result

    docs1_unique = dedup_single_list(docs1)
    docs2_unique = dedup_single_list(docs2)

    if use_wrrf:
        # BM25 权重 0.5，Milvus 权重 0.7
        # WRRF 内部会对两路中都出现的文档累加分数，自动实现跨路去重和重新排名
        return wrrf_fusion(
            [docs1_unique, docs2_unique],
            weights=[0.5, 0.7],
            k=60
        )
    else:
        # 传统方式：简单拼接去重
        final_ids = set()
        final_docs = []
        for doc in docs1_unique + docs2_unique:
            uid = doc.metadata.get("unique_id")
            if uid and uid not in final_ids:
                final_ids.add(uid)
                final_docs.append(doc)
        return final_docs


def post_processing(response, docs):
    """
    对模型回答做后处理：
    1) 解析引用编号
    2) 去除引用标记得到纯答案
    3) 根据引用编号回填页码与相关图片
    """
    all_cites = re.findall("[【](.*?)[】]", response)
    cites = []

    for cite in all_cites:
        cite = re.sub("[{} 【】]", "", cite)
        cite = cite.replace(",", "，")
        cite = [int(k) for k in cite.split("，") if k.isdigit()]
        cites.extend(cite)

    cites = list(set(cites))
    answer = re.sub("[【](.*?)[】]", "", response)
    answer = re.sub("[{}【】]", "", answer)

    related_images = []
    pages = []

    for index in cites:
        if index > len(docs) or index < 1:
            continue
        doc_ref = docs[index - 1]
        images = doc_ref.metadata.get("images_info", [])
        page = doc_ref.metadata.get("page")
        if page is not None:
            pages.append(page)
        for image in images:
            if image.get("title"):
                related_images.append(image)

    pages = sorted(list(set(pages)))

    return {
        "answer": answer,
        "cite_pages": pages,
        "related_images": related_images
    }