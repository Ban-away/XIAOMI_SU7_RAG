# -*- coding: utf-8 -*-

import os
import pickle
import jieba
import hashlib
from langchain.schema import Document
from langchain_community.retrievers import TFIDFRetriever

from src.constant import tfidf_pickle_path


class TFIDF(object):
    def __init__(self, docs, retrieve=False):

        # 创建待编码文档集
        self.documents = docs

        # 初始化TFIDF的知识库
        self.retriever = self.get_TFIDF_retriever(retrieve=retrieve)


    def get_TFIDF_retriever(self, retrieve):
        """
        获取TFIDF检索器，如果已经存在则加载，否则创建并持久化
        """
        # 检索阶段：优先复用历史索引文件
        if retrieve and os.path.exists(tfidf_pickle_path):
            tfidf_retriever = pickle.load(open(tfidf_pickle_path, 'rb'))
        else:
            # 建库阶段：从文档语料创建 TF-IDF 检索器并持久化
            tfidf_retriever = TFIDFRetriever.from_documents(self.documents, preprocess_func=self.tokenize)
            pickle.dump(tfidf_retriever, open(tfidf_pickle_path, 'wb'))
        return tfidf_retriever


    def tokenize(self, text):
        """
        使用jieba进行中文分词
        """
        # 直接使用 jieba 分词，不做停用词过滤
        return jieba.lcut(text)


    def retrieve_topk(self, query, topk=10):
        # 获得得分在topk的文档和分数
        # 设置本次检索条数
        self.retriever.k = topk
        # 对 query 做搜索模式分词，提升召回覆盖
        query = " ".join(jieba.cut_for_search(query))
        # 执行召回并返回文档列表
        ans_docs = self.retriever.get_relevant_documents(query)
        return ans_docs


if __name__ == "__main__":
    texts = ["打开车窗", "空调加热", "加热座椅"]
    docs = []
    for text in texts:
        unique_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        metadata = {"unique_id": unique_id}
        docs.append(Document(page_content=text, metadata=metadata))
    tfidf = TFIDF(docs)
    tfidf_res = tfidf.retrieve_topk("座椅加热", 3)
    print(tfidf_res)

