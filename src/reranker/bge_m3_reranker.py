# -*- coding: utf-8 -*-


import os
import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEM3ReRanker(object):
    def __init__(self, model_path, max_length=4096):
        # 打印模型路径调试信息
        print(f"[DEBUG] 尝试加载重排模型: {model_path}")
        print(f"[DEBUG] 路径是否存在: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            print(f"[DEBUG] 目录内容: {os.listdir(model_path)[:5]}...")  # 显示前5个文件
        else:
            print(f"[ERROR] 模型路径不存在!")

        # 加载 tokenizer 与 sequence classification 模型
        # trust_remote_code=True 用于加载包含自定义代码的模型（如 bge-reranker-v2-minicpm-layerwise）
        # local_files_only=True 强制只使用本地文件，不连接 HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        # 切换到推理模式，关闭 dropout
        self.model.eval()
        # 使用 fp16 降低显存占用
        self.model.half()
        # 模型放到 GPU
        self.model.cuda()
        # 文本对最大长度（query + doc）
        self.max_length = max_length


    def rank(self, query, candidate_docs, topk=10):
        # 输入文档对，返回每一对(query, doc)的相关得分，并从大到小排序
        # 组装 query-doc 文本对
        pairs = [(query, doc.page_content) for doc in candidate_docs]
        # 批量编码后送入模型
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to("cuda")
        with torch.no_grad():
            scores = self.model(**inputs).logits
        # 张量转 numpy，便于 Python 层排序
        scores = scores.detach().cpu().clone().numpy()
        # 依据分数降序排序并截断 topk
        response = [
            doc
            for score, doc in sorted(
                zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
            )
            ][:topk]
        return response


if __name__ == "__main__":
    bge_reranker_large = "./models/BAAI/bge-reranker-v2-m3/"
    # bce_reranker_base = "../../models/bce-reranker-base-v1"
    bge_rerank = BGEM3ReRanker(bge_reranker_large)
    query = "今天天气怎么样"
    docs = ["你好", "今天天气不错", "今天有雨吗"]
    docs = [Document(page_content=doc, metadata={}) for doc in docs]
    response = bge_rerank.rank(query, docs)
    print(response)