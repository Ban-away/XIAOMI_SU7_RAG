# -*- coding: utf-8 -*-


import os
import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel


class BGEM3ReRanker(object):
    def __init__(self, model_path, max_length=4096):
        print(f"[INFO] 加载重排模型: {os.path.basename(model_path)}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 尝试加载模型（双层策略）
        try:
            # 方法1：标准序列分类模型
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
            self.model_type = "sequence_classification"
        except ValueError:
            # 方法2：自定义模型（如 bge-reranker-v2-minicpm-layerwise）
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.model_type = "custom"
        
        # 切换到推理模式，关闭 dropout
        self.model.eval()
        # 使用 fp16 降低显存占用
        self.model.half()
        # 模型放到 GPU
        self.model.cuda()
        # 文本对最大长度（query + doc）
        self.max_length = max_length
        
        print(f"[INFO] 重排模型加载完成")


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
            outputs = self.model(**inputs)
            
            # 处理不同模型的输出格式
            if hasattr(outputs, 'logits'):
                scores = outputs.logits
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                scores = outputs[0]
            else:
                scores = outputs
        
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