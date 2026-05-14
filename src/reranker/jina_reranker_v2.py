# -*- coding: utf-8 -*-


import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class JinaRerankerV2(object):
    def __init__(self, model_path: str, max_length: int = 4096, device: str | None = None):
        # 未指定 device 时自动选择：优先 CUDA
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # 加载 tokenizer 与序列分类模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # GPU 场景下启用 fp16，降低显存占用
        if self.device.startswith("cuda"):
            self.model = self.model.half()
        # 模型移动到目标设备并切换 eval
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    def rank(self, query: str, candidate_docs: list[Document], topk: int = 10) -> list[Document]:
        # 把候选文档转换为 (query, doc) 文本对
        pairs = [(query, doc.page_content) for doc in candidate_docs]
        # 批量编码并搬运到推理设备
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            scores = self.model(**inputs).logits
        # logits 转 numpy，便于排序
        scores = scores.detach().cpu().clone().numpy()
        # 按得分降序排序后截取 topk
        response = [
            doc
            for score, doc in sorted(
                zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
            )
        ][:topk]
        return response


if __name__ == "__main__":
    jina_reranker = JinaRerankerV2("./models/jinaai/jina-reranker-v2-base-multilingual")
    query = "今天天气怎么样"
    docs = ["你好", "今天天气不错", "今天有雨吗"]
    docs = [Document(page_content=doc, metadata={}) for doc in docs]
    response = jina_reranker.rank(query, docs)
    print(response)
