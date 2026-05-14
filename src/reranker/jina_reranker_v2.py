# -*- coding: utf-8 -*-


import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class JinaRerankerV2(object):
    def __init__(self, model_path: str, max_length: int = 4096, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        if self.device.startswith("cuda"):
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    def rank(self, query: str, candidate_docs: list[Document], topk: int = 10) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in candidate_docs]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().clone().numpy()
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
