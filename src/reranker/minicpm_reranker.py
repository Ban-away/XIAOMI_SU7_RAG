# -*- coding: utf-8 -*-
"""MiniCPM 重排器 - bge-reranker-v2-minicpm-layerwise 官方用法

使用 AutoModelForCausalLM 加载，配合官方 prompt 格式：
  "A: {query}\nB: {passage}\n{instruction}"
通过 logits[:, -1, yes_loc] 提取相关性分数。
"""

import os
import torch
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer


class MiniCPMReRanker(object):
    def __init__(self, model_path, max_length=2048, cutoff_layers=None):
        print(f"[INFO] 加载重排模型: {os.path.basename(model_path)}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        self.cutoff_layers = cutoff_layers or 28
        self.max_length = max_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # 预计算 Yes token id
        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

        # 预编码 instruction 和 separator
        self.instruction = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        self.instruction_ids = self.tokenizer(
            self.instruction, return_tensors=None, add_special_tokens=False
        )['input_ids']
        self.sep_ids = self.tokenizer("\n", return_tensors=None, add_special_tokens=False)['input_ids']

        print(f"[INFO] MiniCPM 重排模型加载完成，cutoff_layers={self.cutoff_layers}, device={self.device}")

    def _get_inputs(self, pairs, max_length=None):
        """按官方格式组装输入: BOS + A: query + \n + B: passage + \n + instruction"""
        if max_length is None:
            max_length = self.max_length

        all_inputs = []
        for query, passage in pairs:
            query_ids = self.tokenizer(
                f'A: {query}', add_special_tokens=False, max_length=max_length * 3 // 4, truncation=True
            )['input_ids']
            passage_ids = self.tokenizer(
                f'B: {passage}', add_special_tokens=False, max_length=max_length, truncation=True
            )['input_ids']

            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_ids,
                self.sep_ids + passage_ids,
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item['input_ids'] = item['input_ids'] + self.sep_ids + self.instruction_ids
            item['attention_mask'] = [1] * len(item['input_ids'])
            all_inputs.append(item)

        return self.tokenizer.pad(
            all_inputs,
            padding=True,
            max_length=max_length + len(self.sep_ids) + len(self.instruction_ids),
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

    def rank(self, query, candidate_docs, topk=10):
        if not candidate_docs:
            return []

        pairs = [(query, doc.page_content) for doc in candidate_docs]
        inputs = self._get_inputs(pairs).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                return_dict=True,
                cutoff_layers=[self.cutoff_layers],
            )
            # outputs[0] 是 cutoff_layers 输出的 logits 列表
            all_logits = outputs[0][0]
            scores = all_logits[:, -1, self.yes_loc].view(-1).float()

        scores = scores.detach().cpu().numpy()

        ranked = [
            doc
            for score, doc in sorted(
                zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
            )
        ][:topk]
        return ranked


if __name__ == "__main__":
    model_path = "./models/bge-reranker-v2-minicpm-layerwise/"
    reranker = MiniCPMReRanker(model_path, cutoff_layers=28)
    query = "小米SU7如何开启离车后自动上锁"
    docs = [
        "今天天气不错",
        "离车后自动上锁功能可以在控制>车锁中开启",
        "车辆的最大续航里程为800km",
        "带着手机钥匙离开时车门可以自动锁定",
    ]
    docs = [Document(page_content=doc, metadata={}) for doc in docs]
    response = reranker.rank(query, docs, topk=3)
    for i, doc in enumerate(response):
        print(f"Top {i+1}: {doc.page_content}")
