# -*- coding: utf-8 -*-
"""MiniCPM 重排器 - bge-reranker-v2-minicpm-layerwise 官方用法

使用 FlagEmbedding 库的 FlagAutoReranker 加载，这是官方推荐方式。
"""

import os
import torch
from langchain_core.documents import Document


class MiniCPMReRanker(object):
    def __init__(self, model_path, max_length=2048, cutoff_layers=None):
        print(f"[INFO] 加载重排模型: {os.path.basename(model_path)}")

        # 必须先设置 device：_init_with_transformers() 内部（self.model.to(self.device)）会用到它
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from FlagEmbedding import FlagAutoReranker
            self.reranker = FlagAutoReranker.from_finetuned(
                model_path,
                query_max_length=max_length // 2,
                passage_max_length=max_length // 2,
                use_fp16=True,
                cutoff_layers=cutoff_layers if cutoff_layers else [28]
            )
            self.use_flag_embedding = True
        except ImportError:
            # FlagEmbedding 可能未安装，或与当前 transformers 版本不兼容（如 5.x 移除了 GEMMA2_START_DOCSTRING）
            print("[WARN] FlagEmbedding 不可用（未安装或与 transformers 版本不兼容），改用原生 transformers 方式")
            self._init_with_transformers(model_path, max_length, cutoff_layers)
            self.use_flag_embedding = False

        print(f"[INFO] MiniCPM 重排模型加载完成，cutoff_layers={cutoff_layers or 28}, device={self.device}")

    def _init_with_transformers(self, model_path, max_length, cutoff_layers):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.model.to(self.device)

        self.cutoff_layers = cutoff_layers or 28
        self.max_length = max_length

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

        self.instruction = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        self.instruction_ids = self.tokenizer(
            self.instruction, return_tensors=None, add_special_tokens=False
        )['input_ids']
        self.sep_ids = self.tokenizer("\n", return_tensors=None, add_special_tokens=False)['input_ids']

    def _get_inputs(self, pairs, max_length=None):
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
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

    def rank(self, query, candidate_docs, topk=10):
        if not candidate_docs:
            return []

        if self.use_flag_embedding:
            pairs = [(query, doc.page_content) for doc in candidate_docs]
            scores = self.reranker.compute_score(pairs, normalize=False)

            ranked = []
            for score, doc in sorted(
                zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
            )[:topk]:
                # 将重排分数写入 metadata，供下游使用
                ranked_doc = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "relevance_score": float(score)},
                )
                ranked.append(ranked_doc)
            return ranked
        else:
            pairs = [(query, doc.page_content) for doc in candidate_docs]
            inputs = self._get_inputs(pairs).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    return_dict=True,
                    cutoff_layers=[self.cutoff_layers],
                )

                all_logits = outputs.logits
                if isinstance(all_logits, tuple):
                    if len(all_logits) > 0:
                        all_logits = all_logits[-1]
                    else:
                        raise RuntimeError("cutoff_layers 返回空元组")

                if not isinstance(all_logits, torch.Tensor):
                    raise RuntimeError(f"模型输出格式错误，期望张量但得到: {type(all_logits)}")

                if all_logits.device != self.device:
                    all_logits = all_logits.to(self.device)

                if all_logits.dim() == 2:
                    scores = all_logits[:, self.yes_loc].view(-1).float()
                elif all_logits.dim() == 3:
                    scores = all_logits[:, -1, self.yes_loc].view(-1).float()
                else:
                    raise RuntimeError(f"logits 维度错误: {all_logits.dim()}")

            scores = scores.detach().cpu().numpy()

            ranked = []
            for score, doc in sorted(
                zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
            )[:topk]:
                # 将重排分数写入 metadata，供下游使用
                ranked_doc = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "relevance_score": float(score)},
                )
                ranked.append(ranked_doc)
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