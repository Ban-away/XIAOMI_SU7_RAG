# -*- coding: utf-8 -*-


import warnings
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document
warnings.filterwarnings("ignore")


class Qwen3ReRanker(object):
    def __init__(self, model_path, max_length=4096):
        # 加载 rerank 模型

        # 这里使用 CausalLM 通过 yes/no 概率实现重排打分
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_path).cuda().eval()

        # 预取 yes/no 的 token id，用于提取末位分类概率
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = max_length 

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
                
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'


    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(self, pairs):
        # 先编码正文，再拼接 system/user 包装 token
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        # 统一 pad 成 batch tensor
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            # 全量搬到模型所在设备
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        # 取最后一个 token 位的 logits，并抽取 yes/no 两个通道
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        # 归一化后取 yes 概率作为相关性分数
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rank(self, query, candidate_docs, topk=10):
        # 输入文档对，返回每一对(query, doc)的相关得分，并从大到小排序
        """
        queries = [query] * len(candidate_docs)
        documents = [doc.page_content for doc in candidate_docs]

        # 如果显存不足, 可以改为单条预测
        """
        scores = []
        for docc in candidate_docs:
            # 逐条评分（显存受限场景更稳）
            documents = [docc.page_content]
            queries = [query]

            pairs = [self.format_instruction(self.task, query, doc) for query, doc in zip(queries, documents)]

            # Tokenize the input texts
            inputs = self.process_inputs(pairs)
            # 计算当前文档评分
            score = self.compute_logits(inputs)
            scores.append(score)

        pairs = [self.format_instruction(self.task, query, doc) for query, doc in zip(queries, documents)]

        # 统一再跑一次批量推理，得到可排序分数
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        response = [
            doc
            for score, doc in sorted(
                zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
            )
            ][:topk]
        return response


if __name__ == "__main__":
    qwen3_reranker = "./models/Qwen3-Reranker-4B"
    bge_rerank = Qwen3ReRanker(qwen3_reranker)
    query = "今天天气怎么样"
    docs = ["你好", "今天天气不错", "今天有雨吗"]
    docs = [Document(page_content=doc, metadata={}) for doc in docs]
    response = bge_rerank.rank(query, docs)
    print(response)
