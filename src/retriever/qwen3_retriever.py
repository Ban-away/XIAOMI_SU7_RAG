# -*- coding: utf-8 -*-


import os
import torch
import hashlib
from typing import List
import torch.nn.functional as F
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from src.constant import faiss_qwen_db_path, qwen3_embedding_model_path 
from src.retriever.retriever import BaseRetriever


# ======================
# 自定义 Embedding 类
# ======================
class Qwen3Embeddings(Embeddings):
    """自定义 Transformers 嵌入模型"""

    def __init__(self, model_name: str, device: str = "cuda:0"):
        # 推理设备（默认第一张 GPU）
        self.device = device
        # Qwen tokenizer 使用左填充，方便取最后 token 表征
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        # 文本结尾标记 token id
        self.eod_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        # 最大输入长度
        self.max_length = 8192
        # embedding 指令模板（query 侧）
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'


    def last_token_pool(self, last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def tokenize(self, tokenizer, input_texts, eod_id, max_length):
        # 先做截断编码，为手工追加 EOD 留出位置
        batch_dict = tokenizer(input_texts, padding=False, truncation=True, max_length=max_length-2)
        for seq, att in zip(batch_dict["input_ids"], batch_dict["attention_mask"]):
            # 在每个样本末尾手动追加 EOD
            seq.append(self.eod_id)
            att.append(1)
        # 再统一 pad 成 batch tensor
        batch_dict = tokenizer.pad(batch_dict, padding=True, return_tensors="pt")
        return batch_dict

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文档嵌入"""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """查询嵌入"""
        queries = [
            self.get_detailed_instruct(self.task, text)
        ]
        return self._embed(queries)[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """实际嵌入实现"""
        # Tokenize the input texts
        batch_dict = self.tokenize(self.tokenizer, texts, self.eod_id, self.max_length)
        # 把 batch 放到模型设备
        batch_dict.to(self.model.device)
        with torch.no_grad():
            outputs =self.model(**batch_dict)
            # 取 last token 表征作为句向量
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        # L2 归一化，便于后续向量相似度检索
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().detach().numpy().tolist()


class FaissRetriever(BaseRetriever):
    def __init__(self, docs, retrieve=False):
        # 初始化 Qwen3 向量编码器
        self.embeddings = Qwen3Embeddings(
            model_name=qwen3_embedding_model_path
        )

        if retrieve and os.path.exists(faiss_qwen_db_path):
            # 如果本地已存在索引，直接加载
            self.vector_store = FAISS.load_local(
                faiss_qwen_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # 否则重新构建并落盘
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(faiss_qwen_db_path)

        # 使用完模型后释放显存
        del self.embeddings
        torch.cuda.empty_cache()

    def retrieve_topk(self, query, topk):
        # 获取top-K分数最高的文档块
        # 返回值格式为[(Document, score), ...]
        context = self.vector_store.similarity_search_with_score(query, k=topk)
        return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vector_store


if __name__ == "__main__":
    texts = ["打开车窗", "空调加热", "加热座椅"]
    docs = []
    for text in texts:
        unique_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        metadata = {"unique_id": unique_id}
        docs.append(Document(page_content=text, metadata=metadata))

    # faiss 召回: bce-base, 类似的embedding模型还可以采用gte, m3e, bge
    bce_faissretriever = FaissRetriever(docs)
    bce_faiss_ans = bce_faissretriever.retrieve_topk("座椅加热", 3)
    print(bce_faiss_ans)
