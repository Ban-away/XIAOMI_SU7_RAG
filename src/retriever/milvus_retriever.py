# -*- coding: utf-8 -*-


import os
import time
import hashlib
import pandas as pd
import torch
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker
)
from langchain_core.documents import Document
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from src.fields.manual_images import ManualImages
from src.constant import (
    test_doc_path,
    bge_large_zh_v1_5_model_path,
    splade_v2_model_path,
    milvus_db_path,
)
from src.client.mongodb_config import MongoConfig


EMB_BATCH = 32  # 减小批处理大小，降低显存占用
MAX_TEXT_LENGTH = 2048  # 增加最大文本长度限制
ID_MAX_LENGTH = 100
COL_NAME = "hybrid_bge_large_splade_v2"

# 多GPU配置：自动检测可用GPU数量
NUM_GPUS = torch.cuda.device_count()
print(f"检测到 {NUM_GPUS} 个可用 GPU") 

# 连接 Mongo 文本集合：用于把 Milvus 命中的 unique_id 回表成完整 Document
mongo_collection = MongoConfig.get_collection("manual_text")
# 连接 Milvus Lite 本地库（uri 指向 data/saved_index/milvus.db）
connections.connect(uri=milvus_db_path)


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """BGE dense 向量池化：按 attention mask 做 mean pooling。"""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


class HybridEmbeddingHandler:
    def __init__(self, dense_model_path: str, splade_model_path: str, device: str | None = None):
        # 默认优先走 GPU，GPU 不可用时退化到 CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 多GPU分配策略：将两个模型分别放到不同GPU上
        # Dense 路径：BGE-Large → GPU 0
        self.dense_device = f"cuda:0" if NUM_GPUS >= 1 else device
        print(f"BGE-Large 模型加载到: {self.dense_device}")
        self.dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_path)
        # 使用FP16精度加载模型，大幅减少显存占用
        self.dense_model = AutoModel.from_pretrained(
            dense_model_path, 
            torch_dtype=torch.float16,
            device_map=self.dense_device
        ).eval()
        
        # Sparse 路径：SPLADE → GPU 1（如果有第二个GPU）
        self.sparse_device = f"cuda:1" if NUM_GPUS >= 2 else (f"cuda:0" if NUM_GPUS >= 1 else device)
        print(f"SPLADE 模型加载到: {self.sparse_device}")
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(splade_model_path)
        # 使用FP16精度加载模型
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(
            splade_model_path,
            torch_dtype=torch.float16,
            device_map=self.sparse_device
        ).eval()
        
        # Dense 向量维度（用于 Milvus schema）
        self.dim = {"dense": self.dense_model.config.hidden_size}
        # Sparse 向量每条保留的 topk 维度
        self.sparse_topk = 200

    def _encode_dense(self, texts: list[str]) -> list[list[float]]:
        # 批量 token 化并送入 dense 模型（使用 dense_device）
        inputs = self.dense_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.dense_device)
        with torch.no_grad():
            outputs = self.dense_model(**inputs)
        # mean pooling 得到句向量
        pooled = _mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        # 转为 Python list，便于 Milvus 插入
        return pooled.cpu().detach().numpy().tolist()

    def _encode_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        # 批量 token 化并送入 SPLADE 模型（使用 sparse_device）
        inputs = self.sparse_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.sparse_device)
        with torch.no_grad():
            outputs = self.sparse_model(**inputs)
        logits = outputs.logits
        # padding 位置置为 -inf，避免被错误计入权重
        attn_mask = inputs["attention_mask"].unsqueeze(-1)
        logits = logits.masked_fill(attn_mask == 0, float("-inf"))
        # SPLADE 常见做法：log1p(relu(logits)) 后按 token 维取 max
        weights = torch.log1p(torch.relu(logits)).amax(dim=1)

        sparse_vectors: list[dict[int, float]] = []
        for row in weights:
            # 仅保留 topk 非零稀疏特征，减少存储与检索开销
            if self.sparse_topk and self.sparse_topk < row.numel():
                values, indices = torch.topk(row, self.sparse_topk)
                nonzero = values > 0
                indices = indices[nonzero]
                values = values[nonzero]
            else:
                indices = torch.nonzero(row > 0, as_tuple=False).squeeze(1)
                values = row[indices]
            sparse_vectors.append({int(i): float(v) for i, v in zip(indices, values)})
        return sparse_vectors

    def __call__(self, texts: list[str], batch_size: int = EMB_BATCH, verbose: bool = False) -> dict[str, list]:
        """分批编码文本，避免一次性占用过多显存。
        
        Args:
            texts: 待编码的文本列表
            batch_size: 每批处理的文本数
            verbose: 是否打印进度信息（默认False，查询时不打印）
        """
        dense_embeddings = []
        sparse_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            if verbose:
                print(f"编码批次 {i//batch_size + 1}/{total_batches}, 文本数: {len(batch_texts)}")
            
            # 分别编码 dense 和 sparse
            dense_batch = self._encode_dense(batch_texts)
            sparse_batch = self._encode_sparse(batch_texts)
            
            dense_embeddings.extend(dense_batch)
            sparse_embeddings.extend(sparse_batch)
            
            # 清理显存
            torch.cuda.empty_cache()
        
        return {
            "dense": dense_embeddings,
            "sparse": sparse_embeddings,
        }

    def encode_queries(self, queries: list[str]) -> dict[str, list]:
        return self(queries)


embedding_handler = HybridEmbeddingHandler(
    dense_model_path=bge_large_zh_v1_5_model_path,
    splade_model_path=splade_v2_model_path,
)


class MilvusRetriever:
    def __init__ (self, docs, retrieve=False):
        # 定义 Milvus collection 字段（主键 + 原文 + 稀疏向量 + 稠密向量）
        fields = [
            # 构建查询ID，primary key
            FieldSchema(name="unique_id", dtype=DataType.VARCHAR, is_primary=True, max_length=ID_MAX_LENGTH),
            # 存储原文，dense vector和sparse vector
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_handler.dim["dense"]),
        ]
        schema = CollectionSchema(fields)

        # 建库模式下，先清理同名 collection，确保索引内容与当前语料一致
        if not retrieve and utility.has_collection(COL_NAME):
            Collection(COL_NAME).drop()
        self.col = Collection(COL_NAME, schema, consistency_level="Strong")

        # 为 sparse / dense 分别创建索引
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        self.col.create_index("sparse_vector", sparse_index)
        self.col.create_index("dense_vector", dense_index)
        self.col.load()

        # 如果是非检索阶段，先构建索引
        if not retrieve:
            self.save_vectorstore(docs)


    def save_vectorstore(self, docs: list[str]): 
        # 设置环境变量优化显存分配
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # 拆分出写库所需字段：文本与 unique_id
        raw_texts = [doc.page_content for doc in docs]
        unique_ids = [doc.metadata["unique_id"] for doc in docs]
        
        # 截断过长的文本，确保不超过 Milvus schema 的最大长度限制
        raw_texts = [text[:MAX_TEXT_LENGTH] for text in raw_texts]

        print(f"开始编码 {len(raw_texts)} 条文档...")
        
        # 计算embedding（分批处理）
        texts_embeddings = embedding_handler(raw_texts, batch_size=EMB_BATCH)

        print(f"编码完成，开始插入 Milvus...")
        
        # batch embedding 插入
        for i in range(0, len(docs), EMB_BATCH):
            # Milvus 插入顺序需与 schema 字段顺序对应
            batched_entities = [
                unique_ids[i : i + EMB_BATCH],
                raw_texts[i : i + EMB_BATCH],
                texts_embeddings["sparse"][i : i + EMB_BATCH],
                texts_embeddings["dense"][i : i + EMB_BATCH],
            ]
            self.col.insert(batched_entities)
            # 每批次插入后清理显存
            torch.cuda.empty_cache()
        
        print("索引构建完成，插入了{}条数据:".format(self.col.num_entities))


    def dense_search(self, query_dense_embedding, limit):
        # dense 单路检索（调试或 ablation 使用）
        search_params = {"metric_type": "IP", "params": {}}
        res = self.col.search(
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=limit,
            output_fields=["unique_id", "text"],
            param=search_params,
        )
        return res


    def sparse_search(self, query_sparse_embedding, limit):
        # sparse 单路检索（调试或 ablation 使用）
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        res = self.col.search(
            [query_sparse_embedding],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=["unique_id", "text"],
            param=search_params,
        )
        return res


    def hybrid_search(
        self,
        query_dense_embedding,
        query_sparse_embedding,
        sparse_weight=1.0,
        dense_weight=1.0,
        limit=10,
    ):
        # 组装 dense 检索请求
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
        )
        # 组装 sparse 检索请求
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
        )
        # 使用加权融合器（可通过 sparse_weight/dense_weight 调权）
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = self.col.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["unique_id", "text"]
        )
        return res


    def retrieve_topk(self, query, topk=10):
        # 编码 query：得到 dense + sparse 两路向量
        # 抽取query的embedding 
        query_embeddings = embedding_handler.encode_queries([query])

        # 检索Topk
        hybrid_results = self.hybrid_search(
            query_embeddings["dense"][0],
            query_embeddings["sparse"][0],
            sparse_weight=1.0,
            dense_weight=1.0,
            limit=topk
        )[0]

        # 关联mongo数据
        related_docs = []
        for result in hybrid_results:
            # result["id"] 对应 unique_id，回表拿 page_content 与 metadata
            search_res = mongo_collection.find_one({"unique_id": result["id"]})
            # 防御：若回表为空，跳过该条
            if not search_res:
                continue
            #images_list = []
            #for image in search_res["metadata"]["images_info"]:
            #    images_list.append(ManualImages(**image))
            #search_res["metadata"]["images_info"] =  images_list 
            doc = Document(page_content=search_res["page_content"], metadata=search_res["metadata"])
            related_docs.append(doc)

        return related_docs 


if __name__ == "__main__":
    texts = [k for k in open(test_doc_path).readlines()]
    docs = []
    for text in texts:
        unique_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        metadata = {"unique_id": unique_id}
        docs.append(Document(page_content=text, metadata=metadata))
    retriever = MilvusRetriever(docs)
    query = "小米SU7支持的钥匙类型"
    results = retriever.retrieve_topk(query, 2)
    for res in results:
        print(res)
        print("="*100)