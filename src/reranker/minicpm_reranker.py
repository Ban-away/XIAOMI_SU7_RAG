# -*- coding: utf-8 -*-
"""MiniCPM 重排器 - 专门用于 bge-reranker-v2-minicpm-layerwise 模型"""

import os
import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel


class MiniCPMReRanker(object):
    def __init__(self, model_path, max_length=4096, cutoff_layers=None):
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
        
        # cutoff_layers: 只使用模型的前 N 层（MiniCPM 特有优化）
        self.cutoff_layers = cutoff_layers
        if cutoff_layers is not None and hasattr(self.model, 'encoder'):
            print(f"[INFO] 使用 cutoff_layers={cutoff_layers}，只保留前 {cutoff_layers} 层")
        
        # 检测是否有可用的 GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 尝试使用 FP16，失败则回退到 FP32
        if self.device == "cuda":
            try:
                self.model.half()
                print("[INFO] 使用 FP16 精度")
            except Exception as e:
                print(f"[WARNING] FP16 转换失败，使用 FP32: {e}")
        
        # 模型放到设备上
        self.model.to(self.device)
        # 文本对最大长度（query + doc）
        self.max_length = max_length
        
        print(f"[INFO] MiniCPM 重排模型加载完成，设备: {self.device}")


    def rank(self, query, candidate_docs, topk=10):
        # 输入文档对，返回每一对(query, doc)的相关得分，并从大到小排序
        if not candidate_docs:
            return []
            
        # 组装 query-doc 文本对
        pairs = [(query, doc.page_content) for doc in candidate_docs]
        # 批量编码后送入模型
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 处理不同模型的输出格式
            if hasattr(outputs, 'logits'):
                # 标准序列分类模型
                scores = outputs.logits
            elif hasattr(outputs, 'scores'):
                # 某些自定义重排模型直接输出 scores
                scores = outputs.scores
            elif hasattr(outputs, 'last_hidden_state'):
                # BaseModelOutputWithPast 类型，使用 CLS token 计算相似度
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                # 计算每个文档与查询的相似度（第一个样本作为查询表示）
                query_emb = cls_embeddings[0:1, :]
                scores = torch.matmul(cls_embeddings, query_emb.transpose(0, 1))
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                # 尝试从元组中获取分数
                if hasattr(outputs[0], 'logits'):
                    scores = outputs[0].logits
                elif hasattr(outputs[0], 'scores'):
                    scores = outputs[0].scores
                elif isinstance(outputs[0], torch.Tensor):
                    scores = outputs[0]
                else:
                    raise RuntimeError(f"无法处理模型输出格式: {type(outputs)}")
            elif isinstance(outputs, torch.Tensor):
                # 直接返回张量
                scores = outputs
            else:
                raise RuntimeError(f"无法处理模型输出格式: {type(outputs)}")
        
        # 张量转 numpy，便于 Python 层排序
        scores = scores.detach().cpu().clone().numpy()
        # 确保 scores 是一维数组
        if scores.ndim > 1:
            scores = scores.flatten()
        # 依据分数降序排序并截断 topk
        response = [
            doc
            for score, doc in sorted(
                zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
            )
            ][:topk]
        return response


if __name__ == "__main__":
    model_path = "./models/bge-reranker-v2-minicpm-layerwise/"
    reranker = MiniCPMReRanker(model_path, cutoff_layers=28)
    query = "今天天气怎么样"
    docs = ["你好", "今天天气不错", "今天有雨吗"]
    docs = [Document(page_content=doc, metadata={}) for doc in docs]
    response = reranker.rank(query, docs)
    print(response)