# -*- coding: utf-8 -*-


import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any


class BaseRetriever(ABC):
    def __init__(self,  docs: str, retrieve: bool = False) -> None:
        # 调用父类初始化（保持 ABC 基类链路完整）
        super().__init__()
        # docs: 建库阶段传入语料；检索阶段通常为 None
        self.docs = docs
        # retrieve=True 表示走“加载已有索引”的路径
        self.retrieve = retrieve

    @abstractmethod
    def retrieve_topk(self, query: str, topk=3):
        # 子类必须实现该方法，返回最相关的 topk 文档
        raise NotImplementedError
