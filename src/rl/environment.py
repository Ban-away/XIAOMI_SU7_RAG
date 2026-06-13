# -*- coding: utf-8 -*-
"""
RL 推理环境 - 工具调用路由器

职责：
  1. 接管 vLLM 生成过程，监听 <search_local> / <search_web> 标签
  2. 触发标签时暂停生成，调用对应检索后端
  3. 将检索结果以 <information> 格式拼回上下文
  4. 继续生成直到出现 <answer> 或达到最大步数
  5. 提取最终答案，传给奖励函数

使用方式：
  from src.rl.environment import RetrievalEnvironment
  env = RetrievalEnvironment()
  answer, trajectory = env.run(question)
"""

import re
import os
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.reranker.minicpm_reranker import MiniCPMReRanker
from src.constant import bge_reranker_minicpm_path
from src.utils import merge_docs
from src.rl.web_reader import WebPageReader

# ── 相关性判断阈值 ──────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.35
LOCAL_TOPK          = 3
MAX_SEARCH_STEPS    = 4   # 单次推理最多调用搜索工具的次数（防止死循环）
MAX_READ_PAGE_HOPS  = 2   # 最大页面深度阅读次数（垂直搜索）

# ── 工具标签正则 ──────────────────────────────────────────
_RE_SEARCH_LOCAL = re.compile(r"<search_local>(.*?)</search_local>", re.DOTALL)
_RE_SEARCH_WEB   = re.compile(r"<search_web>(.*?)</search_web>",   re.DOTALL)
_RE_READ_PAGE    = re.compile(r"<read_page>(.*?)</read_page>",     re.DOTALL)
_RE_ANSWER       = re.compile(r"<answer>(.*?)</answer>",            re.DOTALL)
_RE_INFORMATION  = re.compile(r"<information>(.*?)</information>",  re.DOTALL)


class LocalSearchBackend:
    """线程安全的本地检索后端"""

    _instance = None
    _lock      = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.bm25     = BM25(docs=None, retrieve=True)
        self.milvus   = MilvusRetriever(docs=None, retrieve=True)
        # 重排器：bge-reranker-v2-minicpm-layerwise（路径 models/BAAI/...）。
        # layerwise 打分已修复（transformers 4.52 下 outputs.logits 为 (batch, seq)，取 [:, -1]）。
        # search() 内保留 try/except 容错降级。设 RERANKER_DISABLED=1 可禁用。
        self._reranker_disabled = os.getenv("RERANKER_DISABLED", "0") == "1"
        if not self._reranker_disabled:
            self.reranker = MiniCPMReRanker(
                model_path=bge_reranker_minicpm_path, cutoff_layers=28
            )
        else:
            self.reranker = None
            print("[WARN] 重排器已禁用（RERANKER_DISABLED=1），本地检索直接用 BM25+Milvus 融合结果")
        self._bm25_lock   = threading.Lock()
        self._rerank_lock = threading.Lock()
        self.milvus.retrieve_topk("测试", topk=1)

    def search(self, query: str) -> tuple[str, float]:
        """
        返回 (formatted_result, relevance_score)
        score < RELEVANCE_THRESHOLD 表示本地信息不足，需网络兜底
        """
        with self._bm25_lock:
            bm25_docs = self.bm25.retrieve_topk(query, topk=LOCAL_TOPK)
        milvus_docs = self.milvus.retrieve_topk(query, topk=LOCAL_TOPK * 2)
        merged      = merge_docs(bm25_docs, milvus_docs)

        if not merged:
            return "本地知识库中未检索到相关内容。", 0.0

        if self._reranker_disabled or self.reranker is None:
            ranked = merged[:LOCAL_TOPK]
        else:
            with self._rerank_lock:
                try:
                    ranked = self.reranker.rank(query, merged[:10], topk=LOCAL_TOPK)
                except Exception as e:
                    # 重排打分失败（如 layerwise 在 transformers 5.x 下的形状 bug）时，
                    # 降级用 BM25+Milvus 融合排序，不让单点故障拖垮整条检索
                    print(f"[WARN] 重排失败（{type(e).__name__}: {e}），降级用 BM25+Milvus 融合排序")
                    ranked = merged[:LOCAL_TOPK]

        if not ranked:
            return "本地知识库中未检索到相关内容。", 0.0

        # 格式化结果
        parts = []
        for i, doc in enumerate(ranked, 1):
            page = doc.metadata.get("page", "")
            suffix = f"【第{page}页】" if page else ""
            parts.append(f"[{i}]{suffix} {doc.page_content[:400]}")
        result_str = "\n".join(parts)

        # 相关性分数：重排器绝对分不可靠（不同模型/术语差异会误打低分，如"玻璃水"↔"风挡洗涤液"），
        # 只要本地召回到实质性内容，就以"有内容"的保守分兜底，避免误触发网络搜索。
        import math
        top_score = 0.0
        if ranked and hasattr(ranked[0], "metadata"):
            raw = ranked[0].metadata.get("relevance_score",
                     ranked[0].metadata.get("score", None))
            if raw is not None:
                try:
                    top_score = 1 / (1 + math.exp(-float(raw)))
                except (ValueError, OverflowError):
                    top_score = 0.5
        # 有实质性结果时，分数不低于"内容保守分"，防止低分误触发 web 兜底
        content_floor = min(0.3 + len(ranked) * 0.05, 0.6)
        if top_score < content_floor:
            top_score = content_floor
        return result_str, top_score


class WebSearchBackend:
    """网络搜索后端，自动检测可用的 API"""

    def __init__(self):
        # 搜索后端链：按优先级 + 已配置 key 构建；逐个尝试，某个失败（如额度耗尽）顺延到下一个
        chain = []
        if os.getenv("SERPAPI_KEY"):
            chain.append("serpapi")
        if os.getenv("SERPER_API_KEY"):
            chain.append("serper")
        if os.getenv("BING_SEARCH_KEY"):
            chain.append("bing")
        if not chain:
            chain.append("doubao")  # 无任何搜索 key 时用豆包 LLM 模拟
        self._backends = chain
        self.backend = chain[0]  # 兼容旧引用（主后端名）

    def search(self, query: str) -> str:
        # 限定网络搜索只针对小米汽车（query 未含小米车型词时加前缀）
        query = self._scope_to_xiaomi(query)
        last_err = None
        for be in self._backends:
            try:
                if be == "serpapi":
                    return self._serpapi(query)
                elif be == "serper":
                    return self._serper(query)
                elif be == "bing":
                    return self._bing(query)
                else:
                    return self._doubao(query)
            except Exception as e:
                last_err = e
                print(f"[WARN] 搜索后端 {be} 失败（{e}），顺延到下一个")
        return f"网络搜索暂时不可用（所有后端均失败）：{last_err}"

    # 限定网络搜索只针对小米汽车：query 未含小米车型词时加 "小米SU7" 前缀
    _XIAOMI_TERMS = ("小米汽车", "小米SU7", "小米 SU7", "小米YU7", "小米 YU7",
                     "SU7", "YU7", "SU7 Ultra", "Xiaomi", "澎湃")

    def _scope_to_xiaomi(self, query: str) -> str:
        q = query.strip()
        if any(t in q for t in self._XIAOMI_TERMS):
            return q
        return f"小米SU7 {q}"

    def _bing(self, query: str) -> str:
        import requests
        headers = {"Ocp-Apim-Subscription-Key": os.environ["BING_SEARCH_KEY"]}
        params  = {"q": query, "mkt": "zh-CN", "count": 5}
        resp    = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers=headers, params=params, timeout=10
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"  # Bing 返回 UTF-8 JSON；强制避免被默认 ISO-8859-1 解成乱码
        results = resp.json().get("webPages", {}).get("value", [])
        if not results:
            return "网络搜索未找到相关结果。"
        parts = []
        for i, r in enumerate(results[:4], 1):
            parts.append(f"[{i}]【{r['name']}】\n{r['snippet']}\n网址：{r['url']}")
        return "\n\n".join(parts)

    def _serpapi(self, query: str) -> str:
        import requests
        params = {
            "q": query, "hl": "zh-cn", "gl": "cn",
            "api_key": os.environ["SERPAPI_KEY"], "num": 5,
        }
        resp    = requests.get("https://serpapi.com/search", params=params, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"  # 同上，避免中文乱码
        results = resp.json().get("organic_results", [])
        if not results:
            return "网络搜索未找到相关结果。"
        parts = []
        for r in results[:4]:
            parts.append(
                f"【{r.get('title','')}】\n{r.get('snippet','')}\n网址：{r.get('link','')}"
            )
        return "\n\n".join(parts)

    def _serper(self, query: str) -> str:
        # Serper (google.serper.dev) —— SerpAPI 额度耗尽时的兜底后端
        import requests
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": os.environ["SERPER_API_KEY"],
                     "Content-Type": "application/json"},
            json={"q": query, "hl": "zh-cn", "gl": "cn", "num": 5},
            timeout=15,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"  # 避免中文乱码
        results = resp.json().get("organic", [])
        if not results:
            return "网络搜索未找到相关结果。"
        parts = []
        for r in results[:4]:
            parts.append(
                f"【{r.get('title', '')}】\n{r.get('snippet', '')}\n网址：{r.get('link', '')}"
            )
        return "\n\n".join(parts)

    def _doubao(self, query: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["DOUBAO_API_KEY"],
            base_url=os.environ["DOUBAO_BASE_URL"],
        )
        completion = client.chat.completions.create(
            model=os.environ["DOUBAO_MODEL_NAME"],
            messages=[{
                "role": "user",
                "content": (
                    f"请提供关于以下问题的准确网络信息，直接给出内容，"
                    f"不要有任何引导语：\n{query}"
                )
            }],
            max_tokens=600,
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()


# ────────────────────────────────────────────────────────────
# 核心环境类
# ────────────────────────────────────────────────────────────

class RetrievalEnvironment:
    """
    工具调用环境。

    在 GRPO 训练时，vLLM 每生成一个 chunk 后调用 step()；
    在推理时，直接调用 run_with_context() 传入已有生成片段做路由判断。
    """

    def __init__(self):
        self.local_backend = LocalSearchBackend.get_instance()
        self.web_backend   = WebSearchBackend()
        self.page_reader   = WebPageReader()

    # ── 核心接口：处理一段生成文本，返回需要拼回的内容 ──────
    def step(self, generated_text: str) -> tuple[str | None, bool]:
        """
        检测 generated_text 末尾是否有待处理的工具调用标签。

        返回：
          (information_block, is_done)
          - information_block: 需要拼回上下文的 <information>...</information> 字符串
                               为 None 表示无工具调用
          - is_done: True 表示已出现 <answer>，生成结束
        """
        # 检测是否已完成
        if _RE_ANSWER.search(generated_text):
            return None, True

        # 检测各类工具调用
        local_match = _RE_SEARCH_LOCAL.findall(generated_text)
        web_match   = _RE_SEARCH_WEB.findall(generated_text)
        read_match  = _RE_READ_PAGE.findall(generated_text)

        # 已有信息块数量（避免重复处理）
        info_count   = len(_RE_INFORMATION.findall(generated_text))
        total_calls  = len(local_match) + len(web_match) + len(read_match)

        if total_calls <= info_count:
            # 所有已有的调用都已经有了对应的 information，无需处理
            return None, False

        # 检查 read_page 跳数限制
        if len(read_match) > MAX_READ_PAGE_HOPS:
            return "<information>已达到最大页面阅读次数限制，请基于已有信息作答。</information>", False

        if total_calls > MAX_SEARCH_STEPS:
            # 超过最大步数，强制终止
            return "<information>已达到最大检索次数限制。</information>\n<answer>根据已检索到的信息暂时无法给出完整答案，建议访问小米汽车官网获取最新信息。</answer>", True

        # 按出现顺序收集所有调用
        all_calls = []
        for m in _RE_SEARCH_LOCAL.finditer(generated_text):
            all_calls.append(("local", m.group(1).strip(), m.start()))
        for m in _RE_SEARCH_WEB.finditer(generated_text):
            all_calls.append(("web",   m.group(1).strip(), m.start()))
        for m in _RE_READ_PAGE.finditer(generated_text):
            all_calls.append(("read_page", m.group(1).strip(), m.start()))
        all_calls.sort(key=lambda x: x[2])  # 按出现位置排序

        # 取第 info_count 个调用（即下一个待响应的）
        call_type, query, _ = all_calls[info_count]

        if call_type == "local":
            result_str, score = self.local_backend.search(query)

            # 判断是否需要升级到网络搜索
            if score < RELEVANCE_THRESHOLD:
                # 本地信息不足，在 information 中说明，让模型决定是否发起 web search
                info_block = (
                    f"<information>{result_str}\n"
                    f"[提示：本地知识库相关性较低（{score:.2f}），"
                    f"如需更准确信息可调用网络搜索]</information>"
                )
            else:
                info_block = f"<information>{result_str}</information>"

        elif call_type == "web":
            result_str = self.web_backend.search(query)
            info_block = f"<information>{result_str}</information>"

        else:  # read_page
            url = query.strip()
            page_content = self.page_reader.fetch(url)
            info_block = f"<information>{page_content}</information>"

        return info_block, False

    # ── 辅助接口：提取最终答案 ───────────────────────────────
    @staticmethod
    def extract_answer(full_text: str) -> str:
        match = _RE_ANSWER.search(full_text)
        return match.group(1).strip() if match else ""

    # ── 辅助接口：判断是否为领域外拒答 ─────────────────────
    @staticmethod
    def is_refusal(answer: str) -> bool:
        refusal_patterns = [
            "只能回答小米SU7相关问题",
            "无法回答此问题",
            "不在我的服务范围",
            "建议您咨询",
        ]
        return any(p in answer for p in refusal_patterns)

    # ── 工具调用次数统计（用于奖励函数）───────────────────
    @staticmethod
    def count_search_calls(full_text: str) -> dict:
        return {
            "local":     len(_RE_SEARCH_LOCAL.findall(full_text)),
            "web":       len(_RE_SEARCH_WEB.findall(full_text)),
            "read_page": len(_RE_READ_PAGE.findall(full_text)),
        }
