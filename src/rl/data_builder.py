# -*- coding: utf-8 -*-
"""
RL 训练数据构建脚本 - 网络兜底轨迹生成器

功能：
  1. 读取 web_fallback_questions.json 问题库
  2. 每个问题先走本地检索，判断是否信息充足
  3. 本地信息不足时触发网络搜索兜底
  4. 调用 LLM 生成完整工具调用轨迹
  5. 输出符合 GRPO 训练格式的 JSON 文件

输出格式：LLaMA-Factory SFT 格式（同时作为 GRPO 的 warm-up 数据）

运行：
  python src/rl/data_builder.py
  python src/rl/data_builder.py --resume   # 断点续传
  python src/rl/data_builder.py --dry-run  # 只处理前5条，验证流程
"""

import os

# ── 在所有 tqdm 相关 import 之前禁用，防止下游库弹出子进度条 ──
os.environ["TQDM_DISABLE"] = "1"

import re
import json
import time
import hashlib
import argparse
import threading
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 项目内部模块 ────────────────────────────────────────────
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.reranker.minicpm_reranker import MiniCPMReRanker
from src.constant import bge_reranker_minicpm_path
from src.utils import merge_docs
from src.rl.web_reader import WebPageReader

# ── 路径配置 ────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUESTIONS_PATH = os.path.join(BASE_DIR, "data/rl_data/web_fallback_questions.json")
OUTPUT_PATH    = os.path.join(BASE_DIR, "data/rl_data/web_fallback_trajectories.json")
CKPT_PATH      = os.path.join(BASE_DIR, "data/rl_data/web_fallback_ckpt.jsonl")

# ── 超参数 ──────────────────────────────────────────────────
LOCAL_TOPK          = 3      # 本地检索返回条数
RELEVANCE_THRESHOLD = 0.35   # 低于此分数判定为"本地信息不足"
MAX_WORKERS         = 8      # 并发线程数
RETRY_TIMES         = 3      # 单条失败重试次数

# ── 系统提示词（工具协议声明）──────────────────────────────
SYSTEM_PROMPT = """你是小米SU7车型的专业问答助手，服务范围严格限定在小米SU7相关问题。

回答问题时可以调用以下工具：
- 本地知识库检索（优先）：<search_local>检索关键词</search_local>
- 网络搜索（本地信息不足时）：<search_web>检索关键词</search_web>
- 页面深度阅读（搜索结果不够详细时）：<read_page>URL地址</read_page>

工具返回格式：<information>检索结果内容</information>

最终答案格式：<answer>答案内容</answer>

注意：
1. 优先调用本地知识库，本地无结果或信息严重不足时再调用网络搜索
2. 网络搜索结果中包含"网址："字段，可选择最有价值的页面用 <read_page> 深入阅读，最多读取2个页面
3. 与小米SU7无关的问题（闲聊、百科、娱乐等），直接输出 <answer>很抱歉，我只能回答小米SU7相关问题。</answer>
4. 网络搜索结果来源于互联网，答案中需注明"根据网络信息"
5. 涉及页码引用时格式为【页码】"""

# ── 轨迹生成提示词 ─────────────────────────────────────────
TRAJECTORY_GEN_PROMPT = """你是一个数据标注专家，需要为以下问题生成一条高质量的工具调用轨迹。

问题：{question}

本地知识库检索结果（相关性偏低，信息不足）：
{local_result}

网络搜索结果：
{web_result}

{page_content_section}

请严格按照以下格式生成完整轨迹（只输出assistant的回复内容，不要有任何前缀说明）：

<search_local>{local_query}</search_local>
<information>{local_result_placeholder}</information>
<search_web>{web_query}</search_web>
<information>{web_result_placeholder}</information>
{read_page_section}
<answer>基于网络信息的准确回答，语言自然流畅，注明来源为网络信息</answer>

要求：
1. search_local的query要简洁精准，提取问题核心关键词
2. search_web的query要加上"小米SU7"前缀确保搜索精度
3. 如果有页面详细内容，需要用 <read_page>URL</read_page> + <information>详细内容</information> 的形式体现垂直搜索
4. answer要直接回答问题，不重复问题，语言自然，结尾注明"（以上信息来源于网络，请以小米官方最新公告为准）"
5. 如果网络搜索结果也没有有效信息，answer输出"根据目前可获取的信息，暂时无法回答此问题，建议访问小米汽车官网或联系官方客服获取最新信息。"
"""


# ────────────────────────────────────────────────────────────
# 检索模块
# ────────────────────────────────────────────────────────────

class LocalSearchTool:
    """封装现有本地检索栈（Milvus + BM25 + MiniCPM精排）"""

    def __init__(self):
        print("[INFO] 加载本地检索组件...")
        self.bm25     = BM25(docs=None, retrieve=True)
        self.milvus   = MilvusRetriever(docs=None, retrieve=True)
        self.reranker = MiniCPMReRanker(
            model_path=bge_reranker_minicpm_path,
            cutoff_layers=28
        )
        self._lock = threading.Lock()
        # 预热
        self.milvus.retrieve_topk("测试", topk=1)
        print("[INFO] 本地检索组件加载完成")

    def search(self, query: str, topk: int = LOCAL_TOPK) -> tuple[list, float]:
        """
        返回 (docs, max_score)
        max_score 用于判断相关性：< RELEVANCE_THRESHOLD 视为信息不足
        """
        with self._lock:
            bm25_docs   = self.bm25.retrieve_topk(query, topk=topk)
        milvus_docs = self.milvus.retrieve_topk(query, topk=topk * 2)
        merged      = merge_docs(bm25_docs, milvus_docs)

        if not merged:
            return [], 0.0

        with self._lock:
            ranked = self.reranker.rank(query, merged[:10], topk=topk)

        # 用重排分数估算最大相关性（归一化处理）
        import math
        def sigmoid(x):
            try:
                return 1 / (1 + math.exp(-float(x)))
            except (ValueError, OverflowError):
                return 0.5

        # 取第一条的实际重排分数
        max_score = 0.0
        if ranked:
            raw = ranked[0].metadata.get("relevance_score",
                     ranked[0].metadata.get("score", None))
            if raw is not None:
                max_score = sigmoid(raw)
            else:
                # 兜底：有结果但无分数时给保守分
                max_score = min(0.3 + len(ranked) * 0.05, 0.6)
        return ranked, max_score

    def format_result(self, docs: list) -> str:
        if not docs:
            return "本地知识库中未检索到相关内容。"
        parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "")
            page_str = f"（第{page}页）" if page else ""
            parts.append(f"[{i}] {doc.page_content[:300]}{page_str}")
        return "\n".join(parts)


# ────────────────────────────────────────────────────────────
# 网络搜索模块（支持多后端）
# ────────────────────────────────────────────────────────────

class WebSearchTool:
    """
    网络搜索工具，支持三种后端：
    - bing：Bing Search API（推荐，稳定，支持中文）
    - serpapi：SerpAPI（需要 SERPAPI_KEY 环境变量）
    - doubao：通过豆包 LLM 的联网功能模拟搜索（无需额外 API，但结果不是真实爬取）
    """

    def __init__(self, backend: str = "auto"):
        self.backend = self._detect_backend(backend)
        print(f"[INFO] 网络搜索后端：{self.backend}")

    def _detect_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        if os.getenv("BING_SEARCH_KEY"):
            return "bing"
        if os.getenv("SERPAPI_KEY"):
            return "serpapi"
        return "doubao"  # 兜底：用 LLM 模拟

    def search(self, query: str) -> str:
        for attempt in range(RETRY_TIMES):
            try:
                if self.backend == "bing":
                    return self._search_bing(query)
                elif self.backend == "serpapi":
                    return self._search_serpapi(query)
                else:
                    return self._search_via_doubao(query)
            except Exception as e:
                if attempt < RETRY_TIMES - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[WARN] 网络搜索失败（{query}）: {e}")
                    return ""
        return ""

    def _search_bing(self, query: str) -> str:
        import requests
        url     = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": os.environ["BING_SEARCH_KEY"]}
        params  = {"q": query, "mkt": "zh-CN", "count": 5, "responseFilter": "Webpages"}
        resp    = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data    = resp.json()
        results = data.get("webPages", {}).get("value", [])
        if not results:
            return ""
        parts = []
        for r in results[:5]:
            parts.append(f"【{r['name']}】\n{r['snippet']}\n网址：{r['url']}")
        return "\n\n".join(parts)

    def _search_serpapi(self, query: str) -> str:
        import requests
        url    = "https://serpapi.com/search"
        params = {
            "q":       query,
            "hl":      "zh-cn",
            "gl":      "cn",
            "api_key": os.environ["SERPAPI_KEY"],
            "num":     5,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data    = resp.json()
        results = data.get("organic_results", [])
        if not results:
            return ""
        parts = []
        for r in results[:5]:
            snippet = r.get("snippet", "")
            title   = r.get("title", "")
            link    = r.get("link", "")
            parts.append(f"【{title}】\n{snippet}\n网址：{link}")
        return "\n\n".join(parts)

    def _search_via_doubao(self, query: str) -> str:
        """
        使用豆包 LLM 的知识库模拟网络搜索。
        适用于无搜索 API key 的场景，结果基于 LLM 训练数据，
        对实时性要求极高的内容（如当天价格）准确性有限，需后续人工核验。
        """
        client = OpenAI(
            api_key=os.environ["DOUBAO_API_KEY"],
            base_url=os.environ["DOUBAO_BASE_URL"],
        )
        prompt = (
            f"请以网络搜索结果的形式，提供关于以下问题的最新准确信息。"
            f"直接给出信息内容，不要有任何引导语，信息要具体、有数据支撑。\n\n"
            f"搜索问题：{query}\n\n"
            f"请提供3-5条有实质内容的搜索结果摘要："
        )
        completion = client.chat.completions.create(
            model=os.environ["DOUBAO_MODEL_NAME"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()


# ────────────────────────────────────────────────────────────
# 轨迹生成器
# ────────────────────────────────────────────────────────────

class TrajectoryBuilder:
    """将检索结果组装为完整的工具调用轨迹"""

    def __init__(self):
        self.llm_client = OpenAI(
            api_key=os.environ["DOUBAO_API_KEY"],
            base_url=os.environ["DOUBAO_BASE_URL"],
        )
        self.page_reader = WebPageReader()

    def build(
        self,
        question:     str,
        local_query:  str,
        local_docs:   list,
        web_query:    str,
        web_result:   str,
    ) -> str:
        """
        生成完整的 assistant 轨迹文本。
        格式：
          <search_local>...</search_local>
          <information>...</information>
          <search_web>...</search_web>
          <information>...</information>
          <read_page>URL</read_page>          ← 可选，垂直搜索
          <information>...</information>       ← 页面详情
          <answer>...</answer>
        """
        local_result_str = self._format_local_docs(local_docs)

        # 若网络搜索也为空，直接构造降级回复
        if not web_result.strip():
            web_result = "网络搜索暂时未获取到有效结果。"

        # ── 尝试从 web_result 中提取 URL 并抓取页面 ──
        page_url, page_content = self._try_fetch_best_page(web_result)

        # 构建 read_page 段落
        if page_url and page_content:
            page_content_section = f"页面深度阅读内容（来自 {page_url}）：\n{page_content}"
            read_page_section = (
                f"<read_page>{page_url}</read_page>\n"
                f"<information>{page_content}</information>"
            )
        else:
            page_content_section = ""
            read_page_section = ""

        prompt = TRAJECTORY_GEN_PROMPT.format(
            question=question,
            local_result=local_result_str,
            web_result=web_result,
            local_query=local_query,
            web_query=web_query,
            local_result_placeholder=local_result_str[:400] if local_result_str else "本地知识库中未检索到相关内容。",
            web_result_placeholder=web_result[:800],
            page_content_section=page_content_section,
            read_page_section=read_page_section,
        )

        completion = self.llm_client.chat.completions.create(
            model=os.environ["DOUBAO_MODEL_NAME"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.2,
        )
        raw = completion.choices[0].message.content.strip()

        # 后处理：确保格式完整
        raw = self._postprocess(raw, local_query, web_query, local_result_str, web_result, page_url, page_content)
        return raw

    def _format_local_docs(self, docs: list) -> str:
        if not docs:
            return "本地知识库中未检索到相关内容。"
        parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "")
            suffix = f"（第{page}页）" if page else ""
            parts.append(f"[{i}]{suffix} {doc.page_content[:250]}")
        return "\n".join(parts)

    def _try_fetch_best_page(self, web_result: str) -> tuple[str, str]:
        """
        从网络搜索结果中提取最有价值的 URL 并抓取页面内容。
        返回 (url, content)，失败时返回 ("", "")
        """
        # 提取搜索结果中的 URL（匹配 "网址：" 或 "来源：" 前缀）
        urls = re.findall(r"(?:网址|来源)[：:]\s*(https?://[^\s\n]+)", web_result)
        if not urls:
            return "", ""

        # 选择第一个 URL（通常是相关性最高的官方页面）
        best_url = urls[0].strip()
        try:
            content = self.page_reader.fetch(best_url, max_chars=1500)
            if content.startswith("无法") or content.startswith("页面") or content.startswith("不支持"):
                return "", ""
            return best_url, content
        except Exception:
            return "", ""

    def _postprocess(
        self,
        raw:          str,
        local_query:  str,
        web_query:    str,
        local_result: str,
        web_result:   str,
        page_url:     str  = "",
        page_content: str  = "",
    ) -> str:
        """确保关键标签齐全，缺失时补齐"""
        if "<search_local>" not in raw:
            raw = f"<search_local>{local_query}</search_local>\n" + raw
        if "<information>" not in raw:
            raw = raw.replace(
                "</search_local>",
                f"</search_local>\n<information>{local_result[:300]}</information>"
            )
        if "<search_web>" not in raw:
            # 在 answer 前插入 web 搜索
            raw = raw.replace(
                "<answer>",
                f"<search_web>{web_query}</search_web>\n"
                f"<information>{web_result[:600]}</information>\n<answer>"
            )
        # 如果有页面内容但轨迹中没有 read_page，自动插入
        if page_url and page_content and "<read_page>" not in raw:
            raw = raw.replace(
                "<answer>",
                f"<read_page>{page_url}</read_page>\n"
                f"<information>{page_content[:800]}</information>\n<answer>"
            )
        if "<answer>" not in raw:
            raw += "\n<answer>根据目前可获取的信息，暂时无法给出准确回答，建议访问小米汽车官网获取最新信息。（以上信息来源于网络，请以小米官方最新公告为准）</answer>"
        elif "<answer>" in raw and "</answer>" not in raw:
            # max_tokens 截断导致 answer 开标签存在但闭标签缺失，补齐闭标签
            raw += "</answer>"
        return raw


# ────────────────────────────────────────────────────────────
# 格式转换：轨迹 → LLaMA-Factory SFT 格式
# ────────────────────────────────────────────────────────────

def to_sft_target(trajectory: str) -> str:
    """
    SFT warm-up 只学习工具调用和最终答案。

    <information> 是工具/环境返回，不应由 assistant 预测；完整轨迹仍保留
    在 GRPO completion 中用于奖励训练。
    """
    target = re.sub(r"<information>.*?</information>", "", trajectory, flags=re.DOTALL)
    target = re.sub(r"\n{3,}", "\n\n", target)
    return target.strip()


def to_sft_format(question: str, trajectory: str) -> dict:
    """
    转换为 LLaMA-Factory instruction 格式，兼容 GRPO warm-up SFT 训练。
    同时提取 answer 内容作为奖励/评估字段。
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", trajectory, re.DOTALL)
    answer_text  = answer_match.group(1).strip() if answer_match else ""

    return {
        "instruction": question,
        "input":       "",
        "output":      to_sft_target(trajectory),
        "answer":      answer_text,          # 纯答案文本，用于奖励计算
        "system":      SYSTEM_PROMPT,
        "data_source": "web_fallback",
    }


def to_grpo_format(question: str, trajectory: str, category: str) -> dict:
    """
    转换为 GRPO 训练格式。
    prompt = system + user，completion = assistant 轨迹
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", trajectory, re.DOTALL)
    answer_text  = answer_match.group(1).strip() if answer_match else ""

    return {
        "prompt": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question},
        ],
        "completion":  trajectory,
        "answer":      answer_text,
        "category":    category,
        "data_source": "web_fallback",
        "reward_type": "web_answer_quality",  # 奖励函数路由标识
    }


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def process_one(
    item:            dict,
    local_tool:      LocalSearchTool,
    web_tool:        WebSearchTool,
    traj_builder:    TrajectoryBuilder,
) -> dict | None:
    """处理单条问题，生成完整轨迹"""

    question      = item["question"]
    local_query   = item.get("local_query_hint", question)
    web_query_raw = item.get("web_query_hint", f"小米SU7 {question}")

    # ── Step 1：本地检索 ──────────────────────────────────
    local_docs, max_score = local_tool.search(local_query)

    # ── Step 2：判断是否需要网络兜底 ──────────────────────
    # web_fallback_questions.json 中的问题本身就是需要网络兜底的，
    # 因此无条件触发网络搜索，但仍保留本地检索结果作为轨迹中的上下文。
    need_web = True

    if not need_web:
        # 本地信息充足，跳过（此脚本专门处理需要网络兜底的问题）
        print(f"[SKIP] 本地可答（score={max_score:.2f}）: {question[:30]}...")
        return None

    # ── Step 3：网络搜索 ──────────────────────────────────
    web_result = web_tool.search(web_query_raw)

    # ── Step 4：生成轨迹 ──────────────────────────────────
    trajectory = traj_builder.build(
        question=question,
        local_query=local_query,
        local_docs=local_docs,
        web_query=web_query_raw,
        web_result=web_result,
    )

    # ── Step 5：组装输出 ──────────────────────────────────
    unique_id = hashlib.md5(question.encode("utf-8")).hexdigest()
    return {
        "id":          item["id"],
        "unique_id":   unique_id,
        "category":    item.get("category", ""),
        "category_zh": item.get("category_zh", ""),
        "question":    question,
        "trajectory":  trajectory,
        "local_docs_count":   len(local_docs),
        "local_max_score":    round(max_score, 4),
        "web_search_used":    True,
        "sft_format":  to_sft_format(question, trajectory),
        "grpo_format": to_grpo_format(question, trajectory, item.get("category", "")),
    }


def load_checkpoint() -> set:
    """加载已完成的 ID，支持断点续传"""
    done = set()
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj["id"])
                except Exception:
                    pass
    return done


def save_checkpoint(result: dict):
    """实时写入检查点"""
    with open(CKPT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": result["id"]}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Web fallback trajectory builder")
    parser.add_argument("--resume",  action="store_true", help="断点续传，跳过已处理的问题")
    parser.add_argument("--dry-run", action="store_true", help="只处理前5条，验证流程")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "bing", "serpapi", "doubao"],
                        help="网络搜索后端")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发线程数")
    args = parser.parse_args()

    # ── 加载问题库 ──────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    if args.dry_run:
        questions = questions[:5]
        print(f"[DRY-RUN] 仅处理前 {len(questions)} 条")

    # 断点续传
    done_ids = load_checkpoint() if args.resume else set()
    questions = [q for q in questions if q["id"] not in done_ids]
    print(f"[INFO] 待处理：{len(questions)} 条（已跳过 {len(done_ids)} 条）")

    # ── 初始化工具 ─────────────────────────────────────
    local_tool   = LocalSearchTool()
    web_tool     = WebSearchTool(backend=args.backend)
    traj_builder = TrajectoryBuilder()

    # ── 并发处理 ───────────────────────────────────────
    results     = []
    file_lock   = threading.Lock()

    def _process(item):
        for attempt in range(RETRY_TIMES):
            try:
                result = process_one(item, local_tool, web_tool, traj_builder)
                if result:
                    with file_lock:
                        save_checkpoint(result)
                return result
            except Exception as e:
                if attempt < RETRY_TIMES - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[ERROR] 处理失败（{item['id']}）: {e}")
                    return None

    total = len(questions)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process, item): item for item in questions}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
            done_count += 1
            pct = done_count / total * 100
            bar_len = 40
            filled = int(bar_len * done_count / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r生成轨迹：{pct:5.1f}%|{bar}| {done_count}/{total}", end="", flush=True)
    print()  # 进度条换行

    # ── 保存输出 ───────────────────────────────────────
    # 1. 完整数据（含调试信息）
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 2. SFT 格式（用于 LLaMA-Factory warm-up 训练）
    sft_path = OUTPUT_PATH.replace(".json", "_sft.json")
    sft_data = [r["sft_format"] for r in results]
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    # 3. GRPO 格式（用于强化学习训练）
    grpo_path = OUTPUT_PATH.replace(".json", "_grpo.jsonl")
    with open(grpo_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r["grpo_format"], ensure_ascii=False) + "\n")

    # ── 统计报告 ───────────────────────────────────────
    category_counts = {}
    for r in results:
        cat = r.get("category_zh", "其他")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\n" + "=" * 60)
    print("📊 生成完成")
    print("=" * 60)
    print(f"总轨迹数：{len(results)} 条")
    print(f"\n分类分布：")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}：{cnt} 条")
    print(f"\n输出文件：")
    print(f"  完整数据：{OUTPUT_PATH}")
    print(f"  SFT格式：{sft_path}")
    print(f"  GRPO格式：{grpo_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
