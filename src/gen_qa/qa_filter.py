# -*- coding: utf-8 -*-
"""
QA 质量审核与过滤模块

审核流程：
  1. 清理问题中的序号前缀（"1.", "2.", "3." 等 LLM 生成时带出的噪声）
  2. 用 abbr_ch.csv 对问题和答案做缩写展开，统一表达方式
  3. 过滤低质量样本：答案过短、答案含页码引用、问题含无关关键词等
  4. 负样本（无答案）只保留在训练集，不进入测试集
"""

import re
import csv
import jieba
from pathlib import Path
from typing import List, Dict, Tuple

from src.constant import stopwords_path, base_dir

# ── 停用词表 ─────────────────────────────────────────────────
with open(stopwords_path, encoding="utf-8") as f:
    _STOPWORDS = set(line.strip() for line in f if line.strip())

# ── 缩写映射表 ──────────────────────────────────────────────
_ABBR_MAP: Dict[str, str] = {}   # 缩写 → 中文全称
_abbr_path = Path(base_dir) / "data/abbr_ch.csv"
if _abbr_path.exists():
    with open(_abbr_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                abbr, zh = row[0].strip(), row[1].strip()
                if abbr and zh:
                    _ABBR_MAP[abbr] = zh

# ── 低质量问题特征 ───────────────────────────────────────────
# 这些词出现在问题里，说明是无关领域的闲聊或非汽车问题
_IRRELEVANT_KEYWORDS = [
    "诗", "诗歌", "古诗", "唐诗", "宋词", "诗经",
    "还珠格格", "电视剧", "电影", "明星", "演员",
    "导航去", "怎么走", "放歌", "听歌", "播放",
    "天气", "股票", "游戏",
]

# 问题里含有序号前缀的正则（"1.", "2.", "1、" 等）
_NUM_PREFIX_RE = re.compile(r"^\s*\d+\s*[.．。、]\s*")

# 答案里含页码引用的正则（"第X页"、"页码X"）
_PAGE_REF_RE = re.compile(r"第\d+页|页码\s*\d+|参见第|请参阅")

# 答案里含图片引用
_IMAGE_REF_RE = re.compile(r"图\s*\d+|如图所示|见图")


# ── 工具函数 ─────────────────────────────────────────────────
def clean_question_prefix(question: str) -> str:
    """去掉 LLM 生成时附带的序号前缀：'1. 怎么...' → '怎么...'"""
    return _NUM_PREFIX_RE.sub("", question).strip()


def expand_abbreviations(text: str) -> str:
    """
    把文本中出现的缩写替换为中文全称。
    例：'PEPS 功能怎么用' → '无钥匙进入和无钥匙启动 功能怎么用'
    """
    for abbr, zh in _ABBR_MAP.items():
        # 只替换独立出现的缩写词，避免误替换子串
        text = re.sub(rf"(?<![A-Za-z]){re.escape(abbr)}(?![A-Za-z])", zh, text)
    return text


def is_low_quality(item: dict) -> Tuple[bool, str]:
    """
    判断一条 QA 是否为低质量，返回 (是否低质量, 原因)

    规则：
    1. 问题或答案为空
    2. 问题长度 < 5 字
    3. 答案长度 < 4 字（过短的答案通常是截断或无意义的）
    4. 问题包含无关领域关键词
    5. 答案包含页码引用（"第X页"）—— 模型不应引用页码
    6. 答案包含图片引用（"如图所示"）—— 纯文本 QA 不应依赖图片
    """
    q = item.get("question", "").strip()
    a = item.get("answer", "").strip()

    if not q or not a:
        return True, "问题或答案为空"

    if len(q) < 5:
        return True, f"问题过短({len(q)}字)"

    # 无答案样本（负样本）是合法的训练数据，但不进测试集
    if a == "无答案":
        return False, ""

    if len(a) < 4:
        return True, f"答案过短({len(a)}字)"

    for kw in _IRRELEVANT_KEYWORDS:
        if kw in q:
            return True, f"问题含无关关键词：{kw}"

    if _PAGE_REF_RE.search(a):
        return True, "答案含页码引用"

    if _IMAGE_REF_RE.search(a):
        return True, "答案含图片引用"

    return False, ""


# ── 主过滤函数 ────────────────────────────────────────────────
def filter_qa_pairs(
    items: List[dict],
    remove_no_answer: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """
    对 QA 列表执行完整审核流程。

    参数：
      items           - 原始 QA 列表
      remove_no_answer- True 时把无答案样本也过滤掉（测试集用）

    返回：
      (合格列表, 过滤掉的列表)
    """
    passed, rejected = [], []

    for raw_item in items:
        item = dict(raw_item)   # 不修改原始数据

        # Step1：清理序号前缀
        item["question"] = clean_question_prefix(item["question"])

        # Step2：缩写展开（query 和 answer 都做）
        item["question"] = expand_abbreviations(item["question"])
        if item.get("answer") and item["answer"] != "无答案":
            item["answer"] = expand_abbreviations(item["answer"])

        # Step3：过滤低质量
        bad, reason = is_low_quality(item)
        if bad:
            item["_reject_reason"] = reason
            rejected.append(item)
            continue

        # Step4：测试集模式下去掉无答案负样本
        if remove_no_answer and item.get("answer") == "无答案":
            item["_reject_reason"] = "测试集不含负样本"
            rejected.append(item)
            continue

        passed.append(item)

    return passed, rejected


def print_filter_report(total: int, passed: List[dict], rejected: List[dict]):
    """打印过滤报告"""
    from collections import Counter
    reasons = Counter(r.get("_reject_reason", "未知") for r in rejected)
    print(f"\n{'='*60}")
    print(f"QA 质量审核报告")
    print(f"{'='*60}")
    print(f"原始数量：{total}")
    print(f"通过数量：{len(passed)}（{len(passed)/total*100:.1f}%）")
    print(f"过滤数量：{len(rejected)}（{len(rejected)/total*100:.1f}%）")
    print(f"\n过滤原因分布：")
    for reason, count in reasons.most_common():
        print(f"  {reason}：{count} 条")
    print(f"{'='*60}\n")
