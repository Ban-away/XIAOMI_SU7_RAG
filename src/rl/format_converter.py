# -*- coding: utf-8 -*-
"""
RL 训练数据格式转换器

功能：
  1. 读取 data_builder.py 生成的原始轨迹数据
  2. 转换为多种训练框架所需格式：
     - LLaMA-Factory SFT 格式（instruction / input / output）
     - GRPO 强化学习格式（prompt / completion）
     - ShareGPT 对话格式（messages 列表）
  3. 校验轨迹完整性（标签闭合、answer 非空）
  4. 支持多文件合并与去重

运行：
  # 转换为全部格式（默认）
  python src/rl/format_converter.py

  # 只生成 SFT 格式
  python src/rl/format_converter.py --format sft

  # 只生成 GRPO 格式
  python src/rl/format_converter.py --format grpo

  # 指定输入文件
  python src/rl/format_converter.py --input data/rl_data/web_fallback_trajectories.json

  # 合并多个轨迹文件后转换
  python src/rl/format_converter.py --input a.json b.json --merge
"""

import os
import re
import json
import argparse
import hashlib
from pathlib import Path
from typing import Optional


# ── 项目根目录 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 默认路径 ────────────────────────────────────────────────
DEFAULT_INPUT  = os.path.join(BASE_DIR, "data/rl_data/web_fallback_trajectories.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data/rl_data")

# ── 系统提示词（与 data_builder.py 保持一致）──────────────
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

# ── 标签正则 ────────────────────────────────────────────────
_RE_ANSWER       = re.compile(r"<answer>(.*?)</answer>",            re.DOTALL)
_RE_SEARCH_LOCAL = re.compile(r"<search_local>(.*?)</search_local>", re.DOTALL)
_RE_SEARCH_WEB   = re.compile(r"<search_web>(.*?)</search_web>",   re.DOTALL)
_RE_READ_PAGE    = re.compile(r"<read_page>(.*?)</read_page>",     re.DOTALL)
_RE_INFORMATION  = re.compile(r"<information>(.*?)</information>",  re.DOTALL)


# ────────────────────────────────────────────────────────────
# 轨迹校验
# ────────────────────────────────────────────────────────────

def validate_trajectory(trajectory: str) -> dict:
    """
    校验单条轨迹的完整性，返回校验报告。

    Returns:
        {
            "valid":        bool,
            "has_answer":   bool,
            "has_search":   bool,
            "has_read_page": bool,
            "answer_empty": bool,
            "errors":       list[str],
        }
    """
    errors = []
    has_answer    = bool(_RE_ANSWER.search(trajectory))
    has_local     = bool(_RE_SEARCH_LOCAL.search(trajectory))
    has_web       = bool(_RE_SEARCH_WEB.search(trajectory))
    has_read_page = bool(_RE_READ_PAGE.search(trajectory))
    has_info      = bool(_RE_INFORMATION.search(trajectory))
    has_search    = has_local or has_web

    if not has_answer:
        errors.append("缺少 <answer>...</answer> 标签")

    if not has_search:
        errors.append("缺少搜索调用标签（<search_local> 或 <search_web>）")

    if has_search and not has_info:
        errors.append("存在搜索调用但缺少 <information> 响应")

    # read_page 可选校验：存在时验证闭合和 URL 格式
    if has_read_page:
        opens  = trajectory.count("<read_page>")
        closes = trajectory.count("</read_page>")
        if opens != closes:
            errors.append(f"<read_page> 标签不匹配（{opens}开 / {closes}闭）")
        # 检查 URL 格式
        for url in _RE_READ_PAGE.findall(trajectory):
            url = url.strip()
            if not url.startswith(("http://", "https://")):
                errors.append(f"<read_page> URL 格式无效：{url[:60]}")

    # 检查 answer 内容是否为空
    answer_empty = False
    if has_answer:
        answer_text = _RE_ANSWER.search(trajectory).group(1).strip()
        if not answer_text:
            errors.append("<answer> 内容为空")
            answer_empty = True

    return {
        "valid":         len(errors) == 0,
        "has_answer":    has_answer,
        "has_search":    has_search,
        "has_read_page": has_read_page,
        "answer_empty":  answer_empty,
        "errors":        errors,
    }


def extract_answer(trajectory: str) -> str:
    """从轨迹中提取纯答案文本"""
    match = _RE_ANSWER.search(trajectory)
    return match.group(1).strip() if match else ""


def to_sft_target(trajectory: str) -> str:
    """
    构造 SFT warm-up 的监督目标。

    SFT 只学习工具调用决策和最终答案；<information> 是环境/工具返回，
    不应要求模型在 assistant 输出中背诵检索结果。
    """
    target = _RE_INFORMATION.sub("", trajectory)
    target = re.sub(r"\n{3,}", "\n\n", target)
    return target.strip()


# ────────────────────────────────────────────────────────────
# 格式转换函数
# ────────────────────────────────────────────────────────────

def to_sft_format(question: str, trajectory: str, system: str = SYSTEM_PROMPT, **kwargs) -> dict:
    """
    转换为 LLaMA-Factory SFT 格式。

    字段说明：
      instruction: 用户问题
      input:       留空（问题已包含在 instruction 中）
      output:      去掉 <information> 的轻量轨迹（作为训练目标）
      system:      系统提示词
    """
    answer_text = extract_answer(trajectory)
    return {
        "instruction": question,
        "input":       "",
        "output":      to_sft_target(trajectory),
        "answer":      answer_text,
        "system":      system,
        "data_source": "web_fallback",
    }


def to_grpo_format(
    question:    str,
    trajectory:  str,
    category:    str  = "",
    system:      str  = SYSTEM_PROMPT,
) -> dict:
    """
    转换为 GRPO 训练格式。

    prompt:     system + user 消息列表
    completion: assistant 轨迹文本
    answer:     纯答案文本（奖励计算用）
    """
    answer_text = extract_answer(trajectory)
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ],
        "completion":  trajectory,
        "answer":      answer_text,
        "category":    category,
        "data_source": "web_fallback",
        "reward_type": "web_answer_quality",
    }


def to_sharegpt_format(
    question:   str,
    trajectory: str,
    system:     str = SYSTEM_PROMPT,
    **kwargs,
) -> dict:
    """
    转换为 ShareGPT 多轮对话格式。

    messages 列表中包含 system → user → assistant 三轮，
    适合支持 conversation 格式的训练框架。
    """
    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": trajectory},
        ],
        "data_source": "web_fallback",
    }


# ── 格式注册表 ──────────────────────────────────────────────
FORMAT_REGISTRY = {
    "sft":      {"converter": to_sft_format,      "ext": "_sft.json",    "is_jsonl": False},
    "grpo":     {"converter": to_grpo_format,      "ext": "_grpo.jsonl",  "is_jsonl": True},
    "sharegpt": {"converter": to_sharegpt_format,  "ext": "_sharegpt.json", "is_jsonl": False},
}


# ────────────────────────────────────────────────────────────
# 数据加载与合并
# ────────────────────────────────────────────────────────────

def load_trajectories(path: str) -> list[dict]:
    """加载单个轨迹 JSON 文件"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def merge_files(paths: list[str], dedup: bool = True) -> list[dict]:
    """
    合并多个轨迹文件，可选按 question 去重。

    去重策略：相同 question 保留最新的（后加载的覆盖先加载的）。
    """
    all_data: list[dict] = []
    for p in paths:
        all_data.extend(load_trajectories(p))

    if not dedup:
        return all_data

    seen: dict[str, dict] = {}
    for item in all_data:
        question = item.get("question", "")
        key = hashlib.md5(question.encode("utf-8")).hexdigest()
        seen[key] = item  # 后者覆盖前者

    return list(seen.values())


# ────────────────────────────────────────────────────────────
# 核心转换流水线
# ────────────────────────────────────────────────────────────

def repair_trajectory(trajectory: str) -> tuple[str, list[str]]:
    """
    修复轨迹中常见的格式问题（如 LLM 输出被 max_tokens 截断导致闭标签缺失）。

    Returns:
        (repaired_trajectory, list_of_fixes)
    """
    fixes = []

    # 修复 <answer> 开标签存在但闭标签缺失（最常见的截断情况）
    if "<answer>" in trajectory and "</answer>" not in trajectory:
        trajectory += "</answer>"
        fixes.append("补齐 </answer> 闭标签")

    return trajectory, fixes


def convert(
    data:         list[dict],
    target_format: str  = "all",
    skip_invalid:  bool = True,
) -> dict[str, list[dict]]:
    """
    批量转换轨迹数据。

    Args:
        data:          原始轨迹数据列表
        target_format: "all" | "sft" | "grpo" | "sharegpt"
        skip_invalid:  是否跳过校验失败的轨迹

    Returns:
        {format_name: [converted_items]}
    """
    # 确定需要生成的格式
    if target_format == "all":
        formats = list(FORMAT_REGISTRY.keys())
    else:
        formats = [target_format]

    # 预处理：自动修复 + 校验 + 过滤
    repair_count = 0
    valid_items = []
    for item in data:
        trajectory = item.get("trajectory", "")

        # 自动修复已知格式问题
        trajectory, fixes = repair_trajectory(trajectory)
        if fixes:
            item["trajectory"] = trajectory
            q = item.get("question", "")[:30]
            print(f"  [FIX] {q}... → {fixes}")
            repair_count += 1

        report = validate_trajectory(trajectory)

        if not report["valid"]:
            q = item.get("question", "")[:30]
            if skip_invalid:
                print(f"  [SKIP] 轨迹不完整: {q}... → {report['errors']}")
                continue
            else:
                print(f"  [WARN] 轨迹不完整但仍保留: {q}... → {report['errors']}")

        valid_items.append(item)

    if repair_count:
        print(f"[INFO] 自动修复：{repair_count} 条")
    print(f"[INFO] 有效轨迹：{len(valid_items)}/{len(data)} 条")

    # 逐格式转换
    results: dict[str, list[dict]] = {}
    for fmt in formats:
        if fmt not in FORMAT_REGISTRY:
            print(f"[WARN] 未知格式 '{fmt}'，跳过")
            continue

        converter = FORMAT_REGISTRY[fmt]["converter"]
        converted = []
        for item in valid_items:
            try:
                converted.append(converter(
                    question   = item.get("question", ""),
                    trajectory = item.get("trajectory", ""),
                    category   = item.get("category", ""),
                ))
            except Exception as e:
                q = item.get("question", "")[:30]
                print(f"  [ERROR] 转换失败 ({fmt}): {q}... → {e}")

        results[fmt] = converted
        print(f"[INFO] {fmt:>8s} 格式：{len(converted)} 条")

    return results


def save(
    results:     dict[str, list[dict]],
    output_dir:  str           = DEFAULT_OUTPUT,
    base_name:   str           = "web_fallback_trajectories",
) -> list[str]:
    """
    将转换结果写入文件。

    Returns:
        写入的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []

    for fmt, items in results.items():
        if not items:
            continue

        cfg  = FORMAT_REGISTRY[fmt]
        path = os.path.join(output_dir, base_name + cfg["ext"])

        with open(path, "w", encoding="utf-8") as f:
            if cfg["is_jsonl"]:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                json.dump(items, f, ensure_ascii=False, indent=2)

        written.append(path)
        print(f"  [SAVE] {path} ({len(items)} 条)")

    return written


# ────────────────────────────────────────────────────────────
# 统计报告
# ────────────────────────────────────────────────────────────

def print_report(data: list[dict]):
    """打印轨迹数据的分类统计"""
    total = len(data)
    if total == 0:
        print("无数据。")
        return

    # 分类统计
    cat_counts: dict[str, int] = {}
    web_count   = 0
    answer_lens = []

    for item in data:
        cat = item.get("category_zh") or item.get("category", "其他")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        if item.get("web_search_used"):
            web_count += 1

        answer = extract_answer(item.get("trajectory", ""))
        if answer:
            answer_lens.append(len(answer))

    avg_len = sum(answer_lens) / len(answer_lens) if answer_lens else 0

    print("\n" + "=" * 60)
    print("📊 格式转换报告")
    print("=" * 60)
    print(f"总数：{total} 条")
    print(f"网络兜底：{web_count} 条 ({web_count / total * 100:.1f}%)")
    print(f"平均答案长度：{avg_len:.0f} 字")
    print(f"\n分类分布：")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}：{cnt} 条")
    print("=" * 60)


# ────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL 训练数据格式转换器")
    parser.add_argument(
        "--input", nargs="+", default=[DEFAULT_INPUT],
        help="输入轨迹 JSON 文件路径（支持多文件）",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT,
        help="输出目录（默认 data/rl_data/）",
    )
    parser.add_argument(
        "--format", dest="target_format", default="all",
        choices=["all", "sft", "grpo", "sharegpt"],
        help="目标格式（默认 all）",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="合并多个输入文件并按问题去重",
    )
    parser.add_argument(
        "--keep-invalid", action="store_true",
        help="保留校验失败的轨迹（默认跳过）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只校验和统计，不写文件",
    )
    args = parser.parse_args()

    # ── 加载数据 ──────────────────────────────────────────
    print(f"[INFO] 加载数据：{args.input}")
    if args.merge and len(args.input) > 1:
        data = merge_files(args.input, dedup=True)
        print(f"[INFO] 合并去重后：{len(data)} 条")
    else:
        data = []
        for p in args.input:
            data.extend(load_trajectories(p))
    print(f"[INFO] 共加载 {len(data)} 条轨迹")

    # ── 校验 + 转换 ──────────────────────────────────────
    results = convert(data, target_format=args.target_format, skip_invalid=not args.keep_invalid)

    # ── 统计报告 ──────────────────────────────────────────
    print_report(data)

    # ── 写文件 ────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY-RUN] 未写入文件")
    else:
        written = save(results, output_dir=args.output_dir)
        print(f"\n✅ 已写入 {len(written)} 个文件")


if __name__ == "__main__":
    main()
