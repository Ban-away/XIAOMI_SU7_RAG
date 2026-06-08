# -*- coding: utf-8 -*-
"""
RL 奖励函数 - 多维度轨迹质量评分

功能：
  为 GRPO 训练提供奖励信号，评估模型生成的工具调用轨迹质量。

  奖励维度（总分 1.0）：
    1. 格式完整性  (0.20) - 标签是否齐全、正确闭合
    2. 答案质量    (0.35) - 回答是否准确、信息量是否充足
    3. 工具合理性  (0.20) - 检索关键词是否精准、调用顺序是否合理
    4. 来源标注    (0.10) - 网络信息是否正确注明来源
    5. 领域合规    (0.15) - 是否正确拒答非 SU7 问题

  支持两种使用模式：
    - 规则模式（默认）：基于正则 + 启发式规则打分，无需 GPU
    - 模型模式（可选）：调用 LLM 做语义评分，更精准但需 API

运行：
  python src/rl/reward_model.py                                    # 单条测试
  python src/rl/reward_model.py --input data/rl_data/web_fallback_trajectories_grpo.jsonl
"""

import re
import os
import json
import argparse
from typing import Optional

# ── 标签正则 ────────────────────────────────────────────────
_RE_ANSWER       = re.compile(r"<answer>(.*?)</answer>",            re.DOTALL)
_RE_SEARCH_LOCAL = re.compile(r"<search_local>(.*?)</search_local>", re.DOTALL)
_RE_SEARCH_WEB   = re.compile(r"<search_web>(.*?)</search_web>",   re.DOTALL)
_RE_INFORMATION  = re.compile(r"<information>(.*?)</information>",  re.DOTALL)

# ── SU7 领域关键词 ─────────────────────────────────────────
_SU7_KEYWORDS = {
    "小米SU7", "小米 SU7", "小米汽车", "SU7", "SU7 Max", "SU7 Pro",
    "HyperOS", "小米智驾", "澎湃", "米家", "小爱同学",
    "OTA", "弹射起步", "智能驾驶", "辅助驾驶", "领航辅助",
    "激光雷达", "纯电", "续航", "充电", "超级充电",
    "车机", "中控屏", "HUD", "座椅", "空调", "天窗",
    "自动驾驶", "自动泊车", "高速领航", "城市领航",
}

# ── 拒答关键词 ──────────────────────────────────────────────
_REFUSAL_PATTERNS = [
    "只能回答小米SU7相关问题",
    "无法回答此问题",
    "不在我的服务范围",
    "与小米SU7无关",
    "我只能回答",
    "建议您咨询",
]


# ────────────────────────────────────────────────────────────
# 维度 1：格式完整性 (0.0 ~ 0.20)
# ────────────────────────────────────────────────────────────

def score_format(trajectory: str) -> float:
    """
    检查轨迹是否包含完整的标签结构。

    完整轨迹应包含：
      <search_local> → <information> → <search_web> → <information> → <answer>

    每缺失一个关键标签扣 0.05，最低 0 分。
    """
    score = 0.0
    checks = {
        "search_local": bool(_RE_SEARCH_LOCAL.search(trajectory)),
        "information":  bool(_RE_INFORMATION.search(trajectory)),
        "search_web":   bool(_RE_SEARCH_WEB.search(trajectory)),
        "answer":       bool(_RE_ANSWER.search(trajectory)),
    }

    # 每个标签存在得 0.05
    score += sum(0.05 for exists in checks.values() if exists)

    # 标签闭合检查：开闭标签数量是否匹配
    for tag in ["search_local", "search_web", "information", "answer"]:
        opens  = trajectory.count(f"<{tag}>")
        closes = trajectory.count(f"</{tag}>")
        if opens > 0 and opens == closes:
            score += 0.0  # 已在存在性中计分
        elif opens > 0 and opens != closes:
            score -= 0.02  # 标签不匹配扣分

    return max(0.0, min(0.20, score))


# ────────────────────────────────────────────────────────────
# 维度 2：答案质量 (0.0 ~ 0.35)
# ────────────────────────────────────────────────────────────

def score_answer_quality(trajectory: str, question: str = "") -> float:
    """
    评估 <answer> 内容的质量。

    评分标准：
      - 非空得基础分 0.10
      - 答案长度 > 30 字得 0.05
      - 包含具体数据/规格得 0.05
      - 语言自然流畅（非纯复制）得 0.05
      - 答案与问题主题相关得 0.10
    """
    answer_match = _RE_ANSWER.search(trajectory)
    if not answer_match:
        return 0.0

    answer = answer_match.group(1).strip()
    if not answer:
        return 0.0

    score = 0.10  # 非空基础分

    # 长度奖励
    if len(answer) > 30:
        score += 0.05
    if len(answer) > 80:
        score += 0.02

    # 包含具体数据（数字、单位、规格）
    if re.search(r"\d+[\.\d]*\s*(km|kW|N·m|V|Ah|mm|英寸|%|万元|秒|公里|马力)", answer):
        score += 0.05

    # 包含专业术语
    tech_terms = [
        "激光雷达", "毫米波", "摄像头", "算力", "传感器",
        "电池", "电机", "逆变器", "减速器", "悬架",
        "制动", "转向", "扭矩", "功率", "续航",
    ]
    if any(t in answer for t in tech_terms):
        score += 0.03

    # 语言自然度（非纯复制粘贴的标志）
    natural_markers = ["此外", "另外", "需要注意的是", "具体来说", "同时", "因此"]
    if any(m in answer for m in natural_markers):
        score += 0.02

    # 与问题的主题相关性
    if question:
        # 提取问题中的核心关键词
        q_keywords = set(re.findall(r"[一-鿿]{2,}", question))
        a_keywords = set(re.findall(r"[一-鿿]{2,}", answer))
        overlap = q_keywords & a_keywords
        if overlap:
            ratio = len(overlap) / max(len(q_keywords), 1)
            score += min(0.10, ratio * 0.15)

    return max(0.0, min(0.35, score))


# ────────────────────────────────────────────────────────────
# 维度 3：工具调用合理性 (0.0 ~ 0.20)
# ────────────────────────────────────────────────────────────

def score_tool_usage(trajectory: str, question: str = "") -> float:
    """
    评估工具调用的合理性。

    评分标准：
      - search_local 先于 search_web 出现得 0.08
      - 检索关键词简洁（<20 字）且不含冗余得 0.04
      - search_web 关键词包含 "小米SU7" 前缀得 0.04
      - 总调用次数合理（1-3 次 local + 1 次 web）得 0.04
    """
    score = 0.0

    # 检查调用顺序：local 应在 web 之前
    local_pos = trajectory.find("<search_local>")
    web_pos   = trajectory.find("<search_web>")
    if local_pos >= 0 and web_pos >= 0 and local_pos < web_pos:
        score += 0.08
    elif local_pos >= 0 and web_pos < 0:
        score += 0.06  # 只有 local 调用也合理

    # local 检索关键词质量
    local_match = _RE_SEARCH_LOCAL.search(trajectory)
    if local_match:
        local_query = local_match.group(1).strip()
        if len(local_query) <= 20 and len(local_query) >= 2:
            score += 0.04
        elif len(local_query) > 20:
            score += 0.01  # 太长扣分

    # web 检索关键词是否加了 "小米SU7" 前缀
    web_match = _RE_SEARCH_WEB.search(trajectory)
    if web_match:
        web_query = web_match.group(1).strip()
        if "小米" in web_query or "SU7" in web_query:
            score += 0.04

    # 调用次数合理性
    local_count = len(_RE_SEARCH_LOCAL.findall(trajectory))
    web_count   = len(_RE_SEARCH_WEB.findall(trajectory))
    if 1 <= local_count <= 3 and 1 <= web_count <= 2:
        score += 0.04
    elif local_count + web_count > 5:
        score -= 0.04  # 过度调用扣分

    return max(0.0, min(0.20, score))


# ────────────────────────────────────────────────────────────
# 维度 4：来源标注 (0.0 ~ 0.10)
# ────────────────────────────────────────────────────────────

def score_source_attribution(trajectory: str) -> float:
    """
    评估是否正确标注了信息来源。

    评分标准：
      - 使用了网络搜索时，answer 中注明"网络信息"得 0.05
      - 使用了页码引用格式得 0.03
      - 结尾有免责声明得 0.02
    """
    score = 0.0
    answer_match = _RE_ANSWER.search(trajectory)
    if not answer_match:
        return 0.0

    answer = answer_match.group(1).strip()
    has_web = bool(_RE_SEARCH_WEB.search(trajectory))

    # 网络信息来源标注
    if has_web:
        if any(kw in answer for kw in ["网络信息", "来源于网络", "根据网络"]):
            score += 0.05
        elif "网络" in answer:
            score += 0.02

    # 页码引用格式：匹配 【1】或【1,3,5】
    if re.search(r"【\d+(?:,\d+)*】", answer) or re.search(r"第\d+页", answer):
        score += 0.03

    # 免责声明
    disclaimer_keywords = ["请以小米官方", "以官方为准", "最新公告为准", "建议访问"]
    if any(kw in answer for kw in disclaimer_keywords):
        score += 0.02

    return max(0.0, min(0.10, score))


# ────────────────────────────────────────────────────────────
# 维度 5：领域合规性 (0.0 ~ 0.15)
# ────────────────────────────────────────────────────────────

def score_domain_compliance(question: str, trajectory: str) -> float:
    """
    评估领域合规性。

    对于 SU7 相关问题：应正常回答，给出高质量答案。
    对于非 SU7 问题：应正确拒答。

    评分标准：
      - SU7 相关问题给出了实质性回答得 0.10
      - SU7 相关问题没有误触发拒答得 0.05
      - 非 SU7 问题正确拒答得 0.15
      - 非 SU7 问题没有拒答（幻觉风险）得 0.00
    """
    answer_match = _RE_ANSWER.search(trajectory)
    if not answer_match:
        return 0.0

    answer = answer_match.group(1).strip()

    # 判断问题是否属于 SU7 领域
    is_su7_related = _is_su7_question(question)

    if is_su7_related:
        score = 0.05  # 基础分
        # 没有误触发拒答
        if not any(p in answer for p in _REFUSAL_PATTERNS):
            score += 0.05
        # 给出了实质性回答（非空且长度合理）
        if len(answer) > 20:
            score += 0.05
    else:
        # 非 SU7 问题
        if any(p in answer for p in _REFUSAL_PATTERNS):
            score = 0.15  # 正确拒答
        else:
            score = 0.00  # 未拒答，有幻觉风险

    return max(0.0, min(0.15, score))


def _is_su7_question(question: str) -> bool:
    """判断问题是否属于 SU7 领域"""
    # 直接包含 SU7 关键词
    for kw in _SU7_KEYWORDS:
        if kw in question:
            return True

    # 汽车通用关键词（大概率是 SU7 相关）
    car_keywords = {
        "车辆", "驾驶", "充电", "续航", "电池", "轮胎", "刹车",
        "方向盘", "安全带", "空调", "座椅", "车门", "车窗",
        "后备箱", "仪表盘", "导航", "泊车", "灯光", "雨刷",
        "油门", "换挡", "启动", "熄火", "保养", "保险",
    }
    car_hits = sum(1 for kw in car_keywords if kw in question)
    if car_hits >= 2:
        return True

    return False


# ────────────────────────────────────────────────────────────
# 综合奖励函数
# ────────────────────────────────────────────────────────────

def compute_reward(
    question:   str,
    trajectory: str,
    verbose:    bool = False,
) -> dict:
    """
    计算综合奖励分数。

    Args:
        question:   用户问题
        trajectory: 模型生成的完整轨迹
        verbose:    是否打印各维度得分

    Returns:
        {
            "reward":          float,  # 总分 [0, 1]
            "format_score":    float,  # 格式完整性
            "answer_score":    float,  # 答案质量
            "tool_score":      float,  # 工具合理性
            "source_score":    float,  # 来源标注
            "domain_score":    float,  # 领域合规
        }
    """
    fmt    = score_format(trajectory)
    ans    = score_answer_quality(trajectory, question)
    tool   = score_tool_usage(trajectory, question)
    src    = score_source_attribution(trajectory)
    domain = score_domain_compliance(question, trajectory)

    total = fmt + ans + tool + src + domain

    result = {
        "reward":       round(total, 4),
        "format_score": round(fmt,    4),
        "answer_score": round(ans,    4),
        "tool_score":   round(tool,   4),
        "source_score": round(src,    4),
        "domain_score": round(domain, 4),
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  问题: {question[:50]}...")
        print(f"{'='*50}")
        print(f"  格式完整性:  {fmt:.3f} / 0.20")
        print(f"  答案质量:    {ans:.3f} / 0.35")
        print(f"  工具合理性:  {tool:.3f} / 0.20")
        print(f"  来源标注:    {src:.3f} / 0.10")
        print(f"  领域合规:    {domain:.3f} / 0.15")
        print(f"  {'─'*40}")
        print(f"  总奖励:      {total:.3f} / 1.00")
        print(f"{'='*50}")

    return result


# ────────────────────────────────────────────────────────────
# LLaMA-Factory 自定义奖励函数入口
# ────────────────────────────────────────────────────────────

def reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    LLaMA-Factory GRPO 自定义奖励函数接口。

    在 LLaMA-Factory 中通过 custom_reward_config 注册：
      custom_reward_config:
        reward_type: function
        reward_function: src.rl.reward_model.reward_fn

    Args:
        completions: 模型生成的轨迹文本列表

    Returns:
        每条轨迹对应的奖励分数列表
    """
    prompts = kwargs.get("prompts", [""] * len(completions))
    rewards = []

    for prompt_text, completion in zip(prompts, completions):
        # 从 prompt 中提取用户问题
        question = _extract_question_from_prompt(prompt_text)
        result = compute_reward(question, completion)
        rewards.append(result["reward"])

    return rewards


def _extract_question_from_prompt(prompt_text: str) -> str:
    """从 GRPO prompt 消息中提取用户问题"""
    if isinstance(prompt_text, list):
        # messages 格式
        for msg in prompt_text:
            if msg.get("role") == "user":
                return msg.get("content", "")
    elif isinstance(prompt_text, str):
        return prompt_text
    return ""


# ────────────────────────────────────────────────────────────
# 批量评估 CLI
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL 奖励函数评估工具")
    parser.add_argument(
        "--input", type=str, default=None,
        help="GRPO JSONL 文件路径，批量评估奖励分数",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="运行示例演示",
    )
    args = parser.parse_args()

    if args.demo or not args.input:
        # ── 演示模式 ────────────────────────────────────────
        print("=" * 60)
        print("📊 奖励函数演示")
        print("=" * 60)

        demos = [
            {
                "question": "小米SU7的续航里程是多少？",
                "trajectory": (
                    "<search_local>小米SU7 续航里程</search_local>\n"
                    "<information>[1] 小米SU7标准版CLTC续航里程为700km，"
                    "SU7 Pro版续航里程为830km，SU7 Max版续航里程为800km。</information>\n"
                    "<search_web>小米SU7 续航里程 最新数据</search_web>\n"
                    "<information>根据最新网络信息，小米SU7全系续航范围在700-830km之间。</information>\n"
                    "<answer>小米SU7根据不同版本的续航里程如下：\n"
                    "标准版：CLTC续航700km\n"
                    "Pro版：CLTC续航830km\n"
                    "Max版：CLTC续航800km\n"
                    "续航表现处于同级别纯电轿车前列水平。"
                    "（以上信息来源于网络，请以小米官方最新公告为准）</answer>"
                ),
            },
            {
                "question": "今天天气怎么样？",
                "trajectory": (
                    "<search_local>天气</search_local>\n"
                    "<information>本地知识库中未检索到相关内容。</information>\n"
                    "<answer>很抱歉，我只能回答小米SU7相关问题。</answer>"
                ),
            },
        ]

        for d in demos:
            compute_reward(d["question"], d["trajectory"], verbose=True)
        return

    # ── 批量评估模式 ────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        return

    results = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question   = ""
            completion = item.get("completion", "")

            # 从 prompt 中提取问题
            prompt = item.get("prompt", "")
            if isinstance(prompt, list):
                for msg in prompt:
                    if msg.get("role") == "user":
                        question = msg.get("content", "")
            elif isinstance(prompt, str):
                question = prompt

            reward = compute_reward(question, completion)
            results.append(reward)

    # 统计
    if not results:
        print("无数据。")
        return

    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_fmt    = sum(r["format_score"] for r in results) / len(results)
    avg_ans    = sum(r["answer_score"] for r in results) / len(results)
    avg_tool   = sum(r["tool_score"] for r in results) / len(results)
    avg_src    = sum(r["source_score"] for r in results) / len(results)
    avg_domain = sum(r["domain_score"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print(f"📊 批量评估报告（{len(results)} 条）")
    print("=" * 60)
    print(f"  平均总奖励:     {avg_reward:.4f} / 1.00")
    print(f"  格式完整性:     {avg_fmt:.4f} / 0.20")
    print(f"  答案质量:       {avg_ans:.4f} / 0.35")
    print(f"  工具合理性:     {avg_tool:.4f} / 0.20")
    print(f"  来源标注:       {avg_src:.4f} / 0.10")
    print(f"  领域合规:       {avg_domain:.4f} / 0.15")
    print("=" * 60)

    # 分布统计
    brackets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for r in results:
        score = r["reward"]
        if score < 0.2:
            brackets["0.0-0.2"] += 1
        elif score < 0.4:
            brackets["0.2-0.4"] += 1
        elif score < 0.6:
            brackets["0.4-0.6"] += 1
        elif score < 0.8:
            brackets["0.6-0.8"] += 1
        else:
            brackets["0.8-1.0"] += 1

    print("\n奖励分布：")
    for bracket, count in brackets.items():
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"  {bracket}: {count:>4d} ({pct:>5.1f}%) {bar}")


if __name__ == "__main__":
    main()
