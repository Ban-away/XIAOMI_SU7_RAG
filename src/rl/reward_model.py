# -*- coding: utf-8 -*-
"""
RL 奖励函数 - 多维度轨迹质量评分

功能：
  为 GRPO 训练提供奖励信号，评估模型生成的工具调用轨迹质量。

  奖励维度（总分 1.0）：
    1. 格式完整性  (0.05) - 标签是否齐全、正确闭合（SFT 已教格式，GRPO 弱化此维度）
    2. 答案质量    (0.40) - 回答准确性、信息量、基于检索信息的 groundedness
    3. 工具合理性  (0.15) - 检索关键词是否精准、调用顺序是否合理
    4. 来源标注    (0.10) - 网络信息是否正确注明来源
    5. 领域合规    (0.15) - 是否正确拒答非 SU7 问题
    6. 探索深度    (0.15) - 本地充足即停 / 网络搜索有效利用 read_page

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
_RE_READ_PAGE    = re.compile(r"<read_page>(.*?)</read_page>",     re.DOTALL)
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
# 维度 1：格式完整性 (0.0 ~ 0.15)
# ────────────────────────────────────────────────────────────

def score_format(trajectory: str) -> float:
    """
    检查轨迹是否包含完整的标签结构。

    完整轨迹应包含（至少一种搜索方式）：
      方式一：<search_local> → <information> → <answer>          （本地检索）
      方式二：<search_web>   → <information> → <answer>          （网络搜索）
      方式三：<search_local> → <search_web> → <information> → <answer>（混合）
      可选：<read_page> → <information>（垂直搜索）

    评分标准（满分 0.05）：
      - 至少一种搜索方式 + information + answer = 0.03
      - 两种搜索方式均有 → +0.01
      - 所有已出现标签正确闭合 → +0.01
      - 标签不匹配 → -0.01/个
    """
    has_local = bool(_RE_SEARCH_LOCAL.search(trajectory))
    has_web   = bool(_RE_SEARCH_WEB.search(trajectory))
    has_info  = bool(_RE_INFORMATION.search(trajectory))
    has_answer = bool(_RE_ANSWER.search(trajectory))

    # 至少需要一种搜索方式
    if not (has_local or has_web):
        return 0.0

    score = 0.0
    # 核心结构：搜索 + 信息 + 答案
    if has_local or has_web: score += 0.01
    if has_info:             score += 0.01
    if has_answer:           score += 0.01
    # 两种搜索方式都用 → 更完整
    if has_local and has_web: score += 0.01

    # 标签闭合检查
    closure_ok = True
    for tag in ["search_local", "search_web", "read_page", "information", "answer"]:
        opens  = trajectory.count(f"<{tag}>")
        closes = trajectory.count(f"</{tag}>")
        if opens > 0 and opens != closes:
            closure_ok = False
            score -= 0.01  # 标签不匹配扣分
    if closure_ok and (has_local or has_web):
        score += 0.01  # 全部正确闭合加分

    return max(0.0, min(0.05, score))


# ────────────────────────────────────────────────────────────
# 维度 2：答案质量 (0.0 ~ 0.30)
# ────────────────────────────────────────────────────────────

def score_answer_quality(trajectory: str, question: str = "") -> float:
    """
    评估 <answer> 内容的质量（满分 0.40）。

    核心原则：细节奖励（长度/数字/术语）挂钩 groundedness，并对"有具体声称却
    零检索支撑"的幻觉施加惩罚——防止模型"编得详细流畅就拿高分"（reward hacking）。

    评分标准：
      - 非空基础分：0.10
      - 长度充实（>30 / >100 字）：各 +0.04，按 grounded_ratio 缩放（需 >50% 短语被支撑才全额）
      - 包含具体数据/规格：+0.05，按 grounded_ratio 缩放
      - 包含专业术语：+0.04，按 grounded_ratio 缩放
      - 语言自然流畅：+0.03
      - 与问题主题相关：+0.06
      - groundedness（答案短语在 <information> 中出现比例）：+min(0.08, ratio*0.12)
      - 幻觉惩罚：有数字/术语但 grounded_ratio==0 → -0.08
    """
    answer_match = _RE_ANSWER.search(trajectory)
    if not answer_match:
        return 0.0

    answer = answer_match.group(1).strip()
    if not answer:
        return 0.0

    # ── 先算 groundedness（答案是否基于检索信息）──
    grounded_ratio = 0.0
    info_matches = _RE_INFORMATION.findall(trajectory)
    info_text = " ".join(m.strip() for m in info_matches if m.strip())
    if info_text:
        ans_phrases = re.findall(r"[一-鿿]{3,}", answer)
        if ans_phrases:
            grounded_count = sum(1 for p in ans_phrases if p in info_text)
            grounded_ratio = grounded_count / len(ans_phrases)

    # ── 具体声称检测 ──
    has_number = bool(
        re.search(r"\d+[\.\d]*\s*(km|kW|N·m|V|Ah|mm|英寸|%|万元|秒|公里|马力)", answer)
    )
    tech_terms = [
        "激光雷达", "毫米波", "摄像头", "算力", "传感器",
        "电池", "电机", "逆变器", "减速器", "悬架",
        "制动", "转向", "扭矩", "功率", "续航",
    ]
    has_tech = any(t in answer for t in tech_terms)

    score = 0.10  # 非空基础分

    # ── 幻觉惩罚：有具体声称却无任何检索支撑 ──
    if (has_number or has_tech) and grounded_ratio == 0.0:
        score -= 0.08

    # ── 细节奖励：按 grounded_ratio 缩放（需 >50% 短语被支撑才全额发放）──
    g = min(1.0, grounded_ratio * 2.0)
    if len(answer) > 30:
        score += 0.04 * g
    if len(answer) > 100:
        score += 0.04 * g
    if has_number:
        score += 0.05 * g
    if has_tech:
        score += 0.04 * g

    # 语言自然度（风格特征，低权重，不挂钩 grounding）
    natural_markers = ["此外", "另外", "需要注意的是", "具体来说", "同时", "因此"]
    if any(m in answer for m in natural_markers):
        score += 0.03

    # 与问题的主题相关性：问题用 2 字片段，检查是否在答案中出现
    if question:
        q_keywords = set(re.findall(r"[一-鿿]{2,}", question))
        if q_keywords:
            hits = sum(1 for kw in q_keywords if kw in answer)
            ratio = hits / len(q_keywords)
            score += min(0.06, ratio * 0.12)

    # groundedness 强化奖励
    score += min(0.08, grounded_ratio * 0.12)

    return max(0.0, min(0.40, score))


# ────────────────────────────────────────────────────────────
# 维度 3：工具调用合理性 (0.0 ~ 0.15)
# ────────────────────────────────────────────────────────────

def score_tool_usage(trajectory: str, question: str = "") -> float:
    """
    评估工具调用的合理性。

    评分标准：
      - search_local 先于 search_web 出现得 0.06
      - 检索关键词简洁（<20 字）且不含冗余得 0.03
      - search_web 关键词包含 "小米SU7" 前缀得 0.03
      - 总调用次数合理（1-3 次 local + 1 次 web）得 0.03
    """
    score = 0.0

    # 检查调用顺序：local 应在 web 之前
    local_pos = trajectory.find("<search_local>")
    web_pos   = trajectory.find("<search_web>")
    if local_pos >= 0 and web_pos >= 0 and local_pos < web_pos:
        score += 0.06
    elif local_pos >= 0 and web_pos < 0:
        score += 0.05  # 只有 local 调用也合理

    # local 检索关键词质量
    local_match = _RE_SEARCH_LOCAL.search(trajectory)
    if local_match:
        local_query = local_match.group(1).strip()
        if len(local_query) <= 20 and len(local_query) >= 2:
            score += 0.03
        elif len(local_query) > 20:
            score += 0.01  # 太长扣分

    # web 检索关键词是否加了 "小米SU7" 前缀
    web_match = _RE_SEARCH_WEB.search(trajectory)
    if web_match:
        web_query = web_match.group(1).strip()
        if "小米" in web_query or "SU7" in web_query:
            score += 0.03

    # 调用次数合理性
    local_count = len(_RE_SEARCH_LOCAL.findall(trajectory))
    web_count   = len(_RE_SEARCH_WEB.findall(trajectory))
    read_count  = len(_RE_READ_PAGE.findall(trajectory))
    if 1 <= local_count <= 3 and 1 <= web_count <= 2:
        score += 0.03
    elif local_count + web_count + read_count > 6:
        score -= 0.03  # 过度调用扣分

    return max(0.0, min(0.15, score))


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
# 维度 6：探索深度 (0.0 ~ 0.15)
# ────────────────────────────────────────────────────────────

def score_exploration_depth(trajectory: str) -> float:
    """
    评估是否有效利用 read_page 进行垂直搜索（WebWalker 式深度探索）。
    同时公平对待 local-only 轨迹：本地检索已充分时给予合理分数。

    评分标准：
      【Local-only 轨迹】（无 web 搜索）
        - 本地信息充分 + 答案充实：0.07（本地探索已足够；压低上限以鼓励 web 探索）
        - 本地信息充分但答案简短：0.04
        - 本地信息不充分（空结果/低相关性）：0.02
      【Web 轨迹】（有 web 搜索）
        - 有 web 但无 read_page：0.03（仅表面搜索）
        - 有 read_page 且 URL 有效：+0.05
        - read_page 内容被 answer 引用：+0.04
        - 多次 read_page（≤2）：+0.03
        - read_page 在 search_web 之前：-0.03（顺序错误）
    """
    score = 0.0

    has_local = bool(_RE_SEARCH_LOCAL.search(trajectory))
    has_web = bool(_RE_SEARCH_WEB.search(trajectory))
    read_matches = _RE_READ_PAGE.findall(trajectory)
    read_count = len(read_matches)
    answer_match = _RE_ANSWER.search(trajectory)

    # ── Local-only 轨迹：本地检索已足够时给予合理分数 ──────────
    if not has_web:
        if has_local:
            info_matches = _RE_INFORMATION.findall(trajectory)
            # 过滤掉空结果提示（"未检索到"/"相关性较低"）
            substantive = [
                m.strip() for m in info_matches
                if m.strip() and "未检索到" not in m and "相关性较低" not in m
            ]
            if substantive and answer_match:
                answer = answer_match.group(1).strip()
                # 本地探索上限压低（0.10→0.07），使 web+read_page（最高 0.15）相对更有吸引力，
                # 抑制"只搜本地就交差"的偷懒捷径
                score = 0.07 if len(answer) > 20 else 0.04
            else:
                score = 0.02  # 本地结果不充分
        return max(0.0, min(0.15, score))

    # ── Web 轨迹：原有逻辑 ──────────────────────────────────

    # 有 web 搜索但没有深入阅读——仅表面搜索
    if read_count == 0:
        score = 0.03
        return max(0.0, min(0.15, score))

    # read_page URL 有效性
    valid_urls = 0
    for url in read_matches:
        url = url.strip()
        if url.startswith(("http://", "https://")):
            valid_urls += 1

    if valid_urls > 0:
        score += 0.05

    # read_page 内容被 answer 引用
    answer_match = _RE_ANSWER.search(trajectory)
    if answer_match and read_count > 0:
        answer = answer_match.group(1).strip()
        # 检查是否有从 read_page 获得的详细信息被引用
        # 简单启发式：答案长度较长且包含 read_page 的页面域名
        for url in read_matches:
            from urllib.parse import urlparse
            try:
                domain = urlparse(url.strip()).netloc
                if domain and domain in answer:
                    score += 0.04
                    break
            except Exception:
                pass
        # 备选：答案足够详细且出现了 read_page 之后的 information 内容关键词
        if score < 0.09 and len(answer) > 50:
            # 检查 read_page 之后的 information 块
            read_pos = trajectory.find("<read_page>")
            if read_pos >= 0:
                post_text = trajectory[read_pos:]
                info_match = _RE_INFORMATION.search(post_text)
                if info_match:
                    info_text = info_match.group(1).strip()
                    # 取 information 中的关键词，看是否在 answer 中出现
                    info_keywords = set(re.findall(r"[一-鿿]{2,}", info_text[:200]))
                    ans_keywords = set(re.findall(r"[一-鿿]{2,}", answer))
                    if info_keywords & ans_keywords:
                        score += 0.04

    # 多次有效 read_page（≤2 为合理范围）
    if 1 < read_count <= 2:
        score += 0.03

    # 顺序检查：read_page 应出现在 search_web 之后
    web_pos = trajectory.find("<search_web>")
    for rm in _RE_READ_PAGE.finditer(trajectory):
        if rm.start() < web_pos:
            score -= 0.03  # 顺序错误：read_page 不应在 web 搜索之前
            break

    return max(0.0, min(0.15, score))


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
            "reward":            float,  # 总分 [0, 1]
            "format_score":      float,  # 格式完整性 (0.15)
            "answer_score":      float,  # 答案质量 (0.30)
            "tool_score":        float,  # 工具合理性 (0.15)
            "source_score":      float,  # 来源标注 (0.10)
            "domain_score":      float,  # 领域合规 (0.15)
            "exploration_score": float,  # 探索深度 (0.15)
        }
    """
    fmt        = score_format(trajectory)
    ans        = score_answer_quality(trajectory, question)
    tool       = score_tool_usage(trajectory, question)
    src        = score_source_attribution(trajectory)
    domain     = score_domain_compliance(question, trajectory)
    exploration = score_exploration_depth(trajectory)

    total = fmt + ans + tool + src + domain + exploration

    result = {
        "reward":            round(total,       4),
        "format_score":      round(fmt,         4),
        "answer_score":      round(ans,         4),
        "tool_score":        round(tool,        4),
        "source_score":      round(src,         4),
        "domain_score":      round(domain,      4),
        "exploration_score": round(exploration,  4),
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  问题: {question[:50]}...")
        print(f"{'='*50}")
        print(f"  格式完整性:  {fmt:.3f} / 0.05")
        print(f"  答案质量:    {ans:.3f} / 0.40")
        print(f"  工具合理性:  {tool:.3f} / 0.15")
        print(f"  来源标注:    {src:.3f} / 0.10")
        print(f"  领域合规:    {domain:.3f} / 0.15")
        print(f"  探索深度:    {exploration:.3f} / 0.15")
        print(f"  {'─'*40}")
        print(f"  总奖励:      {total:.3f} / 1.00")
        print(f"{'='*50}")

    return result


# ────────────────────────────────────────────────────────────
# 自定义奖励函数入口（兼容 TRL GRPOTrainer）
# ────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """
    从 TRL 传入的 completion/prompt 中提取纯文本。

    TRL GRPOTrainer 可能传入以下格式：
      - str:                         "文本内容"
      - list[dict] (messages):       [{"role": "assistant", "content": "文本"}]
      - list[str]:                   ["文本内容"]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # messages 格式 [{"role": "assistant", "content": "..."}]
        for msg in content:
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
        # 纯字符串列表
        return " ".join(str(c) for c in content)
    return str(content)


def reward_fn(completions: list, **kwargs) -> list[float]:
    """
    GRPO 自定义奖励函数，兼容 TRL GRPOTrainer 的 reward_funcs 接口。

    用法：
      # TRL GRPOTrainer
      trainer = GRPOTrainer(reward_funcs=reward_fn, ...)

    Args:
        completions: 模型生成的轨迹（str / list[dict] / list[str]）
        **kwargs:    TRL 传入 prompts

    Returns:
        每条轨迹对应的奖励分数列表 [0.0, 1.0]
    """
    prompts = kwargs.get("prompts", [""] * len(completions))
    rewards = []

    for prompt_text, completion in zip(prompts, completions):
        question   = _extract_question_from_prompt(_extract_text(prompt_text))
        trajectory = _extract_text(completion)
        result = compute_reward(question, trajectory)
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
                "question": "小米SU7最新OTA更新了什么功能？",
                "trajectory": (
                    "<search_local>小米SU7 OTA更新 功能</search_local>\n"
                    "<information>[提示：本地知识库相关性较低（0.18），"
                    "如需更准确信息可调用网络搜索]</information>\n"
                    "<search_web>小米SU7 最新OTA版本 2025 更新内容</search_web>\n"
                    "<information>"
                    "【小米SU7 OTA v2.4.0 发布公告】新增城市领航辅助、HUD自定义显示等12项更新\n"
                    "网址：https://www.xiaomi.com/ev/su7/ota\n"
                    "【车主社区】OTA v2.4.0 详细体验报告\n"
                    "网址：https://www.autohome.com.cn/news/202501/su7-ota</information>\n"
                    "<read_page>https://www.xiaomi.com/ev/su7/ota</read_page>\n"
                    "<information>[页面来源：www.xiaomi.com]\n"
                    "小米SU7 OTA v2.4.0 正式发布，本次更新包含12项功能升级：\n"
                    "1. 城市领航辅助（City NOA）正式上线\n"
                    "2. HUD抬头显示新增自定义模式\n"
                    "3. 语音助手升级，支持多轮对话\n"
                    "4. 座椅记忆功能优化...</information>\n"
                    "<answer>根据小米官方页面信息，小米SU7最新的OTA v2.4.0更新了以下主要功能：\n"
                    "1. 城市领航辅助（City NOA）正式上线\n"
                    "2. HUD抬头显示新增自定义模式\n"
                    "3. 语音助手升级，支持多轮对话\n"
                    "4. 座椅记忆功能优化\n"
                    "本次更新共包含12项功能升级。"
                    "（以上信息来源于www.xiaomi.com，请以小米官方最新公告为准）</answer>"
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
    avg_explore = sum(r.get("exploration_score", 0) for r in results) / len(results)

    print("\n" + "=" * 60)
    print(f"📊 批量评估报告（{len(results)} 条）")
    print("=" * 60)
    print(f"  平均总奖励:     {avg_reward:.4f} / 1.00")
    print(f"  格式完整性:     {avg_fmt:.4f} / 0.15")
    print(f"  答案质量:       {avg_ans:.4f} / 0.30")
    print(f"  工具合理性:     {avg_tool:.4f} / 0.15")
    print(f"  来源标注:       {avg_src:.4f} / 0.10")
    print(f"  领域合规:       {avg_domain:.4f} / 0.15")
    print(f"  探索深度:       {avg_explore:.4f} / 0.15")
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
