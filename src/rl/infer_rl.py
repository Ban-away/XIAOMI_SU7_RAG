# -*- coding: utf-8 -*-
"""
RL 增强推理脚本 - Search-R1 范式：边生成边检索

核心机制（Search-R1 范式）：
  模型自由生成文本，系统通过 stop token 拦截工具调用标签：
    1. 模型生成 "<search_local>关键词" → vLLM 在 "</search_local>" 处停止
    2. 系统调用本地检索后端，获得真实结果
    3. 拼入 "<information>真实结果</information>"，作为 assistant 消息追加
    4. 模型继续生成，可能触发 "<search_web>" → 同样拦截 + 网络搜索
    5. 最终模型生成 "<answer>...</answer>"，本轮结束

与 infer.py 的区别：
  - infer.py:    检索 → 重排 → 生成（固定管线，模型被动接收上下文）
  - infer_rl.py: 模型主动决定何时检索、检索什么（自主工具调用循环）

运行：
  # 先启动 vLLM 服务（RL 模型）
  python deploy/auto_vllm_server.py --model LLaMA-Factory-main/output/qwen3_lora_rl --port 8000

  # 启动交互式问答
  python src/rl/infer_rl.py

  # 指定 vLLM 地址
  python src/rl/infer_rl.py --vllm-url http://localhost:8000/v1
"""

import os
import re
import sys
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import OpenAI

from src.rl.environment import RetrievalEnvironment
from src.rl.reward_model import compute_reward
from src.constant import qwen3_8b_tune_model_name


# ── 系统提示词（与训练时保持一致）────────────────────────
SYSTEM_PROMPT = """你是小米SU7车型的专业问答助手，服务范围严格限定在小米SU7相关问题。

回答问题时可以调用以下工具：
- 本地知识库检索（优先）：<search_local>检索关键词</search_local>
- 网络搜索（本地信息不足时）：<search_web>检索关键词</search_web>

工具返回格式：<information>检索结果内容</information>

最终答案格式：<answer>答案内容</answer>

注意：
1. 优先调用本地知识库，本地无结果或信息严重不足时再调用网络搜索
2. 与小米SU7无关的问题（闲聊、百科、娱乐等），直接输出 <answer>很抱歉，我只能回答小米SU7相关问题。</answer>
3. 网络搜索结果来源于互联网，答案中需注明"根据网络信息"
4. 涉及页码引用时格式为【页码】"""

# ── 推理参数 ────────────────────────────────────────────────
MAX_GENERATE_ROUNDS  = 8    # 最大生成轮数（防止无限循环）
MAX_TOKENS_PER_ROUND = 512  # 每轮生成最大 token 数

# 关键：stop 在搜索标签的闭合处，让模型暂停以便注入真实检索结果
# 不用 stop=["</answer>"]，因为那会导致 <answer> 内容永远不包含 </answer>
# is_done 判断改由 step() 内部用 _RE_ANSWER 正则检测
SEARCH_STOP_TOKENS = ["</search_local>", "</search_web>"]

# ── 标签正则（与 environment.py 保持一致）──────────────────
_RE_ANSWER       = re.compile(r"<answer>(.*?)</answer>",            re.DOTALL)
_RE_SEARCH_LOCAL = re.compile(r"<search_local>(.*?)</search_local>", re.DOTALL)
_RE_SEARCH_WEB   = re.compile(r"<search_web>(.*?)</search_web>",   re.DOTALL)


# ────────────────────────────────────────────────────────────
# 核心：Search-R1 边生成边检索循环
# ────────────────────────────────────────────────────────────

def run_rl_inference(
    question:    str,
    llm_client:  OpenAI,
    env:         RetrievalEnvironment,
    model_name:  str,
    verbose:     bool = True,
) -> dict:
    """
    Search-R1 范式推理：模型自主生成搜索查询，系统拦截并注入真实检索结果。

    循环流程：
      round 1: model → "<search_local>关键词" (stop at </search_local>)
               system → 执行本地检索 → 注入 <information>...</information>
      round 2: model → "<search_web>关键词" (stop at </search_web>)
               system → 执行网络搜索 → 注入 <information>...</information>
      round 3: model → "<answer>最终答案</answer>"
               system → 检测到 <answer>，结束

    Args:
        question:    用户问题
        llm_client:  OpenAI 客户端（连接 vLLM）
        env:         检索环境
        model_name:  模型名称
        verbose:     是否打印中间过程

    Returns:
        {
            "question":     str,
            "trajectory":   str,   # 完整轨迹
            "answer":       str,   # 纯答案
            "reward":       float, # 奖励分数
            "reward_detail": dict, # 奖励明细
            "rounds":       int,   # 实际生成轮数
            "search_calls": dict,  # 工具调用统计
        }
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    trajectory = ""
    rounds     = 0

    for round_idx in range(MAX_GENERATE_ROUNDS):
        rounds += 1

        # ── 调用模型生成 ───────────────────────────────────
        # 不用 stop=["</answer>"]，否则 vLLM 会吞掉 </answer>，
        # 导致 _RE_ANSWER 无法匹配，is_done 永远 False。
        # 只在搜索标签闭合处 stop，拦截后注入真实检索结果。
        try:
            completion = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=MAX_TOKENS_PER_ROUND,
                temperature=0.3,
                top_p=0.9,
                stop=SEARCH_STOP_TOKENS,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )
            generated = completion.choices[0].message.content or ""
            finish_reason = completion.choices[0].finish_reason
        except Exception as e:
            if verbose:
                print(f"\n[ERROR] 生成失败: {e}")
            break

        trajectory += generated
        if verbose:
            print(generated, end="", flush=True)

        # ── 检查模型是否已经生成了 <answer>...</answer>（自然结束）──
        if _RE_ANSWER.search(trajectory):
            if verbose:
                print()  # 换行
            break

        # ── 检查是否触发了搜索标签（stop 命中）──────────────
        hit_search = (finish_reason == "stop")

        if hit_search:
            # vLLM 在 stop token 处截断，需要补回闭合标签
            # 判断是哪种搜索被触发
            if "<search_local>" in generated and "</search_local>" not in generated:
                trajectory += "</search_local>"
                if verbose:
                    print("</search_local>", end="", flush=True)
                # 提取搜索关键词
                local_match = _RE_SEARCH_LOCAL.findall(trajectory)
                query = local_match[-1].strip() if local_match else question

                # 执行本地检索
                result_str, score = env.local_backend.search(query)
                info_block = f"<information>{result_str}</information>"

                # 如果本地分数低，追加提示
                if score < 0.35:
                    info_block = (
                        f"<information>{result_str}\n"
                        f"[提示：本地知识库相关性较低（{score:.2f}），"
                        f"如需更准确信息可调用网络搜索]</information>"
                    )

            elif "<search_web>" in generated and "</search_web>" not in generated:
                trajectory += "</search_web>"
                if verbose:
                    print("</search_web>", end="", flush=True)
                # 提取搜索关键词
                web_match = _RE_SEARCH_WEB.findall(trajectory)
                query = web_match[-1].strip() if web_match else f"小米SU7 {question}"

                # 执行网络搜索
                result_str = env.web_backend.search(query)
                info_block = f"<information>{result_str}</information>"
            else:
                # 其他 stop 原因，继续
                info_block = None

            if info_block:
                # 注入检索结果
                trajectory += "\n" + info_block + "\n"
                # 将当前生成内容 + 检索结果作为 assistant 消息
                messages.append({
                    "role": "assistant",
                    "content": generated + (
                        "</search_local>" if "<search_local>" in generated and "</search_local>" not in generated
                        else "</search_web>"
                    ) + "\n" + info_block + "\n",
                })
                if verbose:
                    print("\n" + info_block + "\n", end="", flush=True)
        else:
            # finish_reason != "stop"（如 length），说明模型本轮没有触发搜索
            # 将本轮生成内容追加为 assistant 消息，继续下一轮
            messages.append({"role": "assistant", "content": generated})

            # 再检查一次：如果本轮生成包含了 <answer> 但还没闭合
            if "<answer>" in generated and "</answer>" not in generated:
                # 模型可能被 max_tokens 截断了，再给一轮让它写完
                pass

    # ── 确保轨迹格式完整 ────────────────────────────────────
    # 如果 <answer> 未闭合，补齐
    if "<answer>" in trajectory and "</answer>" not in trajectory:
        trajectory += "</answer>"
        if verbose:
            print("</answer>", end="", flush=True)

    if verbose:
        print()  # 最终换行

    # ── 提取结果 ────────────────────────────────────────────
    answer       = RetrievalEnvironment.extract_answer(trajectory)
    search_calls = RetrievalEnvironment.count_search_calls(trajectory)
    reward       = compute_reward(question, trajectory)

    return {
        "question":      question,
        "trajectory":    trajectory,
        "answer":        answer,
        "reward":        reward["reward"],
        "reward_detail": reward,
        "rounds":        rounds,
        "search_calls":  search_calls,
    }


# ────────────────────────────────────────────────────────────
# 交互式问答主循环
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL 增强推理 - Search-R1 边生成边检索")
    parser.add_argument(
        "--vllm-url", type=str,
        default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="vLLM 服务地址",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型名称（默认使用 constant.py 中的配置）",
    )
    parser.add_argument(
        "--show-trajectory", action="store_true",
        help="显示完整轨迹（包含检索过程）",
    )
    parser.add_argument(
        "--show-reward", action="store_true",
        help="显示奖励分数详情",
    )
    args = parser.parse_args()

    model_name = args.model or qwen3_8b_tune_model_name

    # ── 初始化 ──────────────────────────────────────────────
    print("=" * 80)
    print("🚗 小米 SU7 RL 增强推理系统（Search-R1 范式）")
    print("=" * 80)
    print(f"  vLLM 地址: {args.vllm_url}")
    print(f"  模型:      {model_name}")
    print(f"  范式:      边生成边检索（模型自主决定检索时机）")
    print(f"  工具:      <search_local> / <search_web>")
    print("=" * 80)
    print("  输入问题开始对话，输入 'quit' 退出")
    print("=" * 80)

    llm_client = OpenAI(
        api_key="EMPTY",
        base_url=args.vllm_url,
    )

    print("\n[INFO] 加载检索环境...")
    env = RetrievalEnvironment()
    print("[INFO] 检索环境加载完成\n")

    # ── 交互循环 ────────────────────────────────────────────
    while True:
        try:
            question = input("🧑 用户 ➜ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        print(f"\n🤖 助手 ➜ ", end="", flush=True)

        start_time = time.time()
        result = run_rl_inference(
            question   = question,
            llm_client = llm_client,
            env        = env,
            model_name = model_name,
            verbose=True,
        )
        elapsed = time.time() - start_time

        # ── 输出结果 ────────────────────────────────────────
        print(f"  📝 答案: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
        print(f"  ⏱️  耗时: {elapsed:.1f}s | 轮数: {result['rounds']} | "
              f"检索: local×{result['search_calls']['local']} web×{result['search_calls']['web']}")

        if args.show_reward:
            detail = result["reward_detail"]
            print(f"  📊 奖励: {result['reward']:.3f} "
                  f"(格式:{detail['format_score']:.2f} 答案:{detail['answer_score']:.2f} "
                  f"工具:{detail['tool_score']:.2f} 来源:{detail['source_score']:.2f} "
                  f"领域:{detail['domain_score']:.2f})")

        if args.show_trajectory:
            print(f"\n  📋 完整轨迹:")
            print(f"  {'─' * 70}")
            for line in result["trajectory"].split("\n"):
                print(f"  {line}")
            print(f"  {'─' * 70}")

        print()


if __name__ == "__main__":
    main()
