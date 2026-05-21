# -*- coding: utf-8 -*-
"""
vLLM 服务性能压测脚本
测量指标：
  - TTFT（首字延迟，Time To First Token）：量化前 vs 量化后对比
  - 吞吐率（Throughput）：token/s
  - 平均延迟（Latency）：端到端响应时间

运行前确保 vLLM 服务已启动：
  python deploy/auto_vllm_server.py --model /path/to/model --port 8000

运行：
  # 测试量化后模型（默认）
  python deploy/benchmark.py

  # 对比量化前后
  python deploy/benchmark.py --compare
"""

import argparse
import json
import time
import asyncio
import aiohttp
import numpy as np
import os
from tqdm import tqdm
from typing import List, Tuple

# ── 配置 ─────────────────────────────────────────────────────
API_URL      = "http://localhost:8000/v1/chat/completions"
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_FILE  = os.path.join(BASE_DIR, "data/benchmark_result.json")

# 压测用的测试问题，覆盖长短不同的问题
TEST_PROMPTS = [
    "小米SU7的最大续航里程是多少？",
    "如何开启离车后自动上锁功能？",
    "小米SU7支持哪些充电方式？",
    "座椅加热和座椅通风可以同时开启吗？",
    "自动呼叫功能触发后如何取消？",
    "小米SU7的电池容量是多少？支持快充吗？详细介绍一下充电相关的功能。",
    "介绍一下小米SU7的智能驾驶辅助功能，包括哪些传感器和摄像头配置？",
    "如何设置手机钥匙？需要什么前提条件？使用过程中有哪些注意事项？",
    "小米SU7的保养周期是怎么规定的？首保和常规保养分别是多少公里？",
    "弹射模式是什么？如何开启？有哪些使用限制和注意事项？",
]


# ── 工具函数 ─────────────────────────────────────────────────
def build_payload(prompt: str, stream: bool, model_name: str) -> dict:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.001,
        "stream": stream,
        "extra_body": {
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


async def measure_ttft(
    session: aiohttp.ClientSession,
    prompt: str,
    model_name: str,
) -> Tuple[float, float]:
    """
    测量单条请求的首字延迟（TTFT）和端到端延迟。
    返回 (ttft_ms, total_ms)
    """
    payload = build_payload(prompt, stream=True, model_name=model_name)
    start = time.perf_counter()
    ttft = None

    async with session.post(API_URL, json=payload) as resp:
        async for line in resp.content:
            line = line.decode("utf-8").strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content and ttft is None:
                    ttft = (time.perf_counter() - start) * 1000  # ms

    total = (time.perf_counter() - start) * 1000
    return ttft or total, total


async def measure_throughput(
    prompts: List[str],
    model_name: str,
    concurrency: int,
) -> Tuple[float, int]:
    """
    并发压测，返回 (token/s, total_tokens)
    """
    total_tokens = 0

    async def send_one(session, prompt):
        nonlocal total_tokens
        payload = build_payload(prompt, stream=False, model_name=model_name)
        try:
            async with session.post(API_URL, json=payload) as resp:
                result = await resp.json()
                tokens = result["usage"]["completion_tokens"]
                total_tokens += tokens
        except Exception as e:
            print(f"[WARN] 请求失败: {e}")

    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout   = aiohttp.ClientTimeout(total=120)
    start     = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [send_one(session, p) for p in prompts]
        for future in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="压测中"
        ):
            await future

    elapsed   = time.perf_counter() - start
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    return throughput, total_tokens


async def run_ttft_test(model_name: str, n_repeats: int = 3):
    """TTFT 测试：每个问题重复多次取均值"""
    print(f"\n[INFO] 开始 TTFT 测试（每题重复 {n_repeats} 次）...")
    ttft_list = []

    connector = aiohttp.TCPConnector(limit=1)
    timeout   = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for prompt in tqdm(TEST_PROMPTS, desc="TTFT测试"):
            for _ in range(n_repeats):
                ttft, _ = await measure_ttft(session, prompt, model_name)
                ttft_list.append(ttft)
                await asyncio.sleep(0.2)

    avg_ttft = np.mean(ttft_list)
    p50_ttft = np.percentile(ttft_list, 50)
    p95_ttft = np.percentile(ttft_list, 95)

    print(f"  平均 TTFT：{avg_ttft:.0f} ms")
    print(f"  P50  TTFT：{p50_ttft:.0f} ms")
    print(f"  P95  TTFT：{p95_ttft:.0f} ms")

    return {"avg_ms": avg_ttft, "p50_ms": p50_ttft, "p95_ms": p95_ttft, "raw": ttft_list}


async def run_throughput_test(model_name: str, concurrency: int = 8, n_requests: int = 100):
    """吞吐率测试：并发发送请求"""
    print(f"\n[INFO] 开始吞吐率测试（并发={concurrency}，请求数={n_requests}）...")

    # 重复 prompts 到 n_requests 条
    prompts = (TEST_PROMPTS * (n_requests // len(TEST_PROMPTS) + 1))[:n_requests]
    throughput, total_tokens = await measure_throughput(prompts, model_name, concurrency)

    print(f"  吞吐率：{throughput:.0f} token/s")
    print(f"  总 token 数：{total_tokens}")

    return {"token_per_sec": throughput, "total_tokens": total_tokens, "concurrency": concurrency}


def get_model_name() -> str:
    """从 vLLM 服务获取当前加载的模型名"""
    import requests
    try:
        resp = requests.get("http://localhost:8000/v1/models", timeout=5)
        models = resp.json()["data"]
        return models[0]["id"]
    except Exception:
        return "unknown"


async def main_async(args):
    model_name = get_model_name()
    print(f"[INFO] 当前模型：{model_name}")

    if args.compare:
        # 对比模式：需要先测 merged 模型，再测 int4 模型
        # 这里假设两个模型已分别部署，用 --model-a / --model-b 指定
        print("[WARN] 对比模式需要分别启动两个 vLLM 服务，当前只测一个")
        print("[INFO] 建议：先跑一次保存结果，换模型后再跑一次比较")

    # TTFT 测试
    ttft_result = await run_ttft_test(model_name, n_repeats=args.ttft_repeats)

    # 吞吐率测试
    tp_result = await run_throughput_test(
        model_name,
        concurrency=args.concurrency,
        n_requests=args.n_requests,
    )

    # 汇总结果
    result = {
        "model":        model_name,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "ttft":         ttft_result,
        "throughput":   tp_result,
    }

    # 如果已有历史结果，计算对比
    existing = []
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, "r") as f:
            existing = json.load(f)
        # 确保 existing 是 list
        if not isinstance(existing, list):
            existing = [existing]

    print("\n" + "=" * 60)
    print("📊 性能测试结果")
    print("=" * 60)
    print(f"模型：{model_name}")
    print(f"TTFT 均值：{ttft_result['avg_ms']:.0f} ms")
    print(f"TTFT P95：{ttft_result['p95_ms']:.0f} ms")
    print(f"吞吐率：{tp_result['token_per_sec']:.0f} token/s")

    # 对比上次结果（取列表最后一条）
    if existing and len(existing) > 0:
        last_result = existing[-1]
        if "ttft" in last_result:
            prev_ttft = last_result["ttft"]["avg_ms"]
            curr_ttft = ttft_result["avg_ms"]
            ttft_improve = (prev_ttft - curr_ttft) / prev_ttft * 100
            print(f"\n对比上次结果（{last_result.get('model', '上次')}）：")
            print(f"  TTFT：{prev_ttft:.0f} ms → {curr_ttft:.0f} ms  ({ttft_improve:+.1f}%)")

            prev_tp = last_result["throughput"]["token_per_sec"]
            curr_tp = tp_result["token_per_sec"]
            tp_improve = (curr_tp - prev_tp) / prev_tp * 100
            print(f"  吞吐率：{prev_tp:.0f} → {curr_tp:.0f} token/s  ({tp_improve:+.1f}%)")

    print("=" * 60)

    # 追加保存（保留历史）
    existing.append(result)
    with open(RESULT_FILE, "w") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 结果已保存：{RESULT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="vLLM 性能压测")
    parser.add_argument("--concurrency",   type=int, default=8,  help="并发数（吞吐率测试）")
    parser.add_argument("--n-requests",    type=int, default=100, help="总请求数（吞吐率测试）")
    parser.add_argument("--ttft-repeats",  type=int, default=3,  help="TTFT 每题重复次数")
    parser.add_argument("--compare",       action="store_true",   help="对比模式")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()