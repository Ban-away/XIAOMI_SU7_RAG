"""对LLM服务做极限压测的脚本
"""

import aiohttp
import asyncio
import json
import logging
import time
from typing import List, Tuple
import numpy as np


logger = logging.getLogger(__name__)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

API_URL = 'http://localhost:8000/v1/chat/completions'

HEADERS = {
    'Content-Type': 'application/json',
}


async def send_request(session, payload, prompt_len):
    request_start_time = time.time()
    async with session.post(API_URL, data=payload, headers=HEADERS) as response:
        if response.status == 200:
            result = await response.json()
            completion_tokens = len(result['choices'][0]['message']["content"])
            request_end_time = time.time()
            request_latency = request_end_time - request_start_time
            REQUEST_LATENCY.append((prompt_len, completion_tokens, request_latency))
            return result
        else:
            return {'error': response.status, 'message': await response.text()}


class BenchMarkRunner:

    def __init__(
        self,
        requests: List[Tuple[str, int, int]],  # prompt, prompt_len, completion_len
        concurrency: int,
    ):
        self.concurrency = concurrency
        self.requests = requests
        self.request_left = len(requests)
        self.request_queue = asyncio.Queue(concurrency or 100)

    async def run(self):
        tasks = []
        for i in range(self.concurrency):
            tasks.append(asyncio.create_task(self.worker()))
        for req in self.requests:
            await self.request_queue.put(req)
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    async def worker(self):
        timeout = aiohttp.ClientTimeout(total=5 * 60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while self.request_left > 0:
                prompt = await self.request_queue.get()
                payload = json.dumps({
                                      "messages": [{"role": "user", "content": prompt}],
                                      "temperature": 0.001,
                                      "max_tokens": 4096,
                                      "top_k": 1,
                                      "frequency_penalty": 2.0,
                                      "chat_template_kwargs": {"enable_thinking": False},
                                      })

                response = await send_request(session, payload, len(prompt))
                self.request_left -= 1
                print(f"Response {len(self.requests) - self.request_left}")


def main():
    concurrency = 1#50 # 并发数
    logger.info("Preparing for benchmark.")
    testset = json.load(open("./data/summary_test.json"))
    input_requests = [item["instruction"] for item in testset] 

    logger.info("Benchmark starts.")
    benchmark_start_time = time.time()
    asyncio.run(BenchMarkRunner(input_requests, concurrency).run())
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"Total time: {benchmark_time:.4f} s")
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.4f} s")
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in REQUEST_LATENCY
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency:.4f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    print("Average latency per output token: " f"{avg_per_output_token_latency:.4f} s")
    throughput = (
            sum([output_len for _, output_len, _ in REQUEST_LATENCY]) / benchmark_time
    )
    print(f"Throughput: {throughput} tokens/s")


if __name__ == '__main__':
    main()
