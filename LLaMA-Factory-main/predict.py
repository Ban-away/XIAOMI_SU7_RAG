import os
import json
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas import evaluate
from ragas import EvaluationDataset
from openai import OpenAI
from tqdm import tqdm
from typing import Any, List, Optional, Dict, Union

# 尝试从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[INFO] 已从 .env 文件加载环境变量")
except ImportError:
    print("[WARNING] 未安装 python-dotenv，将使用系统环境变量")


class DoubaoRagasWrapper:
    """自定义 LLM wrapper，适配豆包 API 的输出格式"""
    
    def __init__(self, model: str, api_key: str, base_url: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.run_config = None
    
    def set_run_config(self, config):
        """设置运行配置（Ragas 要求的方法）"""
        self.run_config = config
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """生成响应，将豆包 API 的 JSON 输出转换为 Ragas 期望的格式"""
        results = []
        for prompt in prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.01,
                )
                content = response.choices[0].message.content
                
                # 尝试解析豆包 API 返回的 JSON 格式
                try:
                    json_output = json.loads(content)
                    if "classifications" in json_output:
                        # 将分类结果转换为 Ragas 期望的格式
                        classification_str = "\n".join([
                            f"{i+1}. Statement: {item.get('statement', '')}"
                            for i, item in enumerate(json_output["classifications"])
                        ])
                        results.append(classification_str)
                    else:
                        results.append(content)
                except json.JSONDecodeError:
                    # 如果不是 JSON 格式，直接返回原始内容
                    results.append(content)
            except Exception as e:
                print(f"[ERROR] API 调用失败: {e}")
                results.append("")
        return results
    
    async def agenerate(self, prompts: List[str], **kwargs) -> List[str]:
        """异步生成响应"""
        return self.generate(prompts, **kwargs)
    
    @property
    def llm(self):
        return self


# 本地 vLLM 服务配置
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

# 检查豆包 API 环境变量
print("\n[INFO] 检查豆包 API 配置...")
doubao_config = {
    "model": os.environ.get("DOUBAO_MODEL_NAME"),
    "api_key": os.environ.get("DOUBAO_API_KEY"),
    "base_url": os.environ.get("DOUBAO_BASE_URL")
}

missing_config = [key for key, value in doubao_config.items() if not value]
if missing_config:
    raise EnvironmentError(
        f"[ERROR] 缺少必需的豆包 API 配置: {', '.join(missing_config)}\n"
        f"请在 .env 文件中配置或设置环境变量"
    )

print("[INFO] ✅ 豆包 API 配置检查通过")
print(f"[INFO]   - DOUBAO_MODEL_NAME: {doubao_config['model']}")
print(f"[INFO]   - DOUBAO_API_KEY: {'*' * len(doubao_config['api_key'])}")
print(f"[INFO]   - DOUBAO_BASE_URL: {doubao_config['base_url']}")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

test_data_path = os.path.join(os.getcwd(), "data", "summary_test.json")
pred_data_path = os.path.join(os.getcwd(), "data", "summary_test_pred.json")

# 检查预测文件是否已存在
if os.path.exists(pred_data_path):
    print(f"\n[INFO] 检测到已存在预测文件: {pred_data_path}")
    print("[INFO] 将跳过预测阶段，直接加载已有预测结果进行评估")
    test_data = json.load(open(pred_data_path))
else:
    test_data = json.load(open(test_data_path))

    print("\n[INFO] ========== 开始预测任务 ==========")
    print(f"[INFO] 预测模型: 本地 vLLM 服务 (qwen3_lora_sft_int4)")
    print(f"[INFO] 服务地址: {openai_api_base}")
    print(f"[INFO] 预测数据: {len(test_data)} 条")

    for info in tqdm(test_data):
        # 构建模型的完整路径 - 在 LLaMA-Factory-main 目录下
        model_path = os.path.join(os.getcwd(), "output", "qwen3_lora_sft_int4")
        model_path = os.path.abspath(model_path)
        
        chat_response = client.chat.completions.create(
            model=model_path,
            messages=[
                {
                    "role": "user",
                    "content": info["instruction"]
                }
            ],
            max_tokens=4096,
            frequency_penalty=2.0,
            temperature=0.001,
            top_p=0.95,
            extra_body={
                "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        info["response"] = chat_response.choices[0].message.content

    with open(pred_data_path, "w") as fd:
        fd.write(json.dumps(test_data, ensure_ascii=False, indent=4))


"""
以下是RAG评估代码的扩展，利用Ragas框架来对问答系统输出的结果做评估。输入是query，生成的答案，参考答案，以及召回的上下文信息。
评估采用了精确率和召回率两个指标
"""

print("\n[INFO] ========== 开始 RAG 评估 ==========")
print(f"[INFO] 评估模型: 豆包 API ({doubao_config['model']})")
print(f"[INFO] 评估数据: {len(test_data)} 条")
print("[INFO] 评估指标: LLMContextRecall, LLMContextPrecisionWithReference")

# 创建自定义的豆包 API wrapper
evaluator_llm = DoubaoRagasWrapper(
    model=doubao_config["model"],
    api_key=doubao_config["api_key"],
    base_url=doubao_config["base_url"]
)

dataset = []
for g in test_data:
    # 确保字段名一致性，使用预测部分对应的字段
    query = g.get("query", g.get("instruction", ""))  # 尝试获取query或instruction作为输入问题
    reference = g.get("output", "")  # 参考答案
    response = g.get("response", "")  # 生成的答案
    context = [g.get("context", "")]  # 上下文，如果不存在则使用空字符串
    
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": context,
            "response": response,
            "reference": reference
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)

import logging
# 设置日志级别以减少数据样本输出，但保留进度条
logging.getLogger("ragas").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ragas.evaluation").setLevel(logging.ERROR)

# 执行评估，只显示进度条，不打印数据样本
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()],
    llm=evaluator_llm,
    show_progress=True,
    raise_exceptions=False
)
print("评估结果：", result)