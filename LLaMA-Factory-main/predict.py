import os
import json
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from openai import OpenAI
from tqdm import tqdm

# 尝试从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[INFO] 已从 .env 文件加载环境变量")
except ImportError:
    print("[WARNING] 未安装 python-dotenv，将使用系统环境变量")

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

# 使用豆包 API 进行评估
llm = ChatOpenAI(
    model=doubao_config["model"], 
    api_key=doubao_config["api_key"], 
    base_url=doubao_config["base_url"]
)

dataset = []
for g in test_data:
    query = g["query"] # 输入问题
    reference = g["output"] # 参考答案
    response = g["response"] #生成的答案
    context = [g["context"]] # 上下文
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts": context,
            "response":response,
            "reference":reference
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()],llm=evaluator_llm)
print("评估结果：", result)