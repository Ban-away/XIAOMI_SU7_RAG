import os
import json

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
import pandas as pd

load_dotenv()


def main():
    print("[INFO] 已从 .env 文件加载环境变量")
    print()

    print("[INFO] 检查豆包 API 配置...")
    api_key = os.getenv("DOUBAO_API_KEY")
    model_name = os.getenv("DOUBAO_MODEL_NAME", "doubao-1-5-lite-32k-250115")
    base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

    if not api_key:
        raise ValueError("请设置 DOUBAO_API_KEY 环境变量")

    print("[INFO] ✅ 豆包 API 配置检查通过")
    print(f"[INFO]   - DOUBAO_MODEL_NAME: {model_name}")
    print(f"[INFO]   - DOUBAO_API_KEY: {'*' * 48}")
    print(f"[INFO]   - DOUBAO_BASE_URL: `{base_url}`")
    print()

    data_file = "data/summary_test_pred.json"
    if os.path.exists(data_file):
        print(f"[INFO] 检测到已存在预测文件: {os.path.abspath(data_file)}")
        print("[INFO] 将跳过预测阶段，直接加载已有预测结果进行评估")
        print()

    print("[INFO] ========== 开始 RAG 评估 ==========")

    # 使用 ChatOpenAI + LangchainLLMWrapper，RAGas 官方推荐用法
    # 避免自定义 LLM 时 StringPromptValue 类型不兼容的问题
    print(f"[INFO] 初始化豆包 LLM...")
    chat_llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.01,
    )
    evaluator_llm = LangchainLLMWrapper(chat_llm)
    print(f"[INFO] 评估模型: 豆包 API ({model_name})")

    print("[INFO] 加载评估数据集...")
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        print(f"[INFO] 评估数据: {len(test_data)} 条")
    except Exception as e:
        print(f"[ERROR] 加载数据集失败: {e}")
        raise

    print("[INFO] 准备评估数据格式...")
    ragas_data = []
    for item in test_data:
        ragas_data.append({
            "user_input": item.get("question", ""),
            "retrieved_contexts": [item.get("context", "")],
            "answer": item.get("answer", ""),
            "reference": item.get("ground_truth", "")
        })

    df = pd.DataFrame(ragas_data)
    dataset = Dataset.from_pandas(df)

    print("[INFO] 初始化评估指标...")
    metrics = [
        LLMContextRecall(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm)
    ]
    print(f"[INFO] 评估指标: {', '.join([m.__class__.__name__ for m in metrics])}")

    print("[INFO] 开始评估...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )

        print("[INFO] 评估完成!")
        print(result)

        print("[INFO] 保存评估结果...")
        with open("data/ragas_evaluation_result.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print("[INFO] 结果已保存到 data/ragas_evaluation_result.json")

    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")
        raise


if __name__ == "__main__":
    main()
