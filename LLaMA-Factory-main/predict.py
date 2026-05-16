import os
import json

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference

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
        print(f"[INFO] 原始数据: {len(test_data)} 条")
        print("[DEBUG] 第一条数据字段:", list(test_data[0].keys()))
        print("[DEBUG] 第一条数据示例:", json.dumps(test_data[0], ensure_ascii=False, indent=2)[:500])
    except Exception as e:
        print(f"[ERROR] 加载数据集失败: {e}")
        raise

    print("[INFO] 准备评估数据格式...")
    # 过滤无答案条目：
    # 1. RAGas 无法对空答案/无答案做有效分类，会持续报 "did not return a valid classification"
    # 2. 过滤后评估结果更准确，不会被空值拉低得分
    NO_ANSWER_SET = {"无答案", "没有答案", "无", "-", ""}
    ragas_data = []
    skip_count = 0
    for item in test_data:
        answer    = item.get("answer", "").strip()
        reference = item.get("ground_truth", "").strip()
        context   = item.get("context", "").strip()

        if not answer or not reference or not context:
            skip_count += 1
            continue
        if answer in NO_ANSWER_SET or reference in NO_ANSWER_SET:
            skip_count += 1
            continue

        # RAGas EvaluationDataset 要求字段名为 response，不是 answer
        ragas_data.append({
            "user_input":         item.get("question", ""),
            "retrieved_contexts": [context],
            "response":           answer,
            "reference":          reference,
        })

    print(f"[INFO] 过滤后有效数据: {len(ragas_data)} 条，跳过无答案: {skip_count} 条")

    # 使用 EvaluationDataset（RAGas 新版推荐，替代旧版 HuggingFace Dataset）
    dataset = EvaluationDataset.from_list(ragas_data)

    print("[INFO] 初始化评估指标...")
    metrics = [
        LLMContextRecall(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
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
