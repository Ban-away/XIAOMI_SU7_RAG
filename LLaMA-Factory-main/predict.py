import os
import json

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas.run_config import RunConfig
from openai import OpenAI
from tqdm import tqdm

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
    print(f"[INFO]   - DOUBAO_BASE_URL: {base_url}")
    print()

    pred_data_path = "data/summary_test_pred.json"
    test_data_path = "data/summary_test.json"
    
    if os.path.exists(pred_data_path):
        print(f"\n[INFO] 检测到已存在预测文件: {pred_data_path}")
        print("[INFO] 将跳过预测阶段，直接加载已有预测结果进行评估")
        with open(pred_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    else:
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print("\n[INFO] ========== 开始预测任务 ==========")
        print(f"[INFO] 预测数据: {len(test_data)} 条")

        # 初始化 OpenAI 客户端连接本地 vLLM 服务
        openai_api_base = "http://localhost:8000/v1"
        client = OpenAI(
            base_url=openai_api_base,
            api_key="dummy_key",
        )

        print(f"[INFO] 预测模型: 本地 vLLM 服务 (qwen3_lora_sft_int4)")
        print(f"[INFO] 服务地址: {openai_api_base}")

        for info in tqdm(test_data):
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

        with open(pred_data_path, "w", encoding="utf-8") as fd:
            json.dump(test_data, fd, ensure_ascii=False, indent=4)

        print(f"\n[INFO] 预测完成，结果已保存到 {pred_data_path}")

    print()
    print("[INFO] ========== 开始 RAG 评估 ==========")

    print(f"[INFO] 初始化豆包 LLM...")
    chat_llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.01,
        model_kwargs={
            "extra_body": {
                "system": "You are a helpful assistant. Always respond in English with exact JSON format as instructed. Do not add extra fields."
            }
        }
    )
    evaluator_llm = LangchainLLMWrapper(chat_llm)
    print(f"[INFO] 评估模型: 豆包 API ({model_name})")

    print("[INFO] 加载评估数据集...")
    print(f"[INFO] 原始数据: {len(test_data)} 条")
    print("[DEBUG] 第一条数据字段:", list(test_data[0].keys()))
    print("[DEBUG] 第一条数据示例:", json.dumps(test_data[0], ensure_ascii=False, indent=2)[:500])

    print("[INFO] 准备评估数据格式...")
    NO_ANSWER_SET = {"无答案", "没有答案", "无", "-", ""}
    ragas_data = []
    skip_count = 0
    
    for item in test_data:
        response  = item.get("response", "").strip()
        reference = item.get("output", "").strip()
        context   = item.get("context", "").strip()

        if not response or not reference or not context:
            skip_count += 1
            continue
        if response in NO_ANSWER_SET or reference in NO_ANSWER_SET:
            skip_count += 1
            continue

        ragas_data.append({
            "user_input":         item.get("query", ""),
            "retrieved_contexts": [context],
            "response":           response,
            "reference":          reference,
        })

    print(f"[INFO] 过滤后有效数据: {len(ragas_data)} 条，跳过无答案: {skip_count} 条")

    if not ragas_data:
        print("[ERROR] 没有找到有效数据！")
        raise RuntimeError("没有有效数据可评估")

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
            run_config=RunConfig(
                timeout=100,   # 默认30秒，改成100秒
                max_retries=3, # 最大重试次数
                max_wait=50,   # 重试间隔最大等待50秒
            )
        )

        print("[INFO] 评估完成!")
        print(result)

        print("[INFO] 保存评估结果...")
        save_data = {
            "context_recall": result["context_recall"],
            "llm_context_precision_with_reference": result["llm_context_precision_with_reference"],
        }
        with open("data/ragas_evaluation_result.json", "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print("[INFO] 结果已保存到 data/ragas_evaluation_result.json")

    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")
        raise


if __name__ == "__main__":
    main()