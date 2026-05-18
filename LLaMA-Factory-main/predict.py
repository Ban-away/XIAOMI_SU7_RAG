import os
import json

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas.run_config import RunConfig

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

    data_file = "data/summary_test_pred.json"
    abs_data_file = os.path.abspath(data_file)
    
    # 检查预测文件是否存在，如果不存在或不完整则先生成
    need_regenerate = True
    if os.path.exists(data_file):
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
                if test_data and "response" in test_data[0]:
                    need_regenerate = False
                    print(f"[INFO] 检测到已存在有效的预测文件: {abs_data_file}")
                    print("[INFO] 将跳过预测阶段，直接加载已有预测结果进行评估")
        except:
            pass
    
    if need_regenerate:
        print(f"[INFO] 开始生成模型预测...")
        
        # 手动加载模型进行推理
        from llamafactory.chat import ChatModel
        from llamafactory.hparams import get_infer_args
        
        model_args = get_infer_args()
        model_args.model_name_or_path = "output/qwen3_lora_sft_int4"
        model_args.template = "qwen3"
        
        chat_model = ChatModel(model_args)
        
        # 加载测试数据集
        import pandas as pd
        from datasets import load_dataset
        
        try:
            ds = load_dataset("json", data_files="data/summary_test.json")["train"]
        except:
            # 尝试从 data 目录查找
            import glob
            test_files = glob.glob(os.path.join(os.path.dirname(abs_data_file), "*summary*test*.json"))
            if test_files:
                ds = load_dataset("json", data_files=test_files[0])["train"]
            else:
                raise RuntimeError("未找到测试数据集")
        
        # 生成预测
        predictions = []
        for item in ds:
            query = item.get("query", item.get("instruction", ""))
            response = chat_model.chat(query)
            predictions.append({
                "query": query,
                "context": item.get("context", ""),
                "output": item.get("output", ""),
                "response": response,
            })
        
        # 保存预测结果
        os.makedirs(os.path.dirname(abs_data_file), exist_ok=True)
        with open(abs_data_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 预测生成完成，结果已保存到 {abs_data_file}")
        chat_model.close()  # 释放资源
    
    print()
    print("[INFO] ========== 开始 RAG 评估 ==========")

    # 使用 ChatOpenAI + LangchainLLMWrapper，RAGas 官方推荐用法
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
    # 检查数据字段
    if test_data:
        print(f"[DEBUG] 数据字段: {list(test_data[0].keys())}")
    
    # 过滤无答案条目：
    # 1. RAGas 无法对空答案/无答案做有效分类，会持续报 "did not return a valid classification"
    # 2. 过滤后评估结果更准确，不会被空值拉低得分
    NO_ANSWER_SET = {"无答案", "没有答案", "无", "-", ""}
    ragas_data = []
    skip_count = 0
    
    for item in test_data:
        # 支持多种字段名：response 或 prediction（模型预测答案）
        response  = item.get("response", item.get("prediction", "")).strip()
        # 支持多种字段名：output 或 reference（标准答案）
        reference = item.get("output", item.get("reference", "")).strip()
        context   = item.get("context", "").strip()
        query     = item.get("query", item.get("instruction", "")).strip()

        if not response or not reference or not context:
            skip_count += 1
            continue
        if response in NO_ANSWER_SET or reference in NO_ANSWER_SET:
            skip_count += 1
            continue

        ragas_data.append({
            "user_input":         query,
            "retrieved_contexts": [context],
            "response":           response,
            "reference":          reference,
        })

    print(f"[INFO] 过滤后有效数据: {len(ragas_data)} 条，跳过无答案: {skip_count} 条")
    
    # 检查是否有有效数据
    if not ragas_data:
        print("[ERROR] 没有找到有效数据！请检查数据格式")
        print("[DEBUG] 数据可能缺少 'response' 字段（模型预测答案）")
        print("[DEBUG] 第一条数据示例:", json.dumps(test_data[0], ensure_ascii=False, indent=2)[:500])
        raise RuntimeError("没有有效数据可评估")

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