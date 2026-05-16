import os
import json
from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
import pandas as pd

load_dotenv()


class DoubaoLangChainLLM(LLM):
    model: str = "doubao-1-5-lite-32k-250115"
    api_key: str = ""
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        super().__init__()
        if model is not None:
            self.model = model
        if api_key is not None:
            self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url
        self._client = None
        self._async_client = None

    @property
    def _client_instance(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    @property
    def _async_client_instance(self):
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._async_client

    @property
    def _llm_type(self) -> str:
        return "doubao"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = self._client_instance.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01,
            )
            content = response.choices[0].message.content
            
            try:
                json_output = json.loads(content)
                if "classifications" in json_output:
                    result_lines = []
                    for item in json_output["classifications"]:
                        statement = item.get("statement", "").strip()
                        attributed = item.get("attributed", 0)
                        if not statement or statement in ["无答案", "没有答案", "无", "-"]:
                            continue
                        if attributed:
                            result_lines.append(f"SUPPORTED {statement}")
                        else:
                            result_lines.append(f"NOT_SUPPORTED {statement}")
                    if result_lines:
                        return "\n".join(result_lines)
                    else:
                        return "NOT_SUPPORTED No relevant information found in context"
                else:
                    content = str(content).strip()
                    if not content or content in ["无答案", "没有答案", "无", "-"]:
                        return "NOT_SUPPORTED No relevant information found in context"
                    return content
            except json.JSONDecodeError:
                content = str(content).strip()
                if not content or content in ["无答案", "没有答案", "无", "-"]:
                    return "NOT_SUPPORTED No relevant information found in context"
                return content
        except Exception as e:
            print(f"[ERROR] 同步 API 调用失败: {e}")
            return "NOT_SUPPORTED No relevant information found in context"

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = await self._async_client_instance.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01,
            )
            content = response.choices[0].message.content
            
            try:
                json_output = json.loads(content)
                if "classifications" in json_output:
                    result_lines = []
                    for item in json_output["classifications"]:
                        statement = item.get("statement", "").strip()
                        attributed = item.get("attributed", 0)
                        if not statement or statement in ["无答案", "没有答案", "无", "-"]:
                            continue
                        if attributed:
                            result_lines.append(f"SUPPORTED {statement}")
                        else:
                            result_lines.append(f"NOT_SUPPORTED {statement}")
                    if result_lines:
                        return "\n".join(result_lines)
                    else:
                        return "NOT_SUPPORTED No relevant information found in context"
                else:
                    content = str(content).strip()
                    if not content or content in ["无答案", "没有答案", "无", "-"]:
                        return "NOT_SUPPORTED No relevant information found in context"
                    return content
            except json.JSONDecodeError:
                content = str(content).strip()
                if not content or content in ["无答案", "没有答案", "无", "-"]:
                    return "NOT_SUPPORTED No relevant information found in context"
                return content
        except Exception as e:
            print(f"[ERROR] 异步 API 调用失败: {e}")
            return "NOT_SUPPORTED No relevant information found in context"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
        }

    def set_run_config(self, run_config):
        pass


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
    
    print(f"[INFO] 初始化豆包 LLM...")
    llm = DoubaoLangChainLLM(
        model=model_name,
        api_key=api_key,
        base_url=base_url
    )
    print(f"[INFO] 评估模型: 豆包 API ({llm.model})")

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
        LLMContextRecall(llm=llm),
        LLMContextPrecisionWithReference(llm=llm)
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