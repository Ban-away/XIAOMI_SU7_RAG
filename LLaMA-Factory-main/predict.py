import os
import json
from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from datasets import load_dataset
from ragas import evaluate
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference

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
        """同步调用豆包 API - 将 JSON 格式转换为 Ragas 期望的格式"""
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
        """异步调用豆包 API - 将 JSON 格式转换为 Ragas 期望的格式"""
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


def main():
    print("[INFO] 加载环境变量...")
    api_key = os.getenv("DOUBAO_API_KEY")
    if not api_key:
        raise ValueError("请设置 DOUBAO_API_KEY 环境变量")

    print(f"[INFO] 初始化豆包 LLM...")
    llm = DoubaoLangChainLLM(
        model="doubao-1-5-lite-32k-250115",
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )
    print(f"[INFO] 评估模型: 豆包 API ({llm.model})")

    print("[INFO] 加载评估数据集...")
    data_file = "data/summary_test_pred.json"
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        print(f"[INFO] 成功加载数据集: {data_file}")
        print(f"[INFO] 评估数据: {len(test_data)} 条")
    except Exception as e:
        print(f"[ERROR] 加载数据集失败: {e}")
        raise

    print("[INFO] 准备评估数据格式...")
    dataset_dict = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": []
    }
    
    for item in test_data:
        dataset_dict["question"].append(item.get("question", ""))
        dataset_dict["contexts"].append([item.get("context", "")])
        dataset_dict["answer"].append(item.get("answer", ""))
        dataset_dict["ground_truth"].append([item.get("ground_truth", "")])
    
    dataset = load_dataset("json", data_files={"train": data_file})["train"]
    
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