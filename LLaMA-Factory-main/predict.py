import os
import json
import logging
from typing import Any, List, Mapping, Optional

from langchain_core.language_models.llms import LLM
from openai import OpenAI, AsyncOpenAI
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from tqdm import tqdm


try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[INFO] 已从 .env 文件加载环境变量")
except ImportError:
    print("[WARNING] 未安装 python-dotenv，将使用系统环境变量")


class DoubaoLangChainLLM(LLM):
    """使用 LangChain LLM 基类包装豆包 API，确保与 Ragas 兼容"""
    
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
    def _client_instance(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client
    
    @property
    def _async_client_instance(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
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
        """同步调用豆包 API"""
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
                    classification_str = "\n".join([
                        f"{i+1}. Statement: {item.get('statement', '')}"
                        for i, item in enumerate(json_output["classifications"])
                    ])
                    return classification_str
                else:
                    return content
            except json.JSONDecodeError:
                return content
        except Exception as e:
            print(f"[ERROR] 同步 API 调用失败: {e}")
            return ""
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """异步调用豆包 API"""
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
                    classification_str = "\n".join([
                        f"{i+1}. Statement: {item.get('statement', '')}"
                        for i, item in enumerate(json_output["classifications"])
                    ])
                    return classification_str
                else:
                    return content
            except json.JSONDecodeError:
                return content
        except Exception as e:
            print(f"[ERROR] 异步 API 调用失败: {e}")
            return ""
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
        }


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


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


if os.path.exists(pred_data_path):
    print(f"\n[INFO] 检测到已存在预测文件: {pred_data_path}")
    print("[INFO] 将跳过预测阶段，直接加载已有预测结果进行评估")
    with open(pred_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
else:
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("\n[INFO] ========== 开始预测任务 ==========")
    print(f"[INFO] 预测模型: 本地 vLLM 服务 (qwen3_lora_sft_int4)")
    print(f"[INFO] 服务地址: {openai_api_base}")
    print(f"[INFO] 预测数据: {len(test_data)} 条")

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


print("\n[INFO] ========== 开始 RAG 评估 ==========")
print(f"[INFO] 评估模型: 豆包 API ({doubao_config['model']})")
print(f"[INFO] 评估数据: {len(test_data)} 条")
print("[INFO] 评估指标: LLMContextRecall, LLMContextPrecisionWithReference")


evaluator_llm = DoubaoLangChainLLM(
    model=doubao_config["model"],
    api_key=doubao_config["api_key"],
    base_url=doubao_config["base_url"]
)

dataset = []
for g in test_data:
    query = g.get("query", g.get("instruction", ""))
    reference = g.get("output", "")
    response = g.get("response", "")
    context = [g.get("context", "")]
    
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": context,
            "response": response,
            "reference": reference
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)


logging.getLogger("ragas").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("ragas.evaluation").setLevel(logging.ERROR)
logging.getLogger("ragas.metrics").setLevel(logging.ERROR)
logging.getLogger("ragas.executor").setLevel(logging.ERROR)


os.environ["RAGAS_VERBOSE"] = "false"
os.environ["RAGAS_DEBUG"] = "false"


result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()],
    llm=evaluator_llm,
    show_progress=True,
    raise_exceptions=False
)
print("评估结果：", result)