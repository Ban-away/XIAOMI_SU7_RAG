# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------


import os
import json
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from openai import OpenAI
from tqdm import tqdm

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

import os

test_data_path = os.path.join(os.getcwd(), "data", "summary_test.json")
test_data = json.load(open(test_data_path))
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

pred_data_path = os.path.join(os.getcwd(), "data", "summary_test_pred.json")
with open(pred_data_path, "w") as fd:
    fd.write(json.dumps(test_data, ensure_ascii=False, indent=4))

test_data = json.load(open(pred_data_path))


"""
以下是RAG评估代码的扩展，利用Ragas框架来对问答系统输出的结果做评估。输入是query，生成的答案，参考答案，以及召回的上下文信息。
评估采用了精确率和召回率两个指标
"""

llm = ChatOpenAI(model=os.environ["DOUBAO_MODEL_NAME"], api_key=os.environ["DOUBAO_API_KEY"], base_url=os.environ["DOUBAO_BASE_URL"])

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