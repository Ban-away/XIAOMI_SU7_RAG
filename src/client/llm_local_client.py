# -*- coding: utf-8 -*-

import os
import json
import re
from openai import OpenAI
from langchain_core.documents import Document
from src.constant import qwen3_8b_tune_model_name


# 本地推理提示词模板（对接 vLLM 的 OpenAI 兼容接口）
LLM_CHAT_PROMPT = """
### 角色定位
你是小米 SU7 车型的官方用户手册问答专家，根据参考文档精准回答用户问题。

### 核心规则
1. **必须尝试回答**：只要参考文档中有任何相关信息，就必须给出答案，哪怕只是部分相关
2. **简洁优先**：直接回答问题核心，不要扩展无关内容，不要补充文档没有的信息
3. **严格限制"无答案"**：只有当所有参考文档与问题完全无关时，才能回答"无答案"
4. **引用格式**：答案末尾标注引用编号【1, 2, 3】

### 特别注意
- 问题问"是什么/有什么/怎样" → 从文档中找对应描述，直接引用原文关键信息
- 问题问"怎么做/如何操作" → 给出操作步骤，来自文档
- 答案要与问题严格对应，不要答非所问

### 参考文档
{context}

### 用户问题
{query}

### 回答格式
直接给出答案内容，末尾加【引用编号】
如果所有文档均与问题完全无关，则输出：无答案【】
"""


# 本地 vLLM 服务通常无需真实 API Key（默认可用占位值）。
# 只要 base_url 指向 vLLM 的 /v1 入口即可。
llm_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)


def request_chat(query, context, stream=False):
    """
    调用本地 vLLM 生成答案。

    参数：
    - query: 用户问题
    - context: 检索上下文
    - stream: 是否流式返回（True 返回迭代器；False 返回字符串）
    """
    # 统一用模板组织上下文与问题
    prompt = LLM_CHAT_PROMPT.format(context=context, query=query) 

    # 发起本地聊天生成请求
    completion = llm_client.chat.completions.create(
        # 本地模型名配置在 constant.py，便于脚本间复用
        model=qwen3_8b_tune_model_name,
        messages=[
            {"role": "system", "content": "你是一个有用的人工智能助手."},
            {"role": "user", "content": prompt}
        ],
        # 生成参数：尽量降低随机性，提高可复现性
        # max_tokens 需要考虑上下文长度，模型最大长度为 8192
        max_tokens=2048,
        frequency_penalty=2.0,
        temperature=0.001,
        top_p=0.95,
        # 是否启用流式返回
        stream=stream,
        # vLLM 扩展参数：限制采样并关闭思维链显示
        extra_body={
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    # 非流式：直接取文本；流式：把迭代器交给上层消费
    if not stream:
        result = completion.choices[0].message.content
    else:
        result = completion

    # 返回字符串或迭代器
    return result



if __name__ == "__main__":

    context = """
    【1】### 离车后自动上锁
    带着手机钥匙或配对的遥控钥匙离开时，车门和行李箱可以自动锁定（如果订购日期是在大约 2019 年 10 月 1 日之后）。要打开或关闭此功能，可点击控制 > 车锁 > 离车后自动上锁。
    **注**：如果已将 Apple 手表认证为钥匙，也可以将该手表用于离车后自动上锁功能。
    【2】车门锁闭时，外部车灯闪烁一次，后视镜折叠（如果折叠后视镜开启）。要在小米 SU7 锁定时听到提示音，可点击控制 > 车锁 > 锁定提示音。
    【3】### 大灯延时照明
    停止驾驶并将小米 SU7 停在照明较差的环境中时，外部车灯会短暂亮起。它们会在一分钟后或您锁闭小米 SU7 时（以较早者为准）自动关闭。当您使用小米汽车 App 锁定小米 SU7 时，大灯将立即熄灭。但是，如果车辆因启用了“离车后自动上锁”功能而锁定（请参阅离车后自动上锁 页码 7），则大灯将在一分钟后自动熄灭。要打开或关闭此功能，请点击控制 > 车灯 > 大灯延时照明。关闭大灯延时照明后，当换入驻车挡并打开车门时，大灯会立即熄灭。"""

    query = "介绍一下离车后自动上锁功能"

    res = request_chat(query, context, stream=True)
    for r in res:
        uttr = r.choices[0].delta.content
        # 处理流式响应中的 None 值
        if uttr is not None:
            print(uttr, end='')
    print()