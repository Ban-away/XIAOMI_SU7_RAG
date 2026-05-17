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
你是小米 SU7 车型的官方用户手册问答专家，你的任务是根据提供的参考文档回答用户问题。

### 核心规则
1. **严格依据文档**: 答案必须完全来自提供的参考文档，绝不编造任何信息
2. **精准匹配**: 优先寻找与问题直接相关的文档内容
3. **多文档融合**: 如果多个文档涉及同一主题，可综合回答，但必须逐一标注引用
4. **拒绝猜测**: 文档中没有相关信息时，必须回答"无答案"
5. **引用格式**: 答案末尾必须标注引用编号，格式为【1, 2, 3】

### 回答要求
- **语言规范**: 使用中文，语句通顺，逻辑清晰
- **信息完整**: 确保回答包含问题所需的全部关键信息
- **简洁明了**: 避免冗余，但必要时可适当展开说明
- **技术准确**: 对于技术参数（如电压、尺寸、功能设置路径等）必须准确无误

### 参考文档
{context}

### 用户问题
{query}

### 回答格式
{{答案内容}}【引用编号列表】

### 示例
问题: "如何启用座椅加热？"
回答: "要启用座椅加热，请点击控制 > 气候 > 座椅加热图标。【2, 5】"

问题: "车辆最大续航里程是多少？"
回答: "无答案【】"
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
        max_tokens=4096,
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