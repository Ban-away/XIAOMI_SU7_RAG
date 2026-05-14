# -*- coding: utf-8 -*-

import os
import json
import re
from openai import OpenAI
from langchain_core.documents import Document


# 这个提示词用于“云端 LLM”场景。
# 其中 {context} 会在运行时替换为检索到的文档片段，{query} 替换为用户问题。
LLM_CHAT_PROMPT = """
### 信息
{context}

### 任务
你是小米 SU7 车型的用户手册问答系统，你具备{{信息}}中的知识。
请回答问题"{query}"，答案需要精准，语句通顺，并严格按照以下格式输出

{{答案}}【{{引用编号1}}, {{引用编号2}}, ...】
如果无法从中得到答案，请说 "无答案" ，不允许在答案中添加编造成分。
"""


# 初始化 OpenAI 兼容客户端（这里对接豆包或其他兼容服务）。
# 认证信息从环境变量读取，便于不同部署环境切换。
def _validate_doubao_config():
    """验证豆包 API 配置"""
    required_keys = ['DOUBAO_API_KEY', 'DOUBAO_BASE_URL', 'DOUBAO_MODEL_NAME']
    missing_keys = [k for k in required_keys if not os.environ.get(k)]
    if missing_keys:
        raise RuntimeError(f"缺失必需环境变量: {', '.join(missing_keys)}\n请设置: {', '.join([f'export {k}=value' for k in missing_keys])}")

_validate_doubao_config()
llm_client = OpenAI(
    api_key=os.environ['DOUBAO_API_KEY'],
    base_url=os.environ['DOUBAO_BASE_URL']
)


def request_chat(query, context):
    """
    调用云端聊天模型生成答案。

    参数：
    - query: 用户问题
    - context: 检索和重排后的上下文文本
    """
    # 把检索上下文与用户问题填入统一模板，形成最终输入提示词
    prompt = LLM_CHAT_PROMPT.format(context=context, query=query) 

    # 发起 chat.completions 请求
    completion = llm_client.chat.completions.create(
        # 模型名由环境变量控制，部署时可切换
        model=os.environ["DOUBAO_MODEL_NAME"],
        # 这里用单轮 user message，系统约束已放在 prompt 内
        messages=[
            {"role": "user", "content": prompt}
        ],
        # 输出上限，防止返回过长内容
        max_tokens=4096
    )
    # 读取首个候选回答正文
    result = completion.choices[0].message.content

    # 返回字符串答案
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

    res = request_chat(query, context)

    print(res)
