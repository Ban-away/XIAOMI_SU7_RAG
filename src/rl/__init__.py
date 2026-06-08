# -*- coding: utf-8 -*-
"""
src.rl - RL 强化学习训练与推理模块

模块结构：
  data_builder.py     网络兜底轨迹生成器（数据构建）
  format_converter.py 训练数据格式转换器（SFT/GRPO/ShareGPT）
  reward_model.py     多维度奖励函数（格式+答案+工具+来源+领域）
  environment.py      工具调用路由环境（检索后端调度）
  train_grpo.py       GRPO 训练入口脚本（全流程编排）
  infer_rl.py         RL 增强推理（交互式问答）
"""
