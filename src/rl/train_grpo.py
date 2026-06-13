# -*- coding: utf-8 -*-
"""
GRPO 强化学习训练入口脚本

功能：
  1. 整合数据构建 → 格式转换 → GRPO 训练的全流程
  2. 支持 SFT warm-up → GRPO 两阶段训练
  3. 自动注册自定义奖励函数
  4. 训练后导出模型 & 量化

使用方式：
  # 完整流程：轨迹生成 + SFT warm-up + GRPO 训练 + 导出
  python src/rl/train_grpo.py --stage all

  # 只运行数据准备（轨迹生成 + 格式转换）
  python src/rl/train_grpo.py --stage data

  # 只运行 SFT warm-up
  python src/rl/train_grpo.py --stage sft

  # 只运行 GRPO 训练
  python src/rl/train_grpo.py --stage grpo

  # 只导出模型
  python src/rl/train_grpo.py --stage export

  # 指定 GPU
  CUDA_VISIBLE_DEVICES=0,1 python src/rl/train_grpo.py --stage grpo

依赖：
  - LLaMA-Factory (已安装在 LLaMA-Factory-main/)
  - 数据文件：data/rl_data/web_fallback_questions.json
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

# ── 项目路径 ────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
LLAMA_FACTORY_DIR = os.path.join(BASE_DIR, "LLaMA-Factory-main")
DATA_DIR          = os.path.join(BASE_DIR, "data/rl_data")
CONFIG_DIR        = os.path.join(BASE_DIR, "configs")

# ── 模型路径 ────────────────────────────────────────────────
SFT_ADAPTER_DIR   = os.path.join(LLAMA_FACTORY_DIR, "saves/qwen3-8b/lora/rl_sft")
GRPO_OUTPUT_DIR   = os.path.join(LLAMA_FACTORY_DIR, "saves/qwen3-8b/lora/grpo")
RL_EXPORT_DIR     = os.path.join(LLAMA_FACTORY_DIR, "output/qwen3_lora_rl")


# ────────────────────────────────────────────────────────────
# Stage 1: 数据准备（轨迹生成 + 格式转换）
# ────────────────────────────────────────────────────────────

def prepare_data():
    """
    准备 GRPO 训练数据（合并网络兜底 + 本地可答轨迹）。

    优先级：
      1. 已有合并数据 combined_*.jsonl → 直接使用
      2. 已有各子集数据 → 自动合并
      3. 缺少子集 → 自动生成
    """
    print("\n" + "=" * 60)
    print("📦 Stage 1: 数据准备")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 统一使用合并后的数据集 ────────────────────────────
    combined_grpo = os.path.join(DATA_DIR, "combined_trajectories_grpo.jsonl")
    combined_sft  = os.path.join(DATA_DIR, "combined_trajectories_sft.json")

    grpo_data_path = combined_grpo
    sft_data_path  = combined_sft
    raw_path       = os.path.join(DATA_DIR, "web_fallback_trajectories.json")

    # ── Level 1: 已有转换好的数据 ──────────────────────────
    if os.path.exists(grpo_data_path) and os.path.exists(sft_data_path):
        with open(grpo_data_path, encoding="utf-8") as f:
            grpo_count = sum(1 for _ in f)
        with open(sft_data_path, encoding="utf-8") as f:
            sft_data = json.load(f)
        print(f"  ✅ GRPO 数据已存在: {grpo_count} 条 ({grpo_data_path})")
        print(f"  ✅ SFT 数据已存在: {len(sft_data)} 条 ({sft_data_path})")
        return True

    # ── Level 2: 有原始轨迹，需要格式转换 ──────────────────
    if os.path.exists(raw_path):
        print(f"  🔄 发现原始轨迹数据，开始格式转换...")
        _convert_trajectories(raw_path)

    # ── Level 3: 什么都没有，自动生成轨迹 ──────────────────
    else:
        print("  🔄 未找到轨迹数据，自动运行 data_builder.py 生成...")
        if not _run_data_builder():
            return False
        if os.path.exists(raw_path):
            _convert_trajectories(raw_path)
        else:
            print("  ❌ 轨迹生成后仍未找到数据文件")
            return False

    # ── Level 4: 生成本地轨迹 + 合并为 combined 数据 ────────
    if not os.path.exists(combined_grpo) or not os.path.exists(combined_sft):
        print(f"  🔄 合并数据集不存在，运行 build_local_trajectories.py...")
        if not _run_build_local_trajectories():
            print("  ⚠️ 本地轨迹生成失败，将仅使用网络兜底数据")

    return True


def _run_data_builder() -> bool:
    """调用 data_builder.py 生成轨迹"""
    questions_path = os.path.join(DATA_DIR, "web_fallback_questions.json")
    if not os.path.exists(questions_path):
        print(f"  ❌ 问题库不存在: {questions_path}")
        return False

    cmd = [sys.executable, "src/rl/data_builder.py"]
    print(f"  🚀 执行: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=BASE_DIR)

    if result.returncode != 0:
        print(f"\n  ❌ data_builder.py 执行失败 (exit code: {result.returncode})")
        return False

    print(f"\n  ✅ 轨迹生成完成")
    return True


def _convert_trajectories(raw_path: str):
    """使用 format_converter 转换轨迹数据"""
    print("  🔄 使用 format_converter 转换数据...")
    from src.rl.format_converter import convert, save, load_trajectories
    data    = load_trajectories(raw_path)
    results = convert(data, target_format="all")
    save(results, output_dir=DATA_DIR)


def _run_build_local_trajectories() -> bool:
    """调用 build_local_trajectories.py 生成本地轨迹 + 合并数据"""
    cmd = [sys.executable, "src/rl/build_local_trajectories.py"]
    print(f"  🚀 执行: {' '.join(cmd)}（默认加载全部 QA 数据）")
    print()

    result = subprocess.run(cmd, cwd=BASE_DIR)

    if result.returncode != 0:
        print(f"\n  ❌ build_local_trajectories.py 执行失败 (exit code: {result.returncode})")
        return False

    print(f"\n  ✅ 本地轨迹生成 + 合并完成")
    return True


def register_dataset():
    """将训练数据注册到 LLaMA-Factory 的 dataset_info.json"""
    print("\n📋 注册数据集到 LLaMA-Factory...")

    info_path = os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json")
    if not os.path.exists(info_path):
        print(f"  [WARN] dataset_info.json 不存在: {info_path}")
        return True

    with open(info_path, encoding="utf-8") as f:
        dataset_info = json.load(f)

    updated = False

    # 注册 GRPO 数据集
    grpo_data_file = os.path.join(DATA_DIR, "web_fallback_trajectories_grpo.jsonl")
    if os.path.exists(grpo_data_file):
        rel_path = os.path.relpath(grpo_data_file, os.path.join(LLAMA_FACTORY_DIR, "data"))
        dataset_info["web_fallback_grpo"] = {
            "file_name": rel_path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "prompt",
                "completion": "completion",
            },
        }
        print(f"  ✅ 注册 web_fallback_grpo: {rel_path}")
        updated = True

    # 注册合并 SFT 数据集（训练集 + 评估集拆分）
    combined_sft_file = os.path.join(DATA_DIR, "combined_trajectories_sft.json")
    if os.path.exists(combined_sft_file):
        train_file = os.path.join(DATA_DIR, "combined_sft_train.json")
        eval_file = os.path.join(DATA_DIR, "combined_sft_eval.json")

        # 如果拆分文件已存在，直接复用（不重复拆分）
        if os.path.exists(train_file) and os.path.exists(eval_file):
            with open(train_file, encoding="utf-8") as f:
                train_data = json.load(f)
            with open(eval_file, encoding="utf-8") as f:
                eval_data = json.load(f)
            print(f"  ✅ 复用已有拆分: train={len(train_data)} eval={len(eval_data)}")
        else:
            # 读取合并数据
            with open(combined_sft_file, encoding="utf-8") as f:
                all_data = json.load(f)

            # 固定随机种子，按 data_source 分层拆分 80/20
            import random
            random.seed(42)
            web_data = [d for d in all_data if d.get("data_source") == "web_fallback"]
            local_data = [d for d in all_data if d.get("data_source") != "web_fallback"]
            random.shuffle(web_data)
            random.shuffle(local_data)

            web_split = int(len(web_data) * 0.8)
            local_split = int(len(local_data) * 0.8)
            train_data = web_data[:web_split] + local_data[:local_split]
            eval_data = web_data[web_split:] + local_data[local_split:]
            random.shuffle(train_data)
            random.shuffle(eval_data)

            # 保存拆分文件
            with open(train_file, "w", encoding="utf-8") as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=2)

            print(f"  ✅ 拆分合并数据集: train={len(train_data)} eval={len(eval_data)}")

        # 注册到 dataset_info.json
        train_rel = os.path.relpath(train_file, os.path.join(LLAMA_FACTORY_DIR, "data"))
        eval_rel = os.path.relpath(eval_file, os.path.join(LLAMA_FACTORY_DIR, "data"))
        sft_columns = {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
        }
        dataset_info["combined_sft_train"] = {
            "file_name": train_rel,
            "formatting": "alpaca",
            "columns": sft_columns,
        }
        dataset_info["combined_sft_eval"] = {
            "file_name": eval_rel,
            "formatting": "alpaca",
            "columns": sft_columns,
        }
        print(f"  ✅ 注册 combined_sft_train: {train_rel}")
        print(f"  ✅ 注册 combined_sft_eval:  {eval_rel}")
        updated = True

    if updated:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        print("  ✅ dataset_info.json 已更新")
    else:
        print("  ⚠️ 未找到可注册的数据文件")

    return True


# ────────────────────────────────────────────────────────────
# Stage 2: SFT Warm-up
# ────────────────────────────────────────────────────────────

def run_sft_warmup(config_path: str):
    """
    使用 SFT 数据做 warm-up 微调，让模型先学会基本的工具调用格式。
    """
    print("\n" + "=" * 60)
    print("🔥 Stage 2: SFT Warm-up 训练")
    print("=" * 60)

    if not os.path.exists(config_path):
        print(f"  [ERROR] 配置文件不存在: {config_path}")
        return False

    # 确保数据集已注册到 dataset_info.json（支持单独运行 --stage sft）
    register_dataset()

    cmd = [
        sys.executable, "-m", "llamafactory.cli",
        "train", config_path,
    ]

    print(f"  🚀 启动 SFT 训练...")
    print(f"     配置: {config_path}")

    # 不覆盖 PYTHONPATH → 使用 pip 安装的 llamafactory（支持 transformers 5.x）
    # cwd 仍指向 LLaMA-Factory-main，使 output_dir 等相对路径正确解析
    result = subprocess.run(
        cmd, cwd=LLAMA_FACTORY_DIR,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"  ❌ SFT 训练失败 (exit code: {result.returncode})")
        return False

    print(f"  ✅ SFT warm-up 训练完成")
    return True


# ────────────────────────────────────────────────────────────
# Stage 3: GRPO 训练（TRL GRPOTrainer + PEFT LoRA）
# ────────────────────────────────────────────────────────────

# ── GRPO 训练超参数（原 configs/qwen3_lora_grpo.yaml）────────
# 优化策略：
# 1. num_generations=6：增加候选数，提高策略对比学习效果
# 2. per_device_train_batch_size=2 + gradient_accumulation_steps=2：有效batch=4，提升GPU利用率
# 3. beta=0.1：适度增加KL惩罚，平衡探索与稳定
# 4. lora_rank=16：增加LoRA秩，提升表达能力
GRPO_HYPERPARAMS = {
    "num_generations":           6,        # 每个 prompt 生成候选数（改为6）
    "max_completion_length":     768,      # 本地轨迹无 <information>，768 足够
    "per_device_train_batch_size": 2,      # 增大单卡batch（需显存充足）
    "gradient_accumulation_steps": 2,      # 减少累积步数（有效batch=4）
    "learning_rate":            2e-5,      # 适度提高学习率
    "num_train_epochs":         5.0,       # 增加训练轮数，充分学习
    "max_train_samples":        300,       # GRPO 每步需生成 N 条候选，全量数据太慢，采样子集
    "lr_scheduler_type":        "cosine",
    "warmup_ratio":             0.1,
    "bf16":                     True,
    "beta":                     0.1,       # 适度增加KL惩罚，防止策略偏离
    "temperature":              0.7,
    "top_p":                    0.9,
    "lora_rank":                16,        # 增加LoRA秩，提升表达能力
    "lora_alpha":               32,        # 配合rank调整
}


def run_grpo_training(config_path: str):
    """
    使用 TRL GRPOTrainer + PEFT LoRA 进行强化学习训练。

    流程：
      1. 加载 Qwen3-8B 基座模型
      2. 合并 SFT warm-up 适配器（merge_and_unload）
      3. 应用新的 LoRA 配置用于 GRPO 训练
      4. 使用自定义 6 维奖励函数进行 GRPO 训练
      5. 保存 GRPO LoRA 适配器

    依赖：trl >= 0.14, peft, transformers, datasets
    """
    print("\n" + "=" * 60)
    print("🤖 Stage 3: GRPO 强化学习训练 (TRL + PEFT)")
    print("=" * 60)

    # ── 延迟导入（避免非训练阶段加载重型库）──────────────────
    import sys

    # 生成后端：当前使用 HF model.generate()（见下方 GRPOConfig.use_vllm=False）
    # 注意：启用 vLLM 需安装兼容 transformers 5.x 的 vllm 版本，并改用 server 模式
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, LoraConfig, TaskType
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset

        # 确保项目根目录在 sys.path 中，以便导入 src.rl.reward_model
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from src.rl.reward_model import reward_fn
    except ImportError as e:
        print(f"  [ERROR] 导入失败: {e}")
        return False

    # ── 路径配置 ──────────────────────────────────────────
    model_base_path  = os.path.join(BASE_DIR, "models/Qwen3-8B/")
    sft_adapter_path = SFT_ADAPTER_DIR
    grpo_output_dir  = GRPO_OUTPUT_DIR
    grpo_data_path   = os.path.join(DATA_DIR, "combined_trajectories_grpo.jsonl")

    # ── 前置条件检查 ──────────────────────────────────────
    prerequisites = [
        (model_base_path,  "基座模型"),
        (sft_adapter_path, "SFT warm-up 适配器"),
        (grpo_data_path,   "GRPO 训练数据"),
    ]
    for path, name in prerequisites:
        if not os.path.exists(path):
            print(f"  ❌ {name}不存在: {path}")
            return False

    # ── 加载 GRPO 训练数据 ───────────────────────────────
    print("  📊 加载 GRPO 训练数据...")
    data = []
    with open(grpo_data_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    dataset = Dataset.from_list(data)
    print(f"     原始数据: {len(dataset)} 条")

    # GRPO 每步需为每个 prompt 生成 num_generations 条候选，全量数据极慢
    max_samples = GRPO_HYPERPARAMS["max_train_samples"]
    if max_samples > 0 and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        print(f"     采样子集: {len(dataset)} 条（max_train_samples={max_samples}）")

    # ── 加载基座模型 ─────────────────────────────────────
    print("  🔄 加载基座模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 将 prompt 从消息列表转为字符串（GRPOTrainer 要求）──
    def _format_prompt(example):
        prompt = example["prompt"]
        if isinstance(prompt, list):
            example["prompt"] = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True,
            )
        return example

    dataset = dataset.map(_format_prompt, desc="格式化 prompt")
    print(f"     ✅ prompt 已转换为字符串格式")

    # ── 合并 SFT 适配器到基座 ────────────────────────────
    print("  🔄 合并 SFT warm-up 适配器到基座...")
    model = PeftModel.from_pretrained(model, sft_adapter_path)
    model = model.merge_and_unload()
    print("     ✅ 合并完成")

    # ── LoRA 配置（GRPO 训练用新适配器）──────────────────
    lora_config = LoraConfig(
        r=GRPO_HYPERPARAMS["lora_rank"],
        lora_alpha=GRPO_HYPERPARAMS["lora_alpha"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    # ── GRPO 训练配置 ────────────────────────────────────
    # 清理旧输出目录（GRPOConfig 没有 overwrite_output_dir 参数）
    import shutil
    if os.path.exists(grpo_output_dir):
        shutil.rmtree(grpo_output_dir)
    os.makedirs(grpo_output_dir, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=grpo_output_dir,
        num_generations=GRPO_HYPERPARAMS["num_generations"],
        max_completion_length=GRPO_HYPERPARAMS["max_completion_length"],
        per_device_train_batch_size=GRPO_HYPERPARAMS["per_device_train_batch_size"],
        gradient_accumulation_steps=GRPO_HYPERPARAMS["gradient_accumulation_steps"],
        learning_rate=GRPO_HYPERPARAMS["learning_rate"],
        num_train_epochs=GRPO_HYPERPARAMS["num_train_epochs"],
        lr_scheduler_type=GRPO_HYPERPARAMS["lr_scheduler_type"],
        warmup_ratio=GRPO_HYPERPARAMS["warmup_ratio"],
        bf16=GRPO_HYPERPARAMS["bf16"],
        beta=GRPO_HYPERPARAMS["beta"],
        temperature=GRPO_HYPERPARAMS["temperature"],
        top_p=GRPO_HYPERPARAMS["top_p"],
        use_vllm=False,               # 暂用 HF model.generate()；TRL 0.18 的 vLLM(colocate) 与 PEFT 存在挂起风险
        # 如需启用 vLLM 加速：改为 use_vllm=True 并以 `trl vllm-serve` 启动 server 模式，
        # TRL 0.18 已移除 vllm_model_kwargs（不再转发任意 kwargs 到 vLLM 引擎）。
        logging_steps=5,
        save_steps=50,
        report_to="none",
    )

    print(f"  🚀 启动 GRPO 训练...")
    print(f"     基座: {model_base_path}")
    print(f"     SFT:  {sft_adapter_path}")
    print(f"     数据: {len(dataset)} 条")
    print(f"     候选: num_generations={GRPO_HYPERPARAMS['num_generations']}")
    print(f"     输出: {grpo_output_dir}")

    # ── 创建 Trainer 并训练 ─────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # ── 保存模型 ─────────────────────────────────────────
    trainer.save_model(grpo_output_dir)
    tokenizer.save_pretrained(grpo_output_dir)

    print(f"  ✅ GRPO 训练完成，适配器保存至: {grpo_output_dir}")
    return True


# ────────────────────────────────────────────────────────────
# Stage 4: 导出 & 量化
# ────────────────────────────────────────────────────────────

def export_model():
    """
    导出 LoRA 合并后的模型。

    重要：GRPO 训练时基座 = Qwen3-8B + SFT adapter (merge_and_unload)，
    GRPO LoRA 是相对于这个合并模型的增量。导出时必须先合并 SFT 再合并 GRPO，
    否则会丢失 SFT warm-up 的全部学习成果。
    """
    print("\n" + "=" * 60)
    print("📤 Stage 4: 导出模型")
    print("=" * 60)

    model_base_path = os.path.join(BASE_DIR, "models/Qwen3-8B/")

    if not os.path.exists(GRPO_OUTPUT_DIR):
        print(f"  ❌ GRPO 模型目录不存在: {GRPO_OUTPUT_DIR}")
        print(f"     请先完成 GRPO 训练")
        return False

    if not os.path.exists(SFT_ADAPTER_DIR):
        print(f"  ❌ SFT 适配器目录不存在: {SFT_ADAPTER_DIR}")
        print(f"     请先完成 SFT warm-up 训练")
        return False

    # 链式合并：base → +SFT LoRA → +GRPO LoRA → 导出
    adapter_paths = f"{SFT_ADAPTER_DIR},{GRPO_OUTPUT_DIR}"

    print(f"  🔄 导出 LoRA 合并模型...")
    print(f"     基座: {model_base_path}")
    print(f"     适配器链: SFT({SFT_ADAPTER_DIR}) → GRPO({GRPO_OUTPUT_DIR})")
    cmd = [
        sys.executable, "-m", "llamafactory.cli", "export",
        "--model_name_or_path", model_base_path,
        "--adapter_name_or_path", adapter_paths,
        "--template", "qwen3",
        "--finetuning_type", "lora",
        "--export_dir", RL_EXPORT_DIR,
        "--export_size", "2",
        "--export_device", "cpu",
        "--export_legacy_format", "false",
    ]

    # 不覆盖 PYTHONPATH → 使用 pip 安装的 llamafactory（支持 transformers 5.x）
    result = subprocess.run(cmd, cwd=LLAMA_FACTORY_DIR, capture_output=False)

    if result.returncode != 0:
        print(f"  ❌ 导出失败")
        return False

    print(f"  ✅ 模型导出完成: {RL_EXPORT_DIR}")
    print(f"  💡 量化命令: python {LLAMA_FACTORY_DIR}/awq_quant.py")
    return True


# ────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO 强化学习训练流程")
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "data", "sft", "grpo", "export"],
        help="执行阶段：all=全流程, data=数据准备, sft=warm-up, grpo=强化学习, export=导出",
    )
    parser.add_argument(
        "--sft-config", type=str,
        default=os.path.join(CONFIG_DIR, "qwen3_lora_rl_sft.yaml"),
        help="SFT 训练配置文件路径",
    )
    parser.add_argument(
        "--grpo-config", type=str,
        default=os.path.join(CONFIG_DIR, "qwen3_lora_grpo.yaml"),
        help="GRPO 训练配置文件路径",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🚗 小米 SU7 RAG - GRPO 强化学习训练")
    print("=" * 60)
    print(f"  项目根目录:  {BASE_DIR}")
    print(f"  LLaMA-Factory: {LLAMA_FACTORY_DIR}")
    print(f"  数据目录:    {DATA_DIR}")
    print(f"  执行阶段:    {args.stage}")
    print("=" * 60)

    stages = {
        "data":   [
            ("📦 数据准备（轨迹生成 + 格式转换）", prepare_data),
            ("📋 注册数据集到 LLaMA-Factory",      register_dataset),
        ],
        "sft":    [
            ("🔥 SFT Warm-up 训练", lambda: run_sft_warmup(args.sft_config)),
        ],
        "grpo":   [
            ("🤖 GRPO 强化学习训练", lambda: run_grpo_training(args.grpo_config)),
        ],
        "export": [
            ("📤 导出模型", export_model),
        ],
    }

    if args.stage == "all":
        flow = []
        flow.extend(stages["data"])
        flow.extend(stages["sft"])
        flow.extend(stages["grpo"])
        flow.extend(stages["export"])
    else:
        flow = stages[args.stage]

    for name, fn in flow:
        print(f"\n>>> {name}")
        if not fn():
            print(f"\n❌ 阶段 [{name}] 失败，终止流程")
            return

    print("\n" + "=" * 60)
    print("✅ 全部阶段完成！")
    print("=" * 60)
    print(f"  GRPO 模型: {GRPO_OUTPUT_DIR}")
    print(f"  导出路径: {RL_EXPORT_DIR}")
    print(f"\n  💡 启动推理:")
    print(f"     python deploy/auto_vllm_server.py --model {RL_EXPORT_DIR} --port 8000")
    print(f"     python src/rl/infer_rl.py")


if __name__ == "__main__":
    main()