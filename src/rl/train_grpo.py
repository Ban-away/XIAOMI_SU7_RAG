# -*- coding: utf-8 -*-
"""
GRPO 强化学习训练入口脚本

功能：
  1. 整合数据构建 → 格式转换 → GRPO 训练的全流程
  2. 支持 SFT warm-up → GRPO 两阶段训练
  3. 自动注册自定义奖励函数
  4. 训练后导出模型 & 量化

使用方式：
  # 完整流程：数据准备 + SFT warm-up + GRPO 训练
  python src/rl/train_grpo.py --stage all

  # 只运行 GRPO 训练（数据已准备好）
  python src/rl/train_grpo.py --stage grpo

  # 只运行数据准备
  python src/rl/train_grpo.py --stage data

  # 只运行 SFT warm-up
  python src/rl/train_grpo.py --stage sft

  # 指定 GPU
  CUDA_VISIBLE_DEVICES=0,1 python src/rl/train_grpo.py --stage grpo

依赖：
  - LLaMA-Factory (已安装在 LLaMA-Factory-main/)
  - 数据文件：data/rl_data/web_fallback_trajectories_grpo.jsonl
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
LLAMA_FACTORY_DIR = os.path.join(BASE_DIR, "LLaMA-Factory-main")
DATA_DIR          = os.path.join(BASE_DIR, "data/rl_data")
CONFIG_DIR        = os.path.join(BASE_DIR, "configs")

# ── 模型路径 ────────────────────────────────────────────────
SFT_MODEL_DIR     = os.path.join(LLAMA_FACTORY_DIR, "output/qwen3_lora_sft")
GRPO_OUTPUT_DIR   = os.path.join(LLAMA_FACTORY_DIR, "saves/qwen3-8b/lora/grpo")
RL_EXPORT_DIR     = os.path.join(LLAMA_FACTORY_DIR, "output/qwen3_lora_rl")


# ────────────────────────────────────────────────────────────
# Stage 1: 数据准备
# ────────────────────────────────────────────────────────────

def prepare_data():
    """准备 GRPO 训练数据"""
    print("\n" + "=" * 60)
    print("📦 Stage 1: 数据准备")
    print("=" * 60)

    grpo_data_path = os.path.join(DATA_DIR, "web_fallback_trajectories_grpo.jsonl")
    sft_data_path  = os.path.join(DATA_DIR, "web_fallback_trajectories_sft.json")

    # 检查数据是否已存在
    if os.path.exists(grpo_data_path) and os.path.exists(sft_data_path):
        with open(grpo_data_path, encoding="utf-8") as f:
            grpo_count = sum(1 for _ in f)
        with open(sft_data_path, encoding="utf-8") as f:
            sft_data = json.load(f)
        print(f"  ✅ GRPO 数据已存在: {grpo_count} 条 ({grpo_data_path})")
        print(f"  ✅ SFT 数据已存在: {len(sft_data)} 条 ({sft_data_path})")
        return True

    # 如果原始轨迹数据存在，用 format_converter 转换
    raw_path = os.path.join(DATA_DIR, "web_fallback_trajectories.json")
    if os.path.exists(raw_path):
        print("  🔄 使用 format_converter 转换数据...")
        from src.rl.format_converter import convert, save, load_trajectories
        data    = load_trajectories(raw_path)
        results = convert(data, target_format="all")
        save(results, output_dir=DATA_DIR)
        return True

    # 否则需要先用 data_builder 生成
    print("  ⚠️ 未找到已有数据，需要先运行 data_builder.py 生成轨迹数据")
    print(f"     python src/rl/data_builder.py")
    return False


def register_dataset():
    """将训练数据注册到 LLaMA-Factory 的 dataset_info.json"""
    print("\n📋 注册数据集到 LLaMA-Factory...")

    info_path = os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json")
    if not os.path.exists(info_path):
        print(f"  [WARN] dataset_info.json 不存在: {info_path}")
        return

    with open(info_path, encoding="utf-8") as f:
        dataset_info = json.load(f)

    # 注册 GRPO 数据集
    grpo_data_file = os.path.join(DATA_DIR, "web_fallback_trajectories_grpo.jsonl")
    if os.path.exists(grpo_data_file):
        # 计算相对路径
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

    # 注册 SFT warm-up 数据集
    sft_data_file = os.path.join(DATA_DIR, "web_fallback_trajectories_sft.json")
    if os.path.exists(sft_data_file):
        rel_path = os.path.relpath(sft_data_file, os.path.join(LLAMA_FACTORY_DIR, "data"))
        dataset_info["web_fallback_sft"] = {
            "file_name": rel_path,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        }
        print(f"  ✅ 注册 web_fallback_sft: {rel_path}")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print("  ✅ dataset_info.json 已更新")


# ────────────────────────────────────────────────────────────
# Stage 2: SFT Warm-up
# ────────────────────────────────────────────────────────────

def run_sft_warmup(config_path: str):
    """
    使用 SFT 数据做 warm-up 微调，让模型先学会基本的工具调用格式。

    使用 LLaMA-Factory 的 SFT 训练流程。
    """
    print("\n" + "=" * 60)
    print("🔥 Stage 2: SFT Warm-up 训练")
    print("=" * 60)

    if not os.path.exists(config_path):
        print(f"  [ERROR] 配置文件不存在: {config_path}")
        print(f"     请先确认 configs/ 目录下有对应的 YAML 配置")
        return False

    # 检查 SFT 模型是否已存在
    sft_output = os.path.join(LLAMA_FACTORY_DIR, "saves/qwen3-8b/lora/rl_sft")
    if os.path.exists(os.path.join(sft_output, "adapter_model.safetensors")):
        print(f"  ✅ SFT warm-up 模型已存在: {sft_output}")
        print(f"     如需重新训练，请先删除该目录")
        return True

    # 调用 LLaMA-Factory 训练
    cmd = [
        sys.executable, "-m", "llamafactory.cli",
        "train", config_path,
    ]

    print(f"  🚀 启动 SFT 训练...")
    print(f"     配置: {config_path}")
    print(f"     命令: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = LLAMA_FACTORY_DIR

    result = subprocess.run(
        cmd, cwd=LLAMA_FACTORY_DIR, env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"  ❌ SFT 训练失败 (exit code: {result.returncode})")
        return False

    print(f"  ✅ SFT warm-up 训练完成")
    return True


# ────────────────────────────────────────────────────────────
# Stage 3: GRPO 训练
# ────────────────────────────────────────────────────────────

def run_grpo_training(config_path: str):
    """
    使用 GRPO 算法进行强化学习训练。

    GRPO (Group Relative Policy Optimization) 不需要单独的 reward model，
    而是使用自定义奖励函数直接为每个 completion 打分。
    """
    print("\n" + "=" * 60)
    print("🤖 Stage 3: GRPO 强化学习训练")
    print("=" * 60)

    if not os.path.exists(config_path):
        print(f"  [ERROR] 配置文件不存在: {config_path}")
        return False

    cmd = [
        sys.executable, "-m", "llamafactory.cli",
        "train", config_path,
    ]

    print(f"  🚀 启动 GRPO 训练...")
    print(f"     配置: {config_path}")
    print(f"     命令: {' '.join(cmd)}")

    env = os.environ.copy()
    import os.path as osp
    sep = os.pathsep  # Linux: ':', Windows: ';'
    env["PYTHONPATH"] = f"{BASE_DIR}{sep}{LLAMA_FACTORY_DIR}"

    result = subprocess.run(
        cmd, cwd=LLAMA_FACTORY_DIR, env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"  ❌ GRPO 训练失败 (exit code: {result.returncode})")
        return False

    print(f"  ✅ GRPO 训练完成")
    return True


# ────────────────────────────────────────────────────────────
# Stage 4: 导出 & 量化
# ────────────────────────────────────────────────────────────

def export_model():
    """导出 LoRA 合并后的模型"""
    print("\n" + "=" * 60)
    print("📤 Stage 4: 导出模型")
    print("=" * 60)

    export_script = os.path.join(LLAMA_FACTORY_DIR, "export.sh")
    if os.path.exists(export_script):
        print(f"  🔄 使用 export.sh 导出...")
        result = subprocess.run(
            ["bash", export_script],
            cwd=LLAMA_FACTORY_DIR,
            capture_output=False,
        )
        if result.returncode == 0:
            print(f"  ✅ 模型导出完成: {RL_EXPORT_DIR}")
            return True

    # 手动导出
    print(f"  🔄 手动导出 LoRA 合并模型...")
    cmd = [
        sys.executable, "-m", "llamafactory.cli", "export",
        "--model_name_or_path", os.path.join(BASE_DIR, "models/Qwen3-8B/"),
        "--adapter_name_or_path", GRPO_OUTPUT_DIR,
        "--template", "qwen3",
        "--finetuning_type", "lora",
        "--export_dir", RL_EXPORT_DIR,
        "--export_size", "2",
        "--export_device", "cpu",
        "--export_legacy_format", "False",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = LLAMA_FACTORY_DIR

    result = subprocess.run(cmd, cwd=LLAMA_FACTORY_DIR, env=env, capture_output=False)

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
    print(f"  执行阶段:    {args.stage}")
    print("=" * 60)

    # 确保 configs 目录存在
    os.makedirs(CONFIG_DIR, exist_ok=True)

    stages = {
        "data":   [("📦 数据准备", prepare_data), ("📋 注册数据集", register_dataset)],
        "sft":    [("🔥 SFT Warm-up", lambda: run_sft_warmup(args.sft_config))],
        "grpo":   [("🤖 GRPO 训练", lambda: run_grpo_training(args.grpo_config))],
        "export": [("📤 导出模型", export_model)],
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
