# -*- coding: utf-8 -*-
# --------------------------------------------
# AWQ 量化脚本 (使用 llmcompressor)
# 避免 AutoAWQ 版本兼容性问题
# --------------------------------------------

import json
import os
import sys

try:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import AWQModifier
except ImportError:
    print("⚠️ llmcompressor 未安装，尝试自动安装...")
    try:
        import subprocess
        # 安装兼容版本的 llmcompressor 和 transformers
        subprocess.run([sys.executable, "-m", "pip", "install", "llmcompressor", "transformers==4.52.3", "-q"], check=True)
        print("✅ llmcompressor 安装成功！")
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import AWQModifier
    except Exception as e:
        print(f"❌ 安装失败: {str(e)}")
        print("请手动安装: python -m pip install llmcompressor transformers==4.52.3")
        sys.exit(1)

# 定义路径
model_path = "output/qwen3_lora_sft"
quant_path = "output/qwen3_lora_sft_int4"
calib_data_path = "../data/summary_data/train.json"

# 检查模型路径
if not os.path.exists(model_path):
    print(f"❌ 模型路径不存在: {model_path}")
    print("请先运行 LLaMA-Factory 训练或导出 LoRA 权重")
    sys.exit(1)

# 加载校准数据
print(f"⏳ 加载校准数据: {calib_data_path}")
calib_data = []
if os.path.exists(calib_data_path):
    with open(calib_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        for item in raw_data[:200]:  # 使用前200条作为校准数据
            try:
                # 格式化为对话格式
                text = f"用户: {item['instruction']}\n助手: {item['output']}"
                calib_data.append(text)
            except Exception as e:
                continue
    print(f"✅ 校准样本数: {len(calib_data)}")
else:
    print(f"⚠️ 校准数据不存在，使用默认数据集")
    calib_data = None

# AWQ 量化配置
print(f"\n⏳ 开始 AWQ 量化...")
print(f"  模型路径: {model_path}")
print(f"  输出路径: {quant_path}")
print(f"  量化配置: 4-bit, group_size=128")

try:
    # 使用 llmcompressor 进行 AWQ 量化
    oneshot(
        model=model_path,
        dataset=calib_data if calib_data else "wikitext",
        recipe=AWQModifier(
            targets="Linear",
            bits=4,
            group_size=128,
            zero_point=True,
            version="GEMM"
        ),
        output_dir=quant_path,
        overwrite=True
    )
    
    print(f"\n✅ AWQ 量化完成！")
    print(f"📁 量化模型路径: {quant_path}")
    
    # 验证输出
    if os.path.exists(quant_path):
        files = os.listdir(quant_path)
        print(f"📋 输出文件: {files}")
        
except Exception as e:
    print(f"\n❌ 量化失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)