# vllm serve output/qwen3_lora_sft --max-model-len 8192 --gpu-memory-utilization 0.75
# 使用相对路径，从脚本所在目录出发定位模型
cd "$(dirname "$0")" && vllm serve output/qwen3_lora_sft_int4 --max-model-len 8192 --gpu-memory-utilization 0.75