# vllm serve output/qwen3_lora_sft --max-model-len 8192 --gpu-memory-utilization 0.75
# 从项目根目录启动，使用相对路径确保客户端能正确访问
cd /root/autodl-tmp/XIAOMI_SU7_RAG && vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 --max-model-len 8192 --gpu-memory-utilization 0.75