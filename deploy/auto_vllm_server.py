import argparse
import os
import subprocess
import sys


def detect_gpu_count() -> int:
    """Auto detect GPU count; prefer torch, then fallback to nvidia-smi."""
    try:
        import torch

        count = torch.cuda.device_count()
        if count and count > 0:
            return count
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return len(lines)
    except Exception:
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto start vLLM with single-GPU or multi-GPU tensor parallelism."
    )
    parser.add_argument(
        "--model",
        default=os.getenv(
            "VLLM_MODEL", "LLaMA-Factory-main/output/qwen3_lora_sft_int4"
        ),
        help="Model path or model id.",
    )
    parser.add_argument("--port", type=int, default=8000, help="vLLM API port.")
    parser.add_argument(
        "--max-model-len", type=int, default=8192, help="Max model length."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
        help="GPU memory utilization ratio for vLLM.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Computation dtype, e.g. bfloat16/float16/auto.",
    )
    parser.add_argument(
        "--disable-auto-tp",
        action="store_true",
        help="Disable auto tensor parallel even when multiple GPUs are detected.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to `vllm serve`.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    gpu_count = detect_gpu_count()
    if gpu_count <= 0:
        raise RuntimeError("No available CUDA GPU detected. Cannot start vLLM.")

    print(f"[INFO] Detected GPU count: {gpu_count}")
    command = [
        "vllm",
        "serve",
        args.model,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
    ]

    if gpu_count > 1 and not args.disable_auto_tp:
        command.extend(["--tensor-parallel-size", str(gpu_count)])
        print(f"[INFO] Multi-GPU mode: tensor_parallel_size={gpu_count}")
    else:
        print("[INFO] Single-GPU mode.")

    passthrough = args.extra_args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    command.extend(passthrough)

    print(f"[INFO] Launch command: {' '.join(command)}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] vLLM exited with code {exc.returncode}", file=sys.stderr)
        raise

