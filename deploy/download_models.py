#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载项目所需公开模型到本地 models 目录。

默认行为：
1. 读取 RAG_BASE_DIR / XIAOMI_RAG_HOME（未设置则使用当前工作目录）
2. 下载 core 模型集（覆盖主流程必需模型）
3. 将模型按 src/constant.py 的目录结构落盘

示例：
  python deploy/download_models.py
  python deploy/download_models.py --preset all
  python deploy/download_models.py --hf-token <token>
"""

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    target_rel_path: str
    required: bool = True


MODEL_PRESETS: Dict[str, List[ModelSpec]] = {
    "core": [
        ModelSpec("m3e-small", "m3e-small", "models/AI-ModelScope/m3e-small"),
        ModelSpec("bge-large-zh-v1.5", "BAAI/bge-large-zh-v1.5", "models/BAAI/bge-large-zh-v1.5"),
        ModelSpec("splade-v2", "naver/splade-cocondenser-ensembledistil", "models/naver/splade-cocondenser-ensembledistil"),
        ModelSpec("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3", "models/BAAI/bge-reranker-v2-m3"),
    ],
    "all": [
        ModelSpec("m3e-small", "AI-ModelScope/m3e-small", "models/AI-ModelScope/m3e-small"),
        ModelSpec("bge-m3", "BAAI/bge-m3", "models/BAAI/bge-m3"),
        ModelSpec("bge-large-zh-v1.5", "BAAI/bge-large-zh-v1.5", "models/BAAI/bge-large-zh-v1.5"),
        ModelSpec("splade-v2", "naver/splade-cocondenser-ensembledistil", "models/naver/splade-cocondenser-ensembledistil"),
        ModelSpec("bce-embedding-base_v1", "maidalun/bce-embedding-base_v1", "models/maidalun/bce-embedding-base_v1"),
        ModelSpec("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", "models/Qwen3-Embedding-0.6B"),
        ModelSpec("Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-0.6B", "models/Qwen3-Reranker-0.6B"),
        ModelSpec("Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-4B", "models/Qwen3-Reranker-4B", required=False),
        ModelSpec("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3", "models/BAAI/bge-reranker-v2-m3"),
        ModelSpec("bge-reranker-v2-minicpm-layerwise", "BAAI/bge-reranker-v2-minicpm-layerwise", "models/bge-reranker-v2-minicpm-layerwise", required=False),
        ModelSpec("jina-reranker-v2-base-multilingual", "jinaai/jina-reranker-v2-base-multilingual", "models/jinaai/jina-reranker-v2-base-multilingual"),
        ModelSpec("text2vec-base-chinese", "shibing624/text2vec-base-chinese", "models/text2vec-base-chinese"),
    ],
}


def resolve_base_dir(user_base_dir: str = "") -> Path:
    if user_base_dir:
        return Path(user_base_dir).resolve()
    env_base = os.getenv("RAG_BASE_DIR") or os.getenv("XIAOMI_RAG_HOME")
    if env_base:
        return Path(env_base).resolve()
    return Path.cwd().resolve()


def download_one(spec: ModelSpec, base_dir: Path, hf_token: str = "") -> None:
    target_dir = base_dir / Path(spec.target_rel_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] downloading {spec.name} -> {target_dir}")
    snapshot_download(
        repo_id=spec.repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=hf_token or None,
    )
    print(f"[DONE] {spec.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download required models for XIAOMI_SU7_RAG.")
    parser.add_argument(
        "--preset",
        choices=MODEL_PRESETS.keys(),
        default="core",
        help="Model preset to download: core (default) or all.",
    )
    parser.add_argument(
        "--base-dir",
        default="",
        help="Project base directory. Defaults to RAG_BASE_DIR / XIAOMI_RAG_HOME / current directory.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN", ""),
        help="HuggingFace token for gated/private models.",
    )
    args = parser.parse_args()

    base_dir = resolve_base_dir(args.base_dir)
    print(f"[INFO] base_dir = {base_dir}")
    print(f"[INFO] preset = {args.preset}")

    failed: List[str] = []
    for spec in MODEL_PRESETS[args.preset]:
        try:
            download_one(spec, base_dir, args.hf_token)
        except Exception as exc:  # noqa: BLE001
            level = "ERROR" if spec.required else "WARN"
            print(f"[{level}] {spec.name} download failed: {exc}")
            if spec.required:
                failed.append(spec.name)

    print("\n[NOTE] 以下路径无法通过公共下载脚本自动获取，请手动准备：")
    print("       1) LLaMA-Factory-main/output/qwen3_lora_sft_int4  (本地SFT产物)")
    print("       2) RAG-Retrieval/.../checkpoint_0                  (本地重排微调产物)")

    if failed:
        raise SystemExit(f"[FAILED] required models not downloaded: {', '.join(failed)}")
    print("[SUCCESS] model download finished.")


if __name__ == "__main__":
    main()

