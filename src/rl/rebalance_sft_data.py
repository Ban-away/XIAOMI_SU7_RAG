# -*- coding: utf-8 -*-
"""
SFT 数据再平衡：把 web 轨迹占比从 ~0.3% 拉到 ~33%，让模型在 warm-up 阶段真正学到 web 行为。

根因：combined_trajectories_sft.json 里 local 占 99.7%、web 仅 0.3%，模型 SFT 时几乎没见过
web 轨迹 → 学成「永远本地答」。本脚本下采样 local、上采样 web，把 web 占比拉到目标值。

操作对象：data/rl_data/combined_trajectories_sft.json
- 首次运行会备份原文件到 combined_trajectories_sft.original.json
- 再平衡后覆盖写回 combined_trajectories_sft.json（SFT 流程自动读取）

运行（在 build_local_trajectories.py 之后，重做 SFT 之前）：
  python src/rl/rebalance_sft_data.py
  python src/rl/rebalance_sft_data.py --web-ratio 0.33 --local-cap 3000
  python src/rl/rebalance_sft_data.py --restore   # 用备份恢复原始数据
"""

import os
import json
import random
import shutil
import argparse

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH  = os.path.join(BASE_DIR, "data/rl_data/combined_trajectories_sft.json")
BACKUP_PATH = DATA_PATH.replace(".json", ".original.json")


def is_web(item: dict) -> bool:
    return item.get("data_source") == "web_fallback"


def main():
    ap = argparse.ArgumentParser(description="Rebalance SFT data (web vs local ratio)")
    ap.add_argument("--web-ratio", type=float, default=0.33,
                    help="目标 web 占比（默认 0.33）")
    ap.add_argument("--local-cap", type=int, default=3000,
                    help="local 下采样上限（默认 3000；local 足够教会本地行为，不需要 2 万条）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restore", action="store_true", help="用备份恢复原始数据")
    args = ap.parse_args()

    # 恢复模式
    if args.restore:
        if os.path.exists(BACKUP_PATH):
            shutil.copy2(BACKUP_PATH, DATA_PATH)
            print(f"✅ 已从备份恢复：{BACKUP_PATH} → {DATA_PATH}")
        else:
            print(f"❌ 备份不存在：{BACKUP_PATH}")
        return

    if not os.path.exists(DATA_PATH):
        print(f"❌ 文件不存在：{DATA_PATH}")
        print("   请先运行 build_local_trajectories.py 生成合并数据")
        return

    random.seed(args.seed)
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    web = [d for d in data if is_web(d)]
    local = [d for d in data if not is_web(d)]
    print(f"原始：总 {len(data)} | web {len(web)} ({100 * len(web) / max(len(data), 1):.1f}%) | "
          f"local {len(local)} ({100 * len(local) / max(len(data), 1):.1f}%)")

    if not web:
        print("❌ 没有 web 轨迹（data_source=='web_fallback'），无法再平衡。")
        print("   请先运行 data_builder.py 生成 web 轨迹，再 build_local_trajectories.py 合并。")
        return

    # ── 下采样 local ──
    if len(local) > args.local_cap:
        local_sample = random.sample(local, args.local_cap)
    else:
        local_sample = local[:]
    print(f"local 下采样：{len(local)} → {len(local_sample)}（cap={args.local_cap}）")

    # ── 上采样 web 到目标占比 ──
    # 目标 web/(web+local) = web_ratio → web_count = web_ratio/(1-web_ratio) * local_count
    target_web = int(args.web_ratio / (1 - args.web_ratio) * len(local_sample))
    reps, rem = divmod(target_web, len(web))
    web_up = web * reps + random.sample(web, rem)
    rep_factor = len(web_up) / max(len(web), 1)
    print(f"web 上采样：{len(web)} → {len(web_up)}（≈{rep_factor:.1f}x，目标占比 {args.web_ratio:.0%}）")

    # ── 合并 + 打乱 ──
    balanced = web_up + local_sample
    random.shuffle(balanced)

    # ── 备份原文件（仅首次）──
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(DATA_PATH, BACKUP_PATH)
        print(f"✅ 原文件已备份：{BACKUP_PATH}")
    else:
        print(f"ℹ️  备份已存在（不覆盖）：{BACKUP_PATH}")

    # ── 覆盖写回 ──
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(balanced, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ 再平衡完成，已写回：{DATA_PATH}")
    print(f"   总 {len(balanced)} | web {len(web_up)} ({100 * len(web_up) / len(balanced):.1f}%) | "
          f"local {len(local_sample)} ({100 * len(local_sample) / len(balanced):.1f}%)")
    print("=" * 60)
    print("下一步：重做 SFT warm-up（python src/rl/train_grpo.py --stage sft）")
    print("   抽测 OTA 是否会 web；若玻璃水开始 web，回调 --web-ratio（如 0.25）")


if __name__ == "__main__":
    main()
