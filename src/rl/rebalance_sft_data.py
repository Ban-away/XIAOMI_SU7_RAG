# -*- coding: utf-8 -*-
"""
SFT + GRPO 数据再平衡：把 web 轨迹占比从 ~0.3% 拉到 ~33%。

根因：合并数据里 local 占 99.7%、web 仅 0.3%，模型训练时几乎没见过 web 轨迹。
本脚本同时处理两个文件（都按 data_source=='web_fallback' 区分 web/local）：
  - combined_trajectories_sft.json   （SFT warm-up 用，json 数组）
  - combined_trajectories_grpo.jsonl （GRPO 采样用，jsonl）
两个都保留全部 web、按目标占比下采样 local，让 SFT 和 GRPO 都能见到足够的 web 样本。

操作（首次运行自动备份原文件为 *.original.*）：
  python src/rl/rebalance_sft_data.py
  python src/rl/rebalance_sft_data.py --web-ratio 0.33 --local-cap 3000
  python src/rl/rebalance_sft_data.py --restore   # 恢复原始数据
"""

import os
import json
import random
import shutil
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data/rl_data")

FILES = [
    # (路径, 是否 jsonl)
    (os.path.join(DATA_DIR, "combined_trajectories_sft.json"),  False),
    (os.path.join(DATA_DIR, "combined_trajectories_grpo.jsonl"), True),
]


def is_web(item: dict) -> bool:
    return item.get("data_source") == "web_fallback"


def load_items(path: str, is_jsonl: bool) -> list:
    if is_jsonl:
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_items(path: str, items: list, is_jsonl: bool):
    if is_jsonl:
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


def rebalance(items: list, web_ratio: float, local_cap: int, rng: random.Random):
    """返回 (balanced_list, web_kept)；web 全部保留不丢弃，按 web 数量反推 local 数量。"""
    web = [d for d in items if is_web(d)]
    local = [d for d in items if not is_web(d)]
    if not web:
        return None, 0
    # 原则：web 全部保留；按 web 数量反推 local 数量达到目标占比
    # local = web * (1-ratio)/ratio；不超过 local_cap 与实际 local 总数
    target_local = int(len(web) * (1 - web_ratio) / web_ratio)
    n_local = min(target_local, local_cap, len(local))
    local_sample = rng.sample(local, n_local) if n_local < len(local) else local[:]
    web_kept = web[:]   # 全部 web 保留
    balanced = web_kept + local_sample
    rng.shuffle(balanced)
    return balanced, len(web_kept)


def backup_path(path: str) -> str:
    # 在最后扩展名前插入 .original：file.json→file.original.json, file.jsonl→file.original.jsonl
    base, ext = os.path.splitext(path)
    return base + ".original" + ext


def main():
    ap = argparse.ArgumentParser(description="Rebalance SFT+GRPO data (web vs local ratio)")
    ap.add_argument("--web-ratio", type=float, default=0.33, help="目标 web 占比（默认 0.33）")
    ap.add_argument("--local-cap", type=int, default=3000, help="local 下采样上限（默认 3000）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restore", action="store_true", help="用备份恢复原始数据")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    for path, is_jsonl in FILES:
        name = os.path.basename(path)
        bk = backup_path(path)

        if args.restore:
            if os.path.exists(bk):
                shutil.copy2(bk, path)
                print(f"✅ 已恢复：{bk} → {name}")
            else:
                print(f"⚠️  备份不存在：{bk}")
            continue

        if not os.path.exists(path):
            print(f"⏭️  跳过（文件不存在）：{name}")
            continue

        items = load_items(path, is_jsonl)
        web_n = sum(1 for d in items if is_web(d))
        print(f"\n=== {name} ===")
        print(f"原始：总 {len(items)} | web {web_n} ({100 * web_n / max(len(items), 1):.1f}%) | "
              f"local {len(items) - web_n} ({100 * (len(items) - web_n) / max(len(items), 1):.1f}%)")

        balanced, w_up = rebalance(items, args.web_ratio, args.local_cap, rng)
        if balanced is None:
            print("❌ 无 web 轨迹，跳过（请先 data_builder + build_local_trajectories 生成）")
            continue

        # 备份（仅首次）
        if not os.path.exists(bk):
            shutil.copy2(path, bk)
            print(f"✅ 已备份原始：{os.path.basename(bk)}")
        save_items(path, balanced, is_jsonl)
        print(f"再平衡后：总 {len(balanced)} | web {w_up} ({100 * w_up / len(balanced):.1f}%) | "
              f"local {len(balanced) - w_up} ({100 * (len(balanced) - w_up) / len(balanced):.1f}%) "
              f"| web 全部保留")

    if args.restore:
        return
    print("\n" + "=" * 60)
    print("✅ 再平衡完成（SFT + GRPO 均已处理）")
    print("下一步：重做 SFT → 再 GRPO（两个阶段的数据都已是 ~33% web）")
    print("   若玻璃水开始 web，回调：--web-ratio 0.25 重跑")
    print("=" * 60)


if __name__ == "__main__":
    main()
