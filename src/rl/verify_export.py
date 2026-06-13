# -*- coding: utf-8 -*-
"""
验证导出的 RL 模型是否同时保留 SFT + GRPO 两层能力。

原理：SFT 教会模型 Search-R1 格式（<search_local>...</search_local> / <answer>...）。
对同一批问题做三路对比：
  A. base            —— 原始 Qwen3-8B（通常不会稳定产出上述标签）
  B. base + SFT      —— 应稳定产出 SFT 格式
  C. exported        —— base+SFT+GRPO 合并模型

判定：
  - C 与 B 一样稳定产出 SFT 格式标签 → SFT 保留 ✓
  - C 退化为 A 的自然语言（无标签）   → SFT 丢失，需排查 merge_and_unload ✗
  - C 的 <answer> 内容/奖励 ≥ B        → GRPO 生效 ✓

运行（先完成 --stage export）：
  python src/rl/verify_export.py
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 路径（与 train_grpo.py 常量一致）────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_MODEL    = os.path.join(BASE_DIR, "models/Qwen3-8B")
SFT_ADAPTER   = os.path.join(BASE_DIR, "LLaMA-Factory-main/saves/qwen3-8b/lora/rl_sft")
EXPORTED_DIR  = os.path.join(BASE_DIR, "LLaMA-Factory-main/output/qwen3_lora_rl")

# ── 测试问题 ────────────────────────────────────────────────
QUESTIONS = [
    "小米SU7的续航里程是多少？",
    "小米SU7 Ultra 用了什么刹车系统？",
    "小米SU7的轴距是多长？",
]

# ── 精简系统提示（仅用于触发格式；三路用同一 prompt 保证公平）──
SYSTEM_PROMPT = (
    "你是小米SU7车型的专业问答助手。回答时可调用以下工具：\n"
    "- 本地检索：<search_local>关键词</search_local>\n"
    "- 网络搜索：<search_web>关键词</search_web>\n"
    "检索结果格式：<information>内容</information>\n"
    "最终答案格式：<answer>答案内容</answer>"
)

# SFT 教会的格式标签——C 若保留这些即证明 SFT 在场
FORMAT_TAGS = ["<search_local>", "<search_web>", "<answer>", "<information>"]


def generate(model, tokenizer, question, max_new_tokens=256):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # 贪心，保证三路可比
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def format_tags_in(text):
    return [t for t in FORMAT_TAGS if t in text]


def run_variant(name, loader, tokenizer):
    print("\n" + "=" * 70)
    print(f"=== {name} ===")
    print("=" * 70)
    model = loader()
    results = []
    for q in QUESTIONS:
        text = generate(model, tokenizer, q)
        tags = format_tags_in(text)
        results.append((q, tags, text))
        print(f"\nQ: {q}")
        print(f"格式标签: {tags if tags else '（无）'}")
        print(f"输出: {text[:300]}{'...' if len(text) > 300 else ''}")
    del model
    torch.cuda.empty_cache()
    return results


def main():
    for path, label in [(BASE_MODEL, "基座"), (EXPORTED_DIR, "导出模型")]:
        if not os.path.exists(path):
            print(f"❌ {label}不存在: {path}")
            return
    if not os.path.exists(SFT_ADAPTER):
        print(f"❌ SFT 适配器不存在: {SFT_ADAPTER}")
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

    def load_sft():
        from peft import PeftModel
        m = load_base()
        return PeftModel.from_pretrained(m, SFT_ADAPTER)

    def load_exported():
        return AutoModelForCausalLM.from_pretrained(
            EXPORTED_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

    a = run_variant("A. base (原始 Qwen3-8B)",       load_base,     tokenizer)
    b = run_variant("B. base + SFT",                  load_sft,      tokenizer)
    c = run_variant("C. exported (base+SFT+GRPO)",    load_exported, tokenizer)

    # ── 汇总判定 ────────────────────────────────────────────
    print("\n" + "#" * 70)
    print("## 判定汇总（每路每题是否命中 SFT 格式标签）")
    print("#" * 70)
    for i, q in enumerate(QUESTIONS):
        print(f"\nQ: {q}")
        print(f"  A base        : {a[i][1] or '（无）'}")
        print(f"  B base+SFT    : {b[i][1] or '（无）'}")
        print(f"  C exported    : {c[i][1] or '（无）'}")

    c_keeps_format = all(c[i][1] for i in range(len(QUESTIONS)))
    print("\n>> 结论：")
    if c_keeps_format:
        print("   C 与 B 一样稳定产出 SFT 格式标签 → SFT 能力保留 ✓")
        print("   再人工对比 C vs B 的 <answer> 内容：C 更准确/更有条理 → GRPO 生效 ✓")
    else:
        print("   C 未稳定产出 SFT 格式 → SFT 可能丢失，需排查：")
        print("   1) 训练时 merge_and_unload() 是否真正合并了 SFT（而非残留 peft_config）")
        print("   2) export 的 adapter 链顺序是否为 SFT,GRPO")


if __name__ == "__main__":
    main()
