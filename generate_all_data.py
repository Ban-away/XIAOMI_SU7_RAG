#!/usr/bin/env python3
"""
小米 SU7 RAG 项目 - 数据生成一站式脚本

完整审核流程：
  Step1: gen_qa → qa_pair.json          （LLM自动生成原始QA）
  Step2: gen_qa → expand_qa_pair.json   （LLM泛化同义问法）
  Step3: 训练集/测试集切分              （负样本只进训练集，不进测试集）
  Step4: QA质量审核过滤                 （清理序号、缩写展开、低质量过滤）
  Step5: 关键词标注                     （LLM抽取测试集答案关键词）
  Step6: test_qa_pair_verify.json       （最终高质量评估集）
  Step7: SFT训练数据生成
"""
import sys
sys.setrecursionlimit(10000)
import os
import json
import pickle
import hashlib
import random
import argparse
import time
import warnings
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*BaseRetriever.get_relevant_documents.*")
warnings.filterwarnings("ignore", message=".*Asking to truncate to max_length.*")

parser = argparse.ArgumentParser()
parser.add_argument("--force", "-f", action="store_true", help="强制重新生成所有文件")
parser.add_argument("--skip-expand", action="store_true", help="跳过扩展QA生成")
args = parser.parse_args()

from src.gen_qa.run import (
    gen_qa,
    CONTEXT_PROMPT_TPL,
    GENERALIZE_PROMPT_TPL,
    KEYWORDS_PROMPT_TPL,
    QA_PATH,
    OUTPUT_PATH,
    TRAIN_PATH,
    TEST_PATH,
    TEST_KEYWORDS_PATH,
)
from src.gen_qa.qa_filter import filter_qa_pairs, print_filter_report
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.client.llm_chat_client import request_chat
from src.reranker.minicpm_reranker import MiniCPMReRanker
from src.constant import bge_reranker_minicpm_path, split_docs_path
from src.utils import merge_docs, post_processing


# ── Step1：生成原始 QA ────────────────────────────────────────
def step1_generate_raw_qa():
    print("\n" + "="*60)
    print("Step 1: 生成原始 QA 对")
    print("="*60)

    if not os.path.exists(split_docs_path):
        print(f"❌ 错误：{split_docs_path} 不存在，请先运行 build_index.py")
        return False

    if os.path.exists(QA_PATH) and os.path.getsize(QA_PATH) > 0 and not args.force:
        count = sum(1 for _ in open(QA_PATH))
        print(f"✅ {QA_PATH} 已存在 ({count} 条)，跳过")
        return True

    with open(split_docs_path, "rb") as f:
        splitted_docs = pickle.load(f)

    print(f"📄 待处理文档数: {len(splitted_docs)}")
    print(f"[INFO] 已过滤过滤 chunk_size < 100 字符的文档块")
    result = gen_qa(splitted_docs, CONTEXT_PROMPT_TPL, QA_PATH, expand=False, force=args.force)
    print(f"✅ {QA_PATH} 生成完成，共 {len(result)} 条")
    return True


# ── Step2：泛化问法 ───────────────────────────────────────────
def step2_generate_expanded_qa():
    print("\n" + "="*60)
    print("Step 2: 生成扩展 QA 对")
    print("="*60)

    if args.skip_expand:
        print("⏭️ 跳过（--skip-expand）")
        if not os.path.exists(OUTPUT_PATH):
            open(OUTPUT_PATH, "w").close()
        return True

    if not os.path.exists(QA_PATH):
        print(f"❌ {QA_PATH} 不存在")
        return False

    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0 and not args.force:
        count = sum(1 for _ in open(OUTPUT_PATH))
        print(f"✅ {OUTPUT_PATH} 已存在 ({count} 条)，跳过")
        return True

    question_docs = []
    idx = 0
    with open(QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            info = json.loads(line)
            try:
                resp = json.loads(info.get("raw_resp", "[]"))
            except Exception:
                continue
            for qa in resp:
                q = qa.get("question", "").strip()
                if q:
                    question_docs.append(
                        Document(page_content=q, metadata={"unique_id": str(idx)})
                    )
                    idx += 1

    print(f"📄 待扩展问题数（从QA对中提取）: {len(question_docs)}")
    print(f"[INFO] 每个问题生成5个同义问法，实际调用次数取决于断点续传状态")
    
    # 扩展QA使用更高的并发数（参考原始QA的成功经验）
    import src.gen_qa.run as gen_module
    original_workers = gen_module.MAX_WORKERS
    gen_module.MAX_WORKERS = 35  # 直接设置为35（参考原始QA的30并发成功经验）
    print(f"⚠️ 扩展QA已分批处理，设置并发数为 {gen_module.MAX_WORKERS}")
    
    # 分批处理配置（参考原始QA的成功经验）
    BATCH_SIZE = 850  # 每批处理800条（与原始QA的823条相当）
    BASE_WAIT_SECONDS = 5  # 固定等待时间（秒）
    MAX_WAIT_SECONDS = 5   # 固定等待时间，不递增
    
    # 检查是否有已处理的记录（断点续传）
    processed_ids = set()
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    info = json.loads(line)
                    # 同时存储字符串和整数形式，确保匹配
                    uid = info["unique_id"]
                    processed_ids.add(str(uid))
                    try:
                        processed_ids.add(str(int(uid)))
                    except:
                        pass
                except Exception as e:
                    pass
        print(f"📌 已处理 {len(processed_ids)} 条，继续剩余部分...")
    
    # 过滤已处理的问题
    remaining_docs = []
    for doc in question_docs:
        doc_uid = str(doc.metadata["unique_id"])
        if doc_uid not in processed_ids:
            remaining_docs.append(doc)
    print(f"📄 剩余待处理: {len(remaining_docs)} 条")
    
    # 分批处理
    total_batches = (len(remaining_docs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📦 共 {total_batches} 批，每批 {BATCH_SIZE} 条")
    print(f"⏱️ 基础等待时间: {BASE_WAIT_SECONDS}秒，最大等待时间: {MAX_WAIT_SECONDS}秒")
    
    import time
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(remaining_docs))
        batch_docs = remaining_docs[start_idx:end_idx]
        
        print(f"\n🚀 处理批次 {batch_idx+1}/{total_batches} ({len(batch_docs)} 条)")
        
        # 生成扩展 QA（分批处理时强制重新生成，跳过内部断点续传检查）
        # 因为我们已经在外部做了断点续传过滤
        result = gen_qa(batch_docs, GENERALIZE_PROMPT_TPL, OUTPUT_PATH, expand=True, force=True)
        
        # 等待固定时间（最后一批不需要等待）
        if batch_idx < total_batches - 1:
            print(f"⏳ 等待 {BASE_WAIT_SECONDS} 秒...")
            time.sleep(BASE_WAIT_SECONDS)
    
    # 恢复原始并发数
    gen_module.MAX_WORKERS = original_workers
    
    # 统计最终结果
    if os.path.exists(OUTPUT_PATH):
        count = sum(1 for _ in open(OUTPUT_PATH))
        print(f"\n✅ {OUTPUT_PATH} 生成完成，共 {count} 条记录")
        return True
    return False


# ── Step3：切分 + Step4：质量审核 ────────────────────────────
def step3_split_and_filter():
    print("\n" + "="*60)
    print("Step 3+4: 切分训练/测试集 + QA质量审核过滤")
    print("="*60)

    if os.path.exists(TRAIN_PATH) and os.path.getsize(TRAIN_PATH) > 0 and \
       os.path.exists(TEST_PATH) and os.path.getsize(TEST_PATH) > 0 and not args.force:
        print(f"✅ {TRAIN_PATH} 和 {TEST_PATH} 已存在，跳过")
        return True

    if not os.path.exists(QA_PATH):
        print(f"❌ {QA_PATH} 不存在")
        return False

    # 加载原始 QA
    qa_dict = {}
    with open(QA_PATH) as f:
        for line in f:
            info = json.loads(line)
            qa_dict[info["unique_id"]] = info

    # 加载扩展问法
    expand_qa_pairs = {}
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    info = json.loads(line)
                    q = info["unique_id"]
                    raw = info.get("raw_resp", "")
                    questions = [
                        x.strip() for x in raw.split("\n") if x.strip()
                    ]
                    expand_qa_pairs[q] = questions
                except Exception:
                    continue

    print(f"原始 QA 数: {len(qa_dict)}，扩展问法映射: {len(expand_qa_pairs)}")

    # 提取所有 QA，正样本随机切分，负样本全部进训练集
    raw_positive_train, raw_positive_test = [], []
    raw_negative_train = []

    for unique_id, info in qa_dict.items():
        try:
            resp = json.loads(info["raw_resp"])
        except Exception:
            continue
        for qa in resp:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if not q or not a:
                continue
            if "无法准确" in a or "未提及" in a:
                continue

            all_qs = [q] + expand_qa_pairs.get(q, [])
            for query in all_qs:
                item = {
                    "unique_id": hashlib.md5(query.encode()).hexdigest(),
                    "question": query,
                    "answer": a,
                }
                if random.random() < 0.9:
                    raw_positive_train.append(item)
                else:
                    raw_positive_test.append(item)

    # 负样本只进训练集，不进测试集
    chats_path = os.path.join(os.path.dirname(QA_PATH), "..", "ut", "raw_general_chats.txt")
    chats_path2 = os.path.join(os.path.dirname(QA_PATH), "..", "ut", "chats.txt")
    neg_questions = []
    for p in [chats_path, chats_path2]:
        if os.path.exists(p):
            neg_questions += [l.strip() for l in open(p) if l.strip()]

    random.seed(42)
    for line in neg_questions:
        raw_negative_train.append({
            "unique_id": hashlib.md5(line.encode()).hexdigest(),
            "question": line,
            "answer": "无答案",
        })

    print(f"切分前：正样本训练{len(raw_positive_train)}/测试{len(raw_positive_test)}，负样本{len(raw_negative_train)}")

    # ── Step4：质量审核过滤 ──────────────────────────────────
    # 训练集：过滤低质量，保留负样本
    train_passed, train_rejected = filter_qa_pairs(
        raw_positive_train + raw_negative_train, remove_no_answer=False
    )
    print_filter_report(
        len(raw_positive_train) + len(raw_negative_train),
        train_passed, train_rejected
    )

    # 测试集：过滤低质量，同时去掉负样本（测试集只评估有意义的问答）
    test_passed, test_rejected = filter_qa_pairs(
        raw_positive_test, remove_no_answer=True
    )
    print_filter_report(len(raw_positive_test), test_passed, test_rejected)

    random.shuffle(train_passed)
    random.shuffle(test_passed)

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(train_passed, f, ensure_ascii=False, indent=2)
    print(f"✅ 训练集：{TRAIN_PATH}（{len(train_passed)} 条）")

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(test_passed, f, ensure_ascii=False, indent=2)
    print(f"✅ 测试集：{TEST_PATH}（{len(test_passed)} 条）")

    return True


# ── Step5：关键词标注 ─────────────────────────────────────────
def step5_generate_keywords():
    print("\n" + "="*60)
    print("Step 5: 测试集答案关键词标注")
    print("="*60)

    if os.path.exists(TEST_KEYWORDS_PATH) and os.path.getsize(TEST_KEYWORDS_PATH) > 0 and not args.force:
        print(f"✅ {TEST_KEYWORDS_PATH} 已存在，跳过")
        return True

    if not os.path.exists(TEST_PATH):
        print(f"❌ {TEST_PATH} 不存在")
        return False

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_qa_pairs = json.load(f)

    # 对每条答案抽取关键词（去重，按答案内容去重，节省 API 调用）
    unique_answers = list(dict.fromkeys(  # 用 dict 保持顺序去重
        item["answer"] for item in test_qa_pairs
        if item.get("answer") and item["answer"] != "无答案"
    ))
    answer_docs = [
        Document(page_content=a, metadata={"unique_id": str(i)})
        for i, a in enumerate(unique_answers)
    ]

    print(f"📄 待抽取关键词答案数（去重后）: {len(answer_docs)}")
    gen_qa(answer_docs, KEYWORDS_PROMPT_TPL, TEST_KEYWORDS_PATH, expand=True, force=args.force)

    # 把关键词写回测试集
    keywords_mapping = {}
    with open(TEST_KEYWORDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            info = json.loads(line)
            keywords = [k.strip() for k in info["raw_resp"].split(",")
                        if k.strip() not in ["无", "SU7", ""]]
            keywords_mapping[info["unique_id"]] = keywords

    for item in test_qa_pairs:
        ans = item.get("answer", "")
        if ans and ans != "无答案":
            try:
                idx = str(unique_answers.index(ans))
                item["keywords"] = keywords_mapping.get(idx, [])
            except ValueError:
                item["keywords"] = []
        else:
            item["keywords"] = []

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(test_qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"✅ {TEST_KEYWORDS_PATH} 生成完成，测试集关键词已更新")
    return True


# ── Step6：生成最终评估集 ─────────────────────────────────────
def step6_prepare_verify():
    print("\n" + "="*60)
    print("Step 6: 生成最终评估集 test_qa_pair_verify.json")
    print("="*60)

    verify_path = "data/qa_pairs/test_qa_pair_verify.json"

    if os.path.exists(verify_path) and os.path.getsize(verify_path) > 0 and not args.force:
        print(f"✅ {verify_path} 已存在，跳过")
        return True

    import shutil
    shutil.copy(TEST_PATH, verify_path)
    with open(verify_path) as f:
        data = json.load(f)

    has_kw = sum(1 for d in data if d.get("keywords"))
    print(f"✅ 评估集：{verify_path}（{len(data)} 条，有关键词 {has_kw} 条）")
    return True


# ── Step7：SFT 训练数据 ───────────────────────────────────────
def step7_generate_train_data():
    print("\n" + "="*60)
    print("Step 7: 生成 SFT 训练数据")
    print("="*60)

    output_path = "data/qa_pairs/train_data.json"

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and not args.force:
        count = sum(1 for _ in open(output_path))
        print(f"✅ {output_path} 已存在 ({count} 条)，跳过")
        return True

    if not os.path.exists(TRAIN_PATH):
        print(f"❌ {TRAIN_PATH} 不存在")
        return False

    print("🔧 加载检索器和重排器...")
    bm25_retriever   = BM25(docs=None, retrieve=True)
    milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
    reranker         = MiniCPMReRanker(model_path=bge_reranker_minicpm_path, cutoff_layers=28)
    milvus_retriever.retrieve_topk("测试", topk=3)

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_qa_pairs = json.load(f)

    print(f"📄 待处理训练样本数: {len(train_qa_pairs)}")

    BATCH_SIZE  = 700
    MAX_WORKERS = min(32, os.cpu_count() * 2)
    _bm25_lock   = __import__("threading").Lock()
    _rerank_lock = __import__("threading").Lock()

    def process_item(item):
        query = item["question"].strip()
        with _bm25_lock:
            bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
        milvus_docs  = milvus_retriever.retrieve_topk(query, topk=10)
        merged_docs  = merge_docs(bm25_docs, milvus_docs)
        with _rerank_lock:
            ranked_docs = reranker.rank(query, merged_docs, topk=5)
        context  = "\n".join(f"{i+1}.{doc.page_content}" for i, doc in enumerate(ranked_docs))
        response = request_chat(query, context)
        return {
            "query":       query,
            "context":     [doc.page_content for doc in ranked_docs],
            "response":    response,
            "merged_docs": [doc.page_content for doc in merged_docs],
        }

    total_batches = (len(train_qa_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    with open(output_path, "w", encoding="utf-8") as f:
        for batch_idx in range(total_batches):
            batch = train_qa_pairs[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
            print(f"\n🚀 批次 {batch_idx+1}/{total_batches}（{len(batch)} 条）")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_item, item): item for item in batch}
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
            if batch_idx < total_batches - 1:
                time.sleep(5)

    print(f"✅ SFT 训练数据：{output_path}")
    return True


# ── 主入口 ────────────────────────────────────────────────────
def main():
    random.seed(42)

    steps = [
        step1_generate_raw_qa,
        step2_generate_expanded_qa,
        step3_split_and_filter,       # Step3 内含 Step4 质量审核
        step5_generate_keywords,
        step6_prepare_verify,
        step7_generate_train_data,
    ]

    for step in steps:
        if not step():
            print(f"\n❌ {step.__name__} 失败，终止")
            return

    # 最终汇总
    print("\n" + "="*60)
    print("📊 生成结果验证")
    print("="*60)
    files = [
        ("qa_pair.json",             QA_PATH),
        ("expand_qa_pair.json",      OUTPUT_PATH),
        ("train_qa_pair.json",       TRAIN_PATH),
        ("test_qa_pair.json",        TEST_PATH),
        ("test_keywords_pair.json",  TEST_KEYWORDS_PATH),
        ("test_qa_pair_verify.json", "data/qa_pairs/test_qa_pair_verify.json"),
        ("train_data.json",          "data/qa_pairs/train_data.json"),
    ]
    for name, path in files:
        if os.path.exists(path):
            print(f"✅ {name}: {os.path.getsize(path):,} bytes")
        else:
            print(f"❌ {name}: 不存在")


if __name__ == "__main__":
    main()