#!/usr/bin/env python3
"""
小米 SU7 RAG 项目 - 数据生成一站式脚本

生成所有必需的数据文件：
- qa_pair.json: 原始 QA 对
- expand_qa_pair.json: 扩展 QA 对（同义问题）
- train_qa_pair.json: 训练集
- test_qa_pair.json: 测试集
- test_keywords_pair.json: 测试集关键词
- test_qa_pair_verify.json: 评估输入
- train_data.json: SFT 训练数据
"""

import os
import json
import pickle
import hashlib
import random
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document

# 加载环境变量
load_dotenv()

# 命令行参数解析
parser = argparse.ArgumentParser(description='生成小米 SU7 RAG 项目的所有数据文件')
parser.add_argument('--force', '-f', action='store_true', help='强制重新生成所有文件（覆盖已存在的）')
parser.add_argument('--skip-expand', action='store_true', help='跳过扩展 QA 生成（加快速度）')
args = parser.parse_args()

# 导入核心模块
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
    MINMAL_CHUNK_SIZE
)
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.client.llm_chat_client import request_chat
from src.reranker.bge_m3_reranker import BGEM3ReRanker
from src.constant import bge_reranker_model_path, split_docs_path
from src.utils import merge_docs, post_processing


def step1_generate_raw_qa():
    """Step 1: 从切分后的文档生成原始 QA 对"""
    print("\n" + "="*60)
    print("Step 1: 生成原始 QA 对")
    print("="*60)
    
    # 检查切分文档是否存在
    if not os.path.exists(split_docs_path):
        print(f"❌ 错误：{split_docs_path} 不存在")
        print("请先运行: python build_index.py")
        return False
    
    # 检查是否已存在且非空
    if os.path.exists(QA_PATH) and os.path.getsize(QA_PATH) > 0 and not args.force:
        count = sum(1 for _ in open(QA_PATH))
        print(f"✅ {QA_PATH} 已存在 ({count} 条记录)，跳过...")
        return True
    
    # 加载切分后的文档
    with open(split_docs_path, "rb") as f:
        splitted_docs = pickle.load(f)
    
    print(f"📄 待处理文档数: {len(splitted_docs)}")
    
    # 生成原始 QA
    print(f"🚀 正在生成 QA -> {QA_PATH}")
    result = gen_qa(splitted_docs, CONTEXT_PROMPT_TPL, QA_PATH, expand=False, force=args.force)
    
    # 统计结果（使用 gen_qa 返回的结果）
    if result:
        count = len(result)
        print(f"✅ 生成完成，共 {count} 条记录")
        return True
    return False


def step2_generate_expanded_qa():
    """Step 2: 生成扩展 QA 对（同义问题）- 分批处理版本"""
    print("\n" + "="*60)
    print("Step 2: 生成扩展 QA 对")
    print("="*60)
    
    # 检查是否跳过
    if args.skip_expand:
        print("⏭️ 跳过扩展 QA 生成（--skip-expand）")
        # 如果文件不存在，创建空文件
        if not os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, "w") as f:
                pass
        return True
    
    if not os.path.exists(QA_PATH):
        print(f"❌ 错误：{QA_PATH} 不存在")
        return False
    
    # 检查是否已存在且非空
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0 and not args.force:
        count = sum(1 for _ in open(OUTPUT_PATH))
        print(f"✅ {OUTPUT_PATH} 已存在 ({count} 条记录)，跳过...")
        return True
    
    # 从原始 QA 中提取问题
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
                question = qa.get("question", "").strip()
                if question:
                    question_docs.append(Document(page_content=question, metadata={"unique_id": str(idx)}))
                    idx += 1
    
    print(f"📄 待扩展问题数: {len(question_docs)}")
    
    # 扩展QA使用适中的并发数（已分批处理，可适当提高）
    import src.gen_qa.run as gen_module
    original_workers = gen_module.MAX_WORKERS
    gen_module.MAX_WORKERS = 25  # 直接设置为25（参考原始QA的20并发成功经验）
    print(f"⚠️ 扩展QA已分批处理，设置并发数为 {gen_module.MAX_WORKERS}")
    
    # 分批处理配置
    BATCH_SIZE = 800  # 每批处理800条
    BASE_WAIT_SECONDS = 5  # 基础等待时间（秒）
    MAX_WAIT_SECONDS = 60  # 最大等待时间（秒）
    
    # 检查是否有已处理的记录（断点续传）
    processed_ids = set()
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    info = json.loads(line)
                    processed_ids.add(info["unique_id"])
                except:
                    pass
        print(f"📌 已处理 {len(processed_ids)} 条，继续剩余部分...")
    
    # 过滤已处理的问题
    remaining_docs = [doc for doc in question_docs if doc.metadata["unique_id"] not in processed_ids]
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
        
        # 自适应等待时间：随着批次增加逐渐延长（避免累计限流）
        wait_time = min(BASE_WAIT_SECONDS + batch_idx * 5, MAX_WAIT_SECONDS)
        
        print(f"\n🚀 处理批次 {batch_idx+1}/{total_batches} ({len(batch_docs)} 条)")
        
        # 生成扩展 QA（分批处理时强制重新生成，跳过内部断点续传检查）
        # 因为我们已经在外部做了断点续传过滤
        result = gen_qa(batch_docs, GENERALIZE_PROMPT_TPL, OUTPUT_PATH, expand=True, force=True)
        
        # 等待一段时间（最后一批不需要等待）
        if batch_idx < total_batches - 1:
            print(f"⏳ 等待 {wait_time} 秒...")
            time.sleep(wait_time)
    
    # 恢复原始并发数
    gen_module.MAX_WORKERS = original_workers
    
    # 统计最终结果
    if os.path.exists(OUTPUT_PATH):
        count = sum(1 for _ in open(OUTPUT_PATH))
        print(f"\n✅ 生成完成，共 {count} 条记录")
        return True
    return False


def step3_split_train_test():
    """Step 3: 切分训练集和测试集"""
    print("\n" + "="*60)
    print("Step 3: 切分训练集和测试集")
    print("="*60)
    
    # 检查是否已存在且非空
    if os.path.exists(TRAIN_PATH) and os.path.getsize(TRAIN_PATH) > 0 and \
       os.path.exists(TEST_PATH) and os.path.getsize(TEST_PATH) > 0 and not args.force:
        with open(TRAIN_PATH) as f:
            train_count = len(json.load(f))
        with open(TEST_PATH) as f:
            test_count = len(json.load(f))
        print(f"✅ {TRAIN_PATH} ({train_count} 条) 和 {TEST_PATH} ({test_count} 条) 已存在，跳过...")
        return True
    
    if not os.path.exists(QA_PATH):
        print(f"❌ 错误：{QA_PATH} 不存在")
        return False
    
    # 加载原始 QA
    qa_dict = {}
    with open(QA_PATH) as f:
        for line in f:
            info = json.loads(line)
            qa_dict[info["unique_id"]] = info
    
    # 加载扩展 QA（如果存在）
    expand_qa_pairs = {}
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    info = json.loads(line)
                    question = info["unique_id"]
                    expand_questions = info.get("raw_resp", "")
                    if expand_questions:
                        expand_questions = expand_questions.split("\n")
                        expand_questions = [q.strip() for q in expand_questions if q.strip()]
                        expand_qa_pairs[question] = expand_questions
                except:
                    continue
    
    print(f"📄 原始 QA 数: {len(qa_dict)}")
    print(f"📄 扩展问题映射数: {len(expand_qa_pairs)}")
    
    # 生成训练集和测试集
    train_qa_pairs = []
    test_qa_pairs = []
    skipped_count = 0
    
    for unique_id, info in qa_dict.items():
        try:
            resp = json.loads(info["raw_resp"])
        except:
            skipped_count += 1
            continue
        
        for qa in resp:
            # 检查必需字段是否存在
            if "question" not in qa or "answer" not in qa:
                skipped_count += 1
                continue
            
            question = qa["question"].strip()
            answer = qa["answer"].strip()
            
            # 检查空值
            if not question or not answer:
                skipped_count += 1
                continue
            
            if "无法准确" in answer or "未提及" in answer:
                skipped_count += 1
                continue
            
            # 合并原始问题和扩展问题
            all_questions = [question] + expand_qa_pairs.get(question, [])
            
            for query in all_questions:
                item = {
                    "unique_id": hashlib.md5(query.encode('utf-8')).hexdigest(),
                    "question": query,
                    "answer": answer
                }
                
                if random.random() < 0.9:
                    train_qa_pairs.append(item)
                else:
                    test_qa_pairs.append(item)
    
    # 打印跳过信息
    if skipped_count > 0:
        print(f"⚠️ 跳过 {skipped_count} 条无效或不完整的 QA")
    
    # 写入文件
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        random.shuffle(train_qa_pairs)
        json.dump(train_qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"✅ 训练集已写入: {TRAIN_PATH} ({len(train_qa_pairs)} 条)")
    
    with open(TEST_PATH, "w", encoding="utf-8") as f:
        random.shuffle(test_qa_pairs)
        json.dump(test_qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"✅ 测试集已写入: {TEST_PATH} ({len(test_qa_pairs)} 条)")
    
    return True


def step4_generate_keywords():
    """Step 4: 为测试集答案生成关键词标注"""
    print("\n" + "="*60)
    print("Step 4: 生成关键词标注")
    print("="*60)
    
    # 检查是否已存在且非空
    if os.path.exists(TEST_KEYWORDS_PATH) and os.path.getsize(TEST_KEYWORDS_PATH) > 0 and not args.force:
        print(f"✅ {TEST_KEYWORDS_PATH} 已存在，跳过...")
        return True
    
    if not os.path.exists(TEST_PATH):
        print(f"❌ 错误：{TEST_PATH} 不存在")
        return False
    
    # 加载测试集
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_qa_pairs = json.load(f)
    
    # 提取唯一答案
    unique_answers = list(set([item["answer"] for item in test_qa_pairs]))
    print(f"📄 待处理答案数: {len(unique_answers)}")
    
    # 生成关键词
    answer_docs = [Document(page_content=ans, metadata={"unique_id": str(i)}) for i, ans in enumerate(unique_answers)]
    result = gen_qa(answer_docs, KEYWORDS_PROMPT_TPL, TEST_KEYWORDS_PATH, expand=True, force=args.force)
    
    # 将关键词添加到测试集
    if result and os.path.exists(TEST_KEYWORDS_PATH):
        keywords_mapping = {}
        with open(TEST_KEYWORDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                info = json.loads(line)
                keywords = info["raw_resp"].split(",")
                keywords = [k.strip() for k in keywords if k.strip() not in ["无", "SU7"]]
                keywords_mapping[info["unique_id"]] = keywords
        
        # 更新测试集
        updated_test = []
        for item in test_qa_pairs:
            answer_idx = str(unique_answers.index(item["answer"]))
            item["keywords"] = keywords_mapping.get(answer_idx, [])
            updated_test.append(item)
        
        with open(TEST_PATH, "w", encoding="utf-8") as f:
            json.dump(updated_test, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 关键词标注完成，共 {len(result)} 条")
        return True
    return False


def step5_prepare_verify():
    """Step 5: 准备评估输入文件"""
    print("\n" + "="*60)
    print("Step 5: 准备评估输入")
    print("="*60)
    
    verify_path = "data/qa_pairs/test_qa_pair_verify.json"
    
    # 检查是否已存在
    if os.path.exists(verify_path) and os.path.getsize(verify_path) > 0 and not args.force:
        print(f"✅ {verify_path} 已存在，跳过...")
        return True
    
    if os.path.exists(TEST_PATH):
        import shutil
        shutil.copy(TEST_PATH, verify_path)
        print(f"✅ 评估输入已准备: {verify_path}")
        return True
    return False


def step6_generate_train_data():
    """Step 6: 生成 SFT 训练数据"""
    print("\n" + "="*60)
    print("Step 6: 生成 SFT 训练数据")
    print("="*60)
    
    output_path = "data/qa_pairs/train_data.json"
    
    # 检查是否已存在且非空
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and not args.force:
        count = sum(1 for _ in open(output_path))
        print(f"✅ {output_path} 已存在 ({count} 条)，跳过...")
        return True
    
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ 错误：{TRAIN_PATH} 不存在")
        return False
    
    # 加载检索器和重排器
    print("🔧 加载检索器和重排器...")
    bm25_retriever = BM25(docs=None, retrieve=True)
    milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
    bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_model_path)
    
    # 预热
    milvus_retriever.retrieve_topk("测试", topk=3)
    
    # 加载训练集
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_qa_pairs = json.load(f)
    
    print(f"📄 待处理训练样本数: {len(train_qa_pairs)}")
    
    # 生成 train_data.json
    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(train_qa_pairs):
            try:
                query = item["question"].strip()
                
                # 检索
                bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
                milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
                merged_docs = merge_docs(bm25_docs, milvus_docs)
                
                # 重排
                ranked_docs = bge_m3_reranker.rank(query, merged_docs, topk=5)
                
                # 生成答案
                context = "\n".join([str(idx+1) + "." + doc.page_content for idx, doc in enumerate(ranked_docs)])
                response = request_chat(query, context)
                
                # 保存
                info = {
                    "query": query,
                    "context": [doc.page_content for doc in ranked_docs],
                    "response": response,
                    "merged_docs": [doc.page_content for doc in merged_docs]
                }
                f.write(json.dumps(info, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️ 处理失败: {e}")
                continue
    
    print(f"✅ SFT 训练数据已生成: {output_path}")
    return True


def main():
    """主函数：按顺序执行所有步骤"""
    random.seed(42)
    
    steps = [
        step1_generate_raw_qa,
        step2_generate_expanded_qa,
        step3_split_train_test,
        step4_generate_keywords,
        step5_prepare_verify,
        step6_generate_train_data
    ]
    
    for step in steps:
        if not step():
            print(f"\n❌ 步骤 {step.__name__} 失败，终止执行")
            return
    
    # 验证结果
    print("\n" + "="*60)
    print("📊 生成结果验证")
    print("="*60)
    
    files = [
        ("qa_pair.json", QA_PATH),
        ("expand_qa_pair.json", OUTPUT_PATH),
        ("train_qa_pair.json", TRAIN_PATH),
        ("test_qa_pair.json", TEST_PATH),
        ("test_keywords_pair.json", TEST_KEYWORDS_PATH),
        ("test_qa_pair_verify.json", "data/qa_pairs/test_qa_pair_verify.json"),
        ("train_data.json", "data/qa_pairs/train_data.json")
    ]
    
    for name, path in files:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✅ {name}: {size} bytes")
        else:
            print(f"❌ {name}: 不存在")


if __name__ == "__main__":
    main()