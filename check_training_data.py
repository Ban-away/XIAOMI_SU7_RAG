# -*- coding: utf-8 -*-
"""训练数据质量检查脚本：识别和过滤低质量样本"""

import json
import re

# 定义低质量样本模式
LOW_QUALITY_PATTERNS = [
    # 无意义输入
    r'^\s*(哦|啊|嗯|呢|吧|吗|哈|嗨|嘿|喂|好|行|可以)\s*$',
    r'^\s*$',  # 空字符串
    
    # 不完整问题
    r'^\s*[你我他她它]的\s*$',
    r'^\s*(给我|我要|请给|想让)\s*$',
    r'^\s*(看|听|读|唱|背)\s*$',
    
    # 无关主题关键词
    r'(诗|诗歌|背诗|古诗|唐诗|宋词)',
    r'(还珠格格|电视剧|电影|明星|演员)',
    r'(诗经|论语|历史|文学|小说)',
    r'(门|路口|左转|右转|导航|怎么走)',
    r'(放歌|听歌|音乐|唱歌|播放)',
    r'(女儿|儿子|妈妈|爸爸|家庭|家人)',
    r'(天气|新闻|时间|日期)',
    r'(股票|基金|财经|价格)',
    r'(游戏|软件|手机|电脑)',
]

# 定义无效答案模式
INVALID_ANSWER_PATTERNS = [
    r'^无答案\s*$',
    r'^不知道\s*$',
    r'^不清楚\s*$',
    r'^无法回答\s*$',
]


def is_low_quality_question(question):
    """检查问题是否为低质量"""
    if not question or not question.strip():
        return True, "空问题"
    
    question = question.strip()
    
    # 检查问题长度（太短或太长）
    if len(question) < 3:
        return True, "问题过短"
    
    if len(question) > 500:
        return True, "问题过长"
    
    # 检查模式匹配
    for pattern in LOW_QUALITY_PATTERNS:
        if re.search(pattern, question):
            return True, f"匹配模式: {pattern}"
    
    return False, "正常"


def is_invalid_answer(answer):
    """检查答案是否无效"""
    if not answer or not answer.strip():
        return True, "空答案"
    
    answer = answer.strip()
    
    for pattern in INVALID_ANSWER_PATTERNS:
        if re.search(pattern, answer):
            return True, f"无效答案类型"
    
    return False, "正常"


def check_training_data(file_path):
    """检查训练数据质量"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    low_quality_samples = []
    invalid_answer_samples = []
    total_samples = len(data)
    
    for idx, item in enumerate(data):
        question = item.get('question', '')
        answer = item.get('answer', '')
        keywords = item.get('keywords', [])
        
        # 检查问题质量
        is_low_question, reason_question = is_low_quality_question(question)
        if is_low_question:
            low_quality_samples.append({
                'index': idx,
                'unique_id': item.get('unique_id', ''),
                'question': question,
                'answer': answer,
                'reason': reason_question,
                'type': '低质量问题'
            })
        
        # 检查答案质量（非"无答案"样本）
        if answer != '无答案':
            is_invalid, reason_answer = is_invalid_answer(answer)
            if is_invalid:
                invalid_answer_samples.append({
                    'index': idx,
                    'unique_id': item.get('unique_id', ''),
                    'question': question,
                    'answer': answer,
                    'reason': reason_answer,
                    'type': '无效答案'
                })
    
    # 输出报告
    print("=" * 60)
    print("          训练数据质量检查报告")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"低质量问题数: {len(low_quality_samples)}")
    print(f"无效答案数: {len(invalid_answer_samples)}")
    print(f"问题质量占比: {(len(low_quality_samples)/total_samples)*100:.2f}%")
    print(f"答案质量占比: {(len(invalid_answer_samples)/total_samples)*100:.2f}%")
    print("=" * 60)
    
    # 打印低质量样本详情
    if low_quality_samples:
        print("\n【低质量问题样本】")
        print("-" * 60)
        for i, sample in enumerate(low_quality_samples[:20]):  # 只显示前20个
            print(f"序号: {sample['index']}")
            print(f"原因: {sample['reason']}")
            print(f"问题: {sample['question']}")
            print(f"答案: {sample['answer']}")
            print("-" * 60)
            if i >= 19 and len(low_quality_samples) > 20:
                print(f"... 还有 {len(low_quality_samples) - 20} 个低质量样本")
    
    # 打印无效答案样本详情
    if invalid_answer_samples:
        print("\n【无效答案样本】")
        print("-" * 60)
        for i, sample in enumerate(invalid_answer_samples[:20]):
            print(f"序号: {sample['index']}")
            print(f"原因: {sample['reason']}")
            print(f"问题: {sample['question']}")
            print(f"答案: {sample['answer']}")
            print("-" * 60)
            if i >= 19 and len(invalid_answer_samples) > 20:
                print(f"... 还有 {len(invalid_answer_samples) - 20} 个无效答案样本")
    
    return low_quality_samples, invalid_answer_samples


def filter_low_quality_data(input_file, output_file):
    """过滤低质量样本并保存"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = []
    removed_count = 0
    
    for item in data:
        question = item.get('question', '')
        is_low, _ = is_low_quality_question(question)
        
        # 保留条件：问题质量正常，或者是"无答案"样本（用于训练拒绝回答）
        if not is_low:
            filtered_data.append(item)
        else:
            removed_count += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    
    print("\n" + "=" * 60)
    print("            过滤操作完成")
    print("=" * 60)
    print(f"原样本数: {len(data)}")
    print(f"过滤后样本数: {len(filtered_data)}")
    print(f"移除样本数: {removed_count}")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    import sys
    
    input_path = "data/qa_pairs/train_qa_pair.json"
    output_path = "data/qa_pairs/train_qa_pair_filtered.json"
    
    # 检查参数
    if len(sys.argv) > 1 and sys.argv[1] == "--filter":
        # 直接执行过滤
        filter_low_quality_data(input_path, output_path)
    else:
        # 先检查质量
        print("正在检查训练数据质量...\n")
        low_quality, invalid_answer = check_training_data(input_path)
        
        # 询问是否过滤
        if low_quality:
            print("\n是否过滤低质量样本? (y/n)")
            choice = input().strip().lower()
            if choice == 'y':
                filter_low_quality_data(input_path, output_path)
                print(f"\n过滤后的数据已保存到: {output_path}")
            else:
                print("取消过滤操作")
        else:
            print("\n✓ 未发现低质量样本，无需过滤")