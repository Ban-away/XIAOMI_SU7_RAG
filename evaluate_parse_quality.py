# -*- coding: utf-8 -*-
"""文档解析质量评估脚本 - 一行命令运行"""

import sys
import pickle
from src.parser.parse_evaluator import ParseEvaluator
from src.constant import base_dir


def main():
    raw_docs_path = base_dir + "data/processed_docs/raw_docs.pkl"
    split_docs_path = base_dir + "data/processed_docs/split_docs.pkl"
    
    try:
        with open(raw_docs_path, 'rb') as f:
            raw_docs = pickle.load(f)
        raw_text = '\n'.join([doc.page_content for doc in raw_docs])
        
        with open(split_docs_path, 'rb') as f:
            split_docs = pickle.load(f)
        
        chunks = []
        for doc in split_docs:
            chunks.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        clean_docs_path = base_dir + "data/processed_docs/clean_docs.pkl"
        try:
            with open(clean_docs_path, 'rb') as f:
                clean_docs = pickle.load(f)
            parsed_text = '\n'.join([doc.page_content for doc in clean_docs])
        except FileNotFoundError:
            parsed_text = raw_text
            print("[WARNING] 未找到清洗后的文档，使用原始文本进行评估")
        
        evaluator = ParseEvaluator()
        report = evaluator.generate_report(raw_text, parsed_text, chunks)
        
        print(report)
        
        report_path = base_dir + "data/parse_quality_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📝 报告已保存至: {report_path}")
        
    except FileNotFoundError as e:
        print(f"❌ 错误：未找到必要的文件 - {e}")
        print("请先运行 python build_index.py 生成解析数据")
        sys.exit(1)


if __name__ == "__main__":
    main()