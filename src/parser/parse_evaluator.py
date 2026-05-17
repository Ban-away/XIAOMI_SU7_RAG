# -*- coding: utf-8 -*-
"""文档解析质量评估模块"""

import os
import re
from typing import Dict, List


class ParseEvaluator:
    """文档解析质量评估器"""
    
    def __init__(self):
        self.title_pattern = re.compile(r'^[一二三四五六七八九十]+\s*[、.．]|^\d+[\.\)]')
        self.list_pattern = re.compile(r'^[\u2022•●▪-*+>]\s|^\d+[\.\)]\s')
        
    def evaluate_text_quality(self, raw_text: str, parsed_text: str) -> Dict[str, float]:
        """评估文本解析质量"""
        results = {}
        
        raw_chars = len(raw_text.strip())
        parsed_chars = len(parsed_text.strip())
        results['text_retention_rate'] = min(parsed_chars / max(raw_chars, 1), 1.0)
        
        lines = parsed_text.split('\n')
        blank_lines = sum(1 for line in lines if line.strip() == '')
        results['blank_line_ratio'] = blank_lines / max(len(lines), 1)
        
        raw_titles = len(self.title_pattern.findall(raw_text))
        parsed_titles = len(self.title_pattern.findall(parsed_text))
        results['title_retention'] = parsed_titles / max(raw_titles, 1)
        
        raw_lists = len(self.list_pattern.findall(raw_text))
        parsed_lists = len(self.list_pattern.findall(parsed_text))
        results['list_retention'] = parsed_lists / max(raw_lists, 1)
        
        abnormal_chars = self._count_abnormal_chars(parsed_text)
        results['abnormal_char_rate'] = abnormal_chars / max(len(parsed_text), 1)
        
        results['overall_score'] = (
            0.3 * results['text_retention_rate'] +
            0.2 * (1 - results['blank_line_ratio']) +
            0.2 * results['title_retention'] +
            0.15 * results['list_retention'] +
            0.15 * (1 - results['abnormal_char_rate'])
        )
        
        return results
    
    def _count_abnormal_chars(self, text: str) -> int:
        """统计异常字符数量"""
        abnormal_count = 0
        for char in text:
            if '\uFFFD' in char:
                abnormal_count += 1
            if ord(char) < 32 and char not in '\n\r\t':
                abnormal_count += 1
        return abnormal_count
    
    def evaluate_chunk_quality(self, chunks: List[dict]) -> Dict[str, float]:
        """评估切分质量"""
        if not chunks:
            return {'chunk_count': 0, 'avg_chunk_length': 0, 'quality_score': 0.0}
            
        results = {}
        results['chunk_count'] = len(chunks)
        
        total_length = sum(len(chunk.get('content', '')) for chunk in chunks)
        results['avg_chunk_length'] = total_length / len(chunks)
        
        lengths = [len(chunk.get('content', '')) for chunk in chunks]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        results['length_std'] = variance ** 0.5
        
        context_scores = []
        for i in range(len(chunks) - 1):
            current = chunks[i]['content'][:100]
            next_chunk = chunks[i+1]['content'][:100]
            overlap = len(set(current) & set(next_chunk)) / min(len(current), len(next_chunk))
            context_scores.append(overlap)
        
        results['avg_context_similarity'] = sum(context_scores) / max(len(context_scores), 1)
        
        ideal_length = 500
        length_score = max(0, 1 - abs(results['avg_chunk_length'] - ideal_length) / ideal_length)
        uniformity_score = 1 - (results['length_std'] / results['avg_chunk_length']) if results['avg_chunk_length'] > 0 else 0
        
        results['quality_score'] = (
            0.4 * length_score +
            0.3 * uniformity_score +
            0.3 * results['avg_context_similarity']
        )
        
        return results
    
    def generate_report(self, raw_text: str, parsed_text: str, chunks: List[dict]) -> str:
        """生成综合评估报告"""
        text_quality = self.evaluate_text_quality(raw_text, parsed_text)
        chunk_quality = self.evaluate_chunk_quality(chunks)
        
        report = [
            "="*60,
            "📄 文档解析质量评估报告",
            "="*60,
            "",
            "【文本解析质量】",
            f"  ├─ 文本保留率: {text_quality['text_retention_rate']:.2%}",
            f"  ├─ 空白行比例: {text_quality['blank_line_ratio']:.2%}",
            f"  ├─ 标题保留率: {text_quality['title_retention']:.2%}",
            f"  ├─ 列表保留率: {text_quality['list_retention']:.2%}",
            f"  ├─ 异常字符率: {text_quality['abnormal_char_rate']:.2%}",
            f"  └─ 综合评分: {text_quality['overall_score']:.2%}",
            "",
            "【文档切分质量】",
            f"  ├─ 切分数量: {chunk_quality['chunk_count']}",
            f"  ├─ 平均长度: {chunk_quality['avg_chunk_length']:.0f} 字符",
            f"  ├─ 长度标准差: {chunk_quality['length_std']:.0f}",
            f"  ├─ 上下文相关性: {chunk_quality['avg_context_similarity']:.2%}",
            f"  └─ 切分质量评分: {chunk_quality['quality_score']:.2%}",
            "",
            "="*60,
            f"📊 最终解析准确率: {(text_quality['overall_score'] * 0.6 + chunk_quality['quality_score'] * 0.4):.2%}",
            "="*60
        ]
        
        return '\n'.join(report)


if __name__ == "__main__":
    evaluator = ParseEvaluator()
    
    raw_text = """第一章 车辆介绍
1.1 外观特征
• 车身尺寸
• 颜色选项

第二章 操作指南
2.1 启动车辆
请踩下刹车踏板并按下启动按钮。
"""
    
    parsed_text = """第一章 车辆介绍

1.1 外观特征
• 车身尺寸
• 颜色选项

第二章 操作指南

2.1 启动车辆
请踩下刹车踏板并按下启动按钮。
"""
    
    chunks = [
        {'content': '第一章 车辆介绍\n1.1 外观特征\n• 车身尺寸\n• 颜色选项', 'metadata': {'page': 1}},
        {'content': '第二章 操作指南\n2.1 启动车辆\n请踩下刹车踏板并按下启动按钮。', 'metadata': {'page': 2}}
    ]
    
    report = evaluator.generate_report(raw_text, parsed_text, chunks)
    print(report)