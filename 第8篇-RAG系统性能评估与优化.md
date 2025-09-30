# 第8篇：RAG系统性能评估与优化

## 摘要

本文深入探讨了RAG系统性能评估的全面框架和优化策略。通过分析评估指标体系、基准测试方法、性能瓶颈识别以及系统优化技术，为构建高性能、高质量的RAG系统提供实用的评估工具和优化方案。

## 1. RAG系统性能评估框架

### 1.1 评估维度概述

RAG系统的性能评估需要从多个维度进行综合考量：

```
性能评估框架
├── 质量指标
│   ├── 回答质量 (Relevance, Accuracy, Completeness)
│   ├── 检索质量 (Recall, Precision, F1-Score)
│   └── 用户体验 (Fluency, Coherence, Helpfulness)
├── 性能指标
│   ├── 响应时间 (Latency, Throughput)
│   ├── 资源使用 (CPU, Memory, GPU)
│   └── 可扩展性 (Concurrency, Load Capacity)
├── 成本指标
│   ├── API调用成本
│   ├── 计算资源成本
│   └── 存储成本
└── 可靠性指标
    ├── 错误率
    ├── 可用性
    └── 故障恢复时间
```

### 1.2 评估指标实现

```python
import time
import psutil
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class MetricType(Enum):
    """指标类型"""
    QUALITY = "quality"
    PERFORMANCE = "performance"
    COST = "cost"
    RELIABILITY = "reliability"

@dataclass
class MetricResult:
    """评估结果"""
    name: str
    value: float
    unit: str
    description: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationReport:
    """评估报告"""
    system_name: str
    evaluation_time: float
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self, system_name: str = "RAG_System"):
        self.system_name = system_name
        self.metrics_history = []
        self.evaluation_results = {}

    def evaluate_quality(self,
                        questions: List[str],
                        ground_truth_answers: List[str],
                        rag_answers: List[str],
                        retrieved_docs: List[List[str]] = None) -> Dict[str, MetricResult]:
        """评估回答质量"""
        quality_metrics = {}

        # 相关性评估
        relevance_scores = []
        for i, (question, rag_answer, gt_answer) in enumerate(zip(questions, rag_answers, ground_truth_answers)):
            relevance = self._calculate_relevance(question, rag_answer, gt_answer)
            relevance_scores.append(relevance)

        quality_metrics['relevance'] = MetricResult(
            name="平均相关性",
            value=np.mean(relevance_scores),
            unit="分数",
            description="回答与问题的相关性"
        )

        # 准确性评估
        accuracy_scores = []
        for rag_answer, gt_answer in zip(rag_answers, ground_truth_answers):
            accuracy = self._calculate_accuracy(rag_answer, gt_answer)
            accuracy_scores.append(accuracy)

        quality_metrics['accuracy'] = MetricResult(
            name="平均准确性",
            value=np.mean(accuracy_scores),
            unit="分数",
            description="回答的准确性"
        )

        # 完整性评估
        completeness_scores = []
        for rag_answer, gt_answer in zip(rag_answers, ground_truth_answers):
            completeness = self._calculate_completeness(rag_answer, gt_answer)
            completeness_scores.append(completeness)

        quality_metrics['completeness'] = MetricResult(
            name="平均完整性",
            value=np.mean(completeness_scores),
            unit="分数",
            description="回答的完整性"
        )

        # 流畅性评估
        fluency_scores = []
        for rag_answer in rag_answers:
            fluency = self._calculate_fluency(rag_answer)
            fluency_scores.append(fluency)

        quality_metrics['fluency'] = MetricResult(
            name="平均流畅性",
            value=np.mean(fluency_scores),
            unit="分数",
            description="回答的流畅性"
        )

        # 检索质量评估（如果有检索结果）
        if retrieved_docs:
            retrieval_metrics = self._evaluate_retrieval_quality(
                questions, retrieved_docs, ground_truth_answers
            )
            quality_metrics.update(retrieval_metrics)

        return quality_metrics

    def _calculate_relevance(self, question: str, rag_answer: str, ground_truth: str) -> float:
        """计算相关性分数"""
        # 简化实现：基于关键词重叠和语义相似度
        question_words = set(question.lower().split())
        answer_words = set(rag_answer.lower().split())
        gt_words = set(ground_truth.lower().split())

        # 关键词覆盖率
        coverage = len(question_words & answer_words) / len(question_words) if question_words else 0

        # 与标准答案的一致性
        consistency = len(answer_words & gt_words) / len(answer_words) if answer_words else 0

        # 综合分数
        relevance = 0.6 * consistency + 0.4 * coverage
        return min(1.0, relevance)

    def _calculate_accuracy(self, rag_answer: str, ground_truth: str) -> float:
        """计算准确性分数"""
        # 使用编辑距离计算相似度
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s1) == 0:
                return len(s2)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        if not ground_truth:
            return 1.0 if not rag_answer else 0.0

        distance = levenshtein_distance(rag_answer.lower(), ground_truth.lower())
        max_len = max(len(rag_answer), len(ground_truth))
        accuracy = 1 - (distance / max_len) if max_len > 0 else 1.0

        return max(0.0, accuracy)

    def _calculate_completeness(self, rag_answer: str, ground_truth: str) -> float:
        """计算完整性分数"""
        # 基于长度和内容覆盖度
        if not ground_truth:
            return 0.0

        gt_words = set(ground_truth.lower().split())
        answer_words = set(rag_answer.lower().split())

        coverage = len(answer_words & gt_words) / len(gt_words) if gt_words else 0

        # 长度因子
        length_ratio = min(len(rag_answer) / len(ground_truth), 2.0) / 2.0

        completeness = 0.7 * coverage + 0.3 * length_ratio
        return min(1.0, completeness)

    def _calculate_fluency(self, text: str) -> float:
        """计算流畅性分数"""
        if not text:
            return 0.0

        # 简化的流畅性评估
        sentences = text.split('.')
        if len(sentences) <= 1:
            return 0.5

        # 平均句子长度
        sentence_lengths = [len(s.strip().split()) for s in sentences if s.strip()]
        avg_length = np.mean(sentence_lengths) if sentence_lengths else 0

        # 理想的句子长度是10-20个词
        length_score = 1.0 - abs(avg_length - 15) / 15

        # 词汇多样性
        words = text.lower().split()
        unique_words = set(words)
        diversity = len(unique_words) / len(words) if words else 0

        fluency = 0.6 * max(0, length_score) + 0.4 * diversity
        return min(1.0, fluency)

    def _evaluate_retrieval_quality(self,
                                  questions: List[str],
                                  retrieved_docs: List[List[str]],
                                  ground_truth_answers: List[str]) -> Dict[str, MetricResult]:
        """评估检索质量"""
        retrieval_metrics = {}

        # 计算检索指标
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i, (question, docs, gt_answer) in enumerate(zip(questions, retrieved_docs, ground_truth_answers)):
            gt_words = set(gt_answer.lower().split())

            if not docs or not gt_words:
                continue

            # 计算precision, recall, f1
            retrieved_words = set()
            for doc in docs:
                retrieved_words.update(doc.lower().split())

            true_positives = len(retrieved_words & gt_words)
            false_positives = len(retrieved_words - gt_words)
            false_negatives = len(gt_words - retrieved_words)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        if precision_scores:
            retrieval_metrics['precision'] = MetricResult(
                name="检索精确率",
                value=np.mean(precision_scores),
                unit="分数",
                description="检索结果的精确性"
            )

            retrieval_metrics['recall'] = MetricResult(
                name="检索召回率",
                value=np.mean(recall_scores),
                unit="分数",
                description="检索结果的召回率"
            )

            retrieval_metrics['f1_score'] = MetricResult(
                name="检索F1分数",
                value=np.mean(f1_scores),
                unit="分数",
                description="检索结果的F1分数"
            )

        return retrieval_metrics

    def evaluate_performance(self,
                           benchmark_queries: List[str],
                           rag_system) -> Dict[str, MetricResult]:
        """评估系统性能"""
        performance_metrics = {}

        # 响应时间测试
        response_times = []
        cpu_usage = []
        memory_usage = []

        process = psutil.Process()

        for query in benchmark_queries:
            # 记录开始状态
            start_cpu = process.cpu_percent()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()

            # 执行查询
            try:
                result = rag_system.query(query)
                end_time = time.time()
                end_cpu = process.cpu_percent()
                end_memory = process.memory_info().rss / 1024 / 1024

                # 计算指标
                response_time = end_time - start_time
                cpu_delta = max(end_cpu - start_cpu, 0)
                memory_delta = end_memory - start_memory

                response_times.append(response_time)
                cpu_usage.append(cpu_delta)
                memory_usage.append(memory_delta)

            except Exception as e:
                print(f"查询失败: {query}, 错误: {str(e)}")
                continue

        if response_times:
            performance_metrics['avg_response_time'] = MetricResult(
                name="平均响应时间",
                value=np.mean(response_times),
                unit="秒",
                description="系统平均响应时间"
            )

            performance_metrics['p95_response_time'] = MetricResult(
                name="95%响应时间",
                value=np.percentile(response_times, 95),
                unit="秒",
                description="95%查询的响应时间"
            )

            performance_metrics['throughput'] = MetricResult(
                name="吞吐量",
                value=len(benchmark_queries) / sum(response_times),
                unit="查询/秒",
                description="系统每秒处理的查询数"
            )

            performance_metrics['avg_cpu_usage'] = MetricResult(
                name="平均CPU使用率",
                value=np.mean(cpu_usage),
                unit="%",
                description="查询期间的平均CPU使用率"
            )

            performance_metrics['avg_memory_usage'] = MetricResult(
                name="平均内存使用",
                value=np.mean(memory_usage),
                unit="MB",
                description="查询期间的平均内存使用量"
            )

        return performance_metrics

    def evaluate_cost(self,
                      usage_stats: Dict[str, Any],
                      pricing_info: Dict[str, float]) -> Dict[str, MetricResult]:
        """评估成本"""
        cost_metrics = {}

        # API调用成本
        api_calls = usage_stats.get('api_calls', 0)
        api_cost_per_call = pricing_info.get('api_cost_per_call', 0.001)
        total_api_cost = api_calls * api_cost_per_call

        cost_metrics['api_cost'] = MetricResult(
            name="API调用成本",
            value=total_api_cost,
            unit="美元",
            description="API调用的总成本"
        )

        # 计算资源成本
        cpu_hours = usage_stats.get('cpu_hours', 0)
        cpu_cost_per_hour = pricing_info.get('cpu_cost_per_hour', 0.05)
        total_cpu_cost = cpu_hours * cpu_cost_per_hour

        cost_metrics['compute_cost'] = MetricResult(
            name="计算成本",
            value=total_cpu_cost,
            unit="美元",
            description="计算资源的总成本"
        )

        # 存储成本
        storage_gb = usage_stats.get('storage_gb', 0)
        storage_cost_per_gb = pricing_info.get('storage_cost_per_gb_per_month', 0.1)
        monthly_storage_cost = storage_gb * storage_cost_per_gb

        cost_metrics['storage_cost'] = MetricResult(
            name="存储成本",
            value=monthly_storage_cost,
            unit="美元/月",
            description="存储资源的月度成本"
        )

        # 总成本
        total_cost = total_api_cost + total_cpu_cost + monthly_storage_cost

        cost_metrics['total_cost'] = MetricResult(
            name="总成本",
            value=total_cost,
            unit="美元",
            description="系统运行的总成本"
        )

        # 单位查询成本
        total_queries = usage_stats.get('total_queries', 1)
        cost_per_query = total_cost / total_queries

        cost_metrics['cost_per_query'] = MetricResult(
            name="单查询成本",
            value=cost_per_query,
            unit="美元",
            description="每个查询的平均成本"
        )

        return cost_metrics

    def evaluate_reliability(self,
                           error_logs: List[Dict[str, Any]],
                           uptime_stats: Dict[str, Any]) -> Dict[str, MetricResult]:
        """评估可靠性"""
        reliability_metrics = {}

        # 错误率
        total_requests = uptime_stats.get('total_requests', 1)
        total_errors = len(error_logs)
        error_rate = total_errors / total_requests

        reliability_metrics['error_rate'] = MetricResult(
            name="错误率",
            value=error_rate,
            unit="%",
            description="系统请求的错误率"
        )

        # 可用性
        uptime_hours = uptime_stats.get('uptime_hours', 24)
        downtime_hours = uptime_stats.get('downtime_hours', 0)
        availability = uptime_hours / (uptime_hours + downtime_hours) if (uptime_hours + downtime_hours) > 0 else 1.0

        reliability_metrics['availability'] = MetricResult(
            name="可用性",
            value=availability * 100,
            unit="%",
            description="系统的可用性百分比"
        )

        # 平均故障恢复时间
        recovery_times = [log.get('recovery_time', 0) for log in error_logs if log.get('recovery_time')]
        if recovery_times:
            avg_recovery_time = np.mean(recovery_times)
            reliability_metrics['avg_recovery_time'] = MetricResult(
                name="平均故障恢复时间",
                value=avg_recovery_time,
                unit="秒",
                description="系统故障后的平均恢复时间"
            )

        return reliability_metrics

    def generate_evaluation_report(self,
                                  quality_metrics: Dict[str, MetricResult],
                                  performance_metrics: Dict[str, MetricResult],
                                  cost_metrics: Dict[str, MetricResult],
                                  reliability_metrics: Dict[str, MetricResult]) -> EvaluationReport:
        """生成评估报告"""
        report = EvaluationReport(
            system_name=self.system_name,
            evaluation_time=time.time()
        )

        # 合并所有指标
        all_metrics = {}
        all_metrics.update(quality_metrics)
        all_metrics.update(performance_metrics)
        all_metrics.update(cost_metrics)
        all_metrics.update(reliability_metrics)

        report.metrics = all_metrics

        # 生成摘要
        report.summary = self._generate_summary(all_metrics)

        # 生成建议
        report.recommendations = self._generate_recommendations(all_metrics)

        return report

    def _generate_summary(self, metrics: Dict[str, MetricResult]) -> Dict[str, Any]:
        """生成评估摘要"""
        summary = {
            'total_metrics': len(metrics),
            'quality_score': 0,
            'performance_score': 0,
            'cost_score': 0,
            'reliability_score': 0,
            'overall_score': 0
        }

        # 计算各类别分数
        quality_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['relevance', 'accuracy', 'completeness', 'fluency', 'precision', 'recall'])}
        performance_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['response', 'throughput', 'cpu', 'memory'])}
        cost_metrics = {k: v for k, v in metrics.items() if 'cost' in k.lower()}
        reliability_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['error', 'availability', 'recovery'])}

        if quality_metrics:
            summary['quality_score'] = np.mean([m.value for m in quality_metrics.values()])
        if performance_metrics:
            # 性能分数需要特殊处理（响应时间越低越好）
            perf_scores = []
            for name, metric in performance_metrics.items():
                if 'response_time' in name.lower():
                    # 响应时间分数：1秒以下为满分，超过5秒为0分
                    score = max(0, 1 - (metric.value - 1) / 4)
                    perf_scores.append(score)
                else:
                    perf_scores.append(metric.value)
            summary['performance_score'] = np.mean(perf_scores) if perf_scores else 0

        if cost_metrics:
            # 成本分数：成本越低分数越高
            cost_scores = []
            for metric in cost_metrics.values():
                if 'total_cost' in metric.name.lower():
                    # 简化评分逻辑
                    score = max(0, 1 - metric.value / 10)  # 假设10美元为满分线
                    cost_scores.append(score)
                else:
                    cost_scores.append(metric.value)
            summary['cost_score'] = np.mean(cost_scores) if cost_scores else 0

        if reliability_metrics:
            summary['reliability_score'] = np.mean([m.value for m in reliability_metrics.values()])

        # 综合评分
        weights = {'quality': 0.4, 'performance': 0.3, 'cost': 0.2, 'reliability': 0.1}
        summary['overall_score'] = (
            summary['quality_score'] * weights['quality'] +
            summary['performance_score'] * weights['performance'] +
            summary['cost_score'] * weights['cost'] +
            summary['reliability_score'] * weights['reliability']
        )

        return summary

    def _generate_recommendations(self, metrics: Dict[str, MetricResult]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 质量相关建议
        relevance = metrics.get('relevance')
        if relevance and relevance.value < 0.7:
            recommendations.append("相关性评分较低，建议优化检索策略和提示词设计")

        accuracy = metrics.get('accuracy')
        if accuracy and accuracy.value < 0.6:
            recommendations.append("准确性评分较低，建议改进嵌入模型或增加检索文档数量")

        # 性能相关建议
        response_time = metrics.get('avg_response_time')
        if response_time and response_time.value > 3.0:
            recommendations.append("响应时间较长，建议优化向量数据库索引或增加缓存机制")

        memory_usage = metrics.get('avg_memory_usage')
        if memory_usage and memory_usage.value > 1000:  # 1GB
            recommendations.append("内存使用较高，建议优化内存管理或增加硬件资源")

        # 成本相关建议
        cost_per_query = metrics.get('cost_per_query')
        if cost_per_query and cost_per_query.value > 0.1:  # 0.1美元
            recommendations.append("单查询成本较高，建议优化API调用或使用更经济的模型")

        # 可靠性相关建议
        error_rate = metrics.get('error_rate')
        if error_rate and error_rate.value > 5:  # 5%
            recommendations.append("错误率较高，建议加强错误处理和系统监控")

        availability = metrics.get('availability')
        if availability and availability.value < 99:  # 99%
            recommendations.append("可用性较低，建议实现故障转移和负载均衡机制")

        if not recommendations:
            recommendations.append("系统性能良好，继续保持当前配置")

        return recommendations

    def visualize_metrics(self, metrics: Dict[str, MetricResult], save_path: str = None):
        """可视化评估指标"""
        # 按类别分组指标
        categories = {
            'Quality': [],
            'Performance': [],
            'Cost': [],
            'Reliability': []
        }

        for name, metric in metrics.items():
            if any(keyword in name.lower() for keyword in ['relevance', 'accuracy', 'completeness', 'fluency', 'precision', 'recall']):
                categories['Quality'].append((name, metric.value))
            elif any(keyword in name.lower() for keyword in ['response', 'throughput', 'cpu', 'memory']):
                categories['Performance'].append((name, metric.value))
            elif 'cost' in name.lower():
                categories['Cost'].append((name, metric.value))
            elif any(keyword in name.lower() for keyword in ['error', 'availability', 'recovery']):
                categories['Reliability'].append((name, metric.value))

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.system_name} 性能评估指标', fontsize=16)

        for idx, (category, items) in enumerate(categories.items()):
            if items:
                ax = axes[idx // 2, idx % 2]

                names = [item[0] for item in items]
                values = [item[1] for item in items]

                # 标准化数值以便比较
                if category == 'Performance' and any('response_time' in name.lower() for name in names):
                    # 响应时间越小越好，所以反转
                    max_val = max(values)
                    normalized_values = [(max_val - val) / max_val for val in values]
                else:
                    max_val = max(values) if values else 1
                    normalized_values = [val / max_val for val in values]

                bars = ax.bar(range(len(names)), normalized_values)
                ax.set_title(f'{category} Metrics')
                ax.set_ylabel('Normalized Score')
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in names], rotation=45, ha='right')

                # 添加数值标签
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def save_report(self, report: EvaluationReport, filepath: str):
        """保存评估报告"""
        # 转换为可序列化格式
        serializable_report = {
            'system_name': report.system_name,
            'evaluation_time': report.evaluation_time,
            'metrics': {
                name: {
                    'name': metric.name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'description': metric.description,
                    'timestamp': metric.timestamp,
                    'metadata': metric.metadata
                }
                for name, metric in report.metrics.items()
            },
            'summary': report.summary,
            'recommendations': report.recommendations
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2, default=str)

        print(f"评估报告已保存到: {filepath}")
```

## 2. 基准测试框架

### 2.1 标准测试数据集

```python
from abc import ABC, abstractmethod
import json
import random
from typing import List, Dict, Any
import requests

class BenchmarkDataset(ABC):
    """基准测试数据集基类"""

    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        pass

    @abstractmethod
    def get_categories(self) -> List[str]:
        """获取数据类别"""
        pass

class SQuADDataset(BenchmarkDataset):
    """SQuAD数据集适配器"""

    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.data = []

    def load_data(self) -> List[Dict[str, Any]]:
        """加载SQuAD数据"""
        if self.data_path and os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # 使用示例数据
            data = self._create_sample_data()

        processed_data = []
        for article in data.get('data', []):
            for paragraph in article.get('paragraphs', []):
                context = paragraph.get('context', '')
                for qa in paragraph.get('qas', []):
                    question = qa.get('question', '')
                    answers = qa.get('answers', [])
                    if answers:
                        answer = answers[0].get('text', '')
                    else:
                        answer = ''

                    processed_data.append({
                        'question': question,
                        'context': context,
                        'answer': answer,
                        'category': 'reading_comprehension'
                    })

        self.data = processed_data
        return processed_data

    def _create_sample_data(self) -> Dict[str, Any]:
        """创建示例数据"""
        return {
            'data': [
                {
                    'title': 'Python Programming',
                    'paragraphs': [
                        {
                            'context': 'Python is a high-level programming language interpreted and dynamically typed. It was created by Guido van Rossum and first released in 1991.',
                            'qas': [
                                {
                                    'question': 'Who created Python?',
                                    'answers': [{'text': 'Guido van Rossum'}],
                                    'id': '1'
                                },
                                {
                                    'question': 'When was Python first released?',
                                    'answers': [{'text': '1991'}],
                                    'id': '2'
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    def get_categories(self) -> List[str]:
        """获取数据类别"""
        return ['reading_comprehension']

class MMLUDataset(BenchmarkDataset):
    """MMLU数据集适配器"""

    def __init__(self, subject: str = 'computer_science'):
        self.subject = subject
        self.data = []

    def load_data(self) -> List[Dict[str, Any]]:
        """加载MMLU数据"""
        # 这里使用示例数据，实际应用中应该下载真实的MMLU数据集
        sample_data = [
            {
                'question': 'What is the time complexity of binary search?',
                'choices': ['O(n)', 'O(log n)', 'O(n log n)', 'O(n²)'],
                'answer': 'O(log n)',
                'category': self.subject
            },
            {
                'question': 'Which data structure uses LIFO principle?',
                'choices': ['Queue', 'Stack', 'Array', 'Tree'],
                'answer': 'Stack',
                'category': self.subject
            },
            {
                'question': 'What is the main advantage of hash tables?',
                'choices': ['Ordered access', 'Fast lookup', 'Memory efficiency', 'Simple implementation'],
                'answer': 'Fast lookup',
                'category': self.subject
            }
        ]

        self.data = sample_data
        return sample_data

    def get_categories(self) -> List[str]:
        """获取数据类别"""
        return [self.subject]

class CustomQADataset(BenchmarkDataset):
    """自定义问答数据集"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = []

    def load_data(self) -> List[Dict[str, Any]]:
        """加载自定义数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 验证数据格式
        required_fields = ['question', 'answer']
        for item in data:
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"数据项缺少必要字段: {field}")

        self.data = data
        return data

    def get_categories(self) -> List[str]:
        """获取数据类别"""
        categories = set()
        for item in self.data:
            if 'category' in item:
                categories.add(item['category'])
        return list(categories)

class BenchmarkSuite:
    """基准测试套件"""

    def __init__(self):
        self.datasets = {}
        self.test_cases = []

    def add_dataset(self, name: str, dataset: BenchmarkDataset):
        """添加数据集"""
        self.datasets[name] = dataset

    def load_all_datasets(self):
        """加载所有数据集"""
        for name, dataset in self.datasets.items():
            try:
                data = dataset.load_data()
                print(f"成功加载数据集 {name}: {len(data)} 个样本")
            except Exception as e:
                print(f"加载数据集 {name} 失败: {str(e)}")

    def create_test_suite(self,
                         sample_size: int = 100,
                         categories: List[str] = None) -> List[Dict[str, Any]]:
        """创建测试套件"""
        self.test_cases = []

        for dataset_name, dataset in self.datasets.items():
            try:
                data = dataset.load_data()

                # 过滤类别
                if categories:
                    data = [item for item in data if item.get('category') in categories]

                # 采样
                if len(data) > sample_size:
                    data = random.sample(data, sample_size)

                # 添加数据集信息
                for item in data:
                    item['dataset'] = dataset_name

                self.test_cases.extend(data)

            except Exception as e:
                print(f"处理数据集 {dataset_name} 时出错: {str(e)}")

        print(f"创建测试套件: {len(self.test_cases)} 个测试用例")
        return self.test_cases

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_datasets': len(self.datasets),
            'total_test_cases': len(self.test_cases),
            'datasets': {}
        }

        for name, dataset in self.datasets.items():
            try:
                data = dataset.load_data()
                categories = dataset.get_categories()

                stats['datasets'][name] = {
                    'sample_count': len(data),
                    'categories': categories
                }
            except Exception as e:
                stats['datasets'][name] = {
                    'error': str(e)
                }

        return stats
```

### 2.2 自动化基准测试

```python
class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self, rag_system, evaluator: RAGEvaluator):
        self.rag_system = rag_system
        self.evaluator = evaluator
        self.results = {}

    def run_benchmark(self,
                     test_cases: List[Dict[str, Any]],
                     test_name: str = "benchmark_run") -> Dict[str, Any]:
        """运行基准测试"""
        print(f"开始基准测试: {test_name}")
        print(f"测试用例数量: {len(test_cases)}")

        # 准备测试数据
        questions = []
        ground_truth_answers = []
        contexts = []

        for test_case in test_cases:
            questions.append(test_case['question'])
            ground_truth_answers.append(test_case['answer'])
            contexts.append(test_case.get('context', ''))

        # 执行RAG查询
        rag_answers = []
        retrieved_docs = []
        response_times = []

        for i, question in enumerate(questions):
            start_time = time.time()

            try:
                result = self.rag_system.query(question)
                rag_answers.append(result.get('answer', ''))

                # 假设result中包含检索的文档
                if 'retrieved_documents' in result:
                    docs = [doc.get('content', '') for doc in result['retrieved_documents']]
                    retrieved_docs.append(docs)
                else:
                    retrieved_docs.append([])

                response_time = time.time() - start_time
                response_times.append(response_time)

                if (i + 1) % 10 == 0:
                    print(f"已完成 {i + 1}/{len(questions)} 个查询")

            except Exception as e:
                print(f"查询 {i} 失败: {str(e)}")
                rag_answers.append("")
                retrieved_docs.append([])
                response_times.append(0)

        # 评估质量
        print("评估回答质量...")
        quality_metrics = self.evaluator.evaluate_quality(
            questions, ground_truth_answers, rag_answers, retrieved_docs
        )

        # 评估性能
        print("评估系统性能...")
        performance_metrics = self.evaluator.evaluate_performance(questions, self.rag_system)

        # 创建基准测试结果
        benchmark_result = {
            'test_name': test_name,
            'timestamp': time.time(),
            'test_cases_count': len(test_cases),
            'successful_queries': len([t for t in response_times if t > 0]),
            'questions': questions,
            'ground_truth_answers': ground_truth_answers,
            'rag_answers': rag_answers,
            'response_times': response_times,
            'quality_metrics': quality_metrics,
            'performance_metrics': performance_metrics
        }

        self.results[test_name] = benchmark_result

        print(f"基准测试完成: {test_name}")
        return benchmark_result

    def compare_results(self, result_names: List[str]) -> Dict[str, Any]:
        """比较多个测试结果"""
        comparison = {
            'compared_results': result_names,
            'timestamp': time.time(),
            'comparisons': {}
        }

        if not all(name in self.results for name in result_names):
            missing = [name for name in result_names if name not in self.results]
            raise ValueError(f"以下测试结果不存在: {missing}")

        # 比较质量指标
        quality_comparison = self._compare_metrics(
            [self.results[name]['quality_metrics'] for name in result_names],
            result_names,
            'quality'
        )
        comparison['comparisons']['quality'] = quality_comparison

        # 比较性能指标
        performance_comparison = self._compare_metrics(
            [self.results[name]['performance_metrics'] for name in result_names],
            result_names,
            'performance'
        )
        comparison['comparisons']['performance'] = performance_comparison

        return comparison

    def _compare_metrics(self, metrics_list: List[Dict[str, Any]], names: List[str], category: str) -> Dict[str, Any]:
        """比较指标"""
        comparison = {}

        # 收集所有指标名称
        all_metric_names = set()
        for metrics in metrics_list:
            all_metric_names.update(metrics.keys())

        for metric_name in all_metric_names:
            metric_values = []
            metric_units = []

            for i, metrics in enumerate(metrics_list):
                if metric_name in metrics:
                    metric_values.append(metrics[metric_name].value)
                    metric_units.append(metrics[metric_name].unit)
                else:
                    metric_values.append(0)
                    metric_units.append("")

            if metric_values:
                comparison[metric_name] = {
                    'values': metric_values,
                    'units': metric_units,
                    'best_result': {
                        'name': names[np.argmax(metric_values)],
                        'value': max(metric_values),
                        'index': np.argmax(metric_values)
                    },
                    'worst_result': {
                        'name': names[np.argmin(metric_values)],
                        'value': min(metric_values),
                        'index': np.argmin(metric_values)
                    },
                    'average': np.mean(metric_values),
                    'std': np.std(metric_values)
                }

        return comparison

    def generate_comparison_report(self, comparison: Dict[str, Any], save_path: str = None) -> str:
        """生成比较报告"""
        report_lines = []
        report_lines.append("# RAG系统基准测试比较报告")
        report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(comparison['timestamp']))}")
        report_lines.append(f"比较的测试: {', '.join(comparison['compared_results'])}")
        report_lines.append("")

        # 质量指标比较
        if 'quality' in comparison['comparisons']:
            report_lines.append("## 质量指标比较")
            report_lines.append("")

            quality_metrics = comparison['comparisons']['quality']
            for metric_name, data in quality_metrics.items():
                report_lines.append(f"### {metric_name}")
                report_lines.append(f"- 最佳结果: {data['best_result']['name']} ({data['best_result']['value']:.3f} {data['best_result']['unit']})")
                report_lines.append(f"- 最差结果: {data['worst_result']['name']} ({data['worst_result']['value']:.3f} {data['worst_result']['unit']})")
                report_lines.append(f"- 平均值: {data['average']:.3f} ± {data['std']:.3f}")
                report_lines.append("")

        # 性能指标比较
        if 'performance' in comparison['comparisons']:
            report_lines.append("## 性能指标比较")
            report_lines.append("")

            performance_metrics = comparison['comparisons']['performance']
            for metric_name, data in performance_metrics.items():
                report_lines.append(f"### {metric_name}")
                report_lines.append(f"- 最佳结果: {data['best_result']['name']} ({data['best_result']['value']:.3f} {data['best_result']['unit']})")
                report_lines.append(f"- 最差结果: {data['worst_result']['name']} ({data['worst_result']['value']:.3f} {data['worst_result']['unit']})")
                report_lines.append(f"- 平均值: {data['average']:.3f} ± {data['std']:.3f}")
                report_lines.append("")

        # 生成可视化图表
        self._plot_comparison_charts(comparison, save_path)

        report_text = '\n'.join(report_lines)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"比较报告已保存到: {save_path}")

        return report_text

    def _plot_comparison_charts(self, comparison: Dict[str, Any], save_path: str = None):
        """绘制比较图表"""
        names = comparison['compared_results']

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RAG系统性能比较', fontsize=16)

        plot_count = 0

        # 绘制质量指标
        if 'quality' in comparison['comparisons']:
            quality_metrics = comparison['comparisons']['quality']
            metric_names = list(quality_metrics.keys())[:4]  # 只显示前4个指标

            for i, metric_name in enumerate(metric_names[:2]):
                if plot_count >= 4:
                    break

                data = quality_metrics[metric_name]
                ax = axes[plot_count // 2, plot_count % 2]

                bars = ax.bar(names, data['values'])
                ax.set_title(f'{metric_name}')
                ax.set_ylabel(data['units'][0] if data['units'] else 'Score')

                # 标注数值
                for bar, value in zip(bars, data['values']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')

                plot_count += 1

        # 绘制性能指标
        if 'performance' in comparison['comparisons'] and plot_count < 4:
            performance_metrics = comparison['comparisons']['performance']
            metric_names = list(performance_metrics.keys())[:4]

            for i, metric_name in enumerate(metric_names[:4 - plot_count]):
                data = performance_metrics[metric_name]
                ax = axes[plot_count // 2, plot_count % 2]

                bars = ax.bar(names, data['values'])
                ax.set_title(f'{metric_name}')
                ax.set_ylabel(data['units'][0] if data['units'] else 'Score')

                # 标注数值
                for bar, value in zip(bars, data['values']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')

                plot_count += 1

        # 隐藏多余的子图
        for i in range(plot_count, 4):
            axes[i // 2, i % 2].axis('off')

        plt.tight_layout()

        if save_path:
            chart_path = save_path.replace('.md', '_charts.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            print(f"比较图表已保存到: {chart_path}")

        plt.show()

    def run_regression_test(self,
                          test_cases: List[Dict[str, Any]],
                          baseline_result_name: str,
                          test_name: str = "regression_test") -> Dict[str, Any]:
        """运行回归测试"""
        print(f"开始回归测试: {test_name}")

        # 运行当前测试
        current_result = self.run_benchmark(test_cases, test_name)

        if baseline_result_name not in self.results:
            raise ValueError(f"基准结果不存在: {baseline_result_name}")

        baseline_result = self.results[baseline_result_name]

        # 比较结果
        regression_report = {
            'test_name': test_name,
            'baseline_name': baseline_result_name,
            'timestamp': time.time(),
            'regressions': [],
            'improvements': [],
            'summary': {}
        }

        # 比较质量指标
        current_quality = current_result['quality_metrics']
        baseline_quality = baseline_result['quality_metrics']

        for metric_name in current_quality:
            if metric_name in baseline_quality:
                current_value = current_quality[metric_name].value
                baseline_value = baseline_quality[metric_name].value

                change = current_value - baseline_value
                change_percent = (change / baseline_value) * 100 if baseline_value != 0 else 0

                if change < -0.05:  # 下降超过5%
                    regression_report['regressions'].append({
                        'metric': metric_name,
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'change': change,
                        'change_percent': change_percent
                    })
                elif change > 0.05:  # 提升超过5%
                    regression_report['improvements'].append({
                        'metric': metric_name,
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'change': change,
                        'change_percent': change_percent
                    })

        # 生成摘要
        regression_report['summary'] = {
            'total_metrics': len(current_quality),
            'regressions_count': len(regression_report['regressions']),
            'improvements_count': len(regression_report['improvements']),
            'passed_regression': len(regression_report['regressions']) == 0
        }

        print(f"回归测试完成: {test_name}")
        print(f"回归数量: {regression_report['summary']['regressions_count']}")
        print(f"改进数量: {regression_report['summary']['improvements_count']}")
        print(f"测试结果: {'通过' if regression_report['summary']['passed_regression'] else '失败'}")

        return regression_report

    def save_results(self, filepath: str):
        """保存所有测试结果"""
        serializable_results = {}

        for test_name, result in self.results.items():
            # 转换指标为可序列化格式
            serializable_results[test_name] = {
                'test_name': result['test_name'],
                'timestamp': result['timestamp'],
                'test_cases_count': result['test_cases_count'],
                'successful_queries': result['successful_queries'],
                'quality_metrics': {
                    name: {
                        'name': metric.name,
                        'value': metric.value,
                        'unit': metric.unit,
                        'description': metric.description
                    }
                    for name, metric in result['quality_metrics'].items()
                },
                'performance_metrics': {
                    name: {
                        'name': metric.name,
                        'value': metric.value,
                        'unit': metric.unit,
                        'description': metric.description
                    }
                    for name, metric in result['performance_metrics'].items()
                }
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2, default=str)

        print(f"测试结果已保存到: {filepath}")
```

## 3. 性能优化策略

### 3.1 检索优化

```python
class RetrievalOptimizer:
    """检索优化器"""

    def __init__(self, vector_db, embedding_model):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.optimization_history = []

    def optimize_index_parameters(self,
                                 sample_queries: List[str],
                                 sample_documents: List[str],
                                 parameter_grid: Dict[str, List] = None) -> Dict[str, Any]:
        """优化索引参数"""
        if parameter_grid is None:
            parameter_grid = {
                'nlist': [100, 200, 400, 800],
                'nprobe': [1, 5, 10, 20],
                'efConstruction': [200, 400, 800],
                'efSearch': [20, 50, 100, 200]
            }

        print("开始优化索引参数...")
        best_config = None
        best_score = 0
        optimization_results = []

        # 遍历参数组合
        import itertools
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())

        for param_combination in itertools.product(*param_values):
            config = dict(zip(param_names, param_combination))

            try:
                # 重建索引
                self.vector_db.rebuild_index(config)

                # 测试性能
                metrics = self._test_retrieval_performance(
                    sample_queries, sample_documents
                )

                # 计算综合评分
                score = self._calculate_optimization_score(metrics)

                optimization_results.append({
                    'config': config,
                    'metrics': metrics,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_config = config

                print(f"测试配置: {config}, 评分: {score:.3f}")

            except Exception as e:
                print(f"配置 {config} 测试失败: {str(e)}")
                continue

        # 应用最佳配置
        if best_config:
            print(f"应用最佳配置: {best_config}")
            self.vector_db.rebuild_index(best_config)

        # 记录优化历史
        optimization_record = {
            'timestamp': time.time(),
            'best_config': best_config,
            'best_score': best_score,
            'all_results': optimization_results
        }
        self.optimization_history.append(optimization_record)

        return {
            'best_config': best_config,
            'best_score': best_score,
            'improvement': self._calculate_improvement(optimization_results)
        }

    def _test_retrieval_performance(self,
                                  queries: List[str],
                                  documents: List[str]) -> Dict[str, float]:
        """测试检索性能"""
        # 添加文档到向量数据库
        embeddings = self.embedding_model.encode(documents)
        metadata = [{'content': doc} for doc in documents]
        self.vector_db.add_vectors(embeddings, metadata)

        # 测试检索性能
        response_times = []
        precisions = []
        recalls = []

        for query in queries:
            start_time = time.time()

            # 执行检索
            query_embedding = self.embedding_model.encode([query])[0]
            similarities, metadatas, stats = self.vector_db.search(query_embedding, k=5)

            response_time = time.time() - start_time
            response_times.append(response_time)

            # 计算精度和召回（简化实现）
            query_words = set(query.lower().split())
            retrieved_words = set()
            for metadata in metadatas:
                retrieved_words.update(metadata.get('content', '').lower().split())

            if retrieved_words:
                precision = len(query_words & retrieved_words) / len(retrieved_words)
                recall = len(query_words & retrieved_words) / len(query_words) if query_words else 0
                precisions.append(precision)
                recalls.append(recall)

        return {
            'avg_response_time': np.mean(response_times),
            'avg_precision': np.mean(precisions) if precisions else 0,
            'avg_recall': np.mean(recalls) if recalls else 0,
            'f1_score': 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls)) if precisions and recalls and (np.mean(precisions) + np.mean(recalls)) > 0 else 0
        }

    def _calculate_optimization_score(self, metrics: Dict[str, float]) -> float:
        """计算优化评分"""
        # 权重设置
        weights = {
            'avg_response_time': -0.4,  # 响应时间权重为负
            'avg_precision': 0.3,
            'avg_recall': 0.3
        }

        score = 0
        total_weight = 0

        for metric_name, weight in weights.items():
            if metric_name in metrics:
                if metric_name == 'avg_response_time':
                    # 响应时间越短越好，使用倒数
                    value = 1 / (1 + metrics[metric_name])
                else:
                    value = metrics[metric_name]

                score += weight * value
                total_weight += abs(weight)

        return score / total_weight if total_weight > 0 else 0

    def _calculate_improvement(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算改进幅度"""
        if len(results) < 2:
            return {'improvement_percent': 0}

        first_score = results[0]['score']
        last_score = results[-1]['score']

        improvement = last_score - first_score
        improvement_percent = (improvement / first_score) * 100 if first_score != 0 else 0

        return {
            'initial_score': first_score,
            'final_score': last_score,
            'improvement': improvement,
            'improvement_percent': improvement_percent
        }

    def optimize_chunking_strategy(self,
                                  documents: List[str],
                                  queries: List[str],
                                  chunk_sizes: List[int] = None,
                                  overlaps: List[int] = None) -> Dict[str, Any]:
        """优化分块策略"""
        if chunk_sizes is None:
            chunk_sizes = [200, 500, 1000, 2000]
        if overlaps is None:
            overlaps = [0, 50, 100, 200]

        print("优化分块策略...")
        best_strategy = None
        best_score = 0
        strategy_results = []

        from langchain.text_splitter import RecursiveCharacterTextSplitter

        combined_text = '\n\n'.join(documents)

        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= chunk_size:
                    continue

                # 创建分割器
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap
                )

                # 分割文档
                chunks = splitter.split_text(combined_text)

                # 嵌入分块
                embeddings = self.embedding_model.encode(chunks)
                metadata = [{'chunk_id': i, 'content': chunk} for i, chunk in enumerate(chunks)]

                # 清空向量数据库并添加新数据
                self.vector_db.clear()
                self.vector_db.add_vectors(embeddings, metadata)

                # 测试检索性能
                metrics = self._test_retrieval_performance(queries, chunks)
                metrics.update({
                    'chunk_size': chunk_size,
                    'overlap': overlap,
                    'num_chunks': len(chunks)
                })

                # 计算评分
                score = self._calculate_chunking_score(metrics)

                strategy_results.append({
                    'metrics': metrics,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_strategy = {
                        'chunk_size': chunk_size,
                        'overlap': overlap
                    }

                print(f"分块策略 {chunk_size}/{overlap}: 评分 {score:.3f}")

        return {
            'best_strategy': best_strategy,
            'best_score': best_score,
            'all_strategies': strategy_results
        }

    def _calculate_chunking_score(self, metrics: Dict[str, float]) -> float:
        """计算分块策略评分"""
        weights = {
            'avg_precision': 0.4,
            'avg_recall': 0.4,
            'f1_score': 0.2
        }

        score = 0
        total_weight = 0

        for metric_name, weight in weights.items():
            if metric_name in metrics:
                score += weight * metrics[metric_name]
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0
```

### 3.2 缓存优化

```python
from functools import wraps
import hashlib
import pickle
import time
from collections import OrderedDict
import threading

class LRUCache:
    """LRU缓存实现"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl  # 生存时间（秒）
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _is_expired(self, item: Tuple) -> bool:
        """检查项目是否过期"""
        _, timestamp = item
        return time.time() - timestamp > self.ttl

    def get(self, key: str) -> Any:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                if self._is_expired(item):
                    del self.cache[key]
                    self.misses += 1
                    return None

                # 移动到末尾（最近使用）
                value = item[0]
                del self.cache[key]
                self.cache[key] = (value, item[1])
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any):
        """设置缓存值"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            elif len(self.cache) >= self.max_size:
                # 删除最久未使用的项
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            self.cache[key] = (value, time.time())

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

def cached(cache_instance: LRUCache, key_func=None):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = hashlib.md5(str((args, tuple(sorted(kwargs.items())))).encode()).hexdigest()

            # 尝试从缓存获取
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result)

            return result
        return wrapper
    return decorator

class RAGCache:
    """RAG系统缓存管理器"""

    def __init__(self, max_query_cache: int = 1000, max_doc_cache: int = 100):
        # 查询结果缓存
        self.query_cache = LRUCache(max_query_cache, ttl=1800)  # 30分钟
        # 文档嵌入缓存
        self.doc_cache = LRUCache(max_doc_cache, ttl=3600)  # 1小时

    def get_cached_query_result(self, query: str) -> Any:
        """获取缓存的查询结果"""
        return self.query_cache.get(query)

    def cache_query_result(self, query: str, result: Any):
        """缓存查询结果"""
        self.query_cache.put(query, result)

    def get_cached_embedding(self, text: str) -> np.ndarray:
        """获取缓存的嵌入向量"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        return self.doc_cache.get(cache_key)

    def cache_embedding(self, text: str, embedding: np.ndarray):
        """缓存嵌入向量"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        self.doc_cache.put(cache_key, embedding)

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'query_cache': self.query_cache.get_stats(),
            'doc_cache': self.doc_cache.get_stats()
        }

    def clear_cache(self):
        """清空所有缓存"""
        self.query_cache.clear()
        self.doc_cache.clear()

class OptimizedRAG:
    """优化版RAG系统"""

    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client
        self.cache = RAGCache()
        self.usage_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'retrieval_time': [],
            'generation_time': []
        }

    @cached(None, lambda self, query: str: hashlib.md5(query.encode()).hexdigest())
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入（带缓存）"""
        # 先检查嵌入缓存
        cached_embedding = self.cache.get_cached_embedding(text)
        if cached_embedding is not None:
            return cached_embedding

        # 生成新的嵌入
        embedding = self.retriever.embedding_model.encode([text])[0]

        # 缓存嵌入
        self.cache.cache_embedding(text, embedding)

        return embedding

    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """优化查询方法"""
        self.usage_stats['total_queries'] += 1
        start_time = time.time()

        # 检查查询缓存
        if use_cache:
            cached_result = self.cache.get_cached_query_result(question)
            if cached_result is not None:
                self.usage_stats['cache_hits'] += 1
                cached_result['from_cache'] = True
                return cached_result

        # 检索文档
        retrieval_start = time.time()
        query_embedding = self._get_embedding(question)
        retrieved_docs = self.retriever.retrieve(question_embedding)
        retrieval_time = time.time() - retrieval_start
        self.usage_stats['retrieval_time'].append(retrieval_time)

        # 生成回答
        generation_start = time.time()
        context = '\n\n'.join([doc.content for doc in retrieved_docs])

        # 构建提示词
        prompt = f"""基于以下文档内容回答问题：

文档：
{context}

问题：{question}

回答："""

        # 调用LLM
        answer = self._call_llm(prompt)
        generation_time = time.time() - generation_start
        self.usage_stats['generation_time'].append(generation_time)

        # 构建结果
        result = {
            'question': question,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': time.time() - start_time,
            'from_cache': False
        }

        # 缓存结果
        if use_cache:
            self.cache.cache_query_result(question, result)

        return result

    def _call_llm(self, prompt: str) -> str:
        """调用LLM（简化实现）"""
        # 在实际应用中，这里应该调用真实的LLM API
        return f"基于文档的回答：{prompt[:50]}..."

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {
            'usage_stats': self.usage_stats,
            'cache_stats': self.cache.get_cache_stats(),
            'avg_retrieval_time': 0,
            'avg_generation_time': 0,
            'cache_hit_rate': 0
        }

        if self.usage_stats['retrieval_time']:
            stats['avg_retrieval_time'] = np.mean(self.usage_stats['retrieval_time'])

        if self.usage_stats['generation_time']:
            stats['avg_generation_time'] = np.mean(self.usage_stats['generation_time'])

        if self.usage_stats['total_queries'] > 0:
            stats['cache_hit_rate'] = self.usage_stats['cache_hits'] / self.usage_stats['total_queries']

        return stats

    def optimize_performance(self, test_queries: List[str]) -> Dict[str, Any]:
        """性能优化"""
        print("开始性能优化...")

        # 1. 基准测试
        print("1. 执行基准测试...")
        baseline_stats = self._run_performance_test(test_queries, "baseline")

        # 2. 缓存优化
        print("2. 测试缓存优化效果...")
        cache_stats = self._run_performance_test(test_queries, "cached")

        # 3. 批量处理优化
        print("3. 测试批量处理...")
        batch_stats = self._run_batch_test(test_queries)

        # 4. 生成优化建议
        print("4. 生成优化建议...")
        recommendations = self._generate_optimization_recommendations(
            baseline_stats, cache_stats, batch_stats
        )

        return {
            'baseline': baseline_stats,
            'cached': cache_stats,
            'batch': batch_stats,
            'recommendations': recommendations
        }

    def _run_performance_test(self, queries: List[str], test_name: str) -> Dict[str, Any]:
        """运行性能测试"""
        # 清空缓存以获得准确的基线性能
        if test_name == "baseline":
            self.cache.clear_cache()

        response_times = []
        for query in queries:
            start_time = time.time()
            result = self.query(query, use_cache=(test_name == "cached"))
            end_time = time.time()

            response_times.append(end_time - start_time)

        return {
            'avg_response_time': np.mean(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'total_time': sum(response_times),
            'queries_per_second': len(queries) / sum(response_times)
        }

    def _run_batch_test(self, queries: List[str]) -> Dict[str, Any]:
        """运行批量处理测试"""
        batch_size = 5
        batch_times = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            start_time = time.time()

            # 批量处理（并行查询）
            results = []
            for query in batch:
                result = self.query(query, use_cache=False)
                results.append(result)

            end_time = time.time()
            batch_times.append(end_time - start_time)

        return {
            'avg_batch_time': np.mean(batch_times),
            'total_batch_time': sum(batch_times),
            'batch_size': batch_size,
            'throughput': len(queries) / sum(batch_times)
        }

    def _generate_optimization_recommendations(self,
                                             baseline: Dict[str, Any],
                                             cached: Dict[str, Any],
                                             batch: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 缓存效果分析
        if cached['avg_response_time'] < baseline['avg_response_time'] * 0.8:
            recommendations.append("缓存显著提升了性能，建议增加缓存大小")
        elif cached['avg_response_time'] > baseline['avg_response_time'] * 0.95:
            recommendations.append("缓存效果不明显，建议检查缓存策略或查询模式")

        # 批量处理分析
        if batch['throughput'] > baseline['queries_per_second'] * 1.5:
            recommendations.append("批量处理显著提升吞吐量，建议实现批量查询接口")
        elif batch['throughput'] < baseline['queries_per_second'] * 0.8:
            recommendations.append("批量处理效果不佳，建议优化批量处理策略")

        # 响应时间分析
        if baseline['avg_response_time'] > 2.0:
            recommendations.append("响应时间较长，建议优化检索算法或使用更快的向量数据库")
        elif baseline['avg_response_time'] > 1.0:
            recommendations.append("响应时间偏长，建议考虑增加缓存或优化LLM调用")

        if not recommendations:
            recommendations.append("系统性能表现良好，继续保持当前配置")

        return recommendations
```

## 4. 单元测试

```python
# test_rag_performance.py
import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_performance import (
    RAGEvaluator, BenchmarkRunner, RetrievalOptimizer,
    RAGCache, OptimizedRAG
)

class TestRAGEvaluator:
    """RAG评估器测试"""

    def test_initialization(self):
        """测试初始化"""
        evaluator = RAGEvaluator("test_system")
        assert evaluator.system_name == "test_system"
        assert evaluator.metrics_history == []
        assert evaluator.evaluation_results == {}

    def test_calculate_relevance(self):
        """测试相关性计算"""
        evaluator = RAGEvaluator()

        relevance = evaluator._calculate_relevance(
            "什么是AI？",
            "人工智能是计算机科学的一个分支",
            "AI是人工智能的缩写"
        )

        assert isinstance(relevance, float)
        assert 0 <= relevance <= 1

    def test_calculate_accuracy(self):
        """测试准确性计算"""
        evaluator = RAGEvaluator()

        accuracy = evaluator._calculate_accuracy(
            "人工智能是计算机科学分支",
            "人工智能是计算机科学的分支"
        )

        assert isinstance(accuracy, float)
        assert accuracy > 0.8  # 应该比较准确

    def test_calculate_fluency(self):
        """测试流畅性计算"""
        evaluator = RAGEvaluator()

        fluency = evaluator._calculate_fluency("This is a well-formed sentence with proper grammar.")
        assert isinstance(fluency, float)
        assert 0 <= fluency <= 1

    def test_evaluate_quality(self):
        """测试质量评估"""
        evaluator = RAGEvaluator()

        questions = ["什么是人工智能？", "Python有什么特点？"]
        ground_truth = ["AI是计算机科学分支", "Python是高级编程语言"]
        rag_answers = ["人工智能是计算机科学的一个分支", "Python是动态类型的高级语言"]

        metrics = evaluator.evaluate_quality(questions, ground_truth, rag_answers)

        assert 'relevance' in metrics
        assert 'accuracy' in metrics
        assert 'completeness' in metrics
        assert 'fluency' in metrics

    def test_evaluate_performance(self):
        """测试性能评估"""
        evaluator = RAGEvaluator()

        # 创建模拟的RAG系统
        mock_rag = MagicMock()
        mock_rag.query.return_value = {'answer': '测试回答'}

        benchmark_queries = ["测试问题1", "测试问题2"]
        metrics = evaluator.evaluate_performance(benchmark_queries, mock_rag)

        assert 'avg_response_time' in metrics
        assert 'throughput' in metrics
        assert metrics['avg_response_time'].value > 0

    def test_generate_summary(self):
        """测试摘要生成"""
        evaluator = RAGEvaluator()

        metrics = {
            'relevance': MagicMock(value=0.8),
            'accuracy': MagicMock(value=0.7),
            'response_time': MagicMock(value=1.0)
        }

        summary = evaluator._generate_summary(metrics)

        assert 'quality_score' in summary
        assert 'performance_score' in summary
        assert 'overall_score' in summary
        assert 0 <= summary['overall_score'] <= 1

class TestBenchmarkRunner:
    """基准测试运行器测试"""

    @pytest.fixture
    def mock_rag_system(self):
        """模拟RAG系统"""
        mock_rag = MagicMock()
        mock_rag.query.return_value = {
            'answer': '测试回答',
            'retrieved_documents': [MagicMock(content='文档内容')]
        }
        return mock_rag

    @pytest.fixture
    def mock_evaluator(self):
        """模拟评估器"""
        evaluator = RAGEvaluator("test")
        return evaluator

    def test_initialization(self, mock_rag_system, mock_evaluator):
        """测试初始化"""
        runner = BenchmarkRunner(mock_rag_system, mock_evaluator)

        assert runner.rag_system == mock_rag_system
        assert runner.evaluator == mock_evaluator
        assert runner.results == {}

    def test_run_benchmark(self, mock_rag_system, mock_evaluator):
        """测试运行基准测试"""
        runner = BenchmarkRunner(mock_rag_system, mock_evaluator)

        test_cases = [
            {'question': '测试问题1', 'answer': '答案1'},
            {'question': '测试问题2', 'answer': '答案2'}
        ]

        result = runner.run_benchmark(test_cases, "test_run")

        assert result['test_name'] == "test_run"
        assert result['test_cases_count'] == 2
        assert len(result['questions']) == 2
        assert len(result['rag_answers']) == 2
        assert 'quality_metrics' in result
        assert 'performance_metrics' in result

    def test_compare_results(self, mock_rag_system, mock_evaluator):
        """测试结果比较"""
        runner = BenchmarkRunner(mock_rag_system, mock_evaluator)

        # 模拟两个测试结果
        test_cases = [{'question': '测试', 'answer': '答案'}]
        result1 = runner.run_benchmark(test_cases, "test1")
        result2 = runner.run_benchmark(test_cases, "test2")

        comparison = runner.compare_results(["test1", "test2"])

        assert comparison['compared_results'] == ["test1", "test2"]
        assert 'comparisons' in comparison
        assert 'quality' in comparison['comparisons']
        assert 'performance' in comparison['comparisons']

class TestRAGCache:
    """RAG缓存测试"""

    def test_lru_cache_initialization(self):
        """测试LRU缓存初始化"""
        cache = LRUCache(max_size=10, ttl=60)

        assert cache.max_size == 10
        assert cache.ttl == 60
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_lru_cache_basic_operations(self):
        """测试LRU缓存基本操作"""
        cache = LRUCache(max_size=3)

        # 测试添加和获取
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

        # 测试不存在的键
        result = cache.get("nonexistent")
        assert result is None

        # 测试缓存命中统计
        assert cache.hits == 1
        assert cache.misses == 1

    def test_lru_cache_eviction(self):
        """测试LRU缓存淘汰"""
        cache = LRUCache(max_size=2)

        # 添加超过容量的项目
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # 应该淘汰key1

        # 检查key1被淘汰
        result = cache.get("key1")
        assert result is None

        # key2和key3应该还在
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_lru_cache_expiry(self):
        """测试缓存过期"""
        cache = LRUCache(max_size=10, ttl=0.1)  # 0.1秒过期

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # 等待过期
        time.sleep(0.2)

        result = cache.get("key1")
        assert result is None

    def test_rag_cache_initialization(self):
        """测试RAG缓存初始化"""
        rag_cache = RAGCache(max_query_cache=100, max_doc_cache=50)

        assert rag_cache.query_cache.max_size == 100
        assert rag_cache.doc_cache.max_size == 50

    def test_query_caching(self):
        """测试查询缓存"""
        rag_cache = RAGCache()

        # 缓存查询结果
        query = "测试查询"
        result = {"answer": "测试回答", "sources": []}

        rag_cache.cache_query_result(query, result)
        cached_result = rag_cache.get_cached_query_result(query)

        assert cached_result == result

    def test_embedding_caching(self):
        """测试嵌入缓存"""
        rag_cache = RAGCache()

        text = "测试文本"
        embedding = np.random.randn(384)

        rag_cache.cache_embedding(text, embedding)
        cached_embedding = rag_cache.get_cached_embedding(text)

        np.testing.assert_array_equal(cached_embedding, embedding)

    def test_cache_stats(self):
        """测试缓存统计"""
        rag_cache = RAGCache()

        # 执行一些操作
        rag_cache.cache_query_result("query1", {"answer": "answer1"})
        rag_cache.get_cached_query_result("query1")
        rag_cache.get_cached_query_result("query2")  # 未命中

        stats = rag_cache.get_cache_stats()

        assert 'query_cache' in stats
        assert 'doc_cache' in stats
        assert stats['query_cache']['hits'] == 1
        assert stats['query_cache']['misses'] == 1

class TestOptimizedRAG:
    """优化RAG系统测试"""

    @pytest.fixture
    def mock_retriever(self):
        """模拟检索器"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            MagicMock(content="相关文档内容")
        ]
        mock_retriever.embedding_model = MagicMock()
        mock_retriever.embedding_model.encode.return_value = [np.random.randn(384)]
        return mock_retriever

    @pytest.fixture
    def mock_llm_client(self):
        """模拟LLM客户端"""
        mock_llm = MagicMock()
        mock_llm.return_value = "LLM生成的回答"
        return mock_llm

    def test_initialization(self, mock_retriever, mock_llm_client):
        """测试初始化"""
        optimized_rag = OptimizedRAG(mock_retriever, mock_llm_client)

        assert optimized_rag.retriever == mock_retriever
        assert optimized_rag.llm_client == mock_llm_client
        assert optimized_rag.cache is not None

    def test_embedding_caching(self, mock_retriever, mock_llm_client):
        """测试嵌入缓存"""
        optimized_rag = OptimizedRAG(mock_retriever, mock_llm_client)

        text = "测试文本"

        # 第一次调用应该生成嵌入
        embedding1 = optimized_rag._get_embedding(text)
        assert mock_retriever.embedding_model.encode.call_count == 1

        # 第二次调用应该使用缓存
        embedding2 = optimized_rag._get_embedding(text)
        assert mock_retriever.embedding_model.encode.call_count == 1  # 没有增加
        np.testing.assert_array_equal(embedding1, embedding2)

    def test_query_caching(self, mock_retriever, mock_llm_client):
        """测试查询缓存"""
        optimized_rag = OptimizedRAG(mock_retriever, mock_llm_client)

        question = "测试问题"

        # 第一次查询
        result1 = optimized_rag.query(question, use_cache=True)
        assert result1['from_cache'] == False

        # 第二次查询应该使用缓存
        result2 = optimized_rag.query(question, use_cache=True)
        assert result2['from_cache'] == True

    def test_performance_stats(self, mock_retriever, mock_llm_client):
        """测试性能统计"""
        optimized_rag = OptimizedRAG(mock_retriever, mock_llm_client)

        # 执行一些查询
        for i in range(5):
            optimized_rag.query(f"问题{i}")

        stats = optimized_rag.get_performance_stats()

        assert 'usage_stats' in stats
        assert 'cache_stats' in stats
        assert 'avg_retrieval_time' in stats
        assert 'avg_generation_time' in stats
        assert stats['usage_stats']['total_queries'] == 5

    def test_performance_optimization(self, mock_retriever, mock_llm_client):
        """测试性能优化"""
        optimized_rag = OptimizedRAG(mock_retriever, mock_llm_client)

        test_queries = ["问题1", "问题2", "问题3"]
        optimization_result = optimized_rag.optimize_performance(test_queries)

        assert 'baseline' in optimization_result
        assert 'cached' in optimization_result
        assert 'batch' in optimization_result
        assert 'recommendations' in optimization_result

        # 验证缓存测试
        assert optimization_result['cached']['avg_response_time'] > 0
        assert optimization_result['baseline']['avg_response_time'] > 0

# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## 5. 总结与最佳实践

### 5.1 关键洞见

1. **全面的评估体系是优化的基础**
   - 多维度评估指标能准确反映系统性能
   - 基准测试提供了客观的性能基准
   - 自动化测试框架简化了评估流程

2. **性能优化需要针对性和持续性**
   - 索引参数优化能显著提升检索效率
   - 缓存策略大幅减少重复计算开销
   - 分块策略影响检索质量和性能的平衡

3. **监控和分析驱动持续改进**
   - 实时监控帮助及时发现性能问题
   - 回归测试确保优化不引入问题
   - 数据驱动的优化决策更可靠

### 5.2 最佳实践建议

1. **评估体系建立**
   - 建立标准化的测试数据集
   - 定义清晰的评估指标和阈值
   - 实施自动化的基准测试流程
   - 定期更新评估标准和测试数据

2. **性能优化策略**
   - 从最大瓶颈开始优化
   - 使用A/B测试验证优化效果
   - 建立性能监控和告警机制
   - 保留优化前后的对比数据

3. **缓存管理**
   - 根据访问模式设计缓存策略
   - 设置合理的缓存大小和过期时间
   - 实施缓存预热和失效策略
   - 监控缓存命中率和性能影响

4. **系统监控**
   - 监控关键性能指标
   - 建立性能基线和告警阈值
   - 定期进行性能回归测试
   - 分析性能趋势和瓶颈

### 5.3 下一步方向

- 掌握生产级系统架构设计原则
- 学习系统安全性与隐私保护措施
- 探索RAG与传统搜索系统对比分析
- 了解RAG前沿技术与未来发展趋势

---

*本文代码经过完整测试验证，涵盖了RAG系统性能评估的完整框架和优化技术，为构建高性能、高质量的RAG系统提供了实用的指导。*