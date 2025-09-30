# 第12篇：RAG前沿技术与未来展望

## 摘要

本文探讨了RAG（Retrieval-Augmented Generation）技术的最新发展趋势和未来展望。从GraphRAG、Self-RAG等前沿技术到多模态融合、实时学习等新兴方向，全面分析了RAG技术的演进路径，并预测了其在大模型时代的发展前景和商业价值。

## 1. 前沿RAG技术概览

### 1.1 新一代RAG技术图谱

```
RAG技术演进路线：
├── 第一代：基础RAG (2020-2022)
│   ├── 特点：检索+生成简单组合
│   ├── 应用：基础问答系统
│   └── 局限：准确性、可控性不足
├── 第二代：增强RAG (2022-2023)
│   ├── 技术：Self-RAG、Corrective-RAG
│   ├── 特点：自我纠错、质量评估
│   └── 应用：企业级问答系统
├── 第三代：结构化RAG (2023-2024)
│   ├── 技术：GraphRAG、HyDE
│   ├── 特点：知识图谱、假设性文档
│   └── 应用：专业知识系统
└── 第四代：自适应RAG (2024+)
    ├── 技术：Adaptive-RAG、FLARE
    ├── 特点：动态优化、实时学习
    └── 应用：个性化智能助手
```

### 1.2 技术成熟度评估

| 技术名称 | 成熟度 | 性能提升 | 应用难度 | 商业价值 |
|----------|--------|----------|----------|----------|
| Self-RAG | 🟡 中等 | 🟢 高 | 🟡 中等 | 🟢 高 |
| GraphRAG | 🟠 中高 | 🟢 很高 | 🔴 高 | 🟢 很高 |
| Corrective-RAG | 🟢 高 | 🟡 中等 | 🟢 低 | 🟡 中等 |
| HyDE | 🟡 中等 | 🟡 中等 | 🟢 低 | 🟡 中等 |
| Adaptive-RAG | 🔴 低 | 🟢 很高 | 🔴 高 | 🟢 很高 |

## 2. Self-RAG：自反思检索增强生成

### 2.1 Self-RAG核心原理

Self-RAG（Self-Reflective Retrieval-Augmented Generation）通过引入自我反思机制，在生成过程中动态评估和优化检索结果的质量。

```python
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import openai
from enum import Enum

class ReflectionToken(Enum):
    """反思标记类型"""
    RELEVANT = "[Relevant]"
    IRRELEVANT = "[Irrelevant]"
    SUPPORTED = "[Supported]"
    CONTRADICTED = "[Contradicted]"
    NO_INFO = "[No Info]"

@dataclass
class ReflectionResult:
    """反思结果"""
    is_relevant: bool
    is_supported: bool
    confidence: float
    reasoning: str

class SelfRAGSystem:
    """Self-RAG系统实现"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.reflection_prompts = self._init_reflection_prompts()

    def _init_reflection_prompts(self) -> Dict[str, str]:
        """初始化反思提示词"""
        return {
            'relevance_check': """请评估检索到的文档是否与用户查询相关。

查询: {query}
文档: {document}

请从以下选项中选择一个：
[Relevant] - 文档与查询高度相关
[Irrelevant] - 文档与查询不相关

你的选择:""",

            'support_check': """请评估生成的回答是否得到检索文档的支持。

回答: {answer}
文档: {document}

请从以下选项中选择一个：
[Supported] - 回答得到文档充分支持
[Contradicted] - 回答与文档内容矛盾
[No Info] - 文档中没有相关信息

你的选择:""",

            'self_correction': """基于以下评估结果，请改进回答：

原始回答: {answer}
评估反馈: {feedback}
相关文档: {document}

请提供一个改进的回答:""",

            'quality_score': """请为以下回答打分（1-10分）：

查询: {query}
回答: {answer}
支持文档: {document}

评分标准：
- 准确性：回答是否准确
- 完整性：回答是否完整
- 相关性：回答是否相关
- 清晰度：回答是否清晰

评分:"""
        }

    def reflect_on_retrieval(self, query: str, documents: List[Dict]) -> List[ReflectionResult]:
        """对检索结果进行反思评估"""
        reflection_results = []

        for doc in documents:
            # 检查相关性
            relevance_result = self._check_relevance(query, doc)

            reflection_results.append(ReflectionResult(
                is_relevant=relevance_result['is_relevant'],
                is_supported=False,  # 稍后检查
                confidence=relevance_result['confidence'],
                reasoning=relevance_result['reasoning']
            ))

        return reflection_results

    def _check_relevance(self, query: str, document: Dict) -> Dict[str, Any]:
        """检查文档与查询的相关性"""
        prompt = self.reflection_prompts['relevance_check'].format(
            query=query,
            document=document.get('content', '')[:500]  # 限制长度
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的相关性评估助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )

            content = response.choices[0].message.content.strip()

            # 解析反思标记
            is_relevant = ReflectionToken.RELEVANT.value in content
            confidence = 0.8 if is_relevant else 0.2
            reasoning = content

            return {
                'is_relevant': is_relevant,
                'confidence': confidence,
                'reasoning': reasoning
            }

        except Exception as e:
            return {
                'is_relevant': False,
                'confidence': 0.0,
                'reasoning': f"评估失败: {str(e)}"
            }

    def reflect_on_generation(self, query: str, answer: str,
                            documents: List[Dict]) -> ReflectionResult:
        """对生成结果进行反思评估"""
        if not documents:
            return ReflectionResult(
                is_relevant=False,
                is_supported=False,
                confidence=0.0,
                reasoning="没有相关文档"
            )

        # 选择最相关的文档
        relevant_doc = documents[0]  # 简化版，实际应该选择最相关的

        # 检查支持性
        support_result = self._check_support(answer, relevant_doc)

        # 计算质量分数
        quality_score = self._calculate_quality_score(query, answer, relevant_doc)

        return ReflectionResult(
            is_relevant=True,  # 假设已经过相关性过滤
            is_supported=support_result['is_supported'],
            confidence=quality_score,
            reasoning=support_result['reasoning']
        )

    def _check_support(self, answer: str, document: Dict) -> Dict[str, Any]:
        """检查回答是否得到文档支持"""
        prompt = self.reflection_prompts['support_check'].format(
            answer=answer,
            document=document.get('content', '')[:500]
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的事实核查助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )

            content = response.choices[0].message.content.strip()

            # 解析支持性标记
            is_supported = ReflectionToken.SUPPORTED.value in content
            reasoning = content

            return {
                'is_supported': is_supported,
                'reasoning': reasoning
            }

        except Exception as e:
            return {
                'is_supported': False,
                'reasoning': f"评估失败: {str(e)}"
            }

    def _calculate_quality_score(self, query: str, answer: str, document: Dict) -> float:
        """计算回答质量分数"""
        prompt = self.reflection_prompts['quality_score'].format(
            query=query,
            answer=answer,
            document=document.get('content', '')[:300]
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的质量评估助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=20
            )

            content = response.choices[0].message.content.strip()

            # 提取数字分数
            import re
            score_match = re.search(r'\b([1-9]|10)\b', content)
            if score_match:
                score = int(score_match.group(1))
                return score / 10.0  # 标准化到0-1

        except Exception:
            pass

        return 0.5  # 默认分数

    def self_correct(self, query: str, answer: str, feedback: str,
                    document: Dict) -> str:
        """基于反思结果进行自我纠错"""
        prompt = self.reflection_prompts['self_correction'].format(
            answer=answer,
            feedback=feedback,
            document=document.get('content', '')[:500]
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的回答改进助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"纠错失败: {str(e)}"

    def generate_with_self_reflection(self, query: str, documents: List[Dict],
                                   max_iterations: int = 3) -> Dict[str, Any]:
        """带自我反思的生成流程"""
        generation_log = []
        current_answer = ""
        current_iteration = 0

        while current_iteration < max_iterations:
            current_iteration += 1

            # 生成或改进回答
            if current_iteration == 1:
                # 第一次生成
                current_answer = self._generate_initial_answer(query, documents)
                generation_log.append({
                    'iteration': current_iteration,
                    'action': 'initial_generation',
                    'answer': current_answer
                })
            else:
                # 基于反思改进回答
                last_reflection = generation_log[-1].get('reflection')
                if last_reflection:
                    feedback = f"相关性: {last_reflection.is_relevant}, 支持性: {last_reflection.is_supported}"
                    current_answer = self.self_correct(
                        query, current_answer, feedback, documents[0]
                    )
                    generation_log.append({
                        'iteration': current_iteration,
                        'action': 'self_correction',
                        'answer': current_answer,
                        'feedback': feedback
                    })

            # 反思评估
            reflection = self.reflect_on_generation(query, current_answer, documents)
            generation_log[-1]['reflection'] = reflection

            # 如果质量足够好，停止迭代
            if reflection.confidence >= 0.8 and reflection.is_supported:
                break

        return {
            'final_answer': current_answer,
            'iterations': current_iteration,
            'generation_log': generation_log,
            'final_reflection': reflection
        }

    def _generate_initial_answer(self, query: str, documents: List[Dict]) -> str:
        """生成初始回答"""
        context = "\n\n".join([
            f"文档{i+1}: {doc.get('content', '')[:300]}"
            for i, doc in enumerate(documents[:3])
        ])

        prompt = f"""基于以下文档回答用户问题：

查询: {query}

相关文档:
{context}

请提供准确、完整的回答："""

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"生成失败: {str(e)}"
```

### 2.2 Self-RAG性能优化

```python
class SelfRAGOptimizer:
    """Self-RAG性能优化器"""

    def __init__(self):
        self.performance_history = []
        self.optimization_strategies = {
            'cache_reflections': True,
            'batch_reflection': True,
            'early_stopping': True,
            'adaptive_iterations': True
        }

    def optimize_generation_pipeline(self, self_rag: SelfRAGSystem,
                                   query: str, documents: List[Dict]) -> Dict:
        """优化的生成管道"""
        start_time = time.time()

        # 1. 并行相关性检查
        if self.optimization_strategies['batch_reflection']:
            relevance_results = self._batch_relevance_check(query, documents)
            relevant_docs = [doc for doc, result in zip(documents, relevance_results)
                           if result.is_relevant]
        else:
            relevance_results = [self_rag._check_relevance(query, doc) for doc in documents]
            relevant_docs = [doc for doc, result in zip(documents, relevance_results)
                           if result['is_relevant']]

        if not relevant_docs:
            return {
                'answer': "抱歉，没有找到相关的信息来回答您的问题。",
                'confidence': 0.0,
                'optimization_applied': ['no_relevant_docs']
            }

        # 2. 自适应迭代次数
        if self.optimization_strategies['adaptive_iterations']:
            max_iterations = self._determine_optimal_iterations(query, relevant_docs)
        else:
            max_iterations = 3

        # 3. 执行优化的自反思生成
        result = self_rag.generate_with_self_reflection(
            query, relevant_docs, max_iterations
        )

        # 4. 记录性能数据
        end_time = time.time()
        performance_data = {
            'query': query,
            'doc_count': len(documents),
            'relevant_doc_count': len(relevant_docs),
            'iterations_used': result['iterations'],
            'final_confidence': result['final_reflection'].confidence,
            'response_time': end_time - start_time
        }
        self.performance_history.append(performance_data)

        # 5. 添加优化信息
        optimizations_applied = []
        if self.optimization_strategies['batch_reflection']:
            optimizations_applied.append('batch_relevance_check')
        if self.optimization_strategies['adaptive_iterations']:
            optimizations_applied.append('adaptive_iterations')
        if max_iterations < 3:
            optimizations_applied.append('reduced_iterations')

        result['optimization_applied'] = optimizations_applied
        result['performance_metrics'] = performance_data

        return result

    def _batch_relevance_check(self, query: str, documents: List[Dict]) -> List[ReflectionResult]:
        """批量相关性检查"""
        # 构建批量检查的提示词
        batch_prompt = "请评估以下每个文档与查询的相关性：\n\n"
        batch_prompt += f"查询: {query}\n\n"

        for i, doc in enumerate(documents):
            doc_content = doc.get('content', '')[:200]
            batch_prompt += f"文档{i+1}: {doc_content}\n"

        batch_prompt += "\n对每个文档，请从以下选项中选择：\n"
        batch_prompt += "[Relevant] 或 [Irrelevant]\n\n"
        batch_prompt += "请按以下格式回答：\n"
        for i in range(len(documents)):
            batch_prompt += f"文档{i+1}: [选择]\n"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个批量相关性评估助手。"},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()

            # 解析批量结果
            results = []
            lines = content.split('\n')
            for i, line in enumerate(lines[:len(documents)]):
                is_relevant = ReflectionToken.RELEVANT.value in line
                results.append(ReflectionResult(
                    is_relevant=is_relevant,
                    is_supported=False,
                    confidence=0.8 if is_relevant else 0.2,
                    reasoning=line.strip()
                ))

            return results

        except Exception as e:
            # 降级到单个检查
            return [ReflectionResult(
                is_relevant=True,  # 默认认为相关
                is_supported=False,
                confidence=0.5,
                reasoning=f"批量检查失败: {e}"
            ) for _ in documents]

    def _determine_optimal_iterations(self, query: str, documents: List[Dict]) -> int:
        """确定最优迭代次数"""
        query_complexity = self._assess_query_complexity(query)
        doc_relevance = self._assess_document_relevance(documents)

        # 基于复杂度和相关性确定迭代次数
        if query_complexity >= 0.8 and doc_relevance >= 0.7:
            return 3  # 复杂查询，高相关性文档
        elif query_complexity >= 0.6 or doc_relevance >= 0.6:
            return 2  # 中等复杂度
        else:
            return 1  # 简单查询

    def _assess_query_complexity(self, query: str) -> float:
        """评估查询复杂度"""
        complexity_score = 0.0

        # 基于长度
        complexity_score += min(1.0, len(query) / 50.0) * 0.3

        # 基于复杂词汇
        complex_words = ['为什么', '如何', '对比', '影响', '关系', '原理', '机制', '区别']
        complex_count = sum(1 for word in complex_words if word in query.lower())
        complexity_score += min(1.0, complex_count / 3.0) * 0.4

        # 基于标点符号
        if '？' in query or '?' in query:
            complexity_score += 0.1
        if '，' in query or ',' in query:
            complexity_score += 0.1
        if '和' in query or '与' in query:
            complexity_score += 0.1

        return min(1.0, complexity_score)

    def _assess_document_relevance(self, documents: List[Dict]) -> float:
        """评估文档相关性"""
        if not documents:
            return 0.0

        # 简化的相关性评估
        # 实际应用中可以使用更复杂的算法
        return min(1.0, len(documents) / 5.0)

    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.performance_history:
            return {"message": "暂无性能数据"}

        import pandas as pd

        df = pd.DataFrame(self.performance_history)

        summary = {
            'total_queries': len(df),
            'avg_response_time': df['response_time'].mean(),
            'avg_iterations': df['iterations_used'].mean(),
            'avg_confidence': df['final_confidence'].mean(),
            'success_rate': (df['final_confidence'] >= 0.7).mean(),
            'optimization_impact': self._calculate_optimization_impact()
        }

        return summary

    def _calculate_optimization_impact(self) -> Dict:
        """计算优化影响"""
        if len(self.performance_history) < 10:
            return {"message": "数据不足，需要更多查询"}

        # 比较前后性能
        recent = self.performance_history[-10:]
        earlier = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else []

        if not earlier:
            return {"message": "数据不足，需要更多查询"}

        recent_avg_time = np.mean([p['response_time'] for p in recent])
        earlier_avg_time = np.mean([p['response_time'] for p in earlier])

        improvement = ((earlier_avg_time - recent_avg_time) / earlier_avg_time) * 100

        return {
            'response_time_improvement': f"{improvement:.1f}%",
            'recent_avg_time': recent_avg_time,
            'earlier_avg_time': earlier_avg_time
        }
```

## 3. GraphRAG：知识图谱增强的RAG

### 3.1 GraphRAG架构设计

GraphRAG（Graph-based Retrieval-Augmented Generation）将知识图谱与传统RAG结合，通过结构化知识提升检索和生成的准确性。

```python
import networkx as nx
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
import json
import re

@dataclass
class KnowledgeEntity:
    """知识实体"""
    id: str
    name: str
    type: str
    description: str
    properties: Dict[str, Any]

@dataclass
class KnowledgeRelation:
    """知识关系"""
    id: str
    source: str
    target: str
    relation_type: str
    weight: float
    properties: Dict[str, Any]

class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relations = {}
        self.entity_patterns = {
            'person': r'\b(?:张三|李四|王五|赵六)\b',
            'organization': r'\b(?:阿里巴巴|腾讯|百度|字节跳动)\b',
            'technology': r'\b(?:人工智能|机器学习|深度学习|神经网络)\b',
            'concept': r'\b(?:算法|模型|数据|训练)\b'
        }

    def build_from_documents(self, documents: List[Dict]) -> nx.DiGraph:
        """从文档构建知识图谱"""
        for doc in documents:
            self._process_document(doc)

        return self.graph

    def _process_document(self, document: Dict):
        """处理单个文档"""
        content = document.get('content', '')
        doc_id = document.get('id', 'unknown')

        # 提取实体
        entities = self._extract_entities(content, doc_id)

        # 提取关系
        relations = self._extract_relations(content, entities)

        # 添加到图谱
        for entity in entities:
            self._add_entity(entity)

        for relation in relations:
            self._add_relation(relation)

    def _extract_entities(self, text: str, doc_id: str) -> List[KnowledgeEntity]:
        """提取实体"""
        entities = []

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_name = match.group()
                entity_id = f"{entity_type}_{entity_name}_{doc_id}"

                # 提取实体描述（上下文）
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()

                entity = KnowledgeEntity(
                    id=entity_id,
                    name=entity_name,
                    type=entity_type,
                    description=context,
                    properties={
                        'source_doc': doc_id,
                        'position': match.start(),
                        'confidence': 0.8
                    }
                )
                entities.append(entity)

        return entities

    def _extract_relations(self, text: str, entities: List[KnowledgeEntity]) -> List[KnowledgeRelation]:
        """提取实体间关系"""
        relations = []

        # 简单的关系提取规则
        relation_patterns = [
            (r'(\w+)\s*是\s*(\w+)\s*的', 'is_a'),
            (r'(\w+)\s*属于\s*(\w+)', 'belongs_to'),
            (r'(\w+)\s*包含\s*(\w+)', 'contains'),
            (r'(\w+)\s*应用于\s*(\w+)', 'applies_to'),
            (r'(\w+)\s*基于\s*(\w+)', 'based_on')
        ]

        for pattern, relation_type in relation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                source_name = match.group(1)
                target_name = match.group(2)

                # 查找对应的实体
                source_entities = [e for e in entities if e.name == source_name]
                target_entities = [e for e in entities if e.name == target_name]

                if source_entities and target_entities:
                    relation = KnowledgeRelation(
                        id=f"rel_{len(self.relations)}",
                        source=source_entities[0].id,
                        target=target_entities[0].id,
                        relation_type=relation_type,
                        weight=0.8,
                        properties={
                            'source_text': match.group(),
                            'position': match.start()
                        }
                    )
                    relations.append(relation)

        return relations

    def _add_entity(self, entity: KnowledgeEntity):
        """添加实体到图谱"""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
            **entity.properties
        )

    def _add_relation(self, relation: KnowledgeRelation):
        """添加关系到图谱"""
        self.relations[relation.id] = relation
        self.graph.add_edge(
            relation.source,
            relation.target,
            relation_type=relation.relation_type,
            weight=relation.weight,
            **relation.properties
        )

    def get_neighbors(self, entity_id: str, depth: int = 1) -> List[KnowledgeEntity]:
        """获取实体的邻居节点"""
        if entity_id not in self.graph:
            return []

        neighbors = []
        visited = set()
        queue = [(entity_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)
            if current_depth > depth:
                break

            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id != entity_id:  # 排除起始节点
                neighbors.append(self.entities[current_id])

            # 添加邻居到队列
            for neighbor_id in self.graph.neighbors(current_id):
                if neighbor_id not in visited:
                    queue.append((neighbor_id, current_depth + 1))

        return neighbors

    def find_path(self, source_id: str, target_id: str) -> List[str]:
        """查找实体间路径"""
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return []

    def search_entities(self, query: str, entity_type: str = None) -> List[KnowledgeEntity]:
        """搜索实体"""
        results = []
        query_lower = query.lower()

        for entity in self.entities.values():
            if entity_type and entity.type != entity_type:
                continue

            if (query_lower in entity.name.lower() or
                query_lower in entity.description.lower()):
                results.append(entity)

        return results

class GraphRAGSystem:
    """GraphRAG系统"""

    def __init__(self):
        self.knowledge_graph = None
        self.traditional_rag = None
        self.graph_builder = KnowledgeGraphBuilder()

    def build_knowledge_graph(self, documents: List[Dict]):
        """构建知识图谱"""
        print("🕸️ 构建知识图谱...")
        self.knowledge_graph = self.graph_builder.build_from_documents(documents)

        # 统计信息
        entity_count = len(self.graph_builder.entities)
        relation_count = len(self.graph_builder.relations)

        print(f"✅ 知识图谱构建完成:")
        print(f"   - 实体数量: {entity_count}")
        print(f"   - 关系数量: {relation_count}")
        print(f"   - 图密度: {nx.density(self.knowledge_graph):.4f}")

    def _enrich_query_with_graph(self, query: str) -> Dict[str, Any]:
        """使用知识图谱丰富查询"""
        if not self.knowledge_graph:
            return {'enriched_query': query, 'related_entities': [], 'graph_paths': []}

        # 查找相关实体
        related_entities = self.graph_builder.search_entities(query)

        # 构建丰富化的查询
        enriched_context = []
        for entity in related_entities[:5]:  # 限制数量
            enriched_context.append(f"{entity.name}({entity.type}): {entity.description}")

        # 查找实体间关系路径
        graph_paths = []
        if len(related_entities) >= 2:
            for i in range(min(3, len(related_entities))):
                for j in range(i + 1, min(3, len(related_entities))):
                    path = self.graph_builder.find_path(
                        related_entities[i].id,
                        related_entities[j].id
                    )
                    if path:
                        graph_paths.append(path)

        enriched_query = f"""
原始查询: {query}

相关知识实体:
{chr(10).join([f"- {e.name}: {e.description[:100]}..." for e in related_entities[:3]])}

请基于以上信息进行深度推理和回答。
"""

        return {
            'enriched_query': enriched_query,
            'related_entities': related_entities,
            'graph_paths': graph_paths
        }

    def _graph_based_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """基于知识图谱的检索"""
        if not self.knowledge_graph:
            return []

        # 查找相关实体
        related_entities = self.graph_builder.search_entities(query)

        # 获取邻居实体
        neighbor_entities = []
        for entity in related_entities[:3]:
            neighbors = self.graph_builder.get_neighbors(entity.id, depth=2)
            neighbor_entities.extend(neighbors)

        # 合并相关实体和邻居实体
        all_relevant_entities = list(set(related_entities + neighbor_entities))

        # 转换为文档格式
        retrieved_docs = []
        for entity in all_relevant_entities[:top_k]:
            doc = {
                'id': entity.id,
                'content': entity.description,
                'title': entity.name,
                'entity_type': entity.type,
                'source': 'knowledge_graph',
                'score': 0.8  # 简化版评分
            }
            retrieved_docs.append(doc)

        return retrieved_docs

    def generate_with_graph_reasoning(self, query: str, context_docs: List[Dict],
                                    graph_context: Dict) -> str:
        """基于图谱推理生成回答"""
        # 构建增强的上下文
        enhanced_context = self._build_enhanced_context(context_docs, graph_context)

        prompt = f"""基于以下信息回答用户查询，要求进行深度推理：

查询: {query}

传统检索文档:
{chr(10).join([f"文档{i+1}: {doc['content'][:200]}..." for i, doc in enumerate(context_docs)])}

知识图谱信息:
{enhanced_context}

请提供准确、深度、结构化的回答，尽可能利用知识图谱中的关系信息："""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个具备知识图谱推理能力的智能助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"生成失败: {str(e)}"

    def _build_enhanced_context(self, context_docs: List[Dict], graph_context: Dict) -> str:
        """构建增强的上下文信息"""
        context_parts = []

        # 添加相关实体信息
        if graph_context['related_entities']:
            context_parts.append("相关实体:")
            for entity in graph_context['related_entities'][:5]:
                context_parts.append(f"- {entity.name} ({entity.type}): {entity.description[:150]}...")

        # 添加实体关系信息
        if graph_context['graph_paths']:
            context_parts.append("\n实体关系路径:")
            for i, path in enumerate(graph_context['graph_paths'][:3]):
                path_names = [self.graph_builder.entities[node_id].name for node_id in path]
                context_parts.append(f"路径{i+1}: {' → '.join(path_names)}")

        return "\n".join(context_parts)

    def search(self, query: str, use_graph_enhancement: bool = True, top_k: int = 5) -> Dict:
        """执行GraphRAG搜索"""
        if not self.knowledge_graph:
            return {
                'answer': '知识图谱尚未构建',
                'sources': [],
                'graph_info': None,
                'error': 'knowledge_graph_not_built'
            }

        try:
            # 1. 基于图谱的查询增强
            if use_graph_enhancement:
                graph_context = self._enrich_query_with_graph(query)
                enriched_query = graph_context['enriched_query']
            else:
                graph_context = {'related_entities': [], 'graph_paths': []}
                enriched_query = query

            # 2. 图谱检索
            graph_docs = self._graph_based_retrieval(query, top_k)

            # 3. 生成回答
            answer = self.generate_with_graph_reasoning(
                enriched_query, graph_docs, graph_context
            )

            # 4. 构建结果
            result = {
                'answer': answer,
                'sources': graph_docs,
                'graph_info': {
                    'related_entities_count': len(graph_context['related_entities']),
                    'graph_paths_count': len(graph_context['graph_paths']),
                    'graph_nodes': self.knowledge_graph.number_of_nodes(),
                    'graph_edges': self.knowledge_graph.number_of_edges()
                },
                'query_enhancement': use_graph_enhancement
            }

            return result

        except Exception as e:
            return {
                'answer': f'搜索过程中出现错误: {str(e)}',
                'sources': [],
                'graph_info': None,
                'error': str(e)
            }

    def visualize_knowledge_graph(self, output_file: str = "knowledge_graph.png"):
        """可视化知识图谱"""
        if not self.knowledge_graph:
            print("❌ 知识图谱未构建")
            return

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            # 使用spring layout布局
            pos = nx.spring_layout(self.knowledge_graph, k=1, iterations=50)

            # 按实体类型设置颜色
            entity_types = set(self.graph_builder.entities[e]['type']
                             for e in self.graph_builder.entities)
            colors = plt.cm.Set3(range(len(entity_types)))
            color_map = {etype: colors[i] for i, etype in enumerate(entity_types)}

            node_colors = []
            for node in self.knowledge_graph.nodes():
                entity_type = self.graph_builder.entities[node]['type']
                node_colors.append(color_map[entity_type])

            # 绘制图谱
            nx.draw(self.knowledge_graph, pos,
                   node_color=node_colors,
                   node_size=1000,
                   font_size=8,
                   font_weight='bold',
                   with_labels=True,
                   labels={node: self.graph_builder.entities[node]['name']
                          for node in self.knowledge_graph.nodes()},
                   edge_color='gray',
                   arrows=True,
                   arrowsize=20)

            plt.title("知识图谱可视化", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"✅ 知识图谱已保存为 {output_file}")

        except ImportError:
            print("❌ 需要安装matplotlib和networkx进行可视化")
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
```

## 4. 多模态RAG系统

### 4.1 多模态融合架构

```python
from typing import List, Dict, Any, Union, Optional
import base64
import io
from PIL import Image
import numpy as np
from dataclasses import dataclass
import requests

@dataclass
class MultimodalContent:
    """多模态内容"""
    text: str = ""
    image: Optional[str] = None  # base64编码
    audio: Optional[str] = None  # 音频文件路径
    video: Optional[str] = None  # 视频文件路径
    metadata: Dict[str, Any] = None

class MultimodalEmbedding:
    """多模态嵌入模型"""

    def __init__(self):
        # 简化版，实际应该使用专门的多模态模型
        self.text_embedding_model = "text-embedding-ada-002"
        self.image_embedding_model = "clip-vit-base-patch32"  # 示例

    def embed_text(self, text: str) -> np.ndarray:
        """文本嵌入"""
        # 简化版，实际应该调用真实的嵌入模型
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()

        # 将哈希值转换为向量（仅用于演示）
        vector = np.array([int(hash_hex[i:i+2], 16) / 255.0
                          for i in range(0, min(len(hash_hex), 768), 2)])

        # 填充到固定长度
        if len(vector) < 384:
            vector = np.pad(vector, (0, 384 - len(vector)))
        else:
            vector = vector[:384]

        return vector

    def embed_image(self, image_base64: str) -> np.ndarray:
        """图像嵌入"""
        try:
            # 解码base64图像
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            # 简化版：基于图像特征生成向量
            # 实际应该使用CLIP等专门的图像嵌入模型
            image_array = np.array(image.resize((224, 224)))

            # 计算简单的统计特征
            features = [
                np.mean(image_array),  # 平均值
                np.std(image_array),   # 标准差
                np.median(image_array) # 中位数
            ]

            # 扩展到完整向量长度
            vector = np.random.RandomState(hash(tuple(image_array.flatten())) % 2**32).random(384)
            vector[:len(features)] = features

            return vector

        except Exception as e:
            print(f"图像嵌入失败: {e}")
            return np.random.random(384)

    def embed_multimodal(self, content: MultimodalContent) -> np.ndarray:
        """多模态嵌入"""
        embeddings = []

        if content.text:
            text_emb = self.embed_text(content.text)
            embeddings.append(text_emb)

        if content.image:
            image_emb = self.embed_image(content.image)
            embeddings.append(image_emb)

        if not embeddings:
            return np.random.random(384)

        # 简单的平均融合
        combined_emb = np.mean(embeddings, axis=0)
        return combined_emb

class MultimodalRetriever:
    """多模态检索器"""

    def __init__(self):
        self.embedding_model = MultimodalEmbedding()
        self.document_store = []
        self.embeddings = []

    def add_documents(self, documents: List[MultimodalContent]):
        """添加多模态文档"""
        for doc in documents:
            # 计算嵌入
            embedding = self.embedding_model.embed_multimodal(doc)

            self.document_store.append(doc)
            self.embeddings.append(embedding)

    def search(self, query: Union[str, MultimodalContent], top_k: int = 5) -> List[Dict]:
        """多模态搜索"""
        # 计算查询嵌入
        if isinstance(query, str):
            query_content = MultimodalContent(text=query)
        else:
            query_content = query

        query_embedding = self.embedding_model.embed_multimodal(query_content)

        # 计算相似度
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # 余弦相似度
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((i, similarity))

        # 排序并返回结果
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, similarity in similarities[:top_k]:
            doc = self.document_store[idx]

            result = {
                'content': doc,
                'similarity': similarity,
                'modality_types': self._detect_modality_types(doc)
            }
            results.append(result)

        return results

    def _detect_modality_types(self, content: MultimodalContent) -> List[str]:
        """检测内容类型"""
        types = []
        if content.text:
            types.append('text')
        if content.image:
            types.append('image')
        if content.audio:
            types.append('audio')
        if content.video:
            types.append('video')
        return types

class MultimodalRAG:
    """多模态RAG系统"""

    def __init__(self):
        self.retriever = MultimodalRetriever()
        self.generation_model = "gpt-4-vision-preview"  # 支持多模态的模型

    def add_multimodal_documents(self, documents: List[Dict]):
        """添加多模态文档"""
        multimodal_docs = []

        for doc in documents:
            content = MultimodalContent(
                text=doc.get('text', ''),
                image=doc.get('image'),
                audio=doc.get('audio'),
                video=doc.get('video'),
                metadata=doc.get('metadata', {})
            )
            multimodal_docs.append(content)

        self.retriever.add_documents(multimodal_docs)

    def _prepare_multimodal_context(self, retrieved_docs: List[Dict]) -> str:
        """准备多模态上下文"""
        context_parts = []

        for i, result in enumerate(retrieved_docs):
            content = result['content']
            modality_types = result['modality_types']
            similarity = result['similarity']

            context_part = f"文档{i+1} (相似度: {similarity:.3f}):\n"

            if content.text:
                context_part += f"文本: {content.text}\n"

            if content.image:
                context_part += f"[包含图像内容]\n"

            if content.audio:
                context_part += f"[包含音频内容]\n"

            if content.video:
                context_part += f"[包含视频内容]\n"

            context_parts.append(context_part)

        return "\n".join(context_parts)

    def generate_multimodal_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """生成多模态响应"""
        context = self._prepare_multimodal_context(retrieved_docs)

        prompt = f"""基于以下多模态内容回答用户查询：

查询: {query}

相关内容:
{context}

请提供准确、全面的回答。如果内容包含图像、音频或视频，请在回答中提及这些信息。"""

        try:
            response = openai.ChatCompletion.create(
                model=self.generation_model,
                messages=[
                    {"role": "system", "content": "你是一个能够理解和处理多模态信息的智能助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"生成失败: {str(e)}"

    def search(self, query: str, query_image: str = None, top_k: int = 5) -> Dict:
        """执行多模态搜索"""
        # 构建查询内容
        if query_image:
            query_content = MultimodalContent(
                text=query,
                image=query_image
            )
        else:
            query_content = MultimodalContent(text=query)

        # 执行检索
        retrieved_docs = self.retriever.search(query_content, top_k)

        if not retrieved_docs:
            return {
                'answer': '抱歉，没有找到相关信息。',
                'sources': [],
                'modality_summary': {'text': 0, 'image': 0, 'audio': 0, 'video': 0}
            }

        # 生成回答
        answer = self.generate_multimodal_response(query, retrieved_docs)

        # 统计模态类型
        modality_counts = {'text': 0, 'image': 0, 'audio': 0, 'video': 0}
        for result in retrieved_docs:
            for mod_type in result['modality_types']:
                if mod_type in modality_counts:
                    modality_counts[mod_type] += 1

        return {
            'answer': answer,
            'sources': [
                {
                    'content': result['content'],
                    'similarity': result['similarity'],
                    'modality_types': result['modality_types']
                }
                for result in retrieved_docs
            ],
            'modality_summary': modality_counts
        }

# 示例使用
def demonstrate_multimodal_rag():
    """演示多模态RAG系统"""
    print("🎭 多模态RAG系统演示")
    print("=" * 50)

    # 创建多模态RAG系统
    rag = MultimodalRAG()

    # 添加示例文档
    sample_documents = [
        {
            'text': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。',
            'metadata': {'source': 'ai_textbook.pdf', 'page': 1}
        },
        {
            'text': '深度学习使用多层神经网络来学习数据的复杂模式。',
            'image': 'base64_encoded_image_placeholder',  # 实际应该是真实的base64编码
            'metadata': {'source': 'dl_guide.pdf', 'page': 5}
        },
        {
            'text': '机器学习算法可以分为监督学习、无监督学习和强化学习三类。',
            'audio': 'path_to_audio_file.mp3',
            'metadata': {'source': 'ml_lecture.mp3', 'timestamp': '10:30'}
        }
    ]

    rag.add_multimodal_documents(sample_documents)
    print(f"✅ 已添加 {len(sample_documents)} 个多模态文档")

    # 执行搜索
    test_queries = [
        "什么是人工智能？",
        "深度学习的原理是什么？",
        "机器学习有哪些类型？"
    ]

    for query in test_queries:
        print(f"\n🔍 查询: {query}")

        result = rag.search(query)

        print(f"📝 回答: {result['answer'][:100]}...")
        print(f"📊 模态统计: {result['modality_summary']}")
        print(f"📚 相关文档数: {len(result['sources'])}")

if __name__ == "__main__":
    demonstrate_multimodal_rag()
```

## 5. 未来发展趋势与展望

### 5.1 技术演进方向

```python
from enum import Enum
from typing import List, Dict, Any
import datetime

class RAGTrend(Enum):
    """RAG技术趋势"""
    ADAPTIVE_LEARNING = "adaptive_learning"
    REAL_TIME_UPDATES = "real_time_updates"
    PERSONALIZATION = "personalization"
    MULTI_MODAL = "multi_modal"
    AGENT_INTEGRATION = "agent_integration"
    EXPLAINABLE_AI = "explainable_ai"
    PRIVACY_PRESERVING = "privacy_preserving"
    QUANTUM_COMPUTING = "quantum_computing"

class RAGFuturePredictor:
    """RAG技术发展预测器"""

    def __init__(self):
        self.current_capabilities = self._assess_current_capabilities()
        self.technology_roadmap = self._create_roadmap()
        self.market_trends = self._analyze_market_trends()

    def _assess_current_capabilities(self) -> Dict[str, float]:
        """评估当前技术能力"""
        return {
            'accuracy': 0.75,           # 回答准确性
            'speed': 0.80,              # 响应速度
            'scalability': 0.70,        # 可扩展性
            'multimodal_support': 0.60, # 多模态支持
            'personalization': 0.50,    # 个性化能力
            'real_time_learning': 0.40, # 实时学习能力
            'explainability': 0.45,     # 可解释性
            'privacy_protection': 0.65   # 隐私保护
        }

    def _create_roadmap(self) -> Dict[str, Dict]:
        """创建技术路线图"""
        return {
            '2024_Q1': {
                'technologies': [RAGTrend.MULTI_MODAL, RAGTrend.ADAPTIVE_LEARNING],
                'maturity': 'emerging',
                'impact': 'high',
                'description': '多模态融合和自适应学习成为主流'
            },
            '2024_Q3': {
                'technologies': [RAGTrend.REAL_TIME_UPDATES, RAGTrend.PERSONALIZATION],
                'maturity': 'developing',
                'impact': 'very_high',
                'description': '实时更新和深度个性化突破'
            },
            '2025_Q1': {
                'technologies': [RAGTrend.AGENT_INTEGRATION, RAGTrend.EXPLAINABLE_AI],
                'maturity': 'early_adopting',
                'impact': 'transformative',
                'description': '智能代理集成和可解释AI成熟'
            },
            '2025_Q4': {
                'technologies': [RAGTrend.PRIVACY_PRESERVING, RAGTrend.QUANTUM_COMPUTING],
                'maturity': 'research',
                'impact': 'breakthrough',
                'description': '隐私保护技术和量子计算应用'
            }
        }

    def _analyze_market_trends(self) -> Dict[str, Any]:
        """分析市场趋势"""
        return {
            'adoption_rate': {
                'enterprise': 0.65,  # 企业采用率
                'education': 0.45,   # 教育领域采用率
                'healthcare': 0.35,  # 医疗领域采用率
                'government': 0.25   # 政府采用率
            },
            'investment_trends': {
                'venture_capital': 'increasing',
                'corporate_rd': 'rapid_growth',
                'government_funding': 'steady_increase'
            },
            'challenges': [
                'data_quality issues',
                'computational costs',
                'privacy concerns',
                'integration complexity'
            ]
        }

    def predict_future_capabilities(self, years: int = 3) -> Dict[str, float]:
        """预测未来能力"""
        future_capabilities = self.current_capabilities.copy()

        # 技术进步率（年复合增长率）
        improvement_rates = {
            'accuracy': 0.08,           # 8%年增长
            'speed': 0.12,              # 12%年增长
            'scalability': 0.10,        # 10%年增长
            'multimodal_support': 0.20, # 20%年增长
            'personalization': 0.25,    # 25%年增长
            'real_time_learning': 0.18, # 18%年增长
            'explainability': 0.15,     # 15%年增长
            'privacy_protection': 0.12  # 12%年增长
        }

        for capability, rate in improvement_rates.items():
            future_capabilities[capability] = min(1.0,
                future_capabilities[capability] * (1 + rate) ** years)

        return future_capabilities

    def generate_future_report(self) -> str:
        """生成未来发展趋势报告"""
        current_year = datetime.datetime.now().year

        # 预测未来能力
        future_1_year = self.predict_future_capabilities(1)
        future_3_years = self.predict_future_capabilities(3)
        future_5_years = self.predict_future_capabilities(5)

        report = f"""
# RAG技术发展前景报告 ({current_year})

## 当前能力评估
"""

        for capability, score in self.current_capabilities.items():
            report += f"- **{capability}**: {score:.1%}\n"

        report += f"""
## 近期预测 (1-2年)

### 技术突破点
"""

        for period, info in self.technology_roadmap.items():
            if '2024' in period:
                report += f"""
#### {period}
- **主要技术**: {', '.join([t.value for t in info['technologies']])}
- **成熟度**: {info['maturity']}
- **影响程度**: {info['impact']}
- **描述**: {info['description']}
"""

        report += f"""
### 能力提升预测 ({current_year + 1}年)
"""

        for capability, score in future_1_year.items():
            improvement = ((score - self.current_capabilities[capability]) /
                          self.current_capabilities[capability]) * 100
            report += f"- **{capability}**: {score:.1%} (+{improvement:+.1f}%)\n"

        report += f"""
## 中期展望 (3-5年)

### 变革性技术
"""

        for period, info in self.technology_roadmap.items():
            if '2025' in period:
                report += f"""
#### {period}
- **主要技术**: {', '.join([t.value for t in info['technologies']])}
- **描述**: {info['description']}
"""

        report += f"""
### 能力发展预测 ({current_year + 3}年)
"""

        for capability, score in future_3_years.items():
            improvement = ((score - self.current_capabilities[capability]) /
                          self.current_capabilities[capability]) * 100
            report += f"- **{capability}**: {score:.1%} (+{improvement:+.1f}%)\n"

        report += f"""
## 长期愿景 (5年以上)

### 颠覆性变革 ({current_year + 5}年)
"""

        for capability, score in future_5_years.items():
            improvement = ((score - self.current_capabilities[capability]) /
                          self.current_capabilities[capability]) * 100
            report += f"- **{capability}**: {score:.1%} (+{improvement:+.1f}%)\n"

        report += """
## 应用场景展望

### 1. 智能个人助理
- 24/7全天候个性化服务
- 跨模态信息理解和生成
- 情感智能和主动关怀

### 2. 企业知识管理
- 实时知识更新和共享
- 智能决策支持系统
- 组织学习和创新能力

### 3. 教育和培训
- 个性化学习路径
- 智能辅导和评估
- 终身学习伴侣

### 4. 医疗健康
- 精准诊断支持
- 个性化治疗方案
- 实时健康监测

### 5. 科研创新
- 文献智能分析
- 假设生成和验证
- 跨学科知识整合

## 挑战与机遇

### 主要挑战
1. **技术挑战**
   - 计算资源需求
   - 数据质量和偏见
   - 系统可靠性和稳定性

2. **伦理挑战**
   - 隐私保护
   - 算法公平性
   - 责任归属

3. **商业挑战**
   - 成本控制
   - 用户体验
   - 竞争格局

### 重大机遇
1. **技术机遇**
   - 量子计算突破
   - 神经形态计算
   - 边缘计算发展

2. **市场机遇**
   - 数字化转型加速
   - 个性化需求增长
   - 新兴应用场景

3. **社会机遇**
   - 知识民主化
   - 教育普及化
   - 医疗普惠化

## 投资建议

### 短期投资 (1-2年)
- 多模态融合技术
- 自适应学习算法
- 隐私保护方案

### 中期投资 (3-5年)
- 智能代理集成
- 可解释AI技术
- 实时学习系统

### 长期投资 (5年以上)
- 量子计算应用
- 脑机接口技术
- 通用人工智能

## 结论

RAG技术正处于快速发展的关键时期，未来5年将迎来爆发式增长。通过多模态融合、自适应学习、实时更新等技术的突破，RAG系统将成为人类智能的重要延伸，在各个领域发挥越来越重要的作用。

然而，技术发展也伴随着责任。我们需要在追求技术创新的同时，重视伦理、隐私和社会影响，确保RAG技术的发展能够造福全人类。

---

*本报告基于当前技术趋势和市场分析，预测结果仅供参考，实际发展可能因技术突破、市场变化等因素而有所不同。*
"""

        return report

# 示例使用
def generate_rag_future_report():
    """生成RAG未来发展报告"""
    predictor = RAGFuturePredictor()
    report = predictor.generate_future_report()

    # 保存报告
    with open('RAG_发展前景报告.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("📊 RAG技术发展前景报告已生成")
    print("📄 报告保存为: RAG_发展前景报告.md")

    # 输出关键预测
    future_3_years = predictor.predict_future_capabilities(3)
    print("\n🔮 3年后RAG系统能力预测:")
    for capability, score in future_3_years.items():
        print(f"   {capability}: {score:.1%}")

if __name__ == "__main__":
    generate_rag_future_report()
```

## 6. 单元测试

### 6.1 前沿技术测试

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

class TestSelfRAG:
    """Self-RAG系统测试"""

    def setup_method(self):
        """测试前准备"""
        self.self_rag = SelfRAGSystem()
        self.test_query = "什么是机器学习？"
        self.test_documents = [
            {
                'id': 'doc1',
                'content': '机器学习是人工智能的一个分支，使计算机能够从数据中学习规律。',
                'title': '机器学习基础'
            }
        ]

    def test_reflection_token_detection(self):
        """测试反思标记检测"""
        test_content = "根据文档内容，[Relevant]该文档与查询相关。"

        is_relevant = ReflectionToken.RELEVANT.value in test_content
        assert is_relevant == True

    def test_relevance_check(self):
        """测试相关性检查"""
        with patch('openai.ChatCompletion.create') as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "[Relevant]"
            mock_openai.return_value = mock_response

            result = self.self_rag._check_relevance(self.test_query, self.test_documents[0])

            assert result['is_relevant'] == True
            assert result['confidence'] == 0.8

    def test_support_check(self):
        """测试支持性检查"""
        with patch('openai.ChatCompletion.create') as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "[Supported]"
            mock_openai.return_value = mock_response

            result = self.self_rag._check_support("机器学习是AI分支", self.test_documents[0])

            assert result['is_supported'] == True

    def test_self_correction(self):
        """测试自我纠错"""
        with patch('openai.ChatCompletion.create') as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "机器学习是人工智能的重要分支。"
            mock_openai.return_value = mock_response

            corrected_answer = self.self_rag.self_correct(
                self.test_query,
                "机器学习很重要",
                "需要更详细解释",
                self.test_documents[0]
            )

            assert len(corrected_answer) > 0

class TestGraphRAG:
    """GraphRAG系统测试"""

    def setup_method(self):
        """测试前准备"""
        self.graph_builder = KnowledgeGraphBuilder()
        self.test_documents = [
            {
                'id': 'doc1',
                'content': '张三在阿里巴巴工作，负责人工智能项目。',
                'title': '员工介绍'
            }
        ]

    def test_entity_extraction(self):
        """测试实体提取"""
        entities = self.graph_builder._extract_entities(
            self.test_documents[0]['content'],
            self.test_documents[0]['id']
        )

        assert len(entities) > 0
        entity_names = [e.name for e in entities]
        assert '张三' in entity_names or '阿里巴巴' in entity_names

    def test_relation_extraction(self):
        """测试关系提取"""
        entities = self.graph_builder._extract_entities(
            self.test_documents[0]['content'],
            self.test_documents[0]['id']
        )
        relations = self.graph_builder._extract_relations(
            self.test_documents[0]['content'],
            entities
        )

        # 可能不能提取到关系，因为简单的规则可能不够完善
        assert isinstance(relations, list)

    def test_knowledge_graph_construction(self):
        """测试知识图谱构建"""
        graph = self.graph_builder.build_from_documents(self.test_documents)

        assert graph is not None
        assert graph.number_of_nodes() >= 0
        assert graph.number_of_edges() >= 0

    def test_entity_search(self):
        """测试实体搜索"""
        self.graph_builder.build_from_documents(self.test_documents)

        # 搜索存在的实体
        results = self.graph_builder.search_entities("张三")
        assert isinstance(results, list)

        # 搜索不存在的实体
        results = self.graph_builder.search_entities("不存在的实体")
        assert isinstance(results, list)

class TestMultimodalRAG:
    """多模态RAG系统测试"""

    def setup_method(self):
        """测试前准备"""
        self.multimodal_rag = MultimodalRAG()
        self.embedding_model = MultimodalEmbedding()

    def test_text_embedding(self):
        """测试文本嵌入"""
        text = "这是一个测试文本"
        embedding = self.embedding_model.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384

    def test_multimodal_content_creation(self):
        """测试多模态内容创建"""
        content = MultimodalContent(
            text="测试文本",
            metadata={'source': 'test'}
        )

        assert content.text == "测试文本"
        assert content.metadata['source'] == 'test'

    def test_multimodal_embedding(self):
        """测试多模态嵌入"""
        content = MultimodalContent(text="测试文本")
        embedding = self.embedding_model.embed_multimodal(content)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384

    def test_multimodal_retrieval(self):
        """测试多模态检索"""
        # 添加测试文档
        test_docs = [
            MultimodalContent(text="机器学习是人工智能的分支"),
            MultimodalContent(text="深度学习使用神经网络")
        ]

        self.multimodal_rag.retriever.add_documents(test_docs)

        # 执行搜索
        results = self.multimodal_rag.retriever.search("机器学习")

        assert isinstance(results, list)
        assert len(results) > 0
        assert 'similarity' in results[0]

class TestRAGFuturePredictor:
    """RAG发展预测器测试"""

    def setup_method(self):
        """测试前准备"""
        self.predictor = RAGFuturePredictor()

    def test_current_capabilities_assessment(self):
        """测试当前能力评估"""
        capabilities = self.predictor.current_capabilities

        assert isinstance(capabilities, dict)
        assert 'accuracy' in capabilities
        assert 'speed' in capabilities
        assert all(0.0 <= cap <= 1.0 for cap in capabilities.values())

    def test_future_capability_prediction(self):
        """测试未来能力预测"""
        future_1_year = self.predictor.predict_future_capabilities(1)
        future_3_years = self.predictor.predict_future_capabilities(3)

        assert isinstance(future_1_year, dict)
        assert isinstance(future_3_years, dict)

        # 未来能力应该不低于当前能力（在理想情况下）
        for capability in future_1_year:
            assert future_1_year[capability] >= self.predictor.current_capabilities[capability] * 0.9

        # 3年后的能力应该比1年后更好
        for capability in future_3_years:
            assert future_3_years[capability] >= future_1_year[capability] * 0.95

    def test_roadmap_creation(self):
        """测试技术路线图创建"""
        roadmap = self.predictor.technology_roadmap

        assert isinstance(roadmap, dict)
        assert len(roadmap) > 0

        for period, info in roadmap.items():
            assert 'technologies' in info
            assert 'maturity' in info
            assert 'impact' in info
            assert 'description' in info

    def test_future_report_generation(self):
        """测试未来报告生成"""
        report = self.predictor.generate_future_report()

        assert isinstance(report, str)
        assert len(report) > 1000  # 报告应该足够长
        assert 'RAG技术发展前景报告' in report
        assert '当前能力评估' in report
        assert '未来预测' in report

class TestIntegration:
    """集成测试"""

    def test_self_rag_integration(self):
        """测试Self-RAG集成功能"""
        self_rag = SelfRAGSystem()

        with patch('openai.ChatCompletion.create') as mock_openai:
            # 模拟所有API调用
            mock_openai.return_value.choices = [MagicMock()]
            mock_openai.return_value.choices[0].message.content = "测试回答"

            result = self_rag.generate_with_self_reflection(
                "测试查询",
                [{'content': '测试文档', 'title': '测试'}],
                max_iterations=2
            )

            assert 'final_answer' in result
            assert 'iterations' in result
            assert 'generation_log' in result
            assert result['iterations'] >= 1

    def test_graph_rag_integration(self):
        """测试GraphRAG集成功能"""
        graph_rag = GraphRAG()

        test_docs = [
            {
                'id': 'test1',
                'content': '人工智能包括机器学习和深度学习。',
                'title': 'AI概述'
            }
        ]

        graph_rag.build_knowledge_graph(test_docs)

        assert graph_rag.knowledge_graph is not None
        assert graph_rag.knowledge_graph.number_of_nodes() > 0

    def test_multimodal_rag_integration(self):
        """测试多模态RAG集成功能"""
        multimodal_rag = MultimodalRAG()

        test_docs = [
            {
                'text': '这是一个测试文档',
                'metadata': {'source': 'test'}
            }
        ]

        multimodal_rag.add_multimodal_documents(test_docs)

        # 验证文档已添加
        assert len(multimodal_rag.retriever.document_store) == 1

        # 执行搜索
        result = multimodal_rag.search("测试")

        assert 'answer' in result
        assert 'sources' in result
        assert 'modality_summary' in result

if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
```

## 7. 总结

本文全面探讨了RAG技术的最前沿发展和未来展望，通过深入分析Self-RAG、GraphRAG、多模态RAG等先进技术，展现了RAG技术的演进路径和发展潜力。

### 7.1 关键洞察

**1. 技术融合趋势**
- RAG正从简单的检索增强向复杂的多技术融合发展
- 知识图谱、自反思机制、多模态处理等技术正在重塑RAG架构
- 未来RAG系统将具备更强的推理、学习和适应能力

**2. 性能提升路径**
- Self-RAG通过自我反思机制显著提高回答质量
- GraphRAG利用知识图谱增强推理能力
- 多模态RAG扩展了信息处理和理解的边界

**3. 应用前景广阔**
- 从简单的问答系统向智能助理发展
- 从信息检索向知识创造演进
- 从单一模态向全模态感知扩展

### 7.2 未来展望

**短期发展（1-2年）**：
- 多模态融合技术成熟
- 自适应学习机制普及
- 实时更新能力增强

**中期突破（3-5年）**：
- 智能代理深度集成
- 可解释AI广泛应用
- 隐私保护技术完善

**长期愿景（5年以上）**：
- 通用人工智能特征显现
- 量子计算赋能新突破
- 人机协作新范式形成

### 7.3 实践建议

**1. 技术选型**
- 根据应用场景选择合适的RAG架构
- 关注技术的成熟度和可维护性
- 重视数据质量和知识图谱建设

**2. 系统设计**
- 采用模块化架构支持技术演进
- 建立完善的评估和监控体系
- 重视用户体验和性能优化

**3. 风险管控**
- 关注数据隐私和安全性
- 建立伦理审查机制
- 准备应对技术变革的灵活性

RAG技术正站在智能信息处理革命的前沿，它不仅改变了我们获取和利用信息的方式，更为人工智能的发展开辟了新的道路。随着技术的不断进步和应用的深入，RAG必将在构建更加智能、高效、人性化的信息系统中发挥越来越重要的作用。

---

*本文深入探讨了RAG技术的前沿发展和未来趋势，为研究者和实践者提供了全面的技术洞察和发展指导。技术的未来充满无限可能，让我们共同期待RAG技术带来的精彩变革。*