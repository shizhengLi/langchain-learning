# 🦜⛓️ LangChain RAG 学习项目

<p align="center">
  <strong>从基础到进阶的完整 RAG 系统学习指南</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  </a>
  <a href="https://python.langchain.com/">
    <img src="https://img.shields.io/badge/LangChain-0.1.0+-green.svg" alt="LangChain">
  </a>
</p>

---

## 📖 项目简介

这是一个基于 LangChain 框架的 RAG (Retrieval-Augmented Generation) 系统学习项目。通过理论学习和实践代码相结合的方式，帮助你从零开始掌握 RAG 技术的核心概念、实现方法以及生产级应用。

### 🎯 项目特色

- **📚 系统化学习**：从基础概念到高级应用的完整知识体系
- **💻 实战导向**：每个理论都配有可运行的代码示例
- **🧪 测试覆盖**：完整的单元测试和集成测试
- **📈 性能优化**：从原型到生产环境的最佳实践
- **🔧 生产就绪**：包含部署、监控、安全等企业级考虑

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/langchain-learning.git
cd langchain-learning

# 创建虚拟环境
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# rag_env\Scripts\activate  # Windows

# 升级 pip
pip install --upgrade pip
```

### 安装依赖

```bash
# 核心依赖
pip install -r requirements.txt

# 或手动安装
pip install langchain langchain-core langchain-community
pip install openai tiktoken chromadb faiss-cpu
pip install pypdf python-docx unstructured
pip install numpy pandas tqdm python-dotenv
```

### 配置环境变量

创建 `.env` 文件：

```bash
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# 可选配置
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### 运行第一个示例

```bash
# 运行基础 RAG 示例
python simple_rag.py

# 运行测试
python test_simple_rag.py
```

---

## 📚 技术报告

本项目包含 12 篇系统化的技术报告，涵盖从基础到高级的完整 RAG 技术栈：

### 📱 基础篇 (1-4)

| 报告 | 标题 | 难度 | 核心内容 |
|------|------|------|----------|
| [第1篇](./第1篇-LangChain_RAG框架概述与环境搭建.md) | RAG框架概述与环境搭建 | ⭐⭐ | RAG原理、LangChain组件、环境配置 |
| [第2篇](./第2篇-文档加载与文本分割策略.md) | 文档加载与文本分割策略 | ⭐⭐⭐ | 多格式文档处理、分块算法优化 |
| [第3篇](./第3篇-向量嵌入模型深度解析.md) | 向量嵌入模型深度解析 | ⭐⭐⭐ | 嵌入模型对比、质量评估方法 |
| [第4篇](./第4篇-向量数据库存储与检索优化.md) | 向量数据库存储与检索优化 | ⭐⭐⭐⭐ | Chroma/FAISS对比、性能调优 |

### 🚀 进阶篇 (5-8)

| 报告 | 标题 | 难度 | 核心内容 |
|------|------|------|----------|
| [第5篇](./第5篇-检索策略与相似度计算.md) | 检索策略与相似度计算 | ⭐⭐⭐⭐ | 混合检索、重排序算法 |
| [第6篇](./第6篇-提示词工程与上下文构建.md) | 提示词工程与上下文构建 | ⭐⭐⭐⭐ | 上下文管理、Chain设计模式 |
| [第7篇](./第7篇-多模态RAG系统实现.md) | 多模态RAG系统实现 | ⭐⭐⭐⭐⭐ | 图文处理、跨模态检索 |
| [第8篇](./第8篇-RAG系统性能评估与优化.md) | RAG系统性能评估与优化 | ⭐⭐⭐⭐ | 评估指标、性能瓶颈分析 |

### 🏭 高级篇 (9-12)

| 报告 | 标题 | 难度 | 核心内容 |
|------|------|------|----------|
| [第9篇](./第9篇-生产级RAG系统架构设计.md) | 生产级RAG系统架构设计 | ⭐⭐⭐⭐⭐ | 微服务架构、容器化部署 |
| [第10篇](./第10篇-RAG系统安全性与隐私保护.md) | RAG系统安全性与隐私保护 | ⭐⭐⭐⭐⭐ | 数据脱敏、访问控制、恶意防护 |
| [第11篇](./第11篇-RAG与传统搜索系统对比分析.md) | RAG与传统搜索系统对比分析 | ⭐⭐⭐ | 对比实验、混合架构、性能分析 |
| [第12篇](./第12篇-RAG前沿技术与未来展望.md) | RAG前沿技术与未来展望 | ⭐⭐⭐⭐⭐ | Self-RAG、GraphRAG、多模态技术 |

---

## 💻 代码示例

### 核心组件

```
项目结构：
├── simple_rag.py              # 基础RAG实现
├── test_simple_rag.py         # 单元测试
├── test_integration.py        # 集成测试
├── requirements.txt           # 依赖列表
├── .env.example              # 环境变量模板
└── 技术报告/                  # 12篇完整技术报告
    ├── 基础篇 (1-4)           # RAG基础概念与环境搭建
    ├── 进阶篇 (5-8)           # 检索策略与性能优化
    └── 高级篇 (9-12)          # 生产级架构与前沿技术
```

### 快速体验

```python
from simple_rag import SimpleRAG

# 创建RAG实例
rag = SimpleRAG()

# 加载文档
documents = rag.load_documents("your_document.txt")

# 处理文档
chunks = rag.split_documents(documents)

# 创建向量存储
rag.create_vector_store(chunks)

# 创建问答链
rag.create_qa_chain()

# 开始提问
result = rag.query("你的问题是什么？")
print(result['answer'])
```

---

## 🎯 学习路径

### 初学者路径 (2周)

1. **Week 1**: 完成第1-2篇报告
   - 理解 RAG 基本概念
   - 搭建开发环境
   - 实现基础文档处理

2. **Week 2**: 完成第3-4篇报告
   - 掌握嵌入模型原理
   - 学习向量数据库操作
   - 构建完整 RAG 流程

### 进阶开发者路径 (4周)

3. **Week 3-4**: 完成第5-6篇报告
   - 优化检索策略
   - 掌握提示词工程
   - 实现复杂查询处理

4. **Week 5-6**: 完成第7-8篇报告
   - 探索多模态应用
   - 建立评估体系
   - 性能调优实践

### 架构师路径 (6周)

5. **Week 7-10**: 完成第9-10篇报告
   - 设计生产级架构
   - 实施安全措施
   - 部署监控系统

6. **Week 11-12**: 完成第11-12篇报告
   - 对比分析不同搜索技术
   - 掌握Self-RAG、GraphRAG等前沿技术
   - 了解RAG技术发展趋势和商业应用

---

## 🛠️ 技术栈

### 核心框架
- **[LangChain](https://python.langchain.com/)**: LLM 应用开发框架
- **[OpenAI](https://openai.com/)**: 大语言模型 API
- **[Chroma](https://www.trychroma.com/)**: 向量数据库
- **[FAISS](https://faiss.ai/)**: 高效相似度搜索

### 开发工具
- **Python 3.8+**: 编程语言
- **pytest**: 测试框架
- **Jupyter**: 交互式开发
- **Docker**: 容器化部署

### 评估工具
- **ROUGE**: 文本生成评估
- **BLEU**: 翻译质量评估
- **Custom Metrics**: 业务指标评估
- **安全测试**: 漏洞扫描与渗透测试
- **性能监控**: 实时性能指标追踪

---

## 📊 项目进度

<details>
<summary>📈 整体完成度</summary>

### 总体进度: 100% (12/12 篇完成)

- ✅ **基础篇**: 100% 完成 (4/4)
- ✅ **进阶篇**: 100% 完成 (4/4)
- ✅ **高级篇**: 100% 完成 (4/4)
- 📋 **测试覆盖**: 95% 完成
- 📋 **代码示例**: 95% 完成

</details>

<details>
<summary>🧪 测试状态</summary>

```bash
# 运行所有测试
pytest . -v

# 运行特定测试
pytest test_simple_rag.py -v
pytest test_integration.py -v -m integration
```

**测试覆盖情况**:
- ✅ 文档加载功能
- ✅ 文本分割功能
- ✅ 向量存储功能
- ✅ 问答链功能
- ✅ 端到端流程
- ✅ 安全功能测试
- ✅ 性能基准测试
- ✅ 前沿技术验证

</details>

---

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献

1. **Fork** 项目到你的 GitHub
2. **创建** 特性分支 (`git checkout -b feature/AmazingFeature`)
3. **提交** 你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. **推送** 到分支 (`git push origin feature/AmazingFeature`)
5. **创建** Pull Request

### 贡献类型

- 🐛 **Bug 修复**: 发现并修复问题
- 📝 **文档改进**: 完善说明和注释
- 💡 **新功能**: 添加新的示例或工具
- 🧪 **测试增强**: 提高测试覆盖率
- 🎨 **代码优化**: 重构和性能改进
- 🔬 **前沿技术**: 探索最新的RAG技术
- 📊 **基准测试**: 提供性能评估基准

---

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## 🙏 致谢

- [LangChain](https://python.langchain.com/) - 强大的 LLM 开发框架
- [OpenAI](https://openai.com/) - 先进的大语言模型
- [Chroma](https://www.trychroma.com/) - 高性能向量数据库
- [FAISS](https://faiss.ai/) - Facebook AI 相似性搜索库
- 所有贡献者和学习者的支持

---

---

## 🏆 项目亮点

### ✨ 完整的学习体系
- **12篇深度技术报告** - 从基础到前沿的完整知识体系
- **100+可运行代码示例** - 每个概念都有对应的实践代码
- **全面测试覆盖** - 单元测试、集成测试、性能测试、安全测试

### 🚀 前沿技术探索
- **Self-RAG** - 自反思检索增强生成技术
- **GraphRAG** - 知识图谱增强的RAG系统
- **多模态RAG** - 图文音视频的多模态融合
- **混合搜索** - RAG与传统搜索的智能融合

### 🛡️ 企业级应用
- **安全与隐私保护** - GDPR合规、数据脱敏、访问控制
- **性能优化** - 从原型到生产的完整优化方案
- **架构设计** - 微服务、容器化、监控告警
- **质量保证** - 全面的评估指标和基准测试

---

<p align="center">

**🎯 这是最全面的RAG技术学习资源！**<br>
**📚 从入门到精通，从理论到实践**<br>
**🚀 涵盖基础、进阶、高级、前沿的完整技术栈**<br><br>

**⭐ 如果这个项目对你有帮助，请给我们一个 Star！**<br><br>

Made with ❤️ by LangChain Learning Team

</p>