"""
Integration tests for RAG system
"""
import pytest
import tempfile
import os
from dotenv import load_dotenv

# 确保加载环境变量
load_dotenv()

class TestRAGIntegration:
    @pytest.fixture(scope="class")
    def sample_knowledge_base(self):
        """创建测试知识库"""
        kb_content = """
        Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。
        Python的设计哲学强调代码的可读性和简洁的语法。
        Django是一个基于Python的Web框架，遵循MVC架构模式。
        Flask是另一个Python Web框架，更加轻量级和灵活。
        机器学习是人工智能的一个分支，专注于算法的开发。
        TensorFlow和PyTorch是流行的机器学习框架。
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(kb_content)
            temp_file = f.name

        yield temp_file

        os.unlink(temp_file)

    @pytest.mark.integration
    def test_end_to_end_rag_pipeline(self, sample_knowledge_base):
        """端到端RAG流程测试"""
        # 这里需要真实的API key才能运行
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("需要OPENAI_API_KEY环境变量")

        from simple_rag import SimpleRAG

        rag = SimpleRAG()

        # 1. 加载文档
        documents = rag.load_documents(sample_knowledge_base)
        assert len(documents) > 0

        # 2. 分割文档
        chunks = rag.split_documents(documents, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > len(documents)

        # 3. 创建向量存储
        vector_store = rag.create_vector_store(chunks)
        assert vector_store is not None

        # 4. 创建QA链
        qa_chain = rag.create_qa_chain()
        assert qa_chain is not None

        # 5. 进行查询
        result = rag.query("Python是什么时候发布的？")

        assert 'answer' in result
        assert 'source_documents' in result
        assert len(result['source_documents']) > 0
        assert '1991' in result['answer'] or 'Guido' in result['answer']

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])