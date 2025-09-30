"""
Unit tests for SimpleRAG implementation
"""
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from simple_rag import SimpleRAG

class TestSimpleRAG:
    @pytest.fixture
    def rag_instance(self):
        """创建RAG实例的fixture"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            return SimpleRAG()

    @pytest.fixture
    def sample_document(self):
        """创建示例文档的fixture"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("这是一个测试文档。包含一些测试内容。")
            temp_file = f.name

        yield temp_file

        # 清理临时文件
        os.unlink(temp_file)

    def test_load_documents(self, rag_instance, sample_document):
        """测试文档加载功能"""
        documents = rag_instance.load_documents(sample_document)

        assert len(documents) == 1
        assert "测试文档" in documents[0].page_content
        assert documents[0].metadata['source'] == sample_document

    def test_split_documents(self, rag_instance, sample_document):
        """测试文档分割功能"""
        documents = rag_instance.load_documents(sample_document)
        chunks = rag_instance.split_documents(documents, chunk_size=10, chunk_overlap=2)

        assert len(chunks) > 1
        assert all(chunk.page_content for chunk in chunks)

    @patch('simple_rag.OpenAIEmbeddings')
    def test_create_vector_store(self, mock_embeddings, rag_instance, sample_document):
        """测试向量存储创建"""
        mock_embeddings.return_value = MagicMock()

        documents = rag_instance.load_documents(sample_document)
        chunks = rag_instance.split_documents(documents)

        with patch('simple_rag.Chroma') as mock_chroma:
            mock_store = MagicMock()
            mock_chroma.from_documents.return_value = mock_store

            result = rag_instance.create_vector_store(chunks)

            assert result == mock_store
            mock_chroma.from_documents.assert_called_once()

    def test_query_without_qa_chain(self, rag_instance):
        """测试没有QA链时的查询"""
        with pytest.raises(ValueError, match="请先创建QA链"):
            rag_instance.query("测试问题")

# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])