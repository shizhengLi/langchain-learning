"""
Simple RAG implementation for the first technical report
"""
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 加载环境变量
load_dotenv()

class SimpleRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.vector_store = None
        self.qa_chain = None

    def load_documents(self, file_path):
        """加载文档"""
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return documents

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """分割文档"""
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents):
        """创建向量数据库"""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.vector_store.persist()
        return self.vector_store

    def create_qa_chain(self):
        """创建问答链"""
        retriever = self.vector_store.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return self.qa_chain

    def query(self, question):
        """查询RAG系统"""
        if not self.qa_chain:
            raise ValueError("请先创建QA链")

        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }

# 使用示例
def main():
    # 创建RAG实例
    rag = SimpleRAG()

    # 加载示例文档
    sample_text = """
    人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
    机器学习是AI的一个子领域，专注于开发能够从数据中学习的算法。
    深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的工作方式。
    自然语言处理（NLP）是AI的另一个重要分支，专注于计算机与人类语言之间的交互。
    """

    # 保存为临时文件
    with open("sample_text.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    # 加载和处理文档
    documents = rag.load_documents("sample_text.txt")
    chunks = rag.split_documents(documents)

    # 创建向量存储
    rag.create_vector_store(chunks)

    # 创建问答链
    rag.create_qa_chain()

    # 进行查询测试
    questions = [
        "什么是人工智能？",
        "机器学习和深度学习的关系是什么？",
        "自然语言处理的作用是什么？"
    ]

    for question in questions:
        result = rag.query(question)
        print(f"问题: {question}")
        print(f"回答: {result['answer']}")
        print(f"来源: {result['source_documents'][0].page_content[:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()