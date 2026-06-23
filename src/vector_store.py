from typing import Dict, List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


class KnowledgeBaseManager:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = str(persist_dir or config.CHROMA_DB_DIR)
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_LOCAL
        )
        self.client = chromadb.PersistentClient(self.persist_dir)
        self.vectorstore = None
        self._load_or_create()

    def _load_or_create(self):
        """加载或创建向量库。"""
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        print(f"📂 向量库就绪: {self.collection_name}")

    def add_documents(self, docs: List[Document], metadata: Optional[Dict] = None):
        """添加文档到知识库（自动分块）。"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

        chunks = text_splitter.split_documents(docs)

        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        self.vectorstore.add_documents(chunks)
        print(f"✅ 添加 {len(chunks)} 个文档块到知识库")

    def add_faqs(self, faqs: List[Dict]):
        """添加 FAQ 到知识库（整段入库，不分块）。"""
        docs = []
        for faq in faqs:
            content = f"问题：{faq['question']}\n答案：{faq['answer']}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "type": "faq",
                        "category": faq.get("category", ""),
                        "keywords": ",".join(faq.get("keywords", [])),
                        "source": faq.get("source", ""),
                    },
                )
            )

        self.vectorstore.add_documents(docs)
        print(f"✅ 添加 {len(faqs)} 条FAQ到知识库")

    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """检索相关文档。"""
        return self.vectorstore.similarity_search(query, k=k or config.TOP_K)

    def get_collection_stats(self):
        """获取向量库统计信息。"""
        collection = self.client.get_collection(self.collection_name)
        return {
            "total_documents": collection.count(),
            "collection_name": self.collection_name,
        }
