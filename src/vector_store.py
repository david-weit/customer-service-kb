from typing import Dict, List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from src.utils import sanitize_text


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
        self.hybrid_retriever = None
        self.documents: List[Document] = []
        self._load_or_create()

    def create_hybrid_retriever(
        self, documents: Optional[List[Document]] = None
    ) -> None:
        """创建混合检索器（需在文档入库后调用）。"""
        from .hybrid_retriever import HybridRetriever

        docs = documents if documents is not None else self.documents
        if not docs:
            self.hybrid_retriever = None
            return

        self.documents = docs
        self.hybrid_retriever = HybridRetriever(
            self.vectorstore,
            self.documents,
            weights=[0.5, 0.5],
            k=config.TOP_K,
        )
        print("✅ 混合检索器创建完成")

    def _load_or_create(self):
        """加载或创建向量库。"""
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        print(f"📂 向量库就绪: {self.collection_name}")

    def reset_collection(self) -> None:
        """清空向量库 collection，重置文档与混合检索器。"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.documents = []
        self.hybrid_retriever = None
        self._load_or_create()
        print("✅ 向量库已重置")

    def add_documents(self, docs: List[Document], metadata: Optional[Dict] = None):
        """添加文档到知识库。

        默认按 CHUNK_SIZE 分块；若 Document.metadata['pre_chunked'] 为 True
        （如 Excel 行 / JSON 结构块），则跳过二次切分。
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

        ready: List[Document] = []
        to_split: List[Document] = []
        for doc in docs:
            doc.page_content = sanitize_text(doc.page_content)
            if not doc.page_content.strip():
                continue
            if doc.metadata.get("pre_chunked"):
                ready.append(doc)
            else:
                to_split.append(doc)

        if to_split:
            ready.extend(text_splitter.split_documents(to_split))

        if metadata:
            for chunk in ready:
                chunk.metadata.update(metadata)

        if not ready:
            print("⚠️ 没有可入库的文档块")
            return

        self.vectorstore.add_documents(ready)
        self.documents.extend(ready)
        self.create_hybrid_retriever()
        print(f"✅ 添加 {len(ready)} 个文档块到知识库")

    def add_faqs(self, faqs: List[Dict]):
        """添加 FAQ 到知识库（整段入库，不分块）。"""
        docs = []
        for faq in faqs:
            content = sanitize_text(
                f"问题：{faq['question']}\n答案：{faq['answer']}"
            )
            if not content.strip():
                continue
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

        if not docs:
            print("⚠️ 没有可入库的 FAQ")
            return

        self.vectorstore.add_documents(docs)
        self.documents.extend(docs)
        self.create_hybrid_retriever()
        print(f"✅ 添加 {len(docs)} 条FAQ到知识库")

    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """混合检索"""
        limit = k or config.TOP_K
        if self.hybrid_retriever:
            docs = self.hybrid_retriever.retrieve(query)
            return docs[:limit]
        return self.vectorstore.similarity_search(query, k=limit)

    def get_collection_stats(self):
        """获取向量库统计信息。"""
        collection = self.client.get_collection(self.collection_name)
        return {
            "total_documents": collection.count(),
            "collection_name": self.collection_name,
        }
