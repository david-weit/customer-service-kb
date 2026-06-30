from typing import Callable, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever


def _chinese_preprocess(text: str) -> List[str]:
    """中文分词预处理，供 BM25 使用。"""
    try:
        import jieba

        return list(jieba.cut(text))
    except ImportError:
        return list[str](text)


class HybridRetriever:
    def __init__(
        self,
        vectorstore,
        documents: List[Document],
        weights: Optional[List[float]] = None,
        k: int = 5,
        preprocess_func: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        weights: [vector_store_weight, bm25_weight]
        """
        if weights is None:
            weights = [0.5, 0.5]

        tokenize = preprocess_func or _chinese_preprocess
        self.bm25_retriever = BM25Retriever.from_documents(
            documents, preprocess_func=tokenize
        )
        self.bm25_retriever.k = k
        self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=weights,
        )

    def retrieve(self, query: str) -> List[Document]:
        """执行混合检索"""
        return self.ensemble_retriever.invoke(query)

    def retrieve_separate(self, query: str):
        """分别执行向量存储检索和BM25检索"""
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        return {
            "vector_docs": vector_docs,
            "bm25_docs": bm25_docs,
        }
