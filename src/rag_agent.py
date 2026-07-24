"""RAG Agent 模块（LangGraph + Function Calling 薄封装）。"""

from typing import Optional

from src.agent_graph import build_agent_graph
from src.order_api import MockOrderAPI
from src.utils import sanitize_text
from src.vector_store import KnowledgeBaseManager
from .logger import logger


class RAGAgent:
    """基于 LangGraph Function Calling 的客服问答 Agent。"""

    def __init__(
        self,
        llm,
        kb: KnowledgeBaseManager,
        top_k: Optional[int] = None,
        order_api: Optional[MockOrderAPI] = None,
    ):
        self.llm = llm
        self.kb = kb
        self.order_api = order_api or MockOrderAPI()
        logger.info("初始化 RAG Agent (LangGraph + Function Calling)")
        self.graph = build_agent_graph(
            llm=self.llm,
            kb=self.kb,
            order_api=self.order_api,
            top_k=top_k,
        )

    def answer(self, question: str) -> dict:
        """回答问题：由模型决定是否调用 query_order / search_knowledge_base。"""
        question = sanitize_text(question)
        result = self.graph.invoke({"question": question})

        payload = {
            "question": result.get("question", question),
            "answer": result.get("answer", ""),
            "contexts": result.get("contexts") or [],
        }
        if "order_info" in result:
            payload["order_info"] = result.get("order_info")
        return payload

    def invoke(self, question: str) -> str:
        """简化接口，直接返回回答文本。"""
        return self.answer(question)["answer"]


def create_rag_agent(llm, kb: KnowledgeBaseManager) -> RAGAgent:
    """创建 RAG Agent 实例。"""
    return RAGAgent(llm, kb, order_api=MockOrderAPI())
