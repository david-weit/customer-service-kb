"""RAG Agent 模块（LangGraph + Function Calling 薄封装）。"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.agent_graph import build_agent_graph, get_db_pool
from src.message_store import list_messages
from src.order_api import MockOrderAPI
from src.utils import sanitize_text
from src.vector_store import KnowledgeBaseManager
from .logger import logger


class RAGAgent:
    """基于 LangGraph Function Calling + Postgres Checkpointer 的客服问答 Agent。"""

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
        logger.info("初始化 RAG Agent (LangGraph + Function Calling + Checkpointer)")
        self.graph = build_agent_graph(
            llm=self.llm,
            kb=self.kb,
            order_api=self.order_api,
            top_k=top_k,
        )

    @staticmethod
    def new_thread_id() -> str:
        """生成新的会话 thread_id。"""
        return str(uuid4())

    def answer(self, question: str, thread_id: str = "default") -> dict:
        """回答问题：由模型决定是否调用工具；同 thread_id 可多轮续聊。"""
        question = sanitize_text(question)
        result = self.graph.invoke(
            {"question": question},
            config={"configurable": {"thread_id": thread_id}},
        )

        payload = {
            "question": result.get("question", question),
            "answer": result.get("answer", ""),
            "contexts": result.get("contexts") or [],
            "thread_id": thread_id,
            "answer_ok": bool(result.get("answer_ok", True)),
            "verify_reason": result.get("verify_reason") or "",
        }
        if "order_info" in result:
            payload["order_info"] = result.get("order_info")
        return payload

    def invoke(self, question: str, thread_id: str = "default") -> str:
        """简化接口，直接返回回答文本。"""
        return self.answer(question, thread_id=thread_id)["answer"]

    def get_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """读取自建表中的完整对话历史（不受 Checkpointer 窗口裁剪影响）。"""
        return list_messages(get_db_pool(), thread_id)


def create_rag_agent(llm, kb: KnowledgeBaseManager) -> RAGAgent:
    """创建 RAG Agent 实例。"""
    return RAGAgent(llm, kb, order_api=MockOrderAPI())
