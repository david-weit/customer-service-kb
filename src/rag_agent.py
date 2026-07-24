"""RAG Agent 模块。"""

from typing import Optional

from langchain_core.documents import Document

import config
from src.intent import detect_intent
from src.order_api import MockOrderAPI
from src.utils import sanitize_text
from src.vector_store import KnowledgeBaseManager
from .logger import logger
from .query_expansion import QueryExpander


class RAGAgent:
    """基于检索增强生成的客服问答 Agent。"""

    def __init__(
        self,
        llm,
        kb: KnowledgeBaseManager,
        top_k: Optional[int] = None,
        order_api: Optional[MockOrderAPI] = None,
    ):
        self.llm = llm
        self.kb = kb
        self.top_k = top_k or config.TOP_K
        self.order_api = order_api or MockOrderAPI()
        logger.info("初始化 RAG Agent, model")
        self.query_expander = QueryExpander(llm)

    def retrieve(self, question: str) -> list[Document]:
        """检索与问题相关的文档片段。"""
        expanded_queries = self.query_expander.expand_query(question)
        print(f"Expanded queries: {expanded_queries}")

        all_doc: list[Document] = []
        all_doc.extend(self.kb.search(question, self.top_k))

        for q in expanded_queries:
            if q.strip() == question.strip():
                continue
            all_doc.extend(self.kb.search(q, config.EXPANDED_QUERY_K))

        seen_contents: set[str] = set()
        unique_docs: list[Document] = []
        for doc in all_doc:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)

        unique_docs = unique_docs[: config.MAX_RETRIEVED_DOCS]

        print(f"Retrieved {len(unique_docs)} documents:")
        for i, doc in enumerate(unique_docs, 1):
            doc.page_content = sanitize_text(doc.page_content)
            preview = doc.page_content[:50].replace("\n", " ")
            print(f"  [{i}] {preview}...")

        return unique_docs

    def build_prompt(self, question: str, docs: list[Document]) -> str:
        """构建 RAG 提示词。"""
        context_text = "\n\n".join(
            f"[来源: {doc.metadata.get('source', '未知')}]\n"
            f"{sanitize_text(doc.page_content)}"
            for doc in docs
        )

        return sanitize_text(f"""你是一个专业的客服助手。请根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请诚实告知用户，不要编造答案。

参考资料：
{context_text}

用户问题：{sanitize_text(question)}

请用简洁、友好的中文回答：""")

    def build_order_prompt(
        self, question: str, order_info: dict, docs: list[Document]
    ) -> str:
        """构建订单状态 + 知识库联合提示词。"""
        logistics = order_info.get("logistics") or {}
        logistics_lines = "\n".join(
            f"- {k}: {v}" for k, v in logistics.items() if v is not None
        )
        order_text = (
            f"订单号: {order_info.get('order_id')}\n"
            f"状态: {order_info.get('status_text')} ({order_info.get('status')})\n"
            f"更新时间: {order_info.get('updated_at')}\n"
            f"物流信息:\n{logistics_lines or '- 暂无'}"
        )

        context_text = "\n\n".join(
            f"[来源: {doc.metadata.get('source', '未知')}]\n"
            f"{sanitize_text(doc.page_content)}"
            for doc in docs
        ) or "（暂无相关知识库条目）"

        return sanitize_text(f"""你是一个专业的客服助手。请结合「订单实时状态」和「知识库参考资料」回答用户问题。
订单状态为权威事实，不可编造或修改；可用知识库补充运费、签收、异常处理等政策说明。

订单实时状态：
{order_text}

知识库参考资料：
{context_text}

用户问题：{sanitize_text(question)}

请用简洁、友好的中文回答，先说明订单当前状态，再补充必要的政策或操作建议：""")

    def _answer_rag(self, question: str) -> dict:
        """纯知识库检索 + 生成。"""
        question = sanitize_text(question)
        docs = self.retrieve(question)

        if not docs:
            return {
                "question": question,
                "answer": "抱歉，我没有找到相关信息，请联系人工客服。",
                "contexts": [],
            }

        prompt = self.build_prompt(question, docs)
        response = self.llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        return {
            "question": question,
            "answer": answer_text,
            "contexts": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
        }

    def _answer_order_query(self, question: str, order_id: Optional[str]) -> dict:
        """订单查询 + 知识库联合生成。"""
        question = sanitize_text(question)
        if not order_id:
            return {
                "question": question,
                "answer": (
                    "好的，我可以帮您查询订单物流。"
                    "请提供订单号（例如 ORD20260101001），以便我为您查询最新状态。"
                ),
                "contexts": [],
                "order_info": None,
            }

        print(f"📦 查询订单: {order_id}")
        order_info = self.order_api.query(order_id)
        if order_info is None:
            return {
                "question": question,
                "answer": (
                    f"未查询到订单号 {order_id} 的信息，请确认订单号是否正确，"
                    "或联系人工客服进一步核实。"
                ),
                "contexts": [],
                "order_info": None,
            }

        print(
            f"✅ 订单状态: {order_info.get('status_text')} "
            f"({order_info.get('status')})"
        )
        docs = self.retrieve(question)
        prompt = self.build_order_prompt(question, order_info, docs)
        response = self.llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        return {
            "question": question,
            "answer": answer_text,
            "contexts": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
            "order_info": order_info,
        }

    def answer(self, question: str) -> dict:
        """回答问题：订单意图走 API+RAG，否则走纯 RAG。"""
        question = sanitize_text(question)
        intent = detect_intent(question)
        if intent.is_order_query:
            print(
                f"🎯 意图: 订单查询"
                f"{f', 订单号={intent.order_id}' if intent.order_id else '（未提供订单号）'}"
            )
            return self._answer_order_query(question, intent.order_id)
        return self._answer_rag(question)

    def invoke(self, question: str) -> str:
        """简化接口，直接返回回答文本。"""
        return self.answer(question)["answer"]


def create_rag_agent(llm, kb: KnowledgeBaseManager) -> RAGAgent:
    """创建 RAG Agent 实例。"""
    return RAGAgent(llm, kb, order_api=MockOrderAPI())
