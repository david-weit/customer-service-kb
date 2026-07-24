"""客服 Agent 可用工具（Function Calling）。"""

import json
from typing import Callable, List, Optional

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool

import config
from src.order_api import MockOrderAPI
from src.query_expansion import QueryExpander
from src.utils import sanitize_text
from src.vector_store import KnowledgeBaseManager


def _docs_to_contexts(docs: List[Document]) -> List[dict]:
    return [
        {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]


def retrieve_documents(
    kb: KnowledgeBaseManager,
    query_expander: QueryExpander,
    question: str,
    top_k: int,
) -> List[Document]:
    """多查询扩展 + 混合检索。"""
    question = sanitize_text(question)
    expanded_queries = query_expander.expand_query(question)
    print(f"Expanded queries: {expanded_queries}")

    all_doc: List[Document] = []
    all_doc.extend(kb.search(question, top_k))
    for q in expanded_queries:
        if q.strip() == question.strip():
            continue
        all_doc.extend(kb.search(q, config.EXPANDED_QUERY_K))

    seen: set[str] = set()
    unique_docs: List[Document] = []
    for doc in all_doc:
        content = sanitize_text(doc.page_content)
        doc.page_content = content
        if content and content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    unique_docs = unique_docs[: config.MAX_RETRIEVED_DOCS]
    print(f"Retrieved {len(unique_docs)} documents:")
    for i, doc in enumerate(unique_docs, 1):
        preview = doc.page_content[:50].replace("\n", " ")
        print(f"  [{i}] {preview}...")
    return unique_docs


def create_agent_tools(
    kb: KnowledgeBaseManager,
    order_api: MockOrderAPI,
    query_expander: QueryExpander,
    top_k: int,
    on_order_result: Optional[Callable[[Optional[dict]], None]] = None,
    on_search_result: Optional[Callable[[List[Document]], None]] = None,
) -> List[StructuredTool]:
    """创建绑定到 LLM 的工具列表。"""

    def query_order(order_id: str) -> str:
        """查询订单物流状态。当用户询问自己的订单进度、快递到哪了、是否发货，
        并且消息中包含订单号时调用此工具。

        Args:
            order_id: 订单号，例如 ORD20260101001
        """
        order_id = sanitize_text(order_id).strip().upper()
        print(f"🔧 tool call: query_order({order_id})")
        order_info = order_api.query(order_id)
        if on_order_result:
            on_order_result(order_info)
        if order_info is None:
            return json.dumps(
                {
                    "found": False,
                    "order_id": order_id,
                    "message": f"未查询到订单号 {order_id}",
                },
                ensure_ascii=False,
            )
        print(
            f"✅ 订单状态: {order_info.get('status_text')} "
            f"({order_info.get('status')})"
        )
        return json.dumps(
            {"found": True, "order": order_info},
            ensure_ascii=False,
        )

    def search_knowledge_base(query: str) -> str:
        """从客服知识库检索 FAQ 与政策说明。用于回答退货、换货、运费、发票、
        会员、支付等政策问题；也可在查询订单后补充配送/签收相关政策。

        Args:
            query: 检索用的用户问题或关键词
        """
        query = sanitize_text(query)
        print(f"🔧 tool call: search_knowledge_base({query[:40]})")
        docs = retrieve_documents(kb, query_expander, query, top_k)
        if on_search_result:
            on_search_result(docs)
        if not docs:
            return json.dumps(
                {"hits": 0, "documents": [], "message": "知识库未找到相关内容"},
                ensure_ascii=False,
            )
        payload = []
        for doc in docs:
            payload.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "category": doc.metadata.get("category", ""),
                }
            )
        return json.dumps(
            {"hits": len(payload), "documents": payload},
            ensure_ascii=False,
        )

    return [
        StructuredTool.from_function(
            func=query_order,
            name="query_order",
            description=(
                "查询订单物流状态。用户询问个人订单进度/快递位置且已提供订单号时使用。"
                "参数 order_id 为订单号（如 ORD20260101001）。"
            ),
        ),
        StructuredTool.from_function(
            func=search_knowledge_base,
            name="search_knowledge_base",
            description=(
                "检索客服知识库 FAQ 与政策（退货、运费、发票、会员等）。"
                "回答政策类问题，或在查完订单后补充政策说明时使用。"
            ),
        ),
    ]
