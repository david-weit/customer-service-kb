"""LangGraph 客服问答：Function Calling（LLM 选工具 + 执行 + 再生成）。"""

import json
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import config
from src.order_api import MockOrderAPI
from src.query_expansion import QueryExpander
from src.tools import _docs_to_contexts, create_agent_tools
from src.utils import sanitize_text
from src.vector_store import KnowledgeBaseManager

MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """你是专业的电商客服助手，必须通过工具获取事实后再回答，不要编造订单状态或政策。

可用工具：
1. query_order(order_id)：查询个人订单物流。仅当用户消息中已有订单号时调用。
2. search_knowledge_base(query)：检索客服知识库（退货、运费、发票、会员、物流政策等）。

规则：
- 政策/FAQ 问题：调用 search_knowledge_base，再根据工具结果用简洁友好的中文回答。
- 查订单且已提供订单号：先 query_order；如需补充政策可再 search_knowledge_base。
- 查订单但没有订单号：不要调用 query_order，直接请用户提供订单号（例如 ORD20260101001）。
- 订单状态以 query_order 返回为准，不可修改或臆造。
- 工具返回无结果时，诚实告知并建议联系人工客服。
"""


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    order_info: Optional[dict]
    contexts: List[dict]
    tool_rounds: int
    answer: str


def build_agent_graph(
    llm,
    kb: KnowledgeBaseManager,
    order_api: Optional[MockOrderAPI] = None,
    top_k: Optional[int] = None,
):
    """编译基于 Function Calling 的客服问答图。"""
    order_api = order_api or MockOrderAPI()
    top_k = top_k or config.TOP_K
    query_expander = QueryExpander(llm)

    # 用可变容器让工具回调写入本轮状态侧车数据
    side_channel: Dict[str, Any] = {"order_info": None, "docs": []}

    def on_order_result(order_info: Optional[dict]) -> None:
        side_channel["order_info"] = order_info

    def on_search_result(docs: List[Document]) -> None:
        side_channel["docs"] = docs

    tools = create_agent_tools(
        kb=kb,
        order_api=order_api,
        query_expander=query_expander,
        top_k=top_k,
        on_order_result=on_order_result,
        on_search_result=on_search_result,
    )
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def node_prepare(state: AgentState) -> Dict[str, Any]:
        question = sanitize_text(state.get("question", ""))
        side_channel["order_info"] = None
        side_channel["docs"] = []
        return {
            "question": question,
            "tool_rounds": 0,
            "order_info": None,
            "contexts": [],
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
        }

    def node_agent(state: AgentState) -> Dict[str, Any]:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def node_tools(state: AgentState) -> Dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        tool_messages: List[ToolMessage] = []

        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else call["name"]
            args = call.get("args") if isinstance(call, dict) else call["args"]
            call_id = call.get("id") if isinstance(call, dict) else call["id"]
            tool = tools_by_name.get(name)
            if tool is None:
                content = json.dumps(
                    {"error": f"未知工具: {name}"}, ensure_ascii=False
                )
            else:
                try:
                    content = tool.invoke(args)
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                except Exception as e:
                    content = json.dumps(
                        {"error": str(e)}, ensure_ascii=False
                    )
            tool_messages.append(
                ToolMessage(content=sanitize_text(content), tool_call_id=call_id)
            )

        updates: Dict[str, Any] = {
            "messages": tool_messages,
            "tool_rounds": int(state.get("tool_rounds") or 0) + 1,
        }
        if side_channel.get("order_info") is not None:
            updates["order_info"] = side_channel["order_info"]
        docs = side_channel.get("docs") or []
        if docs:
            updates["contexts"] = _docs_to_contexts(docs)
        return updates

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        rounds = int(state.get("tool_rounds") or 0)
        if tool_calls and rounds < MAX_TOOL_ROUNDS:
            return "tools"
        return "finalize"

    def node_finalize(state: AgentState) -> Dict[str, Any]:
        """从最后一条 AI 文本消息提取最终回答。"""
        answer = ""
        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    continue
                content = msg.content
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            parts.append(block)
                    answer = "".join(parts)
                else:
                    answer = str(content or "")
                break

        answer = sanitize_text(answer).strip()
        if not answer:
            answer = "抱歉，我暂时无法回答，请稍后再试或联系人工客服。"

        updates: Dict[str, Any] = {"answer": answer}
        # 确保 contexts / order_info 从 side_channel 回写（若尚未写入 state）
        if not state.get("contexts") and side_channel.get("docs"):
            updates["contexts"] = _docs_to_contexts(side_channel["docs"])
        if state.get("order_info") is None and side_channel.get("order_info") is not None:
            updates["order_info"] = side_channel["order_info"]
        return updates

    graph = StateGraph(AgentState)
    graph.add_node("prepare", node_prepare)
    graph.add_node("agent", node_agent)
    graph.add_node("tools", node_tools)
    graph.add_node("finalize", node_finalize)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()
