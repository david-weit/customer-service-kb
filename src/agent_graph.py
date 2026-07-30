"""LangGraph 客服问答：Function Calling（LLM 选工具 + 执行 + 再生成）。"""

import json
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import RemoveMessage, add_messages
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

import config as app_config
from src.context_manager import ensure_message_id, trim_with_summary
from src.message_store import append_messages, ensure_schema
from src.order_api import MockOrderAPI
from src.query_expansion import QueryExpander
from src.tools import _docs_to_contexts, create_agent_tools
from src.utils import sanitize_text
from src.vector_store import KnowledgeBaseManager

MAX_TOOL_ROUNDS = 6

VERIFY_FAIL_SUFFIX = (
    "\n\n（以上回答经自检仍可能不够准确，建议联系人工客服确认。）"
)

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

# 进程内复用连接池，避免每次建图都新建池
_POOL: Optional[ConnectionPool] = None


class AnswerVerdict(BaseModel):
    """答案自检结果。"""

    is_accurate: bool = Field(description="是否准确、切题地回答了用户问题")
    reason: str = Field(description="简短判断理由")


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    order_info: Optional[dict]
    contexts: List[dict]
    tool_rounds: int
    answer: str
    answer_ok: bool
    verify_reason: str
    verify_retries: int


def _thread_id_from_config(config: Optional[RunnableConfig]) -> str:
    if not config:
        return "default"
    configurable = config.get("configurable") or {}
    return str(configurable.get("thread_id") or "default")


def _persist(thread_id: str, messages: List[BaseMessage]) -> None:
    """双写到业务表；池未就绪时跳过。"""
    if _POOL is None or not messages:
        return
    for msg in messages:
        ensure_message_id(msg)
    append_messages(_POOL, thread_id, messages)


def _ai_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content or "")


def _facts_summary(state: AgentState) -> str:
    """压缩 contexts / order_info，供自检与重写使用。"""
    parts: List[str] = []
    order_info = state.get("order_info")
    if order_info:
        parts.append("订单信息: " + json.dumps(order_info, ensure_ascii=False))
    contexts = state.get("contexts") or []
    if contexts:
        snippets = []
        for ctx in contexts[:5]:
            text = sanitize_text(str(ctx.get("content", "")))[:200]
            if text:
                snippets.append(f"- {text}")
        if snippets:
            parts.append("知识库摘录:\n" + "\n".join(snippets))
    return "\n".join(parts) if parts else "（无工具事实）"


def get_db_pool() -> ConnectionPool:
    """确保连接池与业务表已就绪，供历史查询等复用。"""
    _get_checkpointer()
    assert _POOL is not None
    return _POOL


def _get_checkpointer() -> PostgresSaver:
    """创建（或复用）Postgres Checkpointer，并确保表已建好。"""
    global _POOL
    if _POOL is None:
        _POOL = ConnectionPool(
            conninfo=app_config.DATABASE_URL,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        _POOL.open()
    checkpointer = PostgresSaver(_POOL)
    checkpointer.setup()  # LangGraph 框架表
    ensure_schema(_POOL)  # 业务完整历史表
    return checkpointer


def build_agent_graph(
    llm,
    kb: KnowledgeBaseManager,
    order_api: Optional[MockOrderAPI] = None,
    top_k: Optional[int] = None,
):
    """编译基于 Function Calling + Postgres Checkpointer 的客服问答图。"""
    order_api = order_api or MockOrderAPI()
    top_k = top_k or app_config.TOP_K
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
    verify_llm = llm.with_structured_output(AnswerVerdict)

    def node_prepare(
        state: AgentState, config: RunnableConfig
    ) -> Dict[str, Any]:
        """准备本轮输入：有历史则追加 Human，无历史则写入 System+Human。"""
        question = sanitize_text(state.get("question", ""))
        side_channel["order_info"] = None
        side_channel["docs"] = []
        thread_id = _thread_id_from_config(config)

        updates: Dict[str, Any] = {
            "question": question,
            "tool_rounds": 0,
            "order_info": None,
            "contexts": [],
            "answer_ok": True,
            "verify_reason": "",
            "verify_retries": 0,
        }

        existing = state.get("messages") or []
        if not existing:
            new_msgs: List[BaseMessage] = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        else:
            new_msgs = [HumanMessage(content=question)]
        updates["messages"] = new_msgs
        _persist(thread_id, new_msgs)
        return updates

    def node_manage_context(
        state: AgentState, config: RunnableConfig
    ) -> Dict[str, Any]:
        """超窗时压缩旧消息为摘要，并用 RemoveMessage 写回 Checkpointer。"""
        result = trim_with_summary(
            llm, state.get("messages") or [], app_config.CONTEXT_WINDOW_SIZE
        )
        if result is None:
            return {}
        ensure_message_id(result.summary_msg)
        _persist(_thread_id_from_config(config), [result.summary_msg])
        return {
            "messages": [
                RemoveMessage(id=i) for i in result.delete_ids
            ]
            + [result.summary_msg],
        }

    def node_agent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        response = llm_with_tools.invoke(state["messages"])
        ensure_message_id(response)
        _persist(_thread_id_from_config(config), [response])
        return {"messages": [response]}

    def node_tools(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
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

        for msg in tool_messages:
            ensure_message_id(msg)
        _persist(_thread_id_from_config(config), tool_messages)

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
                answer = _ai_text(msg.content)
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

    def node_verify(state: AgentState) -> Dict[str, Any]:
        """LLM Judge：判断答案是否准确回应用户问题。"""
        question = sanitize_text(state.get("question", ""))
        answer = sanitize_text(state.get("answer", ""))
        facts = _facts_summary(state)

        prompt = f"""你是客服问答质检员。请判断「客服回答」是否准确回应用户问题。

判断标准（只评对错与切题，不评文采）：
1. 是否切题、覆盖用户核心诉求
2. 是否与下方工具/知识库事实冲突（有事实时）
3. 若客服合理要求补充信息（如订单号），可判为准确

用户问题：
{question}

客服回答：
{answer}

已有事实：
{facts}

请输出 is_accurate 与简短 reason。"""

        try:
            verdict: AnswerVerdict = verify_llm.invoke(prompt)
            ok = bool(verdict.is_accurate)
            reason = sanitize_text(verdict.reason).strip()
        except Exception as e:
            # 自检失败时放行，避免拖垮主流程
            return {
                "answer_ok": True,
                "verify_reason": f"自检跳过: {type(e).__name__}",
            }

        updates: Dict[str, Any] = {
            "answer_ok": ok,
            "verify_reason": reason or ("通过" if ok else "未通过"),
        }
        if not ok:
            retries = int(state.get("verify_retries") or 0)
            max_retries = max(0, int(app_config.ANSWER_VERIFY_MAX_RETRIES))
            if retries >= max_retries:
                # 已无重试机会：附带说明后结束
                final = answer
                if VERIFY_FAIL_SUFFIX.strip() not in final:
                    final = final.rstrip() + VERIFY_FAIL_SUFFIX
                updates["answer"] = final
        return updates

    def route_after_verify(state: AgentState) -> str:
        if state.get("answer_ok", True):
            return "end"
        retries = int(state.get("verify_retries") or 0)
        max_retries = max(0, int(app_config.ANSWER_VERIFY_MAX_RETRIES))
        if retries < max_retries:
            return "regenerate"
        return "end"

    def node_regenerate(
        state: AgentState, config: RunnableConfig
    ) -> Dict[str, Any]:
        """未绑定工具的 LLM 根据校验反馈重写回答。"""
        question = sanitize_text(state.get("question", ""))
        prev_answer = sanitize_text(state.get("answer", ""))
        reason = sanitize_text(state.get("verify_reason", ""))
        facts = _facts_summary(state)
        retries = int(state.get("verify_retries") or 0) + 1

        rewrite_prompt = f"""上一版客服回答未通过质检，请基于已有事实重写一版简洁友好的中文回复。

要求：
- 必须切题回应用户问题
- 只能使用已有事实，禁止编造订单状态或政策
- 若事实不足，诚实说明并建议联系人工客服或请用户补充信息

用户问题：
{question}

质检不通过原因：
{reason}

上一版回答：
{prev_answer}

已有事实：
{facts}

请直接输出重写后的客服回复正文，不要解释质检过程。"""

        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=rewrite_prompt),
            ]
        )
        new_answer = sanitize_text(_ai_text(response.content)).strip()
        if not new_answer:
            new_answer = prev_answer

        ai_msg = AIMessage(content=new_answer)
        ensure_message_id(ai_msg)
        _persist(_thread_id_from_config(config), [ai_msg])

        return {
            "messages": [ai_msg],
            "answer": new_answer,
            "verify_retries": retries,
        }

    def route_after_finalize(state: AgentState) -> str:
        if app_config.ANSWER_VERIFY_ENABLED:
            return "verify"
        return "end"

    graph = StateGraph[AgentState, None, AgentState, AgentState](AgentState)
    graph.add_node("prepare", node_prepare)
    graph.add_node("manage_context", node_manage_context)
    graph.add_node("agent", node_agent)
    graph.add_node("tools", node_tools)
    graph.add_node("finalize", node_finalize)
    graph.add_node("verify", node_verify)
    graph.add_node("regenerate", node_regenerate)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "manage_context")
    graph.add_edge("manage_context", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "finalize",
        route_after_finalize,
        {"verify": "verify", "end": END},
    )
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"regenerate": "regenerate", "end": END},
    )
    graph.add_edge("regenerate", "verify")

    return graph.compile(checkpointer=_get_checkpointer())
