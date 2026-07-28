"""上下文管理：滑动窗口 + LLM 滚动摘要压缩。"""

from typing import List, NamedTuple, Optional, Sequence
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)

from src.utils import sanitize_text

SUMMARY_PREFIX = "【对话摘要】"
COMPRESS_BATCH_SIZE = 10


class TrimResult(NamedTuple):
    """压缩结果：待删除 id、新摘要消息、保留的最近消息。"""

    delete_ids: List[str]
    summary_msg: SystemMessage
    recent: List[BaseMessage]


def _msg_content(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content or "")


def is_summary_message(msg: BaseMessage) -> bool:
    """识别摘要消息。"""
    if not isinstance(msg, SystemMessage):
        return False
    return _msg_content(msg).startswith(SUMMARY_PREFIX)


def ensure_message_id(msg: BaseMessage) -> str:
    """保证消息有 id，便于 RemoveMessage。"""
    msg_id = getattr(msg, "id", None)
    if msg_id:
        return str(msg_id)
    msg_id = str(uuid4())
    try:
        msg.id = msg_id
    except Exception:
        pass
    return msg_id


def fix_tool_boundary(
    older: List[BaseMessage], recent: List[BaseMessage]
) -> tuple[List[BaseMessage], List[BaseMessage]]:
    """若窗口以孤立 ToolMessage 开头，向前补齐带 tool_calls 的 AIMessage。"""
    if not recent:
        return older, recent

    while recent and isinstance(recent[0], ToolMessage) and older:
        pulled = older.pop()
        recent.insert(0, pulled)

    # 若仍以 ToolMessage 开头且找不到配对，丢弃孤立 tool 消息
    while recent and isinstance(recent[0], ToolMessage):
        recent.pop(0)

    return older, recent


def _format_messages_for_summary(messages: Sequence[BaseMessage]) -> str:
    lines = []
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        text = sanitize_text(_msg_content(msg)).strip()
        if not text:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                names = [
                    (c.get("name") if isinstance(c, dict) else c.get("name", ""))
                    for c in (msg.tool_calls or [])
                ]
                text = f"(调用工具: {', '.join(n for n in names if n)})"
            else:
                continue
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def build_summary(
    llm,
    older_messages: Sequence[BaseMessage],
    prev_summary: Optional[str] = None,
) -> str:
    """调用 LLM 生成滚动摘要（约 200 字中文）。"""
    older_text = _format_messages_for_summary(older_messages)
    prev = sanitize_text(prev_summary or "").strip()
    prev_block = f"已有摘要：\n{prev}\n" if prev else ""
    prompt = f"""请将以下客服对话历史压缩为简洁中文摘要（约200字）。
要求：保留订单号、用户核心意图、未解决的问题、已确认的事实与政策结论；不要编造。

{prev_block}待压缩的新对话：
{older_text}

只输出摘要正文，不要标题："""
    response = llm.invoke(sanitize_text(prompt))
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "".join(parts)
    text = sanitize_text(str(content or "")).strip()
    if text.startswith(SUMMARY_PREFIX):
        text = text[len(SUMMARY_PREFIX) :].strip()
    return text or "暂无有效历史摘要。"


def split_messages(
    messages: Sequence[BaseMessage],
) -> tuple[Optional[BaseMessage], Optional[BaseMessage], List[BaseMessage]]:
    """拆分为 system_prompt / 旧摘要 / 其余对话消息。"""
    system_prompt: Optional[BaseMessage] = None
    prev_summary: Optional[BaseMessage] = None
    rest: List[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            if is_summary_message(msg):
                prev_summary = msg
            elif system_prompt is None:
                system_prompt = msg
            else:
                rest.append(msg)
            continue
        rest.append(msg)

    return system_prompt, prev_summary, rest


def trim_with_summary(
    llm,
    messages: Sequence[BaseMessage],
    window_size: int,
) -> Optional[TrimResult]:
    """
    若对话消息超过 window_size，则压缩更早部分。

    Returns:
        None 表示无需压缩；否则返回 TrimResult。
    """
    if window_size <= 0:
        return None

    _, prev_summary_msg, rest = split_messages(messages)
    if len(rest) <= window_size:
        return None

    older = list(rest[:COMPRESS_BATCH_SIZE])
    recent = list(rest[COMPRESS_BATCH_SIZE:])
    older, recent = fix_tool_boundary(older, recent)

    # 修正边界后可能不再超窗
    if not older:
        return None

    prev_summary_text = (
        _msg_content(prev_summary_msg)[len(SUMMARY_PREFIX) :].strip()
        if prev_summary_msg
        else None
    )
    summary_body = build_summary(llm, older, prev_summary_text)
    summary_msg = SystemMessage(content=f"{SUMMARY_PREFIX}{summary_body}")
    ensure_message_id(summary_msg)

    delete_ids: List[str] = []
    for msg in older:
        delete_ids.append(ensure_message_id(msg))
    if prev_summary_msg is not None:
        delete_ids.append(ensure_message_id(prev_summary_msg))

    # recent 中因边界修正从 older 拉回的消息，其 id 已在 older 删除列表里，需从 delete_ids 去掉
    recent_ids = {ensure_message_id(m) for m in recent}
    delete_ids = [i for i in delete_ids if i not in recent_ids]

    if not delete_ids:
        return None

    print(
        f"🧹 上下文压缩: 删除 {len(delete_ids)} 条旧消息，"
        f"保留 {len(recent)} 条未压缩消息 + 摘要"
    )
    return TrimResult(delete_ids=delete_ids, summary_msg=summary_msg, recent=recent)
