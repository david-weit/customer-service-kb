"""业务侧完整对话历史存储（与 LangGraph Checkpointer 双写）。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from src.context_manager import ensure_message_id, is_summary_message
from src.utils import sanitize_text

TABLE_NAME = "conversation_messages"

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (thread_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_id
    ON {TABLE_NAME} (thread_id, id);
"""


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


def role_for_message(msg: BaseMessage) -> str:
    if isinstance(msg, SystemMessage):
        return "summary" if is_summary_message(msg) else "system"
    if isinstance(msg, HumanMessage):
        return "human"
    if isinstance(msg, AIMessage):
        return "ai"
    if isinstance(msg, ToolMessage):
        return "tool"
    return type(msg).__name__.lower()


def ensure_schema(pool: ConnectionPool) -> None:
    """创建 conversation_messages 表（不改动 LangGraph 框架表）。"""
    with pool.connection() as conn:
        conn.execute(_SCHEMA_SQL)


def append_messages(
    pool: ConnectionPool,
    thread_id: str,
    messages: Sequence[BaseMessage],
) -> int:
    """将消息追加写入自建表；按 (thread_id, message_id) 幂等。返回尝试写入条数。"""
    if not thread_id or not messages:
        return 0

    rows: List[tuple] = []
    for msg in messages:
        if msg.__class__.__name__ == "RemoveMessage":
            continue
        message_id = ensure_message_id(msg)
        role = role_for_message(msg)
        content = sanitize_text(_msg_content(msg))
        tool_call_id = None
        tool_calls_payload = None
        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", None)
        if isinstance(msg, AIMessage):
            tcs = getattr(msg, "tool_calls", None) or []
            if tcs:
                # 统一为可 JSON 序列化的 list[dict]
                normalized = []
                for call in tcs:
                    if isinstance(call, dict):
                        normalized.append(call)
                    else:
                        normalized.append(
                            {
                                "name": getattr(call, "name", None),
                                "args": getattr(call, "args", None),
                                "id": getattr(call, "id", None),
                            }
                        )
                tool_calls_payload = Json(normalized)

        rows.append(
            (thread_id, message_id, role, content, tool_call_id, tool_calls_payload)
        )

    if not rows:
        return 0

    sql = f"""
        INSERT INTO {TABLE_NAME}
            (thread_id, message_id, role, content, tool_call_id, tool_calls)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (thread_id, message_id) DO NOTHING
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
    return len(rows)


def list_messages(
    pool: ConnectionPool,
    thread_id: str,
) -> List[Dict[str, Any]]:
    """按写入顺序返回某会话的完整历史。"""
    sql = f"""
        SELECT id, thread_id, message_id, role, content,
               tool_call_id, tool_calls, created_at
        FROM {TABLE_NAME}
        WHERE thread_id = %s
        ORDER BY id ASC
    """
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (thread_id,))
            rows = cur.fetchall()
    result: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        # datetime / Json 转成可读结构
        if item.get("created_at") is not None:
            item["created_at"] = item["created_at"].isoformat()
        if item.get("tool_calls") is not None and not isinstance(
            item["tool_calls"], (dict, list)
        ):
            try:
                item["tool_calls"] = json.loads(item["tool_calls"])
            except (TypeError, json.JSONDecodeError):
                pass
        result.append(item)
    return result
