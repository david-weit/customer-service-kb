"""订单意图识别与订单号提取。"""

import re
from dataclasses import dataclass
from typing import Optional


# 强意图：个人订单查询（无订单号时也会引导用户提供）
ORDER_QUERY_KEYWORDS = (
    "还没收到",
    "没收到",
    "到哪了",
    "到哪儿了",
    "查一下订单",
    "查下订单",
    "查询订单",
    "我的订单",
    "订单进度",
    "包裹到哪",
    "快递到哪",
)

# 弱信号：仅有这些词不算订单查询，避免「怎么查看物流」误入 API 分支
_WEAK_LOGISTICS_HINTS = ("订单", "物流", "快递", "包裹", "发货", "签收", "配送", "派送")

# ORD + 数字，或「订单号：xxx」形式
_ORDER_ID_PATTERNS = (
    re.compile(r"\b(ORD\d+)\b", re.IGNORECASE),
    re.compile(r"订单号[:：\s]*([A-Za-z0-9_-]+)"),
    re.compile(r"订单[:：\s]*([A-Za-z0-9_-]{6,})"),
)


@dataclass
class IntentResult:
    """意图识别结果。"""

    is_order_query: bool
    order_id: Optional[str] = None


def extract_order_id(text: str) -> Optional[str]:
    """从文本中提取订单号。"""
    for pattern in _ORDER_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def detect_intent(text: str) -> IntentResult:
    """
    规则识别订单查询意图。

    - 已包含订单号 → 订单查询
    - 含「还没收到 / 我的订单」等强意图词 → 订单查询（可无订单号，后续引导补充）
    - 仅问物流政策（如「怎么查看物流」）→ 非订单查询，走知识库
    """
    order_id = extract_order_id(text)
    if order_id:
        return IntentResult(is_order_query=True, order_id=order_id)

    has_strong_intent = any(kw in text for kw in ORDER_QUERY_KEYWORDS)
    if has_strong_intent:
        return IntentResult(is_order_query=True, order_id=None)

    # 「查一下」需配合订单/物流弱信号，避免误伤其他问题
    if "查一下" in text and any(h in text for h in _WEAK_LOGISTICS_HINTS):
        return IntentResult(is_order_query=True, order_id=None)

    return IntentResult(is_order_query=False, order_id=None)
