"""Mock 订单查询 API。"""

from typing import Any, Dict, Optional


# 样例订单数据（模拟真实订单查询接口返回）
_MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD20260101001": {
        "order_id": "ORD20260101001",
        "status": "in_transit",
        "status_text": "运输中",
        "logistics": {
            "carrier": "顺丰速运",
            "tracking_no": "SF1234567890",
            "current_location": "上海转运中心",
            "eta": "预计2天后送达",
        },
        "updated_at": "2026-01-03 14:30:00",
    },
    "ORD20260101002": {
        "order_id": "ORD20260101002",
        "status": "delivered",
        "status_text": "已签收",
        "logistics": {
            "carrier": "中通快递",
            "tracking_no": "ZT9876543210",
            "current_location": "已送达收件地址",
            "signed_by": "本人签收",
            "signed_at": "2026-01-02 16:20:00",
        },
        "updated_at": "2026-01-02 16:20:00",
    },
    "ORD20260101003": {
        "order_id": "ORD20260101003",
        "status": "pending_shipment",
        "status_text": "待发货",
        "logistics": {
            "carrier": None,
            "tracking_no": None,
            "current_location": "仓库备货中",
            "eta": "预计24小时内发货",
        },
        "updated_at": "2026-01-01 10:00:00",
    },
    "ORD20260101004": {
        "order_id": "ORD20260101004",
        "status": "exception",
        "status_text": "派送异常",
        "logistics": {
            "carrier": "圆通速递",
            "tracking_no": "YT1122334455",
            "current_location": "目的地网点",
            "exception_reason": "联系不上收件人，包裹暂存网点",
            "suggestion": "请确认收货电话畅通，或联系客服改派",
        },
        "updated_at": "2026-01-03 09:15:00",
    },
}


class MockOrderAPI:
    """Mock 订单查询服务，可替换为真实 HTTP API。"""

    def __init__(self, orders: Optional[Dict[str, Dict[str, Any]]] = None):
        self._orders = orders if orders is not None else _MOCK_ORDERS

    def query(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        查询订单物流状态。

        Args:
            order_id: 订单号

        Returns:
            订单详情字典；不存在则返回 None
        """
        order_id = order_id.strip().upper()
        order = self._orders.get(order_id)
        if order is None:
            return None
        return dict(order)
