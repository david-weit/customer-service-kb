import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import config


class ConversationLoader:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or config.CONVERSATIONS_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
    def load_conversations(self) -> List[Dict]:
        """优先加载已有对话数据，不存在时生成示例。"""

        if config.CONVERSATIONS_JSON_PATH.exists():
            print(f"✅ 加载 {config.CONVERSATIONS_JSON_PATH} 文件")
            return self.load_from_json(str(config.CONVERSATIONS_JSON_PATH))

        if config.CONVERSATIONS_CSV_PATH.exists():
            print(f"✅ 加载 {config.CONVERSATIONS_CSV_PATH} 文件")
            return self.load_from_csv(str(config.CONVERSATIONS_CSV_PATH))

        if config.RAW_LOGS_PATH.exists():
            print(f"✅ 加载 {config.RAW_LOGS_PATH} 文件")
            return self._convert_raw_logs(config.RAW_LOGS_PATH)

        return self.load_sample_conversations()

    def load_from_csv(self, filepath: str) -> List[Dict]:
        """从 CSV 加载对话记录。"""
        df = pd.read_csv(filepath)
        conversations = []

        grouped = df.groupby("conversation_id")

        for conv_id, group in grouped:
            messages = []
            for _, row in group.iterrows():
                messages.append(
                    {
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row.get("timestamp", ""),
                    }
                )

            conv_timestamp = ""
            if "timestamp" in group.columns and not group.empty:
                conv_timestamp = group.iloc[0].get("timestamp", "")

            conversations.append(
                {
                    "conversation_id": conv_id,
                    "messages": messages,
                    "timestamp": conv_timestamp,
                }
            )

        return conversations

    def load_from_json(self, filepath: str) -> List[Dict]:
        """从 JSON 加载对话记录。"""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _convert_raw_logs(self, filepath: Path) -> List[Dict]:
        """将旧版 raw_logs.csv 转换为对话格式。"""
        df = pd.read_csv(filepath)
        conversations = []

        session_col = "session_id" if "session_id" in df.columns else "conversation_id"
        grouped = df.groupby(session_col)

        for session_id, group in grouped:
            messages = []
            timestamp = ""
            for _, row in group.iterrows():
                timestamp = row.get("timestamp", timestamp)
                if pd.notna(row.get("user_message")):
                    messages.append(
                        {
                            "role": "customer",
                            "content": row["user_message"],
                            "timestamp": timestamp,
                        }
                    )
                if pd.notna(row.get("agent_message")):
                    messages.append(
                        {
                            "role": "agent",
                            "content": row["agent_message"],
                            "timestamp": timestamp,
                        }
                    )

            conversations.append(
                {
                    "conversation_id": session_id,
                    "messages": messages,
                    "timestamp": timestamp,
                }
            )

        return conversations

    def load_sample_conversations(self):
        """生成示例对话数据用于测试。"""
        sample_data = [
            {
                "conversation_id": "C001",
                "messages": [
                    {"role": "customer", "content": "你们的退换货政策是什么？"},
                    {
                        "role": "agent",
                        "content": "我们提供7天无理由退货，15天内质量问题可以换货。请保留好购物凭证。",
                    },
                    {"role": "customer", "content": "那邮费谁承担？"},
                    {
                        "role": "agent",
                        "content": "如果是质量问题，我们承担邮费。如果是非质量问题，需要您自己承担。",
                    },
                ],
                "timestamp": "2024-01-01 10:00:00",
            },
            {
                "conversation_id": "C002",
                "messages": [
                    {"role": "customer", "content": "订单多久能发货？"},
                    {
                        "role": "agent",
                        "content": "一般下单后24-48小时内发货，节假日顺延。您可以在订单页面查看物流信息。",
                    },
                ],
                "timestamp": "2024-01-02 14:30:00",
            },
            {
                "conversation_id": "C003",
                "messages": [
                    {"role": "customer", "content": "怎么联系客服？"},
                    {
                        "role": "agent",
                        "content": "您可以通过以下方式联系我们：1. 在线客服（9:00-21:00）2. 客服热线 400-123-4567 3. 邮件 service@company.com",
                    },
                ],
                "timestamp": "2024-01-03 09:15:00",
            },
        ]

        self.save_conversations(sample_data)
        return sample_data

    def save_conversations(self, conversations: List[Dict]):
        """保存对话记录。"""
        with open(config.CONVERSATIONS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)

        rows = []
        for conv in conversations:
            for msg in conv["messages"]:
                rows.append(
                    {
                        "conversation_id": conv["conversation_id"],
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg.get("timestamp", conv.get("timestamp", "")),
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(config.CONVERSATIONS_CSV_PATH, index=False)

        print(f"✅ 保存 {len(conversations)} 条对话到 {self.data_dir}")
