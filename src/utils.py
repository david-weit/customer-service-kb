"""工具函数。"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def sanitize_text(text: Any) -> str:
    """清洗文本，移除 UTF-8 无法编码的 surrogate 等非法字符。

    解析 PDF/DOCX 或脏数据入库后，调用 LLM API 时可能触发：
    UnicodeEncodeError: surrogates not allowed
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # 去掉 NUL，再用 utf-8 ignore 丢掉 surrogate（U+D800–U+DFFF）
    text = text.replace("\x00", "")
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    """加载 CSV 文件。"""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """保存 DataFrame 到 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_text_file(path: Path) -> str:
    """加载文本文件内容。"""
    return sanitize_text(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> Any:
    """加载 JSON 文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """保存数据到 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """将文本按固定长度分块。"""
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
