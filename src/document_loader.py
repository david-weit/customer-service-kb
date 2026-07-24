"""文档加载与按扩展名分发（仿 RAGFlow rag/app/naive.py 的 chunk 分发）。"""

import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

from langchain_core.documents import Document

from src.parsers import DocxParser, ExcelParser, JsonParser, PlainParser
from src.utils import sanitize_text

SUPPORTED_SUFFIXES = {
    ".docx",
    ".pdf",
    ".xlsx",
    ".xls",
    ".csv",
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".markdown",
}


def _base_metadata(path: Path, file_type: str, **extra) -> dict:
    meta = {
        "type": "document",
        "source": str(path),
        "filename": path.name,
        "file_type": file_type,
    }
    meta.update(extra)
    return meta


def _sections_to_document(
    path: Path,
    file_type: str,
    sections: Union[List[tuple], List[str]],
    pre_chunked: bool = False,
    category: str = "",
) -> List[Document]:
    """将 parser 产出的 sections 转为 LangChain Document。"""
    texts: List[str] = []
    for sec in sections:
        if isinstance(sec, tuple):
            text = sanitize_text(sec[0] or "").strip()
        else:
            text = sanitize_text(sec or "").strip()
        if text:
            texts.append(text)

    if not texts:
        return []

    if pre_chunked:
        return [
            Document(
                page_content=text,
                metadata=_base_metadata(
                    path,
                    file_type,
                    pre_chunked=True,
                    category=category,
                    chunk_index=i,
                ),
            )
            for i, text in enumerate(texts)
        ]

    return [
        Document(
            page_content="\n".join(texts),
            metadata=_base_metadata(path, file_type, category=category),
        )
    ]


def parse_file(
    path: Union[Path, str],
    binary: Optional[bytes] = None,
    category: str = "",
    chunk_token_num: int = 512,
) -> List[Document]:
    """
    按扩展名选择 parser（对齐 naive.chunk 的分支逻辑）。

    - docx / pdf / txt / md：合并为整篇 Document，交由向量库侧分块
    - excel / csv / json：已按行或结构切块，标记 pre_chunked=True
    """
    path = Path(path)
    filename = path.name
    data = binary if binary is not None else path.read_bytes()

    if re.search(r"\.docx$", filename, re.IGNORECASE):
        secs, tbls = DocxParser()(data)
        parts: List[str] = [t for t, _ in secs if t]
        for table_lines in tbls:
            parts.extend(table_lines)
        return _sections_to_document(path, "docx", parts, category=category)

    if re.search(r"\.pdf$", filename, re.IGNORECASE):
        sections, _ = PlainParser()(data)
        return _sections_to_document(path, "pdf", sections, category=category)

    if re.search(r"\.(csv|xlsx?)$", filename, re.IGNORECASE):
        rows = ExcelParser()(data)
        return _sections_to_document(
            path, "excel", rows, pre_chunked=True, category=category
        )

    if re.search(r"\.(json|jsonl|ldjson)$", filename, re.IGNORECASE):
        sections = JsonParser(chunk_token_num)(data)
        return _sections_to_document(
            path, "json", sections, pre_chunked=True, category=category
        )

    if re.search(r"\.(txt|md|markdown)$", filename, re.IGNORECASE):
        text = sanitize_text(data.decode("utf-8", errors="ignore"))
        if not text.strip():
            return []
        return [
            Document(
                page_content=text,
                metadata=_base_metadata(
                    path, path.suffix.lstrip("."), category=category
                ),
            )
        ]

    raise NotImplementedError(
        f"file type not supported yet: {filename} "
        f"(supported: docx, pdf, xlsx/xls/csv, json/jsonl, txt/md)"
    )


def load_file(path: Union[Path, str], category: str = "") -> List[Document]:
    """从磁盘加载并解析单个文件。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return parse_file(path, category=category)


def load_directory(
    dir_path: Union[Path, str],
    category: str = "",
    recursive: bool = True,
) -> List[Document]:
    """加载目录下所有支持格式的文档。"""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    docs: List[Document] = []
    for path in sorted(dir_path.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            docs.extend(load_file(path, category=category or dir_path.name))
            print(f"  ✅ 解析: {path.name}")
        except Exception as e:
            print(f"  ⚠️ 跳过 {path.name}: {e}")
    return docs


def load_raw_documents(
    directories: Optional[Iterable[Union[Path, str]]] = None,
) -> List[Document]:
    """加载政策/产品等原始文档目录。"""
    import config

    dirs = (
        list(directories)
        if directories is not None
        else [config.POLICIES_DIR, config.PRODUCTS_DIR]
    )
    all_docs: List[Document] = []
    for d in dirs:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            continue
        category = path.name
        loaded = load_directory(path, category=category)
        if loaded:
            print(f"📂 {category}: {len(loaded)} 个文档块/文件单元")
        all_docs.extend(loaded)
    return all_docs
