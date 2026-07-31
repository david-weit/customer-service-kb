"""文档加载与按扩展名分发。"""

import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Union

from langchain_core.documents import Document

from src.parsers import DoclingParser, ExcelParser, JsonParser
from src.utils import sanitize_text

SUPPORTED_SUFFIXES = {
    ".docx",
    ".pdf",
    ".pptx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
    ".xlsx",
    ".xls",
    ".csv",
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".markdown",
}

_DOCLING_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
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
    按扩展名选择 parser。

    - pdf/docx/pptx/html/图片：Docling → Markdown，交由向量库侧分块
    - excel/csv/json：按行或结构切块，标记 pre_chunked=True
    - txt/md：直接读取
    """
    path = Path(path)
    filename = path.name
    suffix = path.suffix.lower()

    if suffix in _DOCLING_SUFFIXES:
        if binary is not None:
            sections, _ = DoclingParser()(binary, suffix=suffix)
        else:
            sections, _ = DoclingParser()(path)
        file_type = suffix.lstrip(".") or "document"
        return _sections_to_document(path, file_type, sections, category=category)

    data = binary if binary is not None else path.read_bytes()

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
        f"(supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
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


def save_uploads(
    files: Iterable[Union[Path, str]],
    dest_dir: Optional[Path] = None,
) -> List[Path]:
    """将上传文件复制到 uploads 目录（同名覆盖），返回保存路径列表。"""
    import config

    dest = Path(dest_dir or config.UPLOADS_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for item in files:
        src = Path(item)
        if not src.is_file():
            continue
        target = dest / src.name
        shutil.copy2(src, target)
        saved.append(target)
    return saved


def list_upload_files(dest_dir: Optional[Path] = None) -> List[Path]:
    """列出 uploads 目录中已导入的支持格式文件。"""
    import config

    dest = Path(dest_dir or config.UPLOADS_DIR)
    if not dest.is_dir():
        return []
    return sorted(
        p
        for p in dest.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_raw_documents(
    directories: Optional[Iterable[Union[Path, str]]] = None,
) -> List[Document]:
    """加载政策/产品/用户上传等原始文档目录。"""
    import config

    dirs = (
        list(directories)
        if directories is not None
        else [config.POLICIES_DIR, config.PRODUCTS_DIR, config.UPLOADS_DIR]
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
