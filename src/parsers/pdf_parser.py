"""PDF 纯文本解析器（仿 RAGFlow PlainParser，不做 OCR/版面分析）。"""

import logging
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, List, Tuple, Union

from pypdf import PdfReader


class PlainParser:
    """仅抽取 PDF 文本行，适合可复制文本的政策/产品文档。"""

    def __call__(
        self,
        filename: Union[str, Path, bytes, BinaryIO],
        from_page: int = 0,
        to_page: int = 100000,
        **kwargs,
    ) -> Tuple[List[Tuple[str, str]], list]:
        """
        Returns:
            sections: [(line, ""), ...]
            tables: []  （Plain 模式不解析表格结构）
        """
        lines: List[str] = []
        try:
            if isinstance(filename, (str, Path)):
                reader = PdfReader(str(filename))
            elif isinstance(filename, bytes):
                reader = PdfReader(BytesIO(filename))
            else:
                reader = PdfReader(filename)

            end = min(to_page, len(reader.pages))
            for page in reader.pages[from_page:end]:
                text = page.extract_text() or ""
                lines.extend(line for line in text.split("\n") if line.strip())
        except Exception:
            logging.exception("PlainParser failed to parse PDF")

        return [(line, "") for line in lines], []
