"""Docling 文档解析器：版面/表格/OCR → Markdown 文本。"""

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

from src.utils import sanitize_text

_CONVERTER = None


def _get_converter():
    """进程内复用 DocumentConverter，避免重复加载模型。"""
    global _CONVERTER
    if _CONVERTER is None:
        from docling.document_converter import DocumentConverter

        _CONVERTER = DocumentConverter()
    return _CONVERTER


class DoclingParser:
    """用 Docling 将 PDF/DOCX/PPTX/HTML/图片等转为 Markdown。"""

    def __call__(
        self,
        fnm: Union[str, Path, bytes, BinaryIO],
        **kwargs,
    ) -> Tuple[List[Tuple[str, str]], list]:
        """
        Returns:
            sections: [(markdown_text, ""), ...]
            tables: []  （表格已并入 markdown）
        """
        converter = _get_converter()
        tmp_path: Optional[Path] = None

        try:
            if isinstance(fnm, (str, Path)):
                source = Path(fnm)
                result = converter.convert(str(source))
            else:
                import tempfile

                data = fnm.read() if hasattr(fnm, "read") else fnm
                suffix = kwargs.get("suffix", ".bin")
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(data if isinstance(data, bytes) else bytes(data))
                    tmp_path = Path(tmp.name)
                result = converter.convert(str(tmp_path))

            markdown = result.document.export_to_markdown()
            text = sanitize_text(markdown or "").strip()
            if not text:
                return [], []
            return [(text, "")], []
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
