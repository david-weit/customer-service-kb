"""文档解析器（仿 RAGFlow deepdoc/parser 轻量实现）。"""

from .docx_parser import DocxParser
from .excel_parser import ExcelParser
from .json_parser import JsonParser
from .pdf_parser import PlainParser

__all__ = [
    "DocxParser",
    "ExcelParser",
    "JsonParser",
    "PlainParser",
]
