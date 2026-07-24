"""JSON 解析器（仿 RAGFlow RAGFlowJsonParser，结构感知切块）。"""

import json
from typing import Any, List, Optional, Union


class JsonParser:
    """将 JSON / JSONL 按体积预算切成保留结构的文本块。"""

    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: Optional[int] = None):
        self.max_chunk_size = max_chunk_size * 2
        self.min_chunk_size = (
            min_chunk_size if min_chunk_size is not None else max(max_chunk_size - 200, 50)
        )

    def __call__(self, binary: Union[bytes, str]) -> List[str]:
        if isinstance(binary, bytes):
            txt = binary.decode("utf-8", errors="ignore")
        else:
            txt = binary

        if self.is_jsonl_format(txt):
            return self._parse_jsonl(txt)
        return self._parse_json(txt)

    @staticmethod
    def _json_size(data: dict) -> int:
        return len(json.dumps(data, ensure_ascii=False))

    @staticmethod
    def _set_nested_dict(d: dict, path: list[str], value: Any) -> None:
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _list_to_dict_preprocessing(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
        if isinstance(data, list):
            return {
                str(i): self._list_to_dict_preprocessing(item)
                for i, item in enumerate(data)
            }
        return data

    def _json_split(
        self,
        data,
        current_path: Optional[List[str]],
        chunks: Optional[List[dict]],
    ) -> List[dict]:
        current_path = current_path or []
        chunks = chunks or [{}]
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = current_path + [key]
                chunk_size = self._json_size(chunks[-1])
                size = self._json_size({key: value})
                remaining = self.max_chunk_size - chunk_size

                if size < remaining:
                    self._set_nested_dict(chunks[-1], new_path, value)
                else:
                    if chunk_size >= self.min_chunk_size:
                        chunks.append({})
                    self._json_split(value, new_path, chunks)
        else:
            self._set_nested_dict(chunks[-1], current_path, data)
        return chunks

    def split_json(self, json_data, convert_lists: bool = False) -> List[dict]:
        if convert_lists:
            preprocessed_data = self._list_to_dict_preprocessing(json_data)
            chunks = self._json_split(preprocessed_data, None, None)
        else:
            chunks = self._json_split(json_data, None, None)
        if chunks and not chunks[-1]:
            chunks.pop()
        return chunks

    def _parse_json(self, content: str) -> List[str]:
        try:
            json_data = json.loads(content)
            chunks = self.split_json(json_data, True)
            return [json.dumps(line, ensure_ascii=False) for line in chunks if line]
        except json.JSONDecodeError:
            return []

    def _parse_jsonl(self, content: str) -> List[str]:
        all_chunks: List[str] = []
        for line in content.strip().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                chunks = self.split_json(data, convert_lists=True)
                all_chunks.extend(
                    json.dumps(chunk, ensure_ascii=False) for chunk in chunks if chunk
                )
            except json.JSONDecodeError:
                continue
        return all_chunks

    def is_jsonl_format(
        self, txt: str, sample_limit: int = 10, threshold: float = 0.8
    ) -> bool:
        lines = [line.strip() for line in txt.strip().splitlines() if line.strip()]
        if not lines:
            return False
        try:
            json.loads(txt)
            return False
        except json.JSONDecodeError:
            pass

        sample_lines = lines[: min(len(lines), sample_limit)]
        valid_lines = sum(1 for line in sample_lines if self._is_valid_json(line))
        if not valid_lines:
            return False
        return (valid_lines / len(sample_lines)) >= threshold

    @staticmethod
    def _is_valid_json(line: str) -> bool:
        try:
            json.loads(line)
            return True
        except json.JSONDecodeError:
            return False
