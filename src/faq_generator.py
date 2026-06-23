import json
import re
from typing import Dict, List, Optional

import pandas as pd
from langchain_core.documents import Document
from pydantic import BaseModel, Field

import config


class FAQItem(BaseModel):
    """FAQ 条目结构。"""

    question: str = Field(description="标准化的问题")
    answer: str = Field(description="简洁清晰的答案")
    category: str = Field(description="问题类别")
    keywords: List[str] = Field(description="关键词列表")
    source: str = Field(description="来源（对话ID或文档）")
    confidence: float = Field(description="置信度 0-1", default=0.8)


class FAQExtractResult(BaseModel):
    """LLM 结构化输出的 FAQ 字段。"""

    question: str = Field(description="标准化的问题")
    answer: str = Field(description="简洁清晰的答案")
    category: str = Field(description="问题类别")
    keywords: List[str] = Field(description="关键词列表")
    confidence: float = Field(description="置信度 0-1", default=0.8)


class FAQExtractList(BaseModel):
    """从文档批量提取的 FAQ 列表。"""

    faqs: List[FAQExtractResult] = Field(description="提取的FAQ列表")


class FAQGenerator:
    def __init__(self, llm):
        self.llm = llm
        self._structured_llm = llm.with_structured_output(FAQExtractResult)

    def extract_qa_from_conversations(self, conversations: List[Dict]) -> List[FAQItem]:
        """从对话记录中提取问题和答案对。"""
        qa_pairs = []

        for conv in conversations:
            for customer_question, agent_answer in self._pair_qa_messages(
                conv["messages"]
            ):
                qa = self._extract_single_qa(
                    customer_question,
                    agent_answer,
                    conv["conversation_id"],
                )
                if qa:
                    qa_pairs.append(qa)

        return qa_pairs

    def _pair_qa_messages(self, messages: List[Dict]) -> List[tuple[str, str]]:
        """按消息时序配对：每个 customer 消息匹配其后最近的 agent 回复。"""
        pairs = []
        for i, msg in enumerate(messages):
            if msg["role"] != "customer":
                continue
            agent_answer = None
            for j in range(i + 1, len(messages)):
                if messages[j]["role"] == "agent":
                    agent_answer = messages[j]["content"]
                    break
            if agent_answer:
                pairs.append((msg["content"], agent_answer))
        return pairs

    def _extract_single_qa(
        self, customer_question: str, agent_answer: str, conv_id: str
    ) -> Optional[FAQItem]:
        """使用 LLM 提取和标准化单个 QA 对。"""
        prompt = f"""从以下客服对话中提取标准化的 FAQ：

顾客问题: {customer_question}
客服回答: {agent_answer}

请返回标准化的问题、答案、类别、关键词和置信度。"""

        try:
            result = self._structured_llm.invoke(prompt)
            return FAQItem(
                question=result.question,
                answer=result.answer,
                category=result.category,
                keywords=result.keywords,
                source=f"conversation_{conv_id}",
                confidence=result.confidence,
            )
        except Exception:
            return self._extract_single_qa_fallback(
                customer_question, agent_answer, conv_id, prompt
            )

    def _extract_single_qa_fallback(
        self, customer_question: str, agent_answer: str, conv_id: str, prompt: str
    ) -> Optional[FAQItem]:
        """结构化输出失败时，尝试从 JSON 文本中解析。"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            data = self._parse_json_from_text(content)
            return FAQItem(
                question=data["question"],
                answer=data["answer"],
                category=data["category"],
                keywords=data["keywords"],
                source=f"conversation_{conv_id}",
                confidence=data.get("confidence", 0.8),
            )
        except Exception:
            return None

    def _parse_json_from_text(self, text: str) -> dict:
        """从 LLM 返回文本中提取 JSON（支持 markdown 代码块）。"""
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            return json.loads(code_block.group(1))
        return json.loads(text)

    def deduplicate_faqs(
        self, faqs: List[FAQItem], similarity_threshold: float = 0.8
    ) -> List[FAQItem]:
        """去重和合并相似的 FAQ。"""
        unique_faqs: List[FAQItem] = []

        for faq in faqs:
            is_duplicate = False
            for i, existing in enumerate(unique_faqs):
                if self._is_similar(
                    faq.question, existing.question, similarity_threshold
                ):
                    is_duplicate = True
                    if faq.confidence > existing.confidence:
                        unique_faqs[i] = faq
                    break

            if not is_duplicate:
                unique_faqs.append(faq)

        return unique_faqs

    def _is_similar(self, q1: str, q2: str, threshold: float) -> bool:
        """判断两个问题是否相似（Jaccard 词重叠）。"""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / len(words1 | words2)
        return overlap > threshold

    def generate_faq_from_docs(self, docs: List[Document]) -> List[FAQItem]:
        """从文档中提取 FAQ。"""
        faqs = []
        for doc in docs:
            prompt = f"""从以下文档内容中提取可能的 FAQ（客户可能问的问题）：

文档: {doc.page_content}

请提取至少 3 个 FAQ，包含 question、answer、category、keywords、confidence 字段。"""
            try:
                result_list = self.llm.with_structured_output(FAQExtractList).invoke(
                    prompt
                )
                for result in result_list.faqs:
                    faqs.append(
                        FAQItem(
                            question=result.question,
                            answer=result.answer,
                            category=result.category,
                            keywords=result.keywords,
                            source=doc.metadata.get("source", "document"),
                            confidence=result.confidence,
                        )
                    )
            except Exception:
                continue
        return faqs

    def export_faqs(self, faqs: List[FAQItem], format: str = "csv"):
        """导出 FAQ 为 CSV 或 JSON。"""
        data = [faq.model_dump() for faq in faqs]

        if format == "csv":
            df = pd.DataFrame(data)
            config.EXTRACTED_QA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(config.EXTRACTED_QA_PATH, index=False)
            print(f"✅ 导出 {len(faqs)} 条FAQ到 {config.EXTRACTED_QA_PATH}")
        elif format == "json":
            config.EXTRACTED_QA_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(config.EXTRACTED_QA_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ 导出 {len(faqs)} 条FAQ到 {config.EXTRACTED_QA_JSON_PATH}")
