"""RAG Agent 模块。"""

from typing import Optional

from langchain_core.documents import Document

import config
from src.vector_store import KnowledgeBaseManager
from .logger import logger


class RAGAgent:
    """基于检索增强生成的客服问答 Agent。"""

    def __init__(
        self,
        llm,
        kb: KnowledgeBaseManager,
        top_k: Optional[int] = None,
    ):
        self.llm = llm
        self.kb = kb
        self.top_k = top_k or config.TOP_K
        logger.info("初始化 RAG Agent, model")

    def retrieve(self, question: str) -> list[Document]:
        """检索与问题相关的文档片段。"""
        return self.kb.search(question, k=self.top_k)

    def build_prompt(self, question: str, docs: list[Document]) -> str:
        """构建 RAG 提示词。"""
        context_text = "\n\n".join(
            f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc in docs
        )

        return f"""你是一个专业的客服助手。请根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请诚实告知用户，不要编造答案。

参考资料：
{context_text}

用户问题：{question}

请用简洁、友好的中文回答："""

    def answer(self, question: str) -> dict:
        """回答问题（检索 + 生成）。"""
        docs = self.retrieve(question)

        if not docs:
            return {
                "question": question,
                "answer": "抱歉，我没有找到相关信息，请联系人工客服。",
                "contexts": [],
            }

        prompt = self.build_prompt(question, docs)
        response = self.llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        return {
            "question": question,
            "answer": answer_text,
            "contexts": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
        }

    def invoke(self, question: str) -> str:
        """简化接口，直接返回回答文本。"""
        return self.answer(question)["answer"]


def create_rag_agent(llm, kb: KnowledgeBaseManager) -> RAGAgent:
    """创建 RAG Agent 实例。"""
    return RAGAgent(llm, kb)
