"""评估模块。"""

import pandas as pd

from src.rag_agent import RAGAgent


class Evaluator:
    """RAG 系统评估器。"""

    def __init__(self, rag_agent: RAGAgent):
        self.rag_agent = rag_agent

    def evaluate_single(self, question: str, expected_answer: str) -> dict:
        """评估单个问答。"""
        result = self.rag_agent.answer(question)
        return {
            "question": question,
            "expected": expected_answer,
            "actual": result["answer"],
            "context_count": len(result.get("contexts", [])),
        }

    def evaluate_batch(self, test_cases: pd.DataFrame) -> pd.DataFrame:
        """批量评估测试集。"""
        results = []
        for _, row in test_cases.iterrows():
            results.append(
                self.evaluate_single(
                    question=row["question"],
                    expected_answer=row.get("answer", ""),
                )
            )
        return pd.DataFrame(results)

    def compute_metrics(self, results: pd.DataFrame) -> dict:
        """计算评估指标。"""
        total = len(results)
        if total == 0:
            return {"total": 0, "with_context": 0, "context_rate": 0.0}

        with_context = (results["context_count"] > 0).sum()
        return {
            "total": total,
            "with_context": int(with_context),
            "context_rate": round(with_context / total, 4),
        }
