"""评估模块。"""

from pathlib import Path
from typing import Optional

import pandas as pd

import config
from src.rag_agent import RAGAgent
from src.utils import save_csv


class Evaluator:
    """RAG 系统评估器。"""

    def __init__(self, rag_agent: RAGAgent):
        self.rag_agent = rag_agent

    @staticmethod
    def validate_test_cases(test_cases: pd.DataFrame) -> None:
        """校验测试集格式。"""
        if "question" not in test_cases.columns:
            raise ValueError("测试集必须包含 question 列")

    def evaluate_single(self, question: str, expected_answer: str = "") -> dict:
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
        self.validate_test_cases(test_cases)
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

    def print_report(
        self, results: pd.DataFrame, metrics: dict, sample_size: int = 3
    ) -> None:
        """打印评估报告。"""
        total = metrics["total"]
        with_context = metrics["with_context"]
        rate = metrics["context_rate"]
        print(f"\n评估完成：共 {total} 条")
        print(f"检索命中率 (context_rate): {rate} ({with_context}/{total})")

        for i, (_, row) in enumerate(results.head(sample_size).iterrows(), 1):
            print(f"\n[{i}] Q: {row['question']}")
            if row.get("expected"):
                print(f"    Expected: {row['expected'][:80]}")
            print(f"    Actual:   {row['actual'][:80]}")
            print(f"    Contexts: {row['context_count']}")

    def save_results(
        self,
        results: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Path:
        """保存评估结果到 CSV。"""
        path = output_path or config.EVAL_RESULTS_PATH
        save_csv(results, path)
        return path
