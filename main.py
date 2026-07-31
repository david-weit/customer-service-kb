import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

import config
from src.evaluator import Evaluator
from src.rag_agent import create_rag_agent
from src.vector_store import KnowledgeBaseManager

load_dotenv()


def setup_agent():
    """初始化 LLM + 空知识库 Agent（文档导入/入库由 Gradio 负责）。"""
    print("=" * 60)
    print("🤖 AI客服知识库系统 v1.0")
    print("=" * 60)

    _model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    llm = init_chat_model(
        model=_model_name,
        model_provider="openai",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )

    print("\n📚 初始化空知识库连接...")
    kb = KnowledgeBaseManager()
    print(f"✅ 向量库就绪: {kb.collection_name}")

    print("\n🚀 启动 RAG Agent...")
    agent = create_rag_agent(llm, kb)
    return llm, agent


def run_evaluation(agent, test_file: Path) -> None:
    """运行批量评估。"""
    import pandas as pd

    if not test_file.exists():
        raise FileNotFoundError(f"测试集不存在: {test_file}")

    print(f"\n📊 开始评估，测试集: {test_file}")
    test_cases = pd.read_csv(test_file)

    evaluator = Evaluator(agent)
    results = evaluator.evaluate_batch(test_cases)
    metrics = evaluator.compute_metrics(results)
    evaluator.print_report(results, metrics)

    output_path = evaluator.save_results(results)
    print(f"\n结果已保存: {output_path}")


def run_chat(agent) -> None:
    """交互式问答（同一进程内共用一个 thread_id，支持多轮续聊）。"""
    print("\n💬 测试问答 (输入 'exit' 退出，输入 'new' 开启新对话)")
    print("提示: 请先通过 Gradio 导入文档并构建知识库，否则检索可能无结果。")
    thread_id = agent.new_thread_id()
    print(f"🧵 thread_id={thread_id}")
    while True:
        query = input("\n👤 用户: ")
        if query.lower() == "exit":
            break
        if query.lower() == "new":
            thread_id = agent.new_thread_id()
            print(f"🆕 新对话 thread_id={thread_id}")
            continue

        response = agent.invoke(query, thread_id=thread_id)
        print(f"🤖 客服: {response}")

    print("\n✅ 系统运行结束")


def main():
    parser = argparse.ArgumentParser(description="AI 客服知识库系统")
    parser.add_argument("--eval", action="store_true", help="运行批量评估")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=config.DEFAULT_TEST_PATH,
        help="评估测试集 CSV 路径（需含 question 列）",
    )
    args = parser.parse_args()

    _, agent = setup_agent()

    if args.eval:
        run_evaluation(agent, args.test_file)
    else:
        run_chat(agent)


if __name__ == "__main__":
    main()
