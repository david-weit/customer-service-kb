import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.data_loader import ConversationLoader
from src.faq_generator import FAQGenerator
from src.rag_agent import create_rag_agent
from src.vector_store import KnowledgeBaseManager

load_dotenv()


def main():
    print("=" * 60)
    print("🤖 AI客服知识库系统 v1.0")
    print("=" * 60)

    llm = init_chat_model(
        model="deepseek-chat",
        model_provider="openai",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )

    print("\n📂 加载历史对话...")
    loader = ConversationLoader()
    conversations = loader.load_conversations()
    print(f"✅ 加载 {len(conversations)} 条对话")

    print("\n🔄 从对话中生成FAQ...")
    faq_gen = FAQGenerator(llm)

    all_faqs = faq_gen.extract_qa_from_conversations(conversations)

    if all_faqs:
        unique_faqs = faq_gen.deduplicate_faqs(all_faqs)
        print(f"✅ 生成 {len(unique_faqs)} 条FAQ (去重前: {len(all_faqs)})")

        faq_gen.export_faqs(unique_faqs, format="csv")
        faq_gen.export_faqs(unique_faqs, format="json")
    else:
        print("⚠️ 没有生成FAQ，请检查对话数据")
        unique_faqs = []

    print("\n📚 构建知识库...")
    kb = KnowledgeBaseManager()

    if unique_faqs:
        faq_dicts = [faq.model_dump() for faq in unique_faqs]
        kb.reset_collection()
        kb.add_faqs(faq_dicts)

    print("\n🚀 启动RAG Agent（已启用多查询检索）...")
    agent = create_rag_agent(llm, kb)

    print("\n💬 测试问答 (输入 'exit' 退出)")
    while True:
        query = input("\n👤 用户: ")
        if query.lower() == "exit":
            break

        response = agent.invoke(query)
        print(f"🤖 客服: {response}")

    print("\n✅ 系统运行结束")


if __name__ == "__main__":
    main()
