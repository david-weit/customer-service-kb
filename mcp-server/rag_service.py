"""
RAG服务包装器 - 调用主项目的RAG系统
"""
import sys
import os
from pathlib import Path

# 获取项目根目录（customer-service-kb的路径）
# mcp-server/rag_service.py -> 上级目录就是项目根目录
project_root = Path(__file__).parent.parent

# 将项目根目录添加到Python路径，方便导入src模块
sys.path.insert(0, str(project_root))

# 现在可以像在主项目中一样导入
from src.rag_agent import create_rag_agent
from src.vector_store import KnowledgeBaseManager
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 加载环境变量（从项目根目录）
load_dotenv(project_root / '.env')


class RAGService:
    """RAG服务包装类，供MCP服务器调用"""
    
    def __init__(self):
        """初始化RAG系统"""
        print("🔧 初始化RAG服务...")
        
        # 1. 初始化LLM（使用主项目配置）
        self.llm = init_chat_model(
            model="deepseek-chat",
            model_provider="openai",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0,
        )
        
        # 2. 加载知识库
        self.kb = KnowledgeBaseManager()
        
        # 3. 创建RAG Agent
        self.agent = create_rag_agent(self.llm, self.kb)
        
        print("✅ RAG服务初始化完成")
    
    def query(self, question: str) -> str:
        """
        查询知识库
        
        Args:
            question: 用户问题
            
        Returns:
            str: RAG系统返回的答案
        """
        try:
            response = self.agent.invoke(question)
            return response
        except Exception as e:
            return f"❌ 查询失败: {str(e)}"
    
    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        try:
            # 这里可以根据你的KnowledgeBaseManager实际方法调整
            return {
                "status": "ready",
                "collection_name": getattr(self.kb, 'collection_name', 'unknown'),
                "document_count": getattr(self.kb, 'doc_count', 0)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}