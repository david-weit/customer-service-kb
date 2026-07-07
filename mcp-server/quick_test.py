"""
快速测试导入是否正常
"""
import sys
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"📂 项目根目录: {project_root}")
print(f"📂 Python路径: {sys.path[0]}")

try:
    from src.rag_agent import create_rag_agent
    print("✅ 成功导入 src.rag_agent")
    
    from src.vector_store import KnowledgeBaseManager
    print("✅ 成功导入 src.vector_store")
    
    print("🎉 所有导入正常！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)