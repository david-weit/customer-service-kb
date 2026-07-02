"""
MCP服务器测试客户端
用于在命令行中测试MCP服务是否正常
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """测试MCP服务器的所有功能"""
    
    # 服务器参数
    server_params = StdioServerParameters(
        command="python",
        args=["mcp-server/server.py"],  # 相对于项目根目录
        env=None  # 使用系统环境变量
    )
    
    print("=" * 60)
    print("🧪 开始测试MCP服务器")
    print("=" * 60)
    
    try:
        # 连接到服务器
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化会话
                await session.initialize()
                print("✅ 成功连接到MCP服务器")
                
                # 1. 列出所有工具
                print("\n📋 可用工具:")
                tools = await session.list_tools()
                for i, tool in enumerate(tools.tools, 1):
                    print(f"  {i}. {tool.name}")
                    print(f"     描述: {tool.description[:80]}...")
                
                # 2. 测试知识库查询
                print("\n💬 测试查询:")
                test_questions = [
                    "你好",
                    "如何退货？",
                    "你们有哪些产品？"
                ]
                
                for question in test_questions:
                    print(f"\n  👤 用户: {question}")
                    result = await session.call_tool(
                        "query_knowledge_base",
                        arguments={"question": question}
                    )
                    answer = result.content[0].text
                    print(f"  🤖 客服: {answer[:100]}...")
                
                # 3. 测试获取统计信息
                print("\n📊 测试统计信息:")
                result = await session.call_tool(
                    "get_knowledge_base_stats",
                    arguments={}
                )
                print(f"  📈 {result.content[0].text}")
                
                # 4. 列出资源
                print("\n📚 可用资源:")
                resources = await session.list_resources()
                for i, resource in enumerate(resources.resources, 1):
                    print(f"  {i}. {resource.name}: {resource.uri}")
                
                # 5. 列出提示
                print("\n💡 可用提示:")
                prompts = await session.list_prompts()
                for i, prompt in enumerate(prompts.prompts, 1):
                    print(f"  {i}. {prompt.name}: {prompt.description}")
                
                print("\n" + "=" * 60)
                print("✅ 所有测试通过！")
                print("=" * 60)
                
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())