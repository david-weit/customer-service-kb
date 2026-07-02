"""
MCP服务器主文件
提供RAG知识库查询的MCP接口
"""
import json
import asyncio
from typing import Any

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# 导入RAG服务
from rag_service import RAGService


# 创建MCP服务器实例
server = Server("rag-knowledge-base-server")

# 初始化RAG服务（启动时加载）
print("🚀 启动RAG知识库MCP服务器...")
rag = RAGService()
print("✅ MCP服务器准备就绪")


# ============ 定义工具 (Tools) ============

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """列出所有可用的工具"""
    return [
        types.Tool(
            name="query_knowledge_base",
            description=(
                "查询客服知识库，获取常见问题解答。"
                "用于回答用户关于产品、服务、政策、流程等方面的问题。"
                "返回基于知识库的准确答案。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的问题，例如：如何办理退货？"
                    }
                },
                "required": ["question"]
            }
        ),
        types.Tool(
            name="get_knowledge_base_stats",
            description="获取知识库的统计信息，如文档数量、状态等",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, 
    arguments: dict[str, Any] | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """处理工具调用"""
    
    if name == "query_knowledge_base":
        # 参数验证
        if not arguments or "question" not in arguments:
            return [types.TextContent(
                type="text",
                text="❌ 错误：请提供 'question' 参数"
            )]
        
        question = arguments["question"]
        print(f"📝 收到查询: {question}")
        
        # 调用RAG系统
        response = rag.query(question)
        print(f"✅ 返回答案: {response[:50]}...")
        
        return [types.TextContent(
            type="text",
            text=response
        )]
    
    elif name == "get_knowledge_base_stats":
        stats = rag.get_stats()
        return [types.TextContent(
            type="text",
            text=json.dumps(stats, ensure_ascii=False, indent=2)
        )]
    
    else:
        raise ValueError(f"❌ 未知工具: {name}")


# ============ 定义资源 (Resources) ============

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """列出可用的资源"""
    return [
        types.Resource(
            uri="knowledge-base://faqs",
            name="FAQ知识库",
            description="当前知识库中的所有FAQ条目（只读）",
            mimeType="application/json",
        ),
        types.Resource(
            uri="knowledge-base://stats",
            name="知识库状态",
            description="知识库的实时状态信息",
            mimeType="application/json",
        )
    ]


@server.read_resource()
async def handle_read_resource(uri: str) -> str | bytes:
    """读取资源内容"""
    if uri == "knowledge-base://faqs":
        # 返回FAQ列表（这里可以扩展为真实数据）
        return json.dumps({
            "total_faqs": 0,
            "faqs": [],
            "message": "详细FAQ列表需要额外实现"
        }, ensure_ascii=False)
    
    elif uri == "knowledge-base://stats":
        stats = rag.get_stats()
        return json.dumps(stats, ensure_ascii=False)
    
    else:
        raise ValueError(f"❌ 未知资源URI: {uri}")


# ============ 定义提示 (Prompts) ============

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """列出可用的提示模板"""
    return [
        types.Prompt(
            name="greeting",
            description="客服问候语模板",
            arguments=[
                types.PromptArgument(
                    name="customer_name",
                    description="客户姓名",
                    required=False
                )
            ]
        ),
        types.Prompt(
            name="query_with_context",
            description="带上下文的查询模板",
            arguments=[
                types.PromptArgument(
                    name="question",
                    description="用户问题",
                    required=True
                ),
                types.PromptArgument(
                    name="context",
                    description="额外上下文信息",
                    required=False
                )
            ]
        )
    ]


@server.get_prompt()
async def handle_get_prompt(
    name: str, 
    arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """获取提示模板内容"""
    
    if name == "greeting":
        customer_name = arguments.get("customer_name", "客户") if arguments else "客户"
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="assistant",
                    content=types.TextContent(
                        type="text",
                        text=f"您好，{customer_name}！我是智能客服助手，请问有什么可以帮您？"
                    )
                )
            ]
        )
    
    elif name == "query_with_context":
        question = arguments.get("question", "") if arguments else ""
        context = arguments.get("context", "") if arguments else ""
        
        prompt_text = f"上下文：{context}\n\n问题：{question}\n\n请根据知识库回答用户问题。"
        
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=prompt_text
                    )
                )
            ]
        )
    
    else:
        raise ValueError(f"❌ 未知提示: {name}")


# ============ 启动服务器 ============

async def main():
    """启动MCP服务器"""
    print("=" * 60)
    print("🧠 RAG知识库 MCP服务器 v1.0")
    print("=" * 60)
    print("\n📋 可用功能:")
    print("  🔧 工具 (Tools):")
    print("     - query_knowledge_base: 查询知识库")
    print("     - get_knowledge_base_stats: 获取统计信息")
    print("  📚 资源 (Resources):")
    print("     - knowledge-base://faqs: FAQ列表")
    print("     - knowledge-base://stats: 状态信息")
    print("  💡 提示 (Prompts):")
    print("     - greeting: 问候语")
    print("     - query_with_context: 带上下文查询")
    print("\n🚀 服务器已启动，等待客户端连接...\n")
    
    # 使用stdio传输运行服务器
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-knowledge-base-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())