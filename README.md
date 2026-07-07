# customer-service-kb

基于 RAG（检索增强生成）的 AI 客服知识库系统。从历史客服对话中自动提取 FAQ，构建混合检索知识库，并通过大语言模型实现智能问答。

## 功能特性

- **对话加载**：支持 JSON、CSV 及旧版 `raw_logs.csv` 格式，无数据时自动生成示例对话
- **FAQ 自动生成**：使用 LLM 从客服对话中提取、标准化问答对，并支持去重
- **混合检索**：ChromaDB 向量检索 + BM25 关键词检索（`EnsembleRetriever` 融合），BM25 使用 jieba 中文分词
- **多查询扩展**：将用户短查询扩展为多条客服场景问法，提升召回率
- **RAG 问答**：检索相关知识后由 LLM 生成简洁、友好的客服回复
- **评估模块**：支持批量测试与检索命中率统计
- **MCP 服务**：通过 Model Context Protocol 对外暴露知识库查询接口，供 Cursor 等 AI 客户端调用
- **CI/CD**：基于 GitHub Actions 自动执行测试，并在推送到 `master` 后自动构建与推送 Docker 镜像到 Docker Hub

## 系统架构

```
历史对话数据
    │
    ▼
ConversationLoader ──► FAQGenerator (LLM 提取 + 去重)
    │                        │
    │                        ▼
    │                  extracted_qa.csv / .json
    │                        │
    ▼                        ▼
KnowledgeBaseManager (ChromaDB + BM25 混合检索)
    │
    ▼
RAGAgent
    ├── QueryExpander   (多查询扩展)
    ├── HybridRetriever (向量 + BM25)
    └── LLM 生成回答
    │
    ▼
客服问答 / MCP 客户端
```

MCP 模式下，外部 AI 客户端通过 stdio 协议调用 `mcp-server/`，底层复用同一套 `RAGAgent` 与知识库。

## 项目结构

```
customer-service-kb/
├── main.py                 # 主入口：加载对话 → 生成 FAQ → 构建知识库 → 交互问答
├── config.py               # 路径、模型、分块与 RAG 参数配置
├── requirements.txt        # Python 依赖
├── data/
│   ├── conversations/      # 对话数据与提取结果
│   │   ├── raw_logs.csv
│   │   ├── extracted_qa.csv
│   │   └── extracted_qa.json
│   ├── raw/
│   │   └── faq_manual.csv  # 手工维护的 FAQ
│   └── eval/               # 评估结果（运行 --eval 后生成）
│       └── eval_results.csv
├── src/
│   ├── data_loader.py      # 对话数据加载
│   ├── faq_generator.py    # FAQ 提取与导出
│   ├── vector_store.py     # ChromaDB 向量库与混合检索管理
│   ├── hybrid_retriever.py # BM25 + 向量混合检索
│   ├── query_expansion.py  # 多查询扩展
│   ├── rag_agent.py        # RAG 问答 Agent
│   ├── evaluator.py        # 批量评估
│   ├── logger.py           # 日志
│   └── utils.py            # 通用工具函数
├── mcp-server/             # MCP 服务（对外暴露 RAG 能力）
│   ├── server.py           # MCP 服务器主入口
│   ├── rag_service.py      # RAG 服务包装器
│   ├── test_client.py      # MCP 客户端测试脚本
│   ├── quick_test.py       # 导入检查脚本
│   └── requirements.txt    # MCP 额外依赖
├── logs/                   # 运行日志（自动生成）
├── docker-compose.yml      # Docker 编排（含 app / mcp 服务）
└── chroma_db/              # 向量库持久化目录（运行后自动生成）
```

## 环境要求

- Python 3.10+
- 可访问 DeepSeek API（或兼容 OpenAI 接口的其他模型服务）

## 快速开始

### 1. 安装依赖

```bash
cd customer-service-kb
pip install -r requirements.txt
```

首次运行会自动下载 HuggingFace Embedding 模型（`sentence-transformers/all-MiniLM-L6-v2`），可能需要一些时间。若下载失败，可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 3. 准备对话数据（可选）

将客服对话放入 `data/conversations/`，支持以下格式（按优先级加载）：

| 文件 | 说明 |
|------|------|
| `conversations.json` | JSON 格式对话 |
| `conversations.csv` | CSV 格式，需包含 `conversation_id`、`role`、`content` 列 |
| `raw_logs.csv` | 旧版格式，需包含 `session_id`、`user_message`、`agent_message` 列 |

若不提供任何数据，程序会自动生成示例对话用于演示。

### 4. 运行

```bash
python main.py
```

程序将依次执行：

1. 加载历史对话
2. 使用 LLM 提取并去重 FAQ，导出到 `data/conversations/extracted_qa.csv` 和 `.json`
3. 重置 ChromaDB 向量库，将 FAQ 写入并构建混合检索器
4. 进入交互式问答模式（输入 `exit` 退出）

问答时会打印扩展查询与召回文档摘要，便于调试检索效果。

### 5. 批量评估

知识库构建完成后，可使用 `--eval` 对测试集运行批量评估（默认使用 `data/conversations/extracted_qa.csv`）：

```bash
python main.py --eval
python main.py --eval --test-file data/conversations/extracted_qa.csv
```

测试集 CSV 需包含 `question` 列，可选 `answer` 列作为期望答案参考。

评估指标：

| 指标 | 说明 |
|------|------|
| `context_rate` | 检索到相关文档的问答占比（检索命中率） |
| `with_context` | 至少召回 1 条上下文的问答数 |

结果保存至 `data/eval/eval_results.csv`，包含 `question`、`expected`、`actual`、`context_count` 字段。

## CI/CD（GitHub Actions + Docker Hub）

项目已配置 GitHub Actions 工作流：`.github/workflows/ci.yml`。

### 触发规则

- `pull_request` 到 `master`：只运行测试任务（`test`）
- `push` 到 `master`：先运行测试，测试通过后执行镜像构建与推送（`docker`）

### 测试任务（test）

测试在 GitHub Hosted Runner（`ubuntu-latest`）上执行，包含：

- `python -m compileall -q src mcp-server`：语法检查
- `python mcp-server/quick_test.py`：关键模块导入冒烟测试

### 镜像构建与发布（docker）

- 登录 Docker Hub（使用仓库 Secrets）
- 使用 Buildx 构建多架构镜像：`linux/amd64`、`linux/arm64`
- 推送镜像：`<DOCKERHUB_USERNAME>/customer-service-kb:latest`

### 必要仓库 Secrets

在 GitHub 仓库 `Settings -> Secrets and variables -> Actions` 中配置：

- `DOCKERHUB_USERNAME`：Docker Hub 用户名
- `DOCKERHUB_TOKEN`：Docker Hub Access Token（建议 Read & Write 权限）

### 本地拉取与运行

```bash
# 拉取最新镜像
docker pull <你的DockerHub用户名>/customer-service-kb:latest

# 运行交互式问答
docker run -it --rm \
  -e DEEPSEEK_API_KEY=your_deepseek_api_key \
  <你的DockerHub用户名>/customer-service-kb:latest
```

## MCP 服务

[`mcp-server/`](mcp-server/) 将 RAG 知识库封装为 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 服务，供 Cursor、Claude Desktop 等 AI 客户端通过 stdio 协议调用。

### 架构

```
AI 客户端 (Cursor / Claude Desktop)
    │  stdio
    ▼
mcp-server/server.py
    │
    ▼
rag_service.py → RAGAgent + KnowledgeBaseManager（复用 src/）
```

### 提供的 MCP 能力

**Tools（工具）**

| 工具名 | 说明 | 参数 |
|--------|------|------|
| `query_knowledge_base` | 查询客服知识库并返回答案 | `question`（必填） |
| `get_knowledge_base_stats` | 获取知识库统计信息 | 无 |

**Resources（资源）**

| URI | 说明 |
|-----|------|
| `knowledge-base://faqs` | FAQ 列表（只读） |
| `knowledge-base://stats` | 知识库实时状态 |

**Prompts（提示模板）**

| 名称 | 说明 |
|------|------|
| `greeting` | 客服问候语（可选 `customer_name`） |
| `query_with_context` | 带上下文的查询模板（`question` + `context`） |

### 本地启动

先确保主项目依赖与知识库已就绪（`chroma_db/` 中有 FAQ 数据，`.env` 已配置 `DEEPSEEK_API_KEY`）：

```bash
# 安装 MCP 额外依赖
pip install -r mcp-server/requirements.txt

# 启动 MCP 服务器（stdio 模式，等待客户端连接）
cd mcp-server
python server.py
```

### 测试 MCP 服务

```bash
# 检查 src 模块导入是否正常
python mcp-server/quick_test.py

# 完整功能测试（需从项目根目录运行）
python mcp-server/test_client.py
```

### Docker 启动

项目 [`docker-compose.yml`](docker-compose.yml) 提供 `mcp` 服务（profile 模式，默认不启动）：

```bash
# 仅启动 MCP 服务
docker compose --profile mcp up mcp

# 同时启动交互式 app 与 MCP
docker compose --profile mcp up
```

| 服务 | 说明 |
|------|------|
| `app` | 运行 `python main.py` 交互问答 |
| `mcp` | 运行 `python server.py` MCP 服务 |

两个服务共享 `data/`、`chroma_db/`、`logs/` 卷与 `.env` 配置。

### 在 Cursor 中配置

在项目或全局 MCP 配置中添加（路径按实际环境修改）：

```json
{
  "mcpServers": {
    "customer-service-kb": {
      "command": "python",
      "args": ["/path/to/customer-service-kb/mcp-server/server.py"],
      "env": {
        "DEEPSEEK_API_KEY": "your_api_key"
      }
    }
  }
}
```

配置完成后，AI 客户端即可通过 `query_knowledge_base` 工具查询客服知识库。

## 检索流程说明

用户提问后，`RAGAgent` 按以下步骤检索：

1. **查询扩展**：`QueryExpander` 将原始问题扩展为多条电商客服场景问法（过滤行业分析类噪声）
2. **混合检索**：原始查询以 `TOP_K` 检索，扩展查询以 `EXPANDED_QUERY_K` 检索
3. **融合去重**：合并结果并截断至 `MAX_RETRIEVED_DOCS` 条，送入 LLM 生成回答

## 数据格式说明

### 对话 JSON 格式

```json
[
  {
    "conversation_id": "C001",
    "timestamp": "2024-01-01 10:00:00",
    "messages": [
      {"role": "customer", "content": "你们的退换货政策是什么？"},
      {"role": "agent", "content": "我们提供7天无理由退货..."}
    ]
  }
]
```

### FAQ 输出格式

| 字段 | 说明 |
|------|------|
| `question` | 标准化问题 |
| `answer` | 简洁答案 |
| `category` | 问题类别 |
| `keywords` | 关键词列表 |
| `source` | 来源（对话 ID 或文档） |
| `confidence` | 置信度（0–1） |

## 配置项

在 `config.py` 中可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL_LOCAL` | `sentence-transformers/all-MiniLM-L6-v2` | 本地 Embedding 模型 |
| `CHUNK_SIZE` | `500` | 文档分块大小 |
| `CHUNK_OVERLAP` | `50` | 分块重叠长度 |
| `TOP_K` | `5` | 原始查询检索返回的文档数量 |
| `EXPANDED_QUERY_K` | `2` | 扩展查询每条检索的文档数量 |
| `MAX_RETRIEVED_DOCS` | `8` | 合并去重后送入 LLM 的最大文档数 |
| `SIMILARITY_THRESHOLD` | `0.7` | 相似度阈值 |

LLM 模型在 `main.py` 中配置，默认使用 DeepSeek Chat。

## 模块使用示例

### 单独使用 RAG Agent

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.vector_store import KnowledgeBaseManager
from src.rag_agent import create_rag_agent

load_dotenv()

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

kb = KnowledgeBaseManager()
kb.reset_collection()
kb.add_faqs([
    {
        "question": "如何查看快递物流信息？",
        "answer": "进入订单详情，点击【查看物流】即可追踪实时位置。",
        "category": "物流查询",
        "keywords": ["物流", "快递"],
        "source": "conversation_sess_005",
    }
])

agent = create_rag_agent(llm, kb)
result = agent.answer("物流")
print(result["answer"])
```

### 批量评估（代码调用）

也可在代码中直接使用 `Evaluator`：

```python
import pandas as pd
from src.evaluator import Evaluator

test_cases = pd.read_csv("data/conversations/extracted_qa.csv")
evaluator = Evaluator(agent)
results = evaluator.evaluate_batch(test_cases)
metrics = evaluator.compute_metrics(results)
evaluator.print_report(results, metrics)
evaluator.save_results(results)
```

## 技术栈

- [LangChain](https://python.langchain.com/) — LLM 编排与结构化输出
- [ChromaDB](https://www.trychroma.com/) — 向量数据库
- [sentence-transformers](https://www.sbert.net/) — 本地文本 Embedding
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) + [jieba](https://github.com/fxsjy/jieba) — BM25 关键词检索与中文分词
- [MCP SDK](https://github.com/modelcontextprotocol/python-sdk) — Model Context Protocol 服务接口
- [DeepSeek API](https://platform.deepseek.com/) — 大语言模型（可替换为其他 OpenAI 兼容接口）

## 许可证

本项目仅供学习与内部使用。
