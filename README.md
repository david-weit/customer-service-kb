# customer-service-kb

基于 RAG（检索增强生成）的 AI 客服知识库系统。从历史客服对话中自动提取 FAQ，构建向量知识库，并通过大语言模型实现智能问答。

## 功能特性

- **对话加载**：支持 JSON、CSV 及旧版 `raw_logs.csv` 格式，无数据时自动生成示例对话
- **FAQ 自动生成**：使用 LLM 从客服对话中提取、标准化问答对，并支持去重
- **向量知识库**：基于 ChromaDB + HuggingFace 本地 Embedding 存储与检索 FAQ
- **RAG 问答**：检索相关知识后由 LLM 生成简洁、友好的客服回复
- **评估模块**：支持批量测试与检索命中率统计

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
KnowledgeBaseManager (ChromaDB 向量库)
    │
    ▼
RAGAgent (检索 + LLM 生成)
    │
    ▼
客服问答
```

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
│   └── raw/
│       └── faq_manual.csv  # 手工维护的 FAQ
├── src/
│   ├── data_loader.py      # 对话数据加载
│   ├── faq_generator.py    # FAQ 提取与导出
│   ├── vector_store.py     # ChromaDB 向量库管理
│   ├── rag_agent.py        # RAG 问答 Agent
│   ├── evaluator.py        # 批量评估
│   └── utils.py            # 通用工具函数
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

首次运行会自动下载 HuggingFace Embedding 模型（`sentence-transformers/all-MiniLM-L6-v2`），可能需要一些时间。

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
3. 将 FAQ 写入 ChromaDB 向量库
4. 进入交互式问答模式（输入 `exit` 退出）

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
| `TOP_K` | `5` | 检索返回的文档数量 |
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
agent = create_rag_agent(llm, kb)

result = agent.answer("如何退货？")
print(result["answer"])
```

### 批量评估

```python
import pandas as pd
from src.evaluator import Evaluator

test_cases = pd.DataFrame([
    {"question": "如何退货？", "answer": "7天无理由退货"},
    {"question": "发货要多久？", "answer": "24-48小时内发货"},
])

evaluator = Evaluator(agent)
results = evaluator.evaluate_batch(test_cases)
metrics = evaluator.compute_metrics(results)
print(metrics)
```

## 技术栈

- [LangChain](https://python.langchain.com/) — LLM 编排与结构化输出
- [ChromaDB](https://www.trychroma.com/) — 向量数据库
- [sentence-transformers](https://www.sbert.net/) — 本地文本 Embedding
- [DeepSeek API](https://platform.deepseek.com/) — 大语言模型（可替换为其他 OpenAI 兼容接口）

## 许可证

本项目仅供学习与内部使用。
