"""项目配置文件。"""

from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent

# 数据路径
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

POLICIES_DIR = RAW_DIR / "policies"
PRODUCTS_DIR = RAW_DIR / "products"
FAQ_MANUAL_PATH = RAW_DIR / "faq_manual.csv"
CHUNKED_DOCS_DIR = PROCESSED_DIR / "chunked_docs"
RAW_LOGS_PATH = CONVERSATIONS_DIR / "raw_logs.csv"
EXTRACTED_QA_PATH = CONVERSATIONS_DIR / "extracted_qa.csv"
EXTRACTED_QA_JSON_PATH = CONVERSATIONS_DIR / "extracted_qa.json"
CONVERSATIONS_JSON_PATH = CONVERSATIONS_DIR / "conversations.json"
CONVERSATIONS_CSV_PATH = CONVERSATIONS_DIR / "conversations.csv"

# 向量数据库
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "customer_service_kb"

# 模型配置
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL_LOCAL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"

# 分块配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# RAG 配置
TOP_K = 5
EXPANDED_QUERY_K = 2
MAX_RETRIEVED_DOCS = 8
SIMILARITY_THRESHOLD = 0.7
