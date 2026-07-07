FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG HF_ENDPOINT=
ENV HF_ENDPOINT=${HF_ENDPOINT}

COPY requirements.txt .
COPY mcp-server/requirements.txt requirements-mcp.txt

RUN pip install --no-cache-dir -r requirements.txt -r requirements-mcp.txt

# 预下载 Embedding 模型（用 huggingface_hub 避免构建时加载 PyTorch）
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

CMD ["python", "main.py"]
