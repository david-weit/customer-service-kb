FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
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

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

CMD ["python", "main.py"]
