# Multi-stage build — minimal production image
FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -e ".[vector-qdrant]"

COPY atlassian_rag_exporter.py auth_providers.py elasticsearch_source.py \
     embedder.py vector_store.py rag_api.py rag_tester.py webhook_listener.py \
     azure_devops_source.py ./

RUN useradd -m -u 1000 rag && chown -R rag:rag /app
USER rag

CMD ["python", "atlassian_rag_exporter.py", "--help"]

LABEL org.opencontainers.image.title="atlassian-rag-exporter"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="Confluence + Jira to RAG corpus exporter"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/OlegKarenkikh/atlassian-rag-exporter"
