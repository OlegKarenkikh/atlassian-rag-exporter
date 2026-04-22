"""rag_api.py — FastAPI REST API поверх RAGTester.

Endpoints:
  POST /ask              — задать вопрос, получить ответ + источники
  POST /ask/stream       — то же, но SSE-стриминг
  POST /eval             — запустить eval по JSONL-файлу или inline-списку
  GET  /health           — статус сервера и LLM
  GET  /metrics          — агрегированные метрики из лога
  DELETE /log            — очистить лог Q&A
  GET  /docs             — встроенный Swagger (автоматически)

Auth: Bearer-токен через env ATLASSIAN_RAG_API_TOKEN (опционально).

Usage:
  uvicorn rag_api:app --host 0.0.0.0 --port 8080
  RAG_CONFIG=config.yaml ATLASSIAN_RAG_API_TOKEN=secret uvicorn rag_api:app
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="Вопрос на естественном языке")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Переопределить top_k для этого запроса")
    stream: Optional[bool] = Field(None, description="Форсировать стриминг/синхронный режим")

class SourceChunk(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    title: str
    space_key: str
    text_preview: str   # первые 300 символов

class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceChunk]
    model: str
    latency: Dict[str, float]   # embed_ms, retrieve_ms, llm_ms
    metrics: Dict[str, float]   # faithfulness, context_relevance
    timestamp: float

class EvalItem(BaseModel):
    question: str
    expected: Optional[str] = None

class EvalRequest(BaseModel):
    questions: Optional[List[EvalItem]] = None
    eval_file: Optional[str] = None

class EvalResult(BaseModel):
    question: str
    answer: str
    faithfulness: float
    context_relevance: float

class EvalResponse(BaseModel):
    results: List[EvalResult]
    avg_faithfulness: float
    avg_context_relevance: float
    total: int
    duration_sec: float

class HealthResponse(BaseModel):
    status: str
    llm_reachable: bool
    llm_model: str
    llm_url: str
    config: str
    uptime_sec: float

class MetricsResponse(BaseModel):
    total_queries: int
    avg_faithfulness: float
    avg_context_relevance: float
    avg_latency_llm_ms: float
    avg_latency_retrieve_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

_START_TIME = time.time()
_tester = None   # lazy-loaded singleton

def _get_tester():
    global _tester
    if _tester is None:
        from rag_tester import RAGTester
        cfg = os.environ.get("RAG_CONFIG", "config.yaml")
        if not Path(cfg).exists():
            raise RuntimeError(f"RAG config not found: {cfg}. Set RAG_CONFIG env var.")
        _tester = RAGTester(cfg)
    return _tester


def _build_ask_response(result) -> AskResponse:
    from rag_tester import _faithfulness_proxy, _context_relevance
    faith = _faithfulness_proxy(result.answer, result.chunks)
    ctx_rel = _context_relevance(result.query, result.chunks)
    sources = [
        SourceChunk(
            doc_id=c.doc_id, chunk_id=c.chunk_id, score=round(c.score, 4),
            title=c.metadata.get("title", ""),
            space_key=c.metadata.get("space_key", ""),
            text_preview=c.text[:300],
        )
        for c in result.chunks
    ]
    return AskResponse(
        query=result.query,
        answer=result.answer,
        sources=sources,
        model=result.model,
        latency={
            "embed_ms": round(result.latency_embed_ms, 2),
            "retrieve_ms": round(result.latency_retrieve_ms, 2),
            "llm_ms": round(result.latency_llm_ms, 2),
        },
        metrics={"faithfulness": round(faith, 3), "context_relevance": round(ctx_rel, 3)},
        timestamp=result.timestamp,
    )


def create_app(config_path: Optional[str] = None) -> FastAPI:
    if config_path:
        os.environ["RAG_CONFIG"] = config_path

    api_token = os.environ.get("ATLASSIAN_RAG_API_TOKEN", "")
    bearer_scheme = HTTPBearer(auto_error=False)

    def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
        if not api_token:
            return  # auth disabled
        if not credentials or credentials.credentials != api_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing Bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    app = FastAPI(
        title="Atlassian RAG API",
        version="1.0.0",
        description="REST API для поиска по Confluence/Jira корпусу с LLM-ответами",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── /health ──────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(_=Depends(verify_token)):
        try:
            t = _get_tester()
            llm_ok = t._llm.test_connection()
            return HealthResponse(
                status="ok",
                llm_reachable=llm_ok,
                llm_model=t._llm.model,
                llm_url=t._llm.url,
                config=os.environ.get("RAG_CONFIG", "config.yaml"),
                uptime_sec=round(time.time() - _START_TIME, 1),
            )
        except Exception as exc:
            return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})

    # ── /ask ─────────────────────────────────────────────────────────────────

    @app.post("/ask", response_model=AskResponse, tags=["rag"])
    def ask(req: AskRequest, _=Depends(verify_token)):
        t = _get_tester()
        # Temporary top_k override
        original_top_k = None
        if req.top_k is not None:
            original_top_k = t._retriever._top_k
            t._retriever._top_k = req.top_k
        try:
            result = t.ask(req.query, stream=req.stream if req.stream is False else None)
        finally:
            if original_top_k is not None:
                t._retriever._top_k = original_top_k
        return _build_ask_response(result)

    # ── /ask/stream ───────────────────────────────────────────────────────────

    @app.post("/ask/stream", tags=["rag"])
    async def ask_stream(req: AskRequest, _=Depends(verify_token)):
        """SSE stream: сначала chunks-metadata, потом токены ответа."""
        t = _get_tester()

        async def _generate() -> AsyncGenerator[str, None]:
            # Retrieval (sync, fast)
            chunks, embed_ms, retr_ms = t._retriever.retrieve(req.query)
            # Send sources first
            sources_event = {
                "event": "sources",
                "sources": [
                    {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": round(c.score, 4),
                     "title": c.metadata.get("title", ""),
                     "space_key": c.metadata.get("space_key", "")}
                    for c in chunks
                ],
                "embed_ms": round(embed_ms, 2),
                "retrieve_ms": round(retr_ms, 2),
            }
            yield f"data: {json.dumps(sources_event, ensure_ascii=False)}\n\n"

            if not chunks:
                done = {"event": "done", "answer": "Информация не найдена в базе знаний."}
                yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
                return

            context = "\n\n---\n\n".join(
                f"Источник: {c.metadata.get('title','')}\n{c.text}" for c in chunks
            )
            messages = t._llm._build_messages(
                t._system, context, req.query, t._ctx_header, t._q_header
            )
            # Stream tokens
            import httpx
            payload = {
                "model": t._llm.model, "messages": messages,
                "temperature": t._llm.temperature,
                "max_tokens": t._llm.max_tokens, "stream": True,
            }
            collected = []
            try:
                with t._llm._client.stream("POST", t._llm.url, json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        try:
                            token = json.loads(line)["choices"][0].get("delta", {}).get("content", "")
                            if token:
                                collected.append(token)
                                yield f"data: {json.dumps({'event':'token','token':token}, ensure_ascii=False)}\n\n"
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
            except Exception as exc:
                yield f"data: {json.dumps({'event':'error','detail':str(exc)})}\n\n"
                return

            answer = "".join(collected)
            from rag_tester import _faithfulness_proxy, _context_relevance, RAGAnswer
            faith = _faithfulness_proxy(answer, chunks)
            ctx_rel = _context_relevance(req.query, chunks)
            done_event = {
                "event": "done", "answer": answer,
                "metrics": {"faithfulness": round(faith, 3), "context_relevance": round(ctx_rel, 3)},
                "model": t._llm.model,
            }
            yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    # ── /eval ─────────────────────────────────────────────────────────────────

    @app.post("/eval", response_model=EvalResponse, tags=["rag"])
    def eval_endpoint(req: EvalRequest, _=Depends(verify_token)):
        from rag_tester import _faithfulness_proxy, _context_relevance
        t = _get_tester()
        questions: List[Dict[str, Any]] = []
        if req.questions:
            questions = [q.model_dump() for q in req.questions]
        elif req.eval_file:
            p = Path(req.eval_file)
            if not p.exists():
                raise HTTPException(404, f"eval_file not found: {req.eval_file}")
            questions = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not questions:
            raise HTTPException(400, "Provide 'questions' list or 'eval_file' path")

        t0 = time.time()
        results: List[EvalResult] = []
        for item in questions:
            result = t.ask(item["question"], stream=False)
            faith = _faithfulness_proxy(result.answer, result.chunks)
            ctx_rel = _context_relevance(result.query, result.chunks)
            results.append(EvalResult(
                question=item["question"], answer=result.answer,
                faithfulness=round(faith, 3), context_relevance=round(ctx_rel, 3),
            ))
        n = len(results)
        return EvalResponse(
            results=results,
            avg_faithfulness=round(sum(r.faithfulness for r in results) / n, 3),
            avg_context_relevance=round(sum(r.context_relevance for r in results) / n, 3),
            total=n,
            duration_sec=round(time.time() - t0, 2),
        )

    # ── /metrics ──────────────────────────────────────────────────────────────

    @app.get("/metrics", response_model=MetricsResponse, tags=["system"])
    def metrics(_=Depends(verify_token)):
        t = _get_tester()
        log_path = Path(t._log_path)
        if not log_path.exists():
            return MetricsResponse(total_queries=0, avg_faithfulness=0, avg_context_relevance=0,
                                   avg_latency_llm_ms=0, avg_latency_retrieve_ms=0)
        entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        n = len(entries)
        if not n:
            return MetricsResponse(total_queries=0, avg_faithfulness=0, avg_context_relevance=0,
                                   avg_latency_llm_ms=0, avg_latency_retrieve_ms=0)
        return MetricsResponse(
            total_queries=n,
            avg_faithfulness=round(sum(e.get("faithfulness", 0) for e in entries) / n, 3),
            avg_context_relevance=round(sum(e.get("context_relevance", 0) for e in entries) / n, 3),
            avg_latency_llm_ms=round(sum(e.get("latency_llm_ms", 0) for e in entries) / n, 1),
            avg_latency_retrieve_ms=round(sum(e.get("latency_retrieve_ms", 0) for e in entries) / n, 1),
        )

    # ── /log ─────────────────────────────────────────────────────────────────

    @app.delete("/log", tags=["system"])
    def clear_log(_=Depends(verify_token)):
        t = _get_tester()
        Path(t._log_path).write_text("")
        return {"cleared": True, "log_file": t._log_path}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag_api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        log_level="info",
    )
