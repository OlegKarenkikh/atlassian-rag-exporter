"""webhook_listener.py — Real-time reindex on Confluence/Jira webhook events.

Architecture:
  FastAPI HTTP server receives Confluence/Jira webhooks → asyncio queue →
  background ReindexWorker re-exports affected pages/issues and upserts them.

Confluence webhooks: page_created, page_updated, page_trashed, page_restored,
  attachment_created, attachment_updated, attachment_removed, space_*

Jira webhooks: jira:issue_created, jira:issue_updated, jira:issue_deleted

Setup in Confluence Admin → General Configuration → Webhooks:
  URL: http://<host>:8765/webhook/confluence
  Secret: <WEBHOOK_SECRET>

Setup in Jira Admin → System → WebHooks:
  URL: http://<host>:8765/webhook/jira
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, status
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)



@dataclass
class ReindexEvent:
    source: str
    event_type: str
    entity_id: str
    space_key: str = ""
    timestamp: float = field(default_factory=time.time)
    raw: Dict[str, Any] = field(default_factory=dict)


def _verify_confluence_signature(body: bytes, secret: str, header_sig: str) -> bool:
    if not secret:
        return True
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header_sig or "")


def _verify_jira_signature(body: bytes, secret: str, header_sig: str) -> bool:
    return _verify_confluence_signature(body, secret, header_sig)


def _parse_confluence_event(payload: Dict[str, Any]) -> Optional[ReindexEvent]:
    evt = payload.get("webhookEvent", "")
    page = payload.get("page", {})
    space = payload.get("space", {})
    page_id = str(page.get("id", ""))
    space_key = space.get("key", "") or page.get("space", {}).get("key", "")
    if not page_id and not space_key:
        return None
    return ReindexEvent(source="confluence", event_type=evt, entity_id=page_id,
                        space_key=space_key, raw=payload)


def _parse_jira_event(payload: Dict[str, Any]) -> Optional[ReindexEvent]:
    evt = payload.get("webhookEvent", "")
    issue_key = payload.get("issue", {}).get("key", "")
    if not issue_key:
        return None
    return ReindexEvent(source="jira", event_type=evt, entity_id=issue_key, raw=payload)


def create_app(
    event_queue: asyncio.Queue,
    webhook_secret: str = "",
    jira_secret: str = "",
) -> "FastAPI":
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi required: pip install fastapi uvicorn")

    app = FastAPI(title="Atlassian RAG Webhook Listener", version="1.0")

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "queue_size": str(event_queue.qsize())}

    @app.post("/webhook/confluence")
    async def confluence_webhook(request: Request) -> JSONResponse:
        body = await request.body()
        sig = request.headers.get("X-Hub-Signature", "")
        if webhook_secret and not _verify_confluence_signature(body, webhook_secret, sig):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad signature")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")
        evt = _parse_confluence_event(payload)
        if evt:
            await event_queue.put(evt)
            logger.info("Queued Confluence event: %s entity=%s", evt.event_type, evt.entity_id)
        return JSONResponse({"queued": bool(evt)})

    @app.post("/webhook/jira")
    async def jira_webhook(request: Request) -> JSONResponse:
        body = await request.body()
        sig = request.headers.get("X-Hub-Signature", "")
        if jira_secret and not _verify_jira_signature(body, jira_secret, sig):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad signature")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")
        evt = _parse_jira_event(payload)
        if evt:
            await event_queue.put(evt)
            logger.info("Queued Jira event: %s entity=%s", evt.event_type, evt.entity_id)
        return JSONResponse({"queued": bool(evt)})

    try:
        connected_ws: List[Any] = []

        @app.websocket("/ws/events")
        async def ws_events(ws: WebSocket) -> None:
            await ws.accept()
            connected_ws.append(ws)
            try:
                while True:
                    await ws.receive_text()
            except Exception:
                if ws in connected_ws:
                    connected_ws.remove(ws)

        app.state.connected_ws = connected_ws
    except Exception:
        pass

    return app


class ReindexWorker:
    def __init__(
        self,
        queue: asyncio.Queue,
        exporter_factory: Callable[[], Any],
        indexer_factory: Callable[[], Any],
        debounce_seconds: float = 5.0,
        max_batch: int = 20,
    ) -> None:
        self._queue = queue
        self._exporter_factory = exporter_factory
        self._indexer_factory = indexer_factory
        self._debounce = debounce_seconds
        self._max_batch = max_batch
        self._pending: Dict[str, ReindexEvent] = {}
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("ReindexWorker started (debounce=%.1fs)", self._debounce)
        while self._running:
            try:
                evt: ReindexEvent = await asyncio.wait_for(
                    self._queue.get(), timeout=self._debounce
                )
                self._pending[evt.entity_id] = evt
                while not self._queue.empty() and len(self._pending) < self._max_batch:
                    extra = self._queue.get_nowait()
                    self._pending[extra.entity_id] = extra
            except asyncio.TimeoutError:
                pass
            if self._pending:
                events = list(self._pending.values())
                self._pending.clear()
                await asyncio.get_event_loop().run_in_executor(
                    None, self._process_batch, events
                )

    def _process_batch(self, events: List[ReindexEvent]) -> None:
        logger.info("Processing reindex batch: %d events", len(events))
        exporter = self._exporter_factory()
        indexer = self._indexer_factory()
        for evt in events:
            if evt.source == "confluence":
                if "trash" in evt.event_type or "delete" in evt.event_type:
                    indexer.delete([evt.entity_id])
                    continue
                try:
                    doc = exporter.export_page_by_id(evt.entity_id)
                    if doc:
                        indexer.upsert([doc])
                        logger.info("Re-indexed page %s (%s)", evt.entity_id, evt.event_type)
                except Exception as exc:
                    logger.error("Failed to re-index page %s: %s", evt.entity_id, exc)
            elif evt.source == "jira":
                if "deleted" in evt.event_type:
                    indexer.delete([evt.entity_id])
                    continue
                try:
                    doc = exporter.export_jira_issue_by_key(evt.entity_id)
                    if doc:
                        indexer.upsert([doc])
                        logger.info("Re-indexed Jira %s (%s)", evt.entity_id, evt.event_type)
                except Exception as exc:
                    logger.error("Failed to re-index Jira %s: %s", evt.entity_id, exc)

    def stop(self) -> None:
        self._running = False


def run_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    webhook_secret: str = "",
    jira_secret: str = "",
    exporter_factory: Optional[Callable] = None,
    indexer_factory: Optional[Callable] = None,
    debounce: float = 5.0,
) -> None:
    import uvicorn
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    queue: asyncio.Queue = asyncio.Queue()
    app = create_app(queue, webhook_secret=webhook_secret, jira_secret=jira_secret)
    if exporter_factory and indexer_factory:
        worker = ReindexWorker(queue, exporter_factory, indexer_factory, debounce)
        loop.create_task(worker.run())
    config = uvicorn.Config(app, host=host, port=port, loop="none", log_level="info")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
