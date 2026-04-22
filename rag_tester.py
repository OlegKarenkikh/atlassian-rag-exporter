"""rag_tester.py — Interactive RAG test harness with LLM endpoint support.

Supports:
  • Any OpenAI-compatible /v1/chat/completions endpoint (Ollama, vLLM, LM Studio,
    LocalAI, GigaChat, YandexGPT, OpenAI, Mistral…)
  • Streaming token output
  • Russian-first prompt templates
  • RAGAS-lite evaluation (faithfulness proxy + context relevance)
  • Interactive REPL + one-shot --query CLI
  • Q&A log in JSONL format

Usage:
  python rag_tester.py --print-example-config > rag_test_config.yaml
  python rag_tester.py --config rag_test_config.yaml
  python rag_tester.py --config rag_test_config.yaml --query "Как настроить SSO?"
  python rag_tester.py --config rag_test_config.yaml --eval eval_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml

logger = logging.getLogger(__name__)

EXAMPLE_RAG_TEST_CONFIG = """\
# rag_test_config.yaml

embedder:
  backend: openai_compatible        # stub | openai_compatible | sentence_transformers
  model: "multilingual-e5-large"    # must match model used during indexing
  api_url: "http://localhost:11434/v1"
  api_key: ""
  ru_normalize: true

vector_store:
  backend: qdrant
  host: localhost
  port: 6333
  collection: atlassian_rag
  distance: cosine

llm:
  api_url: "http://localhost:11434/v1"
  api_key: ""
  model: "mistral"
  temperature: 0.1
  max_tokens: 1024
  stream: true
  timeout: 120

retrieval:
  top_k: 5
  score_threshold: 0.35
  rerank: false
  rerank_model: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

prompt:
  system: |
    Ты — умный помощник, отвечающий на вопросы на основе документации.
    Отвечай строго на основе предоставленного контекста.
    Если ответ не содержится в контексте, скажи «Информация не найдена в базе знаний».
    Ответ давай на том же языке, что и вопрос.
  context_header: "### Контекст из базы знаний:\\n"
  question_header: "### Вопрос:\\n"

log_file: "rag_qa_log.jsonl"
"""


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        title = self.metadata.get("title", "")
        space = self.metadata.get("space_key", "")
        page_id = self.metadata.get("page_id", self.doc_id)
        return f"[{space}] {title} (id={page_id})"


@dataclass
class RAGAnswer:
    query: str
    answer: str
    chunks: List[RetrievedChunk]
    latency_embed_ms: float = 0.0
    latency_retrieve_ms: float = 0.0
    latency_llm_ms: float = 0.0
    model: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMClient:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        base = cfg.get("api_url", "http://localhost:11434/v1").rstrip("/")
        self.url = f"{base}/chat/completions"
        self.model = cfg.get("model", "mistral")
        self.temperature = cfg.get("temperature", 0.1)
        self.max_tokens = cfg.get("max_tokens", 1024)
        self.stream = cfg.get("stream", True)
        self.timeout = cfg.get("timeout", 120)
        headers = {"Content-Type": "application/json"}
        key = cfg.get("api_key", "") or os.environ.get("LLM_API_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(headers=headers, timeout=self.timeout)

    def _build_messages(
        self, system: str, context: str, question: str,
        context_header: str, question_header: str,
    ) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context_header}{context}\n\n{question_header}{question}"},
        ]

    def complete(self, messages: List[Dict[str, str]], stream: Optional[bool] = None) -> str:
        use_stream = stream if stream is not None else self.stream
        payload = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            "stream": use_stream,
        }
        if use_stream:
            return self._stream_complete(payload)
        resp = self._client.post(self.url, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _stream_complete(self, payload: Dict[str, Any]) -> str:
        collected = []
        with self._client.stream("POST", self.url, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    token = json.loads(line)["choices"][0].get("delta", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        collected.append(token)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        print()
        return "".join(collected)

    def test_connection(self) -> bool:
        try:
            resp = self._client.get(self.url.replace("/chat/completions", "/models"), timeout=5)
            return resp.status_code < 500
        except Exception:
            return False


class Retriever:
    def __init__(self, vs_cfg: Dict[str, Any], embedder: Any, retrieval_cfg: Dict[str, Any]) -> None:
        from vector_store import VectorStoreConfig, build_adapter
        self._embedder = embedder
        self._top_k = retrieval_cfg.get("top_k", 5)
        self._threshold = retrieval_cfg.get("score_threshold", 0.0)
        cfg = VectorStoreConfig.from_dict(vs_cfg)
        self._adapter = build_adapter(cfg)
        self._adapter.connect()
        self._reranker = None
        if retrieval_cfg.get("rerank"):
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
                self._reranker = CrossEncoder(retrieval_cfg.get(
                    "rerank_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
            except Exception as exc:
                logger.warning("Reranker load failed: %s", exc)

    def retrieve(self, query: str) -> Tuple[List[RetrievedChunk], float, float]:
        t0 = time.perf_counter()
        vec = self._embedder.embed_one(query)
        embed_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        raw = self._adapter.search(vec, k=self._top_k * 2)
        retrieve_ms = (time.perf_counter() - t1) * 1000
        chunks = [
            RetrievedChunk(doc_id=r.doc_id, chunk_id=r.doc_id, score=r.score,
                           text=r.text, metadata=r.metadata)
            for r in raw if r.score >= self._threshold
        ]
        if self._reranker and chunks:
            scores = self._reranker.predict([[query, c.text] for c in chunks])
            chunks = [c for c, _ in sorted(zip(chunks, scores), key=lambda x: -x[1])]
        return chunks[: self._top_k], embed_ms, retrieve_ms


def _faithfulness_proxy(answer: str, chunks: List[RetrievedChunk]) -> float:
    if not chunks or not answer:
        return 0.0
    context = " ".join(c.text.lower() for c in chunks)
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not sentences:
        return 0.0
    supported = sum(
        1 for s in sentences
        if any(word in context for word in s.lower().split() if len(word) > 4)
    )
    return round(supported / len(sentences), 3)


def _context_relevance(query: str, chunks: List[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    return round(sum(c.score for c in chunks) / len(chunks), 3)


class RAGTester:
    def __init__(self, config_path: str) -> None:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._cfg = raw
        self._log_path = raw.get("log_file", "rag_qa_log.jsonl")
        from embedder import EmbedderConfig, build_embedder
        self._embedder = build_embedder(EmbedderConfig.from_dict(raw.get("embedder", {})))
        self._retriever = Retriever(raw.get("vector_store", {}), self._embedder, raw.get("retrieval", {}))
        self._llm = LLMClient(raw.get("llm", {}))
        prompt = raw.get("prompt", {})
        self._system = prompt.get("system",
            "Ты помощник. Отвечай строго на основе контекста. "
            "Если ответа нет — так и скажи.")
        self._ctx_header = prompt.get("context_header", "### Контекст:\n")
        self._q_header = prompt.get("question_header", "### Вопрос:\n")

    def ask(self, query: str, stream: Optional[bool] = None) -> RAGAnswer:
        chunks, embed_ms, retr_ms = self._retriever.retrieve(query)
        if not chunks:
            return RAGAnswer(query=query, answer="Информация не найдена в базе знаний.",
                             chunks=[], model=self._llm.model)
        context = "\n\n---\n\n".join(f"Источник: {c.source_label}\n{c.text}" for c in chunks)
        messages = self._llm._build_messages(self._system, context, query, self._ctx_header, self._q_header)
        t0 = time.perf_counter()
        answer = self._llm.complete(messages, stream=stream)
        llm_ms = (time.perf_counter() - t0) * 1000
        result = RAGAnswer(query=query, answer=answer, chunks=chunks,
                           latency_embed_ms=embed_ms, latency_retrieve_ms=retr_ms,
                           latency_llm_ms=llm_ms, model=self._llm.model)
        self._log(result)
        return result

    def print_result(self, result: RAGAnswer) -> None:
        print("\n" + "─" * 60)
        print(f"❓ Вопрос:  {result.query}")
        print(f"🤖 Ответ:   {result.answer}")
        print(f"\n📚 Источники ({len(result.chunks)} чанков):")
        for i, c in enumerate(result.chunks, 1):
            print(f"  {i}. [{c.score:.3f}] {c.source_label}")
        print(f"\n⏱  embed={result.latency_embed_ms:.0f}ms  "
              f"retrieve={result.latency_retrieve_ms:.0f}ms  "
              f"llm={result.latency_llm_ms:.0f}ms")
        faith = _faithfulness_proxy(result.answer, result.chunks)
        ctx_rel = _context_relevance(result.query, result.chunks)
        print(f"📊 faithfulness≈{faith:.2f}  context_relevance≈{ctx_rel:.2f}")
        print("─" * 60)

    def run_eval(self, eval_file: str) -> None:
        path = Path(eval_file)
        if not path.exists():
            print(f"Eval file not found: {eval_file}")
            return
        questions = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(f"Running eval on {len(questions)} questions...")
        scores = []
        for i, item in enumerate(questions, 1):
            q = item["question"]
            result = self.ask(q, stream=False)
            faith = _faithfulness_proxy(result.answer, result.chunks)
            ctx_rel = _context_relevance(q, result.chunks)
            scores.append({"faithfulness": faith, "context_relevance": ctx_rel})
            print(f"[{i}/{len(questions)}] faith={faith:.2f} ctx={ctx_rel:.2f} | {q[:60]}")
        if scores:
            print(f"\n── Eval summary ──────────────────────────────────────")
            print(f"   avg faithfulness:      {sum(s['faithfulness'] for s in scores)/len(scores):.3f}")
            print(f"   avg context_relevance: {sum(s['context_relevance'] for s in scores)/len(scores):.3f}")
            print(f"   questions evaluated:   {len(scores)}")

    def repl(self) -> None:
        llm_ok = self._llm.test_connection()
        status = "✅ доступен" if llm_ok else "⚠️  недоступен (работаем без LLM)"
        print(f"RAG Tester REPL  |  LLM {self._llm.model}: {status}")
        print("Введите вопрос, 'eval <файл>' или 'exit'")
        while True:
            try:
                query = input("\n❓ Вопрос: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nПока!")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "выход"):
                break
            if query.lower().startswith("eval "):
                self.run_eval(query[5:].strip())
                continue
            result = self.ask(query)
            self.print_result(result)

    def _log(self, result: RAGAnswer) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.debug("Log write failed: %s", exc)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RAG Tester — интерактивное тестирование RAG")
    parser.add_argument("--config", help="Path to rag_test_config.yaml")
    parser.add_argument("--query", help="Single query (non-interactive)")
    parser.add_argument("--eval", help="JSONL eval file")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--print-example-config", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.print_example_config:
        print(EXAMPLE_RAG_TEST_CONFIG)
        return 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.config:
        parser.print_help()
        return 2

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    try:
        tester = RAGTester(args.config)
    except Exception as exc:
        print(f"Init error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback; traceback.print_exc()
        return 1

    if args.eval:
        tester.run_eval(args.eval)
        return 0
    if args.query:
        result = tester.ask(args.query, stream=not args.no_stream)
        tester.print_result(result)
        return 0
    tester.repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
