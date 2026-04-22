"""embedder.py — Russian-aware text embedding with pluggable backends.

Supported backends:
  • sentence-transformers (local)  — best for Russian: intfloat/multilingual-e5-large,
                                     deepvk/USER-bge-m3, ai-forever/sbert_large_nlu_ru
  • openai-compatible REST API     — any /v1/embeddings endpoint (OpenAI, vLLM, LM Studio,
                                     Ollama, LocalAI, GigaChat, YandexGPT, etc.)
  • cohere                         — multilingual-22-12 (good Russian)
  • stub                           — returns zero-vectors, useful for dry-run / CI
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    backend: str = "openai_compatible"
    model: str = "text-embedding-3-small"
    api_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    dim: int = 0
    batch_size: int = 32
    max_tokens: int = 512
    ru_normalize: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmbedderConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        base = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        obj = cls(**base)
        obj.extra = extra
        return obj

    RUSSIAN_PRESETS: ClassVar[Dict[str, Dict[str, str]]] = {
        "sentence_transformers": {"model": "intfloat/multilingual-e5-large"},
        "openai_compatible": {"model": "multilingual-e5-large"},
        "cohere": {"model": "embed-multilingual-v3.0"},
    }


def _ru_preprocess(text: str, max_chars: int = 8192) -> str:
    """Light normalisation for Russian text before embedding."""
    text = unicodedata.normalize("NFC", text)
    for ch in ("\u00ad", "\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(ch, "")
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()[:max_chars]


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64, separator: str = "\n") -> List[str]:
    words = text.split(separator)
    chunks, buf = [], []
    for w in words:
        buf.append(w)
        if len(" ".join(buf)) >= chunk_size:
            chunks.append(separator.join(buf))
            buf = buf[-overlap:] if overlap else []
    if buf:
        chunks.append(separator.join(buf))
    return chunks or [text]


class Embedder(ABC):
    def __init__(self, cfg: EmbedderConfig) -> None:
        self.cfg = cfg
        self._dim: Optional[int] = cfg.dim or None

    @property
    def dim(self) -> int:
        if self._dim is None:
            probe = self.embed(["probe"])
            self._dim = len(probe[0])
        return self._dim  # type: ignore[return-value]

    def preprocess(self, texts: List[str]) -> List[str]:
        if self.cfg.ru_normalize:
            return [_ru_preprocess(t) for t in texts]
        return texts

    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> List[List[float]]: ...

    def embed(self, texts: List[str]) -> List[List[float]]:
        texts = self.preprocess(texts)
        results: List[List[float]] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i : i + self.cfg.batch_size]
            results.extend(self._embed_batch(batch))
        return results

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class StubEmbedder(Embedder):
    """Returns deterministic zero-like vectors — no external deps."""
    _DIM = 384

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * self._DIM for _ in texts]

    @property
    def dim(self) -> int:
        return self._DIM


class SentenceTransformersEmbedder(Embedder):
    """Local inference. Best Russian: intfloat/multilingual-e5-large."""

    def __init__(self, cfg: EmbedderConfig) -> None:
        super().__init__(cfg)
        from sentence_transformers import SentenceTransformer  # type: ignore
        device = cfg.extra.get("device", "cpu")
        logger.info("Loading SentenceTransformer model %r on %s", cfg.model, device)
        self._model = SentenceTransformer(cfg.model, device=device)
        self._e5_prefix = "query: " if "e5" in cfg.model.lower() else ""

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        prefixed = [self._e5_prefix + t if self._e5_prefix else t for t in texts]
        vecs = self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class OpenAICompatibleEmbedder(Embedder):
    """Any /v1/embeddings REST endpoint: Ollama, vLLM, LM Studio, OpenAI, GigaChat, etc."""

    def __init__(self, cfg: EmbedderConfig) -> None:
        super().__init__(cfg)
        import httpx
        base = cfg.api_url.rstrip("/")
        self._url = f"{base}/embeddings"
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self._client = httpx.Client(headers=headers, timeout=60)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.post(self._url, json={"model": self.cfg.model, "input": texts})
        resp.raise_for_status()
        items = sorted(resp.json()["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


class CohereEmbedder(Embedder):
    """Cohere embed-multilingual-v3.0 — solid Russian support."""

    def __init__(self, cfg: EmbedderConfig) -> None:
        super().__init__(cfg)
        import cohere  # type: ignore
        self._co = cohere.Client(cfg.api_key)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._co.embed(texts=texts, model=self.cfg.model, input_type="search_document")
        return [list(e) for e in resp.embeddings]


_BACKEND_MAP: Dict[str, type] = {
    "stub": StubEmbedder,
    "sentence_transformers": SentenceTransformersEmbedder,
    "openai_compatible": OpenAICompatibleEmbedder,
    "cohere": CohereEmbedder,
}


def build_embedder(cfg: EmbedderConfig) -> Embedder:
    cls = _BACKEND_MAP.get(cfg.backend)
    if cls is None:
        raise ValueError(
            f"Unknown embedder backend {cfg.backend!r}. Supported: {', '.join(_BACKEND_MAP)}"
        )
    return cls(cfg)


@dataclass
class Chunk:
    doc_id: str
    chunk_idx: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}__chunk_{self.chunk_idx}"


def chunk_document(
    doc_id: str, text: str, metadata: Dict[str, Any],
    chunk_size: int = 512, overlap: int = 64,
) -> List[Chunk]:
    parts = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            doc_id=doc_id, chunk_idx=i, text=part,
            metadata={**metadata, "chunk_idx": i, "chunk_total": len(parts)},
        )
        for i, part in enumerate(parts)
    ]
