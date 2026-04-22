"""
vector_store.py — Unified vector database adapters for the Atlassian RAG Exporter.

Supported backends:
  chromadb    : Chroma (local persistent or HTTP client)
  qdrant      : Qdrant (local or cloud)
  weaviate    : Weaviate v4 (local or WCS cloud)
  pinecone    : Pinecone Serverless / Pod
  pgvector    : PostgreSQL + pgvector extension
  opensearch  : OpenSearch with knn_vector (also covers AWS OpenSearch Service)
  milvus      : Milvus / Zilliz Cloud
  redis       : Redis Stack (RedisSearch / VSS)
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("atlassian_rag_exporter.vector_store")


@dataclass
class RAGDocument:
    doc_id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_markdown_file(cls, content_path: Path) -> "RAGDocument":
        raw = content_path.read_text(encoding="utf-8")
        meta_path = content_path.parent / "metadata.json"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        doc_id = (
            meta.get("page_id")
            or meta.get("doc_id")
            or hashlib.sha256(raw[:200].encode()).hexdigest()[:16]
        )
        return cls(doc_id=doc_id, text=raw, metadata=meta)


@dataclass
class SearchResult:
    doc_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: str = ""


SUPPORTED_BACKENDS = [
    "chromadb",
    "qdrant",
    "weaviate",
    "pinecone",
    "pgvector",
    "opensearch",
    "milvus",
    "redis",
]


@dataclass
class VectorStoreConfig:
    backend: str = "chromadb"
    collection: str = "rag_corpus"
    embedding_dim: int = 1536
    distance: str = "cosine"
    host: str = "localhost"
    port: int = 0
    api_key: Optional[str] = None
    url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict) -> "VectorStoreConfig":
        valid = {k for k in cls.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in valid}
        cfg = cls(**{k: v for k, v in d.items() if k in valid})
        cfg.extra.update(extra)
        return cfg

    @classmethod
    def from_yaml(cls, path: str, section: str = "vector_store") -> "VectorStoreConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw.get(section, {}))


class VectorStoreAdapter(ABC):
    def __init__(self, cfg: VectorStoreConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def upsert(self, docs: List[RAGDocument]) -> int: ...

    @abstractmethod
    def delete(self, ids: List[str]) -> int: ...

    @abstractmethod
    def search(self, vector: List[float], k: int = 5) -> List[SearchResult]: ...

    def close(self) -> None:
        pass


class ChromaAdapter(VectorStoreAdapter):
    def connect(self) -> None:
        import chromadb  # type: ignore[import]

        if self.cfg.url:
            self._client = chromadb.HttpClient(host=self.cfg.host, port=self.cfg.port or 8000)
        else:
            persist_path = self.cfg.extra.get("persist_path", "./chroma_db")
            self._client = chromadb.PersistentClient(path=persist_path)
        distance_map = {"cosine": "cosine", "l2": "l2", "dot": "ip"}
        self._col = self._client.get_or_create_collection(
            name=self.cfg.collection,
            metadata={"hnsw:space": distance_map.get(self.cfg.distance, "cosine")},
        )
        logger.info("Chroma: connected, collection=%r", self.cfg.collection)

    def upsert(self, docs: List[RAGDocument]) -> int:
        ids = [d.doc_id for d in docs]
        texts = [d.text for d in docs]
        metas = [d.metadata for d in docs]
        embeddings = [d.embedding for d in docs if d.embedding]
        if embeddings and len(embeddings) == len(docs):
            self._col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
        else:
            self._col.upsert(ids=ids, documents=texts, metadatas=metas)
        return len(docs)

    def delete(self, ids: List[str]) -> int:
        self._col.delete(ids=ids)
        return len(ids)

    def search(self, vector: List[float], k: int = 5) -> List[SearchResult]:
        res = self._col.query(
            query_embeddings=[vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out: List[SearchResult] = []
        for i, doc_id in enumerate(res["ids"][0]):
            out.append(
                SearchResult(
                    doc_id=doc_id,
                    score=1 - res["distances"][0][i],
                    text=res["documents"][0][i],
                    metadata=res["metadatas"][0][i],
                )
            )
        return out


class QdrantAdapter(VectorStoreAdapter):
    def connect(self) -> None:
        from qdrant_client import QdrantClient  # type: ignore[import]
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

        dist_map = {"cosine": Distance.COSINE, "l2": Distance.EUCLID, "dot": Distance.DOT}
        kw: Dict[str, Any] = {}
        if self.cfg.url:
            kw["url"] = self.cfg.url
        else:
            kw["host"] = self.cfg.host
            kw["port"] = self.cfg.port or 6333
        if self.cfg.api_key:
            kw["api_key"] = self.cfg.api_key
        self._client = QdrantClient(**kw)
        existing = [c.name for c in self._client.get_collections().collections]
        if self.cfg.collection not in existing:
            self._client.create_collection(
                collection_name=self.cfg.collection,
                vectors_config=VectorParams(
                    size=self.cfg.embedding_dim,
                    distance=dist_map.get(self.cfg.distance, Distance.COSINE),
                ),
            )
        logger.info("Qdrant: connected, collection=%r", self.cfg.collection)

    def upsert(self, docs: List[RAGDocument]) -> int:
        from qdrant_client.models import PointStruct  # type: ignore[import]

        points = [
            PointStruct(
                id=abs(hash(d.doc_id)) % (2**63),
                vector=d.embedding or [],
                payload={**d.metadata, "_text": d.text[:2000], "_doc_id": d.doc_id},
            )
            for d in docs
            if d.embedding
        ]
        if points:
            self._client.upsert(collection_name=self.cfg.collection, points=points)
        return len(points)

    def delete(self, ids: List[str]) -> int:
        from qdrant_client.models import (  # type: ignore[import]
            FieldCondition,
            Filter,
            FilterSelector,
            MatchAny,
        )

        self._client.delete(
            collection_name=self.cfg.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="_doc_id", match=MatchAny(any=ids))])
            ),
        )
        return len(ids)

    def search(self, vector: List[float], k: int = 5) -> List[SearchResult]:
        hits = self._client.search(
            collection_name=self.cfg.collection,
            query_vector=vector,
            limit=k,
            with_payload=True,
        )
        return [
            SearchResult(
                doc_id=h.payload.get("_doc_id", str(h.id)),
                score=h.score,
                text=h.payload.get("_text", ""),
                metadata=h.payload,
            )
            for h in hits
        ]

    def close(self) -> None:
        self._client.close()


class WeaviateAdapter(VectorStoreAdapter):
    def connect(self) -> None:
        import weaviate  # type: ignore[import]

        if self.cfg.url:
            self._client = weaviate.connect_to_custom(
                http_host=self.cfg.host,
                http_port=self.cfg.port or 8080,
                http_secure=self.cfg.url.startswith("https"),
            )
        elif self.cfg.api_key:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cfg.host,
                auth_credentials=weaviate.auth.AuthApiKey(self.cfg.api_key),
            )
        else:
            self._client = weaviate.connect_to_local(host=self.cfg.host, port=self.cfg.port or 8080)
        col_name = self.cfg.collection.capitalize()
        if not self._client.collections.exists(col_name):
            import weaviate.classes.config as wc  # type: ignore[import]

            dist_map = {
                "cosine": wc.VectorDistances.COSINE,
                "l2": wc.VectorDistances.L2_SQUARED,
                "dot": wc.VectorDistances.DOT,
            }
            self._client.collections.create(
                col_name,
                vectorizer_config=wc.Configure.Vectorizer.none(),
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    distance_metric=dist_map.get(self.cfg.distance, wc.VectorDistances.COSINE)
                ),
            )
        self._col = self._client.collections.get(col_name)
        logger.info("Weaviate: connected, class=%r", col_name)

    def upsert(self, docs: List[RAGDocument]) -> int:
        from weaviate.classes.data import DataObject  # type: ignore[import]

        objects = [
            DataObject(
                properties={**d.metadata, "_text": d.text[:2000]},
                uuid=d.doc_id,
                vector=d.embedding or [],
            )
            for d in docs
            if d.embedding
        ]
        if objects:
            self._col.data.insert_many(objects)
        return len(objects)

    def delete(self, ids: List[str]) -> int:
        for doc_id in ids:
            self._col.data.delete_by_id(doc_id)
        return len(ids)

    def search(self, vector: List[float], k: int = 5) -> List[SearchResult]:
        from weaviate.classes.query import MetadataQuery  # type: ignore[import]

        res = self._col.query.near_vector(
            near_vector=vector,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            SearchResult(
                doc_id=str(o.uuid),
                score=1 - (o.metadata.distance or 0),
                text=o.properties.get("_text", ""),
                metadata=o.properties,
            )
            for o in res.objects
        ]

    def close(self) -> None:
        self._client.close()


class PineconeAdapter(VectorStoreAdapter):
    def connect(self) -> None:
        from pinecone import Pinecone, ServerlessSpec  # type: ignore[import]

        self._pc = Pinecone(api_key=self.cfg.api_key or "")
        existing = [i.name for i in self._pc.list_indexes()]
        if self.cfg.collection not in existing:
            cloud = self.cfg.extra.get("cloud", "aws")
            region = self.cfg.extra.get("region", "us-east-1")
            metric_map = {"cosine": "cosine", "l2": "euclidean", "dot": "dotproduct"}
            self._pc.create_index(
                name=self.cfg.collection,
                dimension=self.cfg.embedding_dim,
                metric=metric_map.get(self.cfg.distance, "cosine"),
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self._index = self._pc.Index(self.cfg.collection)
        logger.info("Pinecone: connected, index=%r", self.cfg.collection)

    def upsert(self, docs: List[RAGDocument]) -> int:
        vectors = [
            {
                "id": d.doc_id,
                "values": d.embedding or [],
                "metadata": dict(d.metadata, **{"_text": d.text[:1000]}),
            }
            for d in docs
            if d.embedding
        ]
        if vectors:
            batch = self.cfg.extra.get("batch_size", 100)
            for i in range(0, len(vectors), batch):
                self._index.upsert(vectors=vectors[i : i + batch])
        return len(vectors)

    def delete(self, ids: List[str]) -> int:
        self._index.delete(ids=ids)
        return len(ids)

    def search(self, vector: List[float], k: int = 5) -> List[SearchResult]:
        res = self._index.query(vector=vector, top_k=k, include_metadata=True)
        return [
            SearchResult(
                doc_id=m.id,
                score=m.score,
                text=m.metadata.get("_text", ""),
                metadata=m.metadata,
            )
            for m in res.matches
        ]


class PgVectorAdapter(VectorStoreAdapter):
    def connect(self) -> None:
        import psycopg2  # type: ignore[import]
        from pgvector.psycopg2 import register_vector  # type: ignore[import]

        dsn = self.cfg.extra.get("dsn") or (
            f"host={self.cfg.host} port={self.cfg.port or 5432} "
            f"dbname={self.cfg.extra.get('dbname', 'rag')} "
            f"user={self.cfg.extra.get('user', 'postgres')} "
            f"password={self.cfg.extra.get('password', '')}"
        )
        self._conn = psycopg2.connect(dsn)
        register_vector(self._conn)
        dist_ops = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "dot": "vector_ip_ops",
        }
        op = dist_ops.get(self.cfg.distance, "vector_cosine_ops")
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS {self.cfg.collection} (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    embedding vector({self.cfg.embedding_dim}),
                    metadata JSONB
                )""")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.cfg.collection}_vec_idx "
                f"ON {self.cfg.collection} USING ivfflat (embedding {op})"
            )
        self._conn.commit()
        logger.info("pgvector: connected, table=%r", self.cfg.collection)

    def upsert(self, docs: List[RAGDocument]) -> int:
        with self._conn.cursor() as cur:
            for d in docs:
                if d.embedding:
                    cur.execute(
                        f"INSERT INTO {self.cfg.collection} (id, text, embedding, metadata) "
                        "VALUES (%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE "
                        "SET text=EXCLUDED.text, embedding=EXCLUDED.embedding, metadata=EXCLUDED.metadata",
                        (d.doc_id, d.text[:8000], d.embedding, json.dumps(d.metadata)),
                    )
        self._conn.commit()
        return len(docs)

    def delete(self, ids: List[str]) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.cfg.collection} WHERE id = ANY(%s)", (ids,))
        self._conn.commit()
        return len(ids)

    def search(self, vector: List[float], k: int = 5) -> List[SearchResult]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, metadata, 1-(embedding <=> %s::vector) AS score "
                f"FROM {self.cfg.collection} ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector, vector, k),
            )
            rows = cur.fetchall()
        return [
            SearchResult(doc_id=r[0], text=r[1], metadata=r[2] or {}, score=float(r[3]))
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()


class OpenSearchAdapter(VectorStoreAdapter):
    def connect(self) -> None:
        from opensearchpy import OpenSearch, RequestsHttpConnection  # type: ignore[import]

        hosts = [{"host": self.cfg.host, "port": self.cfg.port or 9200}]
        auth = (
            (self.cfg.extra.get("user", "admin"), self.cfg.extra.get("password", "admin"))
            if self.cfg.extra.get("user")
            else None
        )
        self._os = OpenSearch(
            hosts=hosts,
            http_auth=auth,
            use_ssl=self.cfg.extra.get("use_ssl", False),
            verify_certs=self.cfg.extra.get("verify_certs", False),
            connection_class=RequestsHttpConnection,
        )
        engine_map = {"cosine": "cosinesimil", "l2": "l2", "dot": "innerproduct"}
        if not self._os.indices.exists(self.cfg.collection):
            self._os.indices.create(
                self.cfg.collection,
                body={
                    "settings": {"index": {"knn": True}},
                    "mappings": {
                        "properties": {
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": self.cfg.embedding_dim,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": engine_map.get(self.cfg.distance, "cosinesimil"),
                                    "engine": "nmslib",
                                },
                            },
                            "text": {"type": "text"},
                            "doc_id": {"type": "keyword"},
                        }
                    },
                },
            )
        logger.info("OpenSearch: connected, index=%r", self.cfg.collection)

    def upsert(self, docs: List[RAGDocument]) -> int:
        for d in docs:
            if d.embedding:
                self._os.index(
                    index=self.cfg.collection,
                    id=d.doc_id,
                    body={
                        "doc_id": d.doc_id,
                        "text": d.text[:8000],
                        "embedding": d.embedding,
                        **d.metadata,
                    },
                )
        return len(docs)

    def delete(self, ids: List[str]) -> int:
        for doc_id in ids:
            self._os.delete(index=self.cfg.collection, id=doc_id, ignore=[404])
        return len(ids)

    def search(self, vector: List[float], k: int = 5) -> List[SearchResult]:
        body = {"size": k, "query": {"knn": {"embedding": {"vector": vector, "k": k}}}}
        res = self._os.search(index=self.cfg.collection, body=body)
        return [
            SearchResult(
                doc_id=h["_source"].get("doc_id", h["_id"]),
                score=h["_score"],
                text=h["_source"].get("text", ""),
                metadata=h["_source"],
            )
            for h in res["hits"]["hits"]
        ]


class MilvusAdapter(VectorStoreAdapter):
    def connect(self) -> None:
        from pymilvus import MilvusClient  # type: ignore[import]

        uri = self.cfg.url or f"http://{self.cfg.host}:{self.cfg.port or 19530}"
        token = self.cfg.api_key or self.cfg.extra.get("token", "")
        self._client = MilvusClient(uri=uri, token=token)
        dist_map = {"cosine": "COSINE", "l2": "L2", "dot": "IP"}
        if not self._client.has_collection(self.cfg.collection):
            self._client.create_collection(
                collection_name=self.cfg.collection,
                dimension=self.cfg.embedding_dim,
                metric_type=dist_map.get(self.cfg.distance, "COSINE"),
                auto_id=False,
            )
        logger.info("Milvus: connected, collection=%r", self.cfg.collection)

    def upsert(self, docs: List[RAGDocument]) -> int:
        data = [
            {
                "id": d.doc_id,
                "vector": d.embedding,
                "text": d.text[:4000],
                **{k: str(v)[:512] for k, v in d.metadata.items()},
            }
            for d in docs
            if d.embedding
        ]
        if data:
            self._client.upsert(collection_name=self.cfg.collection, data=data)
        return len(data)

    def delete(self, ids: List[str]) -> int:
        self._client.delete(collection_name=self.cfg.collection, ids=ids)
        return len(ids)

    def search(self, vector: List[float], k: int = 5) -> List[SearchResult]:
        res = self._client.search(
            collection_name=self.cfg.collection,
            data=[vector],
            limit=k,
            output_fields=["text", "doc_id"],
        )
        out: List[SearchResult] = []
        for hit in res[0]:
            entity = hit.get("entity", {})
            out.append(
                SearchResult(
                    doc_id=entity.get("doc_id", hit.get("id", "")),
                    score=hit.get("distance", 0.0),
                    text=entity.get("text", ""),
                    metadata=entity,
                )
            )
        return out

    def close(self) -> None:
        self._client.close()


class RedisAdapter(VectorStoreAdapter):
    def connect(self) -> None:
        import redis as _redis  # type: ignore[import]
        from redis.commands.search.field import (  # type: ignore[import]
            TagField,
            TextField,
            VectorField,
        )
        from redis.commands.search.indexDefinition import (  # type: ignore[import]
            IndexDefinition,
            IndexType,
        )

        url = self.cfg.url or f"redis://{self.cfg.host}:{self.cfg.port or 6379}"
        self._r = _redis.from_url(url, password=self.cfg.api_key)
        dist_map = {"cosine": "COSINE", "l2": "L2", "dot": "IP"}
        algo = self.cfg.extra.get("algorithm", "HNSW")
        try:
            self._r.ft(self.cfg.collection).info()
        except Exception:
            self._r.ft(self.cfg.collection).create_index(
                fields=[
                    TagField("$.doc_id", as_name="doc_id"),
                    TextField("$.text", as_name="text"),
                    VectorField(
                        "$.embedding",
                        algo,
                        {
                            "TYPE": "FLOAT32",
                            "DIM": self.cfg.embedding_dim,
                            "DISTANCE_METRIC": dist_map.get(self.cfg.distance, "COSINE"),
                        },
                        as_name="embedding",
                    ),
                ],
                definition=IndexDefinition(
                    prefix=[f"{self.cfg.collection}:"], index_type=IndexType.JSON
                ),
            )
        logger.info("Redis: connected, index=%r", self.cfg.collection)

    def upsert(self, docs: List[RAGDocument]) -> int:
        pipe = self._r.pipeline(transaction=False)
        for d in docs:
            if d.embedding:
                key = f"{self.cfg.collection}:{d.doc_id}"
                pipe.json().set(
                    key,
                    "$",
                    {
                        "doc_id": d.doc_id,
                        "text": d.text[:4000],
                        "embedding": d.embedding,
                        **{k: str(v)[:256] for k, v in d.metadata.items()},
                    },
                )
        pipe.execute()
        return len(docs)

    def delete(self, ids: List[str]) -> int:
        pipe = self._r.pipeline()
        for doc_id in ids:
            pipe.delete(f"{self.cfg.collection}:{doc_id}")
        pipe.execute()
        return len(ids)

    def search(self, vector: List[float], k: int = 5) -> List[SearchResult]:
        import struct

        from redis.commands.search.query import Query  # type: ignore[import]

        vec_bytes = struct.pack(f"{len(vector)}f", *vector)
        q = (
            Query(f"*=>[KNN {k} @embedding $vec AS score]")
            .sort_by("score")
            .paging(0, k)
            .return_fields("doc_id", "text", "score")
            .dialect(2)
        )
        res = self._r.ft(self.cfg.collection).search(q, query_params={"vec": vec_bytes})
        return [
            SearchResult(
                doc_id=d.doc_id,
                score=1 - float(d.score),
                text=getattr(d, "text", ""),
                metadata={},
            )
            for d in res.docs
        ]


_ADAPTER_REGISTRY: Dict[str, type] = {
    "chromadb": ChromaAdapter,
    "qdrant": QdrantAdapter,
    "weaviate": WeaviateAdapter,
    "pinecone": PineconeAdapter,
    "pgvector": PgVectorAdapter,
    "opensearch": OpenSearchAdapter,
    "milvus": MilvusAdapter,
    "redis": RedisAdapter,
}


def build_adapter(cfg: VectorStoreConfig) -> VectorStoreAdapter:
    cls = _ADAPTER_REGISTRY.get(cfg.backend)
    if cls is None:
        raise ValueError(
            f"Unknown vector store backend: {cfg.backend!r}. "
            f"Supported: {', '.join(SUPPORTED_BACKENDS)}"
        )
    return cls(cfg)


def load_corpus(
    adapter: VectorStoreAdapter,
    corpus_dir: str,
    embedder: Any = None,
    batch_size: int = 64,
) -> Dict[str, int]:
    root = Path(corpus_dir)
    md_files = sorted(root.rglob("content.md"))
    logger.info("load_corpus: found %d content.md files in %s", len(md_files), corpus_dir)
    total = errors = 0
    for i in range(0, len(md_files), batch_size):
        batch_files = md_files[i : i + batch_size]
        docs: List[RAGDocument] = []
        for f in batch_files:
            try:
                doc = RAGDocument.from_markdown_file(f)
                if embedder is not None:
                    doc.embedding = embedder(doc.text)
                docs.append(doc)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", f, exc)
                errors += 1
        if docs:
            saved = adapter.upsert(docs)
            total += saved
        logger.debug(
            "load_corpus: batch %d/%d done (%d docs)",
            i // batch_size + 1,
            -(-len(md_files) // batch_size),
            len(docs),
        )
    logger.info("load_corpus done: %d upserted, %d errors", total, errors)
    return {"upserted": total, "errors": errors}
