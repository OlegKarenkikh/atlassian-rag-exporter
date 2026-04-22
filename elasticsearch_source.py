"""
elasticsearch_source.py — Configurable Elasticsearch/OpenSearch importer for RAG corpus.

Features:
  - Any ES/OpenSearch cluster (cloud or self-hosted)
  - Configurable index pattern, query DSL, field mapping
  - Scroll API + Point-in-Time (ES 7.10+)
  - Same output schema as Confluence pages (content.md + metadata.json)
  - Image/attachment download with SHA-256 dedup
  - Incremental sync via timestamp range filter
  - Auth: api_key, bearer, basic, none
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional

import requests
import yaml

logger = logging.getLogger("atlassian_rag_exporter.elasticsearch")


@dataclass
class ESFieldMapping:
    title: str = "title"
    body: str = "body"
    body_html: Optional[str] = None
    author: str = "author"
    created_at: str = "created_at"
    updated_at: str = "updated_at"
    url: str = "url"
    labels: Optional[str] = "tags"
    space: Optional[str] = "space"
    attachment_urls: List[str] = field(default_factory=list)


@dataclass
class ESSourceConfig:
    hosts: List[str] = field(default_factory=lambda: ["http://localhost:9200"])
    index: str = "*"
    auth_type: str = "none"
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    verify_ssl: bool = True
    ca_cert: Optional[str] = None
    query: Dict = field(default_factory=lambda: {"match_all": {}})
    size: int = 500
    scroll_ttl: str = "5m"
    use_pit: bool = False
    pit_keep_alive: str = "5m"
    source_fields: Optional[List[str]] = None
    fields: ESFieldMapping = field(default_factory=ESFieldMapping)
    output_dir: str = "./rag_corpus/elasticsearch"
    source_name: str = "elasticsearch"
    incremental: bool = True
    updated_at_gte: Optional[str] = None
    download_attachments: bool = False
    attachment_timeout: int = 30

    @classmethod
    def from_dict(cls, d: Dict) -> "ESSourceConfig":
        fields_cfg = d.pop("fields", {})
        valid = {k for k in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in d.items() if k in valid})
        valid_f = {k for k in ESFieldMapping.__dataclass_fields__}
        cfg.fields = ESFieldMapping(**{k: v for k, v in fields_cfg.items() if k in valid_f})
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "ESSourceConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


class ESSession:
    SUPPORTED_IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}

    def __init__(self, cfg: ESSourceConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        if cfg.auth_type == "api_key":
            self.session.headers["Authorization"] = f"ApiKey {cfg.api_key}"
        elif cfg.auth_type == "bearer":
            self.session.headers["Authorization"] = f"Bearer {cfg.bearer_token}"
        elif cfg.auth_type == "basic":
            self.session.auth = (cfg.username or "", cfg.password or "")
        if not cfg.verify_ssl:
            self.session.verify = False
        elif cfg.ca_cert:
            self.session.verify = cfg.ca_cert
        self.host = cfg.hosts[0].rstrip("/")

    def post(self, path: str, body: Dict) -> Dict:
        url = self.host + path
        for attempt in range(4):
            resp = self.session.post(url, json=body, timeout=60)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                logger.warning("ES rate-limited, sleeping %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        return {}

    def get(self, path: str, params: Optional[Dict] = None) -> Dict:
        resp = self.session.get(self.host + path, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def delete(self, path: str, body: Optional[Dict] = None) -> None:
        self.session.delete(self.host + path, json=body, timeout=30)

    def download(self, url: str, dest: Path) -> bool:
        try:
            resp = self.session.get(url, timeout=self.cfg.attachment_timeout, stream=True)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(65536):
                    f.write(chunk)
            return True
        except Exception as exc:
            logger.warning("Download failed %s: %s", url, exc)
            return False


class ESScrollIterator:
    def __init__(self, session: ESSession, cfg: ESSourceConfig, query_dsl: Dict) -> None:
        self.session = session
        self.cfg = cfg
        self.query_dsl = query_dsl

    def _body(self) -> Dict:
        b: Dict[str, Any] = {"query": self.query_dsl, "size": self.cfg.size}
        if self.cfg.source_fields:
            b["_source"] = self.cfg.source_fields
        return b

    def scroll(self) -> Generator[Dict, None, None]:
        data = self.session.post(
            f"/{self.cfg.index}/_search?scroll={self.cfg.scroll_ttl}", self._body()
        )
        scroll_id = data.get("_scroll_id")
        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {})
        count = total.get("value", 0) if isinstance(total, dict) else total
        logger.info("ES scroll: ~%d docs in %r", count, self.cfg.index)
        while hits:
            yield from hits
            if not scroll_id:
                break
            data = self.session.post(
                "/_search/scroll",
                {"scroll": self.cfg.scroll_ttl, "scroll_id": scroll_id},
            )
            scroll_id = data.get("_scroll_id", scroll_id)
            hits = data.get("hits", {}).get("hits", [])
        if scroll_id:
            try:
                self.session.delete("/_search/scroll", {"scroll_id": scroll_id})
            except Exception:
                pass

    def pit(self) -> Generator[Dict, None, None]:
        pit_data = self.session.post(
            f"/{self.cfg.index}/_pit?keep_alive={self.cfg.pit_keep_alive}", {}
        )
        pit_id = pit_data.get("id")
        if not pit_id:
            logger.warning("PIT not supported, fallback to scroll")
            yield from self.scroll()
            return
        body = self._body()
        body["pit"] = {"id": pit_id, "keep_alive": self.cfg.pit_keep_alive}
        body["sort"] = [{"_shard_doc": "asc"}]
        search_after = None
        while True:
            if search_after:
                body["search_after"] = search_after
            data = self.session.post("/_search", body)
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break
            yield from hits
            search_after = hits[-1].get("sort")
        try:
            self.session.delete("/_pit", {"id": pit_id})
        except Exception:
            pass

    def __iter__(self) -> Iterator[Dict]:
        if self.cfg.use_pit:
            yield from self.pit()
        else:
            yield from self.scroll()


class ESDocumentConverter:
    def __init__(self, cfg: ESSourceConfig) -> None:
        self.cfg = cfg
        self.fm = cfg.fields
        self.output_dir = Path(cfg.output_dir)
        self.attachments_dir = self.output_dir / "attachments"
        self.attachments_dir.mkdir(parents=True, exist_ok=True)
        self._seen: set = set()

    def _get(self, source: Dict, path: str, default: Any = "") -> Any:
        val: Any = source
        for p in path.split("."):
            val = val.get(p, default) if isinstance(val, dict) else default
        return val if val is not None else default

    def _html_to_markdown(self, html: str) -> str:
        try:
            from markdownify import markdownify as md

            return md(html, heading_style="ATX", bullets="-")
        except ImportError:
            from html.parser import HTMLParser

            class _S(HTMLParser):
                def __init__(self) -> None:
                    super().__init__()
                    self._p: List[str] = []

                def handle_data(self, d: str) -> None:
                    self._p.append(d)

            s = _S()
            s.feed(html)
            return " ".join(s._p)

    def _markdown(self, source: Dict, att_lines: List[str]) -> str:
        title = str(self._get(source, self.fm.title, "Untitled"))
        if self.fm.body_html:
            raw = str(self._get(source, self.fm.body_html, ""))
            body = self._html_to_markdown(raw) if raw else ""
        else:
            body = str(self._get(source, self.fm.body, ""))
        lines = [f"# {title}", "", body.strip()]
        if att_lines:
            lines += ["", "## Attached Files", ""] + att_lines
        return "\n".join(lines)

    def _frontmatter(self, doc_id: str, source: Dict, att_paths: List[str]) -> str:
        import yaml as _yaml

        data = {
            "doc_id": doc_id,
            "source": self.cfg.source_name,
            "index": self.cfg.index,
            "title": str(self._get(source, self.fm.title, "")),
            "url": str(self._get(source, self.fm.url, "")),
            "author": str(self._get(source, self.fm.author, "")),
            "created_at": str(self._get(source, self.fm.created_at, "")),
            "updated_at": str(self._get(source, self.fm.updated_at, "")),
            "labels": self._get(source, self.fm.labels, []) if self.fm.labels else [],
            "space": str(self._get(source, self.fm.space, "")) if self.fm.space else "",
            "attachment_count": len(att_paths),
        }
        return "---\n" + _yaml.dump(data, allow_unicode=True, sort_keys=False) + "---\n\n"

    def _download_attachments(
        self, session: ESSession, source: Dict
    ) -> tuple[List[str], List[str]]:
        import hashlib
        from pathlib import Path as _P

        if not self.cfg.download_attachments or not self.fm.attachment_urls:
            return [], []
        paths: List[str] = []
        md_lines: List[str] = []
        IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}
        for field_name in self.fm.attachment_urls:
            urls = self._get(source, field_name, [])
            if isinstance(urls, str):
                urls = [urls]
            for url in (u for u in urls if u):
                sha = hashlib.sha256(url.encode()).hexdigest()[:16]
                ext = _P(url.split("?")[0]).suffix.lower() or ".bin"
                fname = f"{sha}{ext}"
                dest = self.attachments_dir / fname
                if sha in self._seen or dest.exists():
                    self._seen.add(sha)
                    rel = str(dest.relative_to(self.output_dir))
                    paths.append(rel)
                    continue
                if session.download(url, dest):
                    self._seen.add(sha)
                    rel = str(dest.relative_to(self.output_dir))
                    paths.append(rel)
                    md_lines.append(f"![{fname}]({rel})" if ext in IMG_EXT else f"[{fname}]({rel})")
        return paths, md_lines

    @staticmethod
    def _slugify(text: str, max_len: int = 60) -> str:
        import re

        text = re.sub(r"[^\w\s-]", "", text.lower())
        return re.sub(r"[\s_-]+", "-", text).strip("-")[:max_len]

    def save(self, hit: Dict, session: ESSession) -> Optional[Path]:
        doc_id = hit.get("_id", "unknown")
        source = hit.get("_source", {})
        slug = self._slugify(str(self._get(source, self.fm.title, doc_id)))
        doc_dir = self.output_dir / f"{doc_id}_{slug}"
        doc_dir.mkdir(parents=True, exist_ok=True)
        att_paths, att_md = self._download_attachments(session, source)
        fm = self._frontmatter(doc_id, source, att_paths)
        body = self._markdown(source, att_md)
        content_path = doc_dir / "content.md"
        content_path.write_text(fm + body, encoding="utf-8")
        metadata = {
            "doc_id": doc_id,
            "source": self.cfg.source_name,
            "index": hit.get("_index", self.cfg.index),
            "title": str(self._get(source, self.fm.title, "")),
            "url": str(self._get(source, self.fm.url, "")),
            "author": str(self._get(source, self.fm.author, "")),
            "created_at": str(self._get(source, self.fm.created_at, "")),
            "updated_at": str(self._get(source, self.fm.updated_at, "")),
            "labels": self._get(source, self.fm.labels, []) if self.fm.labels else [],
            "space": str(self._get(source, self.fm.space, "")) if self.fm.space else "",
            "attachment_paths": att_paths,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        (doc_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return content_path


class ElasticsearchImporter:
    """Full pipeline: connect → scroll → convert → save."""

    def __init__(self, cfg: ESSourceConfig) -> None:
        self.cfg = cfg
        self.session = ESSession(cfg)
        self.converter = ESDocumentConverter(cfg)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.output_dir / ".es_sync_state.json"
        self._state: Dict = self._load_state()

    def _load_state(self) -> Dict:
        if self._state_file.exists():
            return json.loads(self._state_file.read_text())
        return {}

    def _save_state(self) -> None:
        self._state["last_sync"] = datetime.now(timezone.utc).isoformat()
        self._state_file.write_text(json.dumps(self._state, indent=2))

    def _build_query(self) -> Dict:
        base = self.cfg.query
        if not self.cfg.incremental:
            return base
        cutoff = self.cfg.updated_at_gte or self._state.get("last_sync")
        if not cutoff:
            return base
        return {
            "bool": {
                "must": base,
                "filter": [{"range": {self.cfg.fields.updated_at: {"gte": cutoff}}}],
            }
        }

    def run(self) -> Dict:
        query = self._build_query()
        logger.info(
            "ES import: %s index=%r incremental=%s",
            self.cfg.hosts,
            self.cfg.index,
            self.cfg.incremental,
        )
        saved = errors = 0
        for hit in ESScrollIterator(self.session, self.cfg, query):
            try:
                if self.converter.save(hit, self.session):
                    saved += 1
                    if saved % 100 == 0:
                        logger.info("ES: saved %d...", saved)
            except Exception as exc:
                errors += 1
                logger.error("Failed doc %s: %s", hit.get("_id"), exc)
        self._save_state()
        (self.output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "source": self.cfg.source_name,
                    "index": self.cfg.index,
                    "hosts": self.cfg.hosts,
                    "total_saved": saved,
                    "errors": errors,
                    "exported_at": self._state["last_sync"],
                },
                indent=2,
            )
        )
        logger.info("ES done: %d saved, %d errors", saved, errors)
        return {"saved": saved, "errors": errors, "output_dir": str(self.output_dir)}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python elasticsearch_source.py <config.yaml>")
        sys.exit(1)
    cfg = ESSourceConfig.from_yaml(sys.argv[1])
    print(ElasticsearchImporter(cfg).run())
