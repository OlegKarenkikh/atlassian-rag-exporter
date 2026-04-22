"""
azure_devops_source.py - Azure DevOps / Azure DevOps Server importer for RAG.

Covers:
  - Wiki pages (project wikis + code wikis)
  - Work Items (WIQL queries, all types: Epic/Feature/Story/Bug/Task)
  - Repositories: Markdown files, inline images
  - Table-aware post-processing -> JSON + CSV side-cars
  - Multimodal image captioning via local LLM (Ollama / OpenAI-compatible)

Auth modes:
  - Personal Access Token (PAT)
  - Azure Entra ID client_credentials (service principal)
  - Managed Identity / Workload Identity (IMDS)
  - Interactive device-flow (local dev)
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
import yaml
from bs4 import BeautifulSoup

logger = logging.getLogger("atlassian_rag_exporter.azure_devops")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AzureDevOpsConfig:
    """Configuration for Azure DevOps importer."""

    org_url: str = ""
    project: str = ""
    is_server: bool = False

    auth_type: str = "pat"
    pat: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    resource: str = "499b84ac-1321-427f-aa17-267ca6975798"

    import_wikis: bool = True
    import_work_items: bool = True
    import_repos: bool = False
    wiki_names: List[str] = field(default_factory=list)
    work_item_query: Optional[str] = None
    work_item_types: List[str] = field(default_factory=lambda: ["Epic", "Feature", "Story", "Bug", "Task"])
    repo_names: List[str] = field(default_factory=list)

    output_dir: str = "./rag_corpus/azure"
    incremental: bool = True

    extract_tables: bool = True
    table_min_rows: int = 2

    image_captioning: bool = False
    caption_backend: str = "ollama"
    caption_model: str = "qwen2.5vl:latest"
    caption_endpoint: str = "http://localhost:11434"
    caption_api_key: Optional[str] = None
    caption_max_tokens: int = 512
    caption_prompt: str = (
        "Describe this image in detail. Include: visual elements, text visible in the image, "
        "chart/diagram type if applicable, data values or trends shown, colors, structure. "
        "Be precise and comprehensive. Output in the same language as visible text."
    )

    requests_per_second: float = 10.0
    max_retries: int = 5
    retry_backoff: float = 2.0

    @classmethod
    def from_dict(cls, d: Dict) -> "AzureDevOpsConfig":
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str, section: str = "azure_devops") -> "AzureDevOpsConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw.get(section, {}))


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _build_session(cfg: AzureDevOpsConfig) -> requests.Session:
    session = requests.Session()
    session.headers["Accept"] = "application/json"

    if cfg.auth_type == "pat":
        token = base64.b64encode(f":{cfg.pat}".encode()).decode()
        session.headers["Authorization"] = f"Basic {token}"
    elif cfg.auth_type == "entra_client_credentials":
        token = _get_entra_token(cfg)
        session.headers["Authorization"] = f"Bearer {token}"
    elif cfg.auth_type == "managed_identity":
        token = _get_managed_identity_token(cfg.resource)
        session.headers["Authorization"] = f"Bearer {token}"
    elif cfg.auth_type == "device_flow":
        token = _get_device_flow_token(cfg)
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        raise ValueError(f"Unknown auth_type: {cfg.auth_type!r}")

    return session


def _get_entra_token(cfg: AzureDevOpsConfig) -> str:
    url = f"https://login.microsoftonline.com/{cfg.tenant_id}/oauth2/v2.0/token"
    resp = requests.post(url, data={
        "grant_type": "client_credentials",
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
        "scope": f"{cfg.resource}/.default",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def _get_managed_identity_token(resource: str) -> str:
    resp = requests.get(
        "http://169.254.169.254/metadata/identity/oauth2/token",
        params={"api-version": "2018-02-01", "resource": resource},
        headers={"Metadata": "true"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _get_device_flow_token(cfg: AzureDevOpsConfig) -> str:
    try:
        from azure.identity import DeviceCodeCredential  # type: ignore[import]
        cred = DeviceCodeCredential(tenant_id=cfg.tenant_id, client_id=cfg.client_id)
        token = cred.get_token(f"{cfg.resource}/.default")
        return token.token
    except ImportError:
        raise ImportError("Install azure-identity: pip install azure-identity")


# ---------------------------------------------------------------------------
# Rate-limited API client
# ---------------------------------------------------------------------------


class AzureDevOpsClient:
    API_VERSION = "7.1"

    def __init__(self, cfg: AzureDevOpsConfig) -> None:
        self.cfg = cfg
        self.session = _build_session(cfg)
        self._min_interval = 1.0 / max(cfg.requests_per_second, 0.1)
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def get(self, url: str, **kwargs) -> Any:
        params = kwargs.pop("params", {})
        params.setdefault("api-version", self.API_VERSION)
        for attempt in range(self.cfg.max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=60, **kwargs)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    logger.warning("Rate limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                if attempt == self.cfg.max_retries - 1:
                    raise
                time.sleep(self.cfg.retry_backoff ** attempt)

    def get_raw(self, url: str, **kwargs) -> bytes:
        params = kwargs.pop("params", {})
        params.setdefault("api-version", self.API_VERSION)
        for attempt in range(self.cfg.max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=120, **kwargs)
                if resp.status_code == 429:
                    time.sleep(int(resp.headers.get("Retry-After", 5)))
                    continue
                resp.raise_for_status()
                return resp.content
            except requests.HTTPError as exc:
                if attempt == self.cfg.max_retries - 1:
                    raise
                time.sleep(self.cfg.retry_backoff ** attempt)

    def _base(self, api: str = "") -> str:
        base = self.cfg.org_url.rstrip("/")
        project = quote(self.cfg.project, safe="")
        if api:
            return f"{base}/{project}/_apis/{api}"
        return f"{base}/{project}/_apis"

    def list_wikis(self) -> List[Dict]:
        data = self.get(self._base("wiki/wikis"))
        return data.get("value", [])

    def list_wiki_pages(self, wiki_id: str) -> List[Dict]:
        url = self._base(f"wiki/wikis/{wiki_id}/pages")
        data = self.get(url, params={"recursionLevel": "full", "includeContent": "false"})
        return self._flatten_pages(data)

    def get_wiki_page_by_path(self, wiki_id: str, path: str) -> Dict:
        url = self._base(f"wiki/wikis/{wiki_id}/pages")
        return self.get(url, params={"path": path, "includeContent": "true"})

    def get_wiki_attachment(self, wiki_id: str, filename: str) -> bytes:
        url = self._base(f"wiki/wikis/{wiki_id}/attachments")
        return self.get_raw(url, params={"name": filename})

    def _flatten_pages(self, node: Dict, acc: Optional[List] = None) -> List[Dict]:
        if acc is None:
            acc = []
        if "path" in node:
            acc.append(node)
        for child in node.get("subPages", []):
            self._flatten_pages(child, acc)
        return acc

    def query_work_items(self, wiql: str) -> List[int]:
        url = self._base("wit/wiql")
        resp = self.session.post(
            url, json={"query": wiql},
            params={"api-version": self.API_VERSION}, timeout=60,
        )
        resp.raise_for_status()
        return [wi["id"] for wi in resp.json().get("workItems", [])]

    def get_work_items_batch(self, ids: List[int], fields: Optional[List[str]] = None) -> List[Dict]:
        if not ids:
            return []
        url = self._base("wit/workitemsbatch")
        body: Dict[str, Any] = {"ids": ids[:200]}
        if fields:
            body["fields"] = fields
        resp = self.session.post(url, json=body, params={"api-version": self.API_VERSION}, timeout=60)
        resp.raise_for_status()
        return resp.json().get("value", [])

    def get_work_item_comments(self, wi_id: int) -> List[Dict]:
        url = self._base(f"wit/workItems/{wi_id}/comments")
        data = self.get(url, params={"api-version": "7.1-preview.3"})
        return data.get("comments", [])

    def list_repos(self) -> List[Dict]:
        data = self.get(self._base("git/repositories"))
        return data.get("value", [])

    def list_repo_items(self, repo_id: str, path: str = "/", recursive: bool = True) -> List[Dict]:
        url = self._base(f"git/repositories/{repo_id}/items")
        data = self.get(url, params={
            "scopePath": path,
            "recursionLevel": "full" if recursive else "none",
            "includeContentMetadata": "true",
        })
        return data.get("value", [])

    def get_repo_item(self, repo_id: str, path: str, ref: str = "main") -> bytes:
        url = self._base(f"git/repositories/{repo_id}/items")
        return self.get_raw(url, params={
            "path": path,
            "versionDescriptor.version": ref,
            "versionDescriptor.versionType": "branch",
            "$format": "octetStream",
        })


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------


@dataclass
class ExtractedTable:
    index: int
    source: str
    headers: List[str]
    rows: List[List[str]]
    caption: str = ""

    def to_dict(self) -> Dict:
        return {"index": self.index, "source": self.source, "caption": self.caption,
                "headers": self.headers, "rows": self.rows}

    def to_markdown(self) -> str:
        if not self.headers:
            return ""
        sep = ["-" * max(len(h), 3) for h in self.headers]
        lines = [
            "| " + " | ".join(self.headers) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in self.rows:
            padded = row + [""] * (len(self.headers) - len(row))
            lines.append("| " + " | ".join(str(c) for c in padded[:len(self.headers)]) + " |")
        return "\n".join(lines)

    def to_csv_string(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(self.headers)
        writer.writerows(self.rows)
        return buf.getvalue()


def extract_tables_from_html(html: str, min_rows: int = 2) -> List[ExtractedTable]:
    soup = BeautifulSoup(html, "html.parser")
    tables: List[ExtractedTable] = []
    for idx, table_tag in enumerate(soup.find_all("table")):
        headers: List[str] = []
        rows: List[List[str]] = []
        caption = ""
        cap_tag = table_tag.find("caption")
        if cap_tag:
            caption = cap_tag.get_text(strip=True)
        thead = table_tag.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                headers = [th.get_text(separator=" ", strip=True) for th in header_row.find_all(["th", "td"])]
        for tr in table_tag.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            row = [c.get_text(separator=" ", strip=True) for c in cells]
            if headers and row == headers:
                continue
            if not headers and all(c.name == "th" for c in cells):
                headers = row
                continue
            rows.append(row)
        if not headers and rows:
            headers = [f"col_{i}" for i in range(len(rows[0]))]
        if len(rows) >= min_rows:
            tables.append(ExtractedTable(index=idx, source="html", headers=headers, rows=rows, caption=caption))
    return tables


def extract_tables_from_markdown(md: str, min_rows: int = 2) -> List[ExtractedTable]:
    tables: List[ExtractedTable] = []
    lines = md.splitlines()
    i = 0
    idx = 0
    while i < len(lines):
        if re.match(r"^\s*\|(.+\|)+", lines[i]):
            if i + 1 < len(lines) and re.match(r"^\s*\|[\s\-:|]+\|", lines[i + 1]):
                headers = [c.strip() for c in lines[i].split("|") if c.strip()]
                rows: List[List[str]] = []
                j = i + 2
                while j < len(lines) and re.match(r"^\s*\|(.+\|)+", lines[j]):
                    row = [c.strip() for c in lines[j].split("|") if c.strip()]
                    rows.append(row)
                    j += 1
                if len(rows) >= min_rows:
                    tables.append(ExtractedTable(index=idx, source="markdown", headers=headers, rows=rows))
                    idx += 1
                i = j
                continue
        i += 1
    return tables


def save_tables(tables: List[ExtractedTable], out_dir: Path) -> List[Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for t in tables:
        stem = f"table_{t.index:02d}"
        (out_dir / f"{stem}.json").write_text(json.dumps(t.to_dict(), ensure_ascii=False, indent=2))
        (out_dir / f"{stem}.csv").write_text(t.to_csv_string(), encoding="utf-8")
        meta.append({
            "index": t.index, "source": t.source, "caption": t.caption,
            "headers": t.headers, "row_count": len(t.rows),
            "json_path": f"tables/{stem}.json", "csv_path": f"tables/{stem}.csv",
            "markdown_preview": t.to_markdown()[:500],
        })
    return meta


# ---------------------------------------------------------------------------
# Multimodal image captioner
# ---------------------------------------------------------------------------


@dataclass
class ImageCaption:
    filename: str
    sha256: str
    width: Optional[int]
    height: Optional[int]
    description: str
    model: str
    structured: Optional[Dict] = None


class MultimodalCaptioner:
    def __init__(self, cfg: AzureDevOpsConfig) -> None:
        self.cfg = cfg

    def caption(self, image_bytes: bytes, filename: str) -> ImageCaption:
        sha = hashlib.sha256(image_bytes).hexdigest()
        try:
            from PIL import Image as PILImage  # type: ignore[import]
            img = PILImage.open(io.BytesIO(image_bytes))
            w, h = img.size
        except Exception:
            w, h = None, None
        description = self._call_backend(image_bytes)
        return ImageCaption(
            filename=filename, sha256=sha, width=w, height=h,
            description=description, model=self.cfg.caption_model,
            structured=self._parse_structured(description),
        )

    def _call_backend(self, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        if self.cfg.caption_backend == "ollama":
            return self._ollama(b64)
        elif self.cfg.caption_backend in ("openai_compatible", "azure_openai"):
            return self._openai_compatible(b64)
        raise ValueError(f"Unknown caption backend: {self.cfg.caption_backend!r}")

    def _ollama(self, b64: str) -> str:
        url = f"{self.cfg.caption_endpoint.rstrip('/')}/api/generate"
        payload = {
            "model": self.cfg.caption_model, "prompt": self.cfg.caption_prompt,
            "images": [b64], "stream": False,
            "options": {"num_predict": self.cfg.caption_max_tokens},
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _openai_compatible(self, b64: str) -> str:
        url = self.cfg.caption_endpoint.rstrip("/")
        if not url.endswith("/v1/chat/completions"):
            url = f"{url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.caption_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.caption_api_key}"
        payload = {
            "model": self.cfg.caption_model,
            "max_tokens": self.cfg.caption_max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self.cfg.caption_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _parse_structured(self, description: str) -> Dict:
        lower = description.lower()
        image_type = "unknown"
        for keyword, itype in [
            ("chart", "chart"), ("graph", "chart"), ("diagram", "diagram"),
            ("table", "table"), ("screenshot", "screenshot"),
            ("photo", "photo"), ("logo", "logo"), ("icon", "icon"),
            ("map", "map"), ("flow", "flowchart"), ("architecture", "architecture"),
        ]:
            if keyword in lower:
                image_type = itype
                break
        text_fragments = re.findall(r'"([^"]{3,100})"', description)
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?(?:\s*%|\s*[kKmMbB])?\b', description)
        return {"type": image_type, "text_in_image": text_fragments[:10],
                "data_values": numbers[:20], "char_count": len(description)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower()).strip()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:max_len].strip("-") or "page"


def _html_to_text(html: Optional[str]) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)


def _inject_table_refs(md: str, tables_meta: List[Dict]) -> str:
    if not tables_meta:
        return md
    lines = ["\n\n## Extracted Tables\n"]
    for t in tables_meta:
        lines.append(f"- **Table {t['index']}** ({t['row_count']} rows): [JSON]({t['json_path']}) · [CSV]({t['csv_path']})")
        if t.get("caption"):
            lines.append(f"  _Caption: {t['caption']}_")
        lines.append(f"\n{t['markdown_preview']}\n")
    return md + "\n".join(lines)


def _fix_image_links(md: str, images_dir: Path, wiki_id: str, client: "AzureDevOpsClient") -> str:
    def _replace(m: re.Match) -> str:
        alt = m.group(1)
        url_or_path = m.group(2)
        filename = url_or_path.split("/")[-1].split("?")[0]
        local_path = images_dir / filename
        if not local_path.exists():
            try:
                if url_or_path.startswith("http"):
                    data = client.session.get(url_or_path, timeout=60).content
                else:
                    data = client.get_wiki_attachment(wiki_id, filename)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(data)
            except Exception as exc:
                logger.warning("Could not download image %s: %s", filename, exc)
                return m.group(0)
        return f"![{alt}](images/{filename})"
    return re.compile(r"!\[([^\]]*)\]\(([^)]+)\)").sub(_replace, md)


def _load_sync_state(out_dir: str) -> Dict:
    p = Path(out_dir) / ".azure_sync_state.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_sync_state(out_dir: str, state: Dict) -> None:
    p = Path(out_dir) / ".azure_sync_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Wiki, WorkItem, Repo exporters
# ---------------------------------------------------------------------------

WI_FIELDS = [
    "System.Id", "System.Title", "System.WorkItemType", "System.State",
    "System.AssignedTo", "System.CreatedBy", "System.CreatedDate",
    "System.ChangedDate", "System.Tags", "System.AreaPath",
    "System.IterationPath", "System.Description",
    "Microsoft.VSTS.Common.Priority", "Microsoft.VSTS.Common.Severity",
    "Microsoft.VSTS.Common.AcceptanceCriteria",
    "Microsoft.VSTS.Scheduling.StoryPoints",
]

ALLOWED_IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}


class WikiExporter:
    def __init__(self, cfg: AzureDevOpsConfig, client: AzureDevOpsClient,
                 captioner: Optional[MultimodalCaptioner], sync_state: Dict) -> None:
        self.cfg = cfg
        self.client = client
        self.captioner = captioner
        self.sync_state = sync_state
        self.out_root = Path(cfg.output_dir) / _slugify(cfg.project) / "wikis"

    def run(self) -> Dict[str, int]:
        wikis = self.client.list_wikis()
        if self.cfg.wiki_names:
            wikis = [w for w in wikis if w.get("name") in self.cfg.wiki_names]
        logger.info("Found %d wikis", len(wikis))
        total = errors = skipped = 0
        for wiki in wikis:
            wiki_id = wiki["id"]
            wiki_name = wiki.get("name", wiki_id)
            pages = self.client.list_wiki_pages(wiki_id)
            for page_stub in pages:
                path = page_stub.get("path", "")
                try:
                    page = self.client.get_wiki_page_by_path(wiki_id, path)
                    updated = page.get("lastVersion", {}).get("pushedDate", "")
                    cache_key = f"wiki:{wiki_id}:{path}"
                    if self.cfg.incremental and self.sync_state.get(cache_key) == updated and updated:
                        skipped += 1
                        continue
                    self._save_page(wiki_id, wiki_name, page)
                    self.sync_state[cache_key] = updated
                    total += 1
                except Exception as exc:
                    logger.error("Failed wiki page %s: %s", path, exc)
                    errors += 1
        return {"saved": total, "skipped": skipped, "errors": errors}

    def _save_page(self, wiki_id: str, wiki_name: str, page: Dict) -> None:
        path = page.get("path", "untitled")
        slug = _slugify(path.replace("/", "_"))
        out_dir = self.out_root / wiki_name / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        content_md = page.get("content", "")
        images_dir = out_dir / "images"
        content_md = _fix_image_links(content_md, images_dir, wiki_id, self.client)
        tables_meta: List[Dict] = []
        if self.cfg.extract_tables:
            tables = extract_tables_from_markdown(content_md, self.cfg.table_min_rows)
            if tables:
                tables_meta = save_tables(tables, out_dir / "tables")
                content_md = _inject_table_refs(content_md, tables_meta)
        images_meta: List[Dict] = []
        if self.captioner and images_dir.exists():
            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() in ALLOWED_IMG_EXT:
                    try:
                        cap = self.captioner.caption(img_file.read_bytes(), img_file.name)
                        sidecar = images_dir / f"{img_file.stem}.caption.json"
                        sidecar.write_text(json.dumps({
                            "description": cap.description, "structured": cap.structured,
                            "model": cap.model, "sha256": cap.sha256,
                        }, ensure_ascii=False, indent=2))
                        content_md += f"\n\n**Image** ({img_file.name}): {cap.description}\n"
                        images_meta.append({
                            "filename": img_file.name, "sha256": cap.sha256,
                            "width": cap.width, "height": cap.height,
                            "description": cap.description, "model": cap.model,
                            "structured": cap.structured,
                        })
                    except Exception as exc:
                        logger.warning("Caption failed %s: %s", img_file.name, exc)
        ancestors = [p for p in path.split("/") if p]
        front_matter = {
            "source": "azure_devops_wiki", "wiki_id": wiki_id, "wiki_name": wiki_name,
            "path": path, "title": path.split("/")[-1] or "Home",
            "url": f"{self.cfg.org_url}/{quote(self.cfg.project)}/_wiki/wikis/{wiki_id}?pagePath={quote(path)}",
            "ancestors": ancestors[:-1],
            "updated_at": page.get("lastVersion", {}).get("pushedDate", ""),
            "version": page.get("lastVersion", {}).get("version", ""),
            "table_count": len(tables_meta), "image_count": len(images_meta),
        }
        final_md = f"---\n{yaml.dump(front_matter, allow_unicode=True, default_flow_style=False)}---\n\n{content_md}"
        (out_dir / "content.md").write_text(final_md, encoding="utf-8")
        (out_dir / "metadata.json").write_text(
            json.dumps({**front_matter, "tables": tables_meta, "images": images_meta}, ensure_ascii=False, indent=2)
        )


class WorkItemExporter:
    DEFAULT_WIQL = (
        "SELECT [System.Id] FROM WorkItems "
        "WHERE [System.TeamProject] = @project "
        "AND [System.WorkItemType] IN ({types}) "
        "ORDER BY [System.ChangedDate] DESC"
    )

    def __init__(self, cfg: AzureDevOpsConfig, client: AzureDevOpsClient, sync_state: Dict) -> None:
        self.cfg = cfg
        self.client = client
        self.sync_state = sync_state
        self.out_root = Path(cfg.output_dir) / _slugify(cfg.project) / "work_items"
        self.out_root.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, int]:
        wiql = self.cfg.work_item_query or self._default_wiql()
        ids = self.client.query_work_items(wiql)
        total = errors = skipped = 0
        for i in range(0, len(ids), 200):
            try:
                items = self.client.get_work_items_batch(ids[i:i + 200], WI_FIELDS)
            except Exception as exc:
                logger.error("Batch fetch failed: %s", exc)
                errors += len(ids[i:i + 200])
                continue
            for wi in items:
                wi_id = wi["id"]
                try:
                    changed = wi.get("fields", {}).get("System.ChangedDate", "")
                    cache_key = f"wi:{wi_id}"
                    if self.cfg.incremental and self.sync_state.get(cache_key) == changed and changed:
                        skipped += 1
                        continue
                    self._save_work_item(wi)
                    self.sync_state[cache_key] = changed
                    total += 1
                except Exception as exc:
                    logger.error("Failed WI #%s: %s", wi_id, exc)
                    errors += 1
        return {"saved": total, "skipped": skipped, "errors": errors}

    def _default_wiql(self) -> str:
        types_csv = ", ".join(f"'{t}'" for t in self.cfg.work_item_types)
        return self.DEFAULT_WIQL.format(types=types_csv)

    def _save_work_item(self, wi: Dict) -> None:
        f = wi.get("fields", {})
        wi_id = wi["id"]
        wi_type = f.get("System.WorkItemType", "WorkItem")
        title = f.get("System.Title", f"#{wi_id}")
        state = f.get("System.State", "")
        assigned_to = f.get("System.AssignedTo", {})
        if isinstance(assigned_to, dict):
            assigned_to = assigned_to.get("displayName", "")
        desc_html = f.get("System.Description", "")
        desc_text = _html_to_text(desc_html)
        acceptance_text = _html_to_text(f.get("Microsoft.VSTS.Common.AcceptanceCriteria", ""))
        tags = [t.strip() for t in f.get("System.Tags", "").split(";") if t.strip()]
        comments_text = ""
        try:
            comments = self.client.get_work_item_comments(wi_id)
            parts = []
            for c in comments[:20]:
                author = c.get("createdBy", {}).get("displayName", "")
                date = c.get("createdDate", "")[:10]
                text = _html_to_text(c.get("text", ""))
                if text:
                    parts.append(f"**{author}** ({date}):\n{text}")
            comments_text = "\n\n".join(parts)
        except Exception:
            pass
        tables_meta: List[Dict] = []
        if self.cfg.extract_tables and desc_html:
            tables = extract_tables_from_html(desc_html, self.cfg.table_min_rows)
            if tables:
                tables_meta = save_tables(tables, self.out_root / f"{wi_id}_tables")
        created_by = f.get("System.CreatedBy", {})
        if isinstance(created_by, dict):
            created_by = created_by.get("displayName", "")
        front_matter = {
            "source": "azure_devops_work_item", "id": wi_id, "type": wi_type,
            "title": title, "state": state, "priority": f.get("Microsoft.VSTS.Common.Priority"),
            "severity": f.get("Microsoft.VSTS.Common.Severity"),
            "story_points": f.get("Microsoft.VSTS.Scheduling.StoryPoints"),
            "assigned_to": assigned_to, "created_by": created_by,
            "created_at": f.get("System.CreatedDate", ""),
            "updated_at": f.get("System.ChangedDate", ""),
            "area_path": f.get("System.AreaPath", ""),
            "iteration_path": f.get("System.IterationPath", ""),
            "tags": tags,
            "url": f"{self.cfg.org_url}/{quote(self.cfg.project)}/_workitems/edit/{wi_id}",
            "table_count": len(tables_meta),
        }
        sections = [
            f"---\n{yaml.dump(front_matter, allow_unicode=True, default_flow_style=False)}---",
            f"# [{wi_type} #{wi_id}] {title}",
            f"**State:** {state}  |  **Assigned:** {assigned_to}",
        ]
        if desc_text:
            sections.append(f"## Description\n\n{desc_text}")
        if acceptance_text:
            sections.append(f"## Acceptance Criteria\n\n{acceptance_text}")
        if tables_meta:
            sections.append("## Tables\n\n" + "\n\n".join(
                f"**Table {t['index']}** ({t['row_count']} rows):\n{t['markdown_preview']}"
                for t in tables_meta))
        if comments_text:
            sections.append(f"## Comments\n\n{comments_text}")
        out_path = self.out_root / f"{wi_id}.md"
        out_path.write_text("\n\n".join(sections), encoding="utf-8")


class RepoMarkdownExporter:
    def __init__(self, cfg: AzureDevOpsConfig, client: AzureDevOpsClient,
                 captioner: Optional[MultimodalCaptioner], sync_state: Dict) -> None:
        self.cfg = cfg
        self.client = client
        self.captioner = captioner
        self.sync_state = sync_state
        self.out_root = Path(cfg.output_dir) / _slugify(cfg.project) / "repos"

    def run(self) -> Dict[str, int]:
        repos = self.client.list_repos()
        if self.cfg.repo_names:
            repos = [r for r in repos if r.get("name") in self.cfg.repo_names]
        total = errors = 0
        for repo in repos:
            repo_id, repo_name = repo["id"], repo["name"]
            try:
                items = self.client.list_repo_items(repo_id)
            except Exception as exc:
                logger.error("Could not list %s: %s", repo_name, exc)
                continue
            for item in items:
                if item.get("gitObjectType") == "blob" and item.get("path", "").endswith(".md"):
                    path = item.get("path", "")
                    try:
                        data = self.client.get_repo_item(repo_id, path)
                        content = data.decode("utf-8", errors="replace")
                        self._save_md(repo_name, repo_id, path, content)
                        total += 1
                    except Exception as exc:
                        logger.error("Failed %s/%s: %s", repo_name, path, exc)
                        errors += 1
        return {"saved": total, "errors": errors}

    def _save_md(self, repo_name: str, repo_id: str, path: str, content: str) -> None:
        slug = _slugify(path.replace("/", "_").replace(".md", ""))
        out_dir = self.out_root / repo_name / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        tables_meta: List[Dict] = []
        if self.cfg.extract_tables:
            tables = extract_tables_from_markdown(content, self.cfg.table_min_rows)
            if tables:
                tables_meta = save_tables(tables, out_dir / "tables")
                content = _inject_table_refs(content, tables_meta)
        front_matter = {
            "source": "azure_devops_repo", "repo_name": repo_name, "repo_id": repo_id,
            "path": path,
            "url": f"{self.cfg.org_url}/{quote(self.cfg.project)}/_git/{repo_name}?path={quote(path)}",
            "table_count": len(tables_meta),
        }
        final = f"---\n{yaml.dump(front_matter, allow_unicode=True, default_flow_style=False)}---\n\n{content}"
        (out_dir / "content.md").write_text(final, encoding="utf-8")
        (out_dir / "metadata.json").write_text(
            json.dumps({**front_matter, "tables": tables_meta}, ensure_ascii=False, indent=2)
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_azure_export(cfg: AzureDevOpsConfig) -> Dict[str, Any]:
    sync_state = _load_sync_state(cfg.output_dir)
    client = AzureDevOpsClient(cfg)
    captioner = MultimodalCaptioner(cfg) if cfg.image_captioning else None
    results: Dict[str, Any] = {}
    if cfg.import_wikis:
        results["wikis"] = WikiExporter(cfg, client, captioner, sync_state).run()
    if cfg.import_work_items:
        results["work_items"] = WorkItemExporter(cfg, client, sync_state).run()
    if cfg.import_repos:
        results["repos"] = RepoMarkdownExporter(cfg, client, captioner, sync_state).run()
    _save_sync_state(cfg.output_dir, sync_state)
    return results
