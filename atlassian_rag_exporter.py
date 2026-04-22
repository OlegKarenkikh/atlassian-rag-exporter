#!/usr/bin/env python3
"""
atlassian_rag_exporter.py
=========================
Production-ready exporter for Atlassian Confluence → structured RAG corpus.

Supports:
  - Confluence Cloud & Server via REST API v2 (Cloud) / v1 (Server fallback)
  - Authentication: API Token (Cloud), Personal Access Token (Server), OAuth2 Bearer
  - Full page content (storage + view format → Markdown)
  - All attachments (images, PDFs, Office docs) saved to disk & indexed
  - Inline image references resolved to local paths in Markdown
  - Jira issues export (optional)
  - Structured JSON metadata per document (for vector DB ingestion)
  - Incremental sync via last-modified tracking
  - Rate-limit-aware retry with exponential backoff
  - YAML-front-matter output compatible with LlamaIndex / LangChain loaders

Usage:
  python atlassian_rag_exporter.py --config config.yaml

Requirements:
  pip install requests markdownify beautifulsoup4 PyYAML tenacity tqdm
"""

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("atlassian_rag_exporter")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AttachmentRecord:
    filename: str
    local_path: str
    checksum: str
    media_type: str
    mime_type: str
    size_bytes: int
    created: str

    @property
    def is_image(self) -> bool:
        return self.media_type == "image"

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "local_path": self.local_path,
            "checksum": self.checksum,
            "media_type": self.media_type,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "created": self.created,
        }

    def __getitem__(self, key: str):
        return self.to_dict()[key]

    def __repr__(self) -> str:
        return f"AttachmentRecord(filename={self.filename!r}, media_type={self.media_type!r})"


@dataclass
class ExportResult:
    confluence_pages: int = 0
    jira_issues: int = 0
    errors: int = 0

    @property
    def total_documents(self) -> int:
        return self.confluence_pages + self.jira_issues

    def to_dict(self) -> Dict:
        return {
            "confluence_pages": self.confluence_pages,
            "jira_issues": self.jira_issues,
            "errors": self.errors,
            "total_documents": self.total_documents,
        }

    def __repr__(self) -> str:
        return (
            f"ExportResult(confluence_pages={self.confluence_pages}, "
            f"jira_issues={self.jira_issues}, errors={self.errors})"
        )


# ---------------------------------------------------------------------------
# Custom Markdown Converter  (preserves local image references)
# ---------------------------------------------------------------------------
class ConfluenceMarkdownConverter(MarkdownConverter):
    """Extends markdownify to handle Confluence-specific HTML and local images."""

    def __init__(self, attachment_map: Dict[str, str] = None, **kwargs):
        super().__init__(**kwargs)
        self.attachment_map = attachment_map or {}  # filename → local_rel_path

    def convert_img(self, el, text, parent_tags):
        """Replace remote/internal image src with local path when available."""
        src = el.get("src", "")
        alt = el.get("alt", "image")
        filename = el.get("data-linked-resource-default-alias") or os.path.basename(src)
        if filename in self.attachment_map:
            local = self.attachment_map[filename]
            return f"![ {alt}]({local})\n\n"
        return super().convert_img(el, text, parent_tags) + "\n\n"

    def convert_ac_image(self, el, text, parent_tags):
        """Handle <ac:image> Confluence storage format tags."""
        ri_att = el.find("ri:attachment")
        if ri_att:
            fname = ri_att.get("ri:filename", "unknown")
            local = self.attachment_map.get(fname, f"attachments/{fname}")
            return f"![{fname}]({local})\n\n"
        ri_url = el.find("ri:url")
        if ri_url:
            url = ri_url.get("ri:value", "")
            return f"![image]({url})\n\n"
        return ""


# ---------------------------------------------------------------------------
# HTTP Session with retry + rate-limit handling
# ---------------------------------------------------------------------------
class AtlassianSession:
    """Authenticated requests session with retry and rate-limit handling."""

    RETRY_STATUSES = {429, 500, 502, 503, 504}

    def __init__(self, base_url: str, auth_type: str, **auth_kwargs):
        if not base_url:
            raise ValueError("base_url must not be empty")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        if auth_type == "token":
            email = auth_kwargs["email"]
            token = auth_kwargs["token"]
            self.session.auth = (email, token)
        elif auth_type == "pat":
            self.session.headers["Authorization"] = f"Bearer {auth_kwargs['token']}"
        elif auth_type == "oauth2":
            self.session.headers["Authorization"] = f"Bearer {auth_kwargs['access_token']}"
        else:
            raise ValueError(f"Unknown auth_type: {auth_type}")

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def get(self, url: str, **kwargs) -> requests.Response:
        if not url.startswith("http"):
            url = self.base_url + url
        while True:
            resp = self.session.get(url, timeout=30, **kwargs)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                logger.warning("Rate limited. Sleeping %ds", retry_after)
                time.sleep(retry_after)
                continue
            if resp.status_code in self.RETRY_STATUSES:
                resp.raise_for_status()
            return resp

    def get_json(self, url: str, **kwargs) -> Any:
        resp = self.get(url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def get_binary(self, url: str) -> bytes:
        resp = self.get(url)
        resp.raise_for_status()
        return resp.content


# ---------------------------------------------------------------------------
# Confluence API v2 Client (Cloud) with v1 fallback
# ---------------------------------------------------------------------------
class ConfluenceClient:
    """
    Wraps Confluence REST API v2 (Cloud) for pages, spaces, and attachments.
    Falls back to v1 endpoints where v2 is not yet available.
    """

    def __init__(self, session: AtlassianSession, is_cloud: bool = True):
        self.session = session
        self.is_cloud = is_cloud
        self.api_v2 = "/wiki/api/v2" if is_cloud else "/rest/api"
        self.api_v1 = "/wiki/rest/api" if is_cloud else "/rest/api"

    def _paginate_v2(self, path: str, params: Dict = None) -> Generator:
        """Cursor-based pagination for API v2 endpoints."""
        params = params or {}
        params.setdefault("limit", 50)
        url = self.api_v2 + path
        while url:
            data = self.session.get_json(url, params=params)
            for item in data.get("results", []):
                yield item
            next_link = data.get("_links", {}).get("next")
            if next_link:
                url = self.api_v2 + next_link if not next_link.startswith("http") else next_link
                params = {}
            else:
                url = None

    def _paginate_v1(self, path: str, params: Dict = None) -> Generator:
        """Offset-based pagination for API v1 endpoints."""
        params = params or {}
        params.setdefault("limit", 50)
        params.setdefault("start", 0)
        url = self.api_v1 + path
        while True:
            data = self.session.get_json(url, params=params)
            results = data.get("results", [])
            for item in results:
                yield item
            size = data.get("size", 0)
            limit = data.get("limit", params["limit"])
            start = data.get("start", 0)
            if start + size < data.get("totalSize", 0):
                params["start"] = start + limit
            else:
                break

    def get_spaces(self, space_keys: List[str] = None) -> List[Dict]:
        spaces = []
        for sp in self._paginate_v2("/spaces"):
            if not space_keys or sp.get("key") in space_keys:
                spaces.append(sp)
        return spaces

    def get_pages_in_space(self, space_id: str, modified_since: Optional[str] = None) -> Generator:
        params = {"space-id": space_id, "status": "current", "limit": 50}
        if modified_since:
            params["lastModified"] = modified_since
        for page in self._paginate_v2("/pages", params=params):
            yield page

    def get_page_body(self, page_id: str) -> Tuple[str, str, Dict]:
        url = f"{self.api_v1}/content/{page_id}"
        params = {"expand": "body.storage,body.view,version,ancestors,metadata.labels"}
        data = self.session.get_json(url, params=params)
        storage = data.get("body", {}).get("storage", {}).get("value", "")
        view = data.get("body", {}).get("view", {}).get("value", "")
        return storage, view, data

    def get_attachments(self, page_id: str) -> List[Dict]:
        attachments = []
        try:
            for att in self._paginate_v2(f"/pages/{page_id}/attachments"):
                attachments.append(att)
        except Exception:
            for att in self._paginate_v1(f"/content/{page_id}/child/attachment"):
                attachments.append(att)
        return attachments

    def get_attachment_download_url(self, att: Dict) -> str:
        download = att.get("downloadUrl") or att.get("webuiLink")
        if download:
            return download
        links = att.get("_links", {})
        download = links.get("download", "")
        base = self.session.base_url
        return urljoin(base + "/wiki", download)

    def get_page_labels(self, page_id: str) -> List[str]:
        url = f"{self.api_v1}/content/{page_id}/label"
        data = self.session.get_json(url)
        return [lbl["name"] for lbl in data.get("results", [])]

    def get_children(self, page_id: str) -> Generator:
        for child in self._paginate_v1(f"/content/{page_id}/child/page"):
            yield child


# ---------------------------------------------------------------------------
# Jira Client (optional)
# ---------------------------------------------------------------------------
class JiraClient:
    def __init__(self, session: AtlassianSession):
        self.session = session
        self.api = "/rest/api/3"

    def search_issues(self, jql: str, fields: List[str] = None) -> Generator:
        fields_str = ",".join(fields) if fields else "*navigable"
        start = 0
        page_size = 50
        while True:
            url = f"{self.api}/search"
            params = {"jql": jql, "startAt": start, "maxResults": page_size, "fields": fields_str}
            data = self.session.get_json(url, params=params)
            issues = data.get("issues", [])
            for issue in issues:
                yield issue
            total = data.get("total", 0)
            start += len(issues)
            if start >= total or not issues:
                break


# ---------------------------------------------------------------------------
# RAG Document Builder
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_TYPES = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tiff",
}
SUPPORTED_ATTACHMENT_TYPES = SUPPORTED_IMAGE_TYPES | {
    ".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".csv", ".md",
}


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_len]


def compute_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class RAGExporter:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.attachments_dir = self.output_dir / "attachments"
        self.pages_dir = self.output_dir / "pages"
        self.jira_dir = self.output_dir / "jira"
        self._setup_dirs()
        self._load_state()
        self._manifest: List[Dict] = []
        self._result: ExportResult = ExportResult()

        auth = config["auth"]
        session = AtlassianSession(
            base_url=config["base_url"],
            auth_type=auth["type"],
            **{k: v for k, v in auth.items() if k != "type"},
        )
        is_cloud = config.get("is_cloud", True)
        self.confluence = ConfluenceClient(session, is_cloud=is_cloud)

        if config.get("jira"):
            jira_cfg = config["jira"]
            jira_session = AtlassianSession(
                base_url=jira_cfg.get("base_url", config["base_url"]),
                auth_type=auth["type"],
                **{k: v for k, v in auth.items() if k != "type"},
            )
            self.jira = JiraClient(jira_session)
        else:
            self.jira = None

    def _setup_dirs(self):
        for d in [self.output_dir, self.attachments_dir, self.pages_dir, self.jira_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_state(self):
        state_file = self.output_dir / ".sync_state.json"
        self._state: Dict = {}
        if state_file.exists():
            with open(state_file) as f:
                self._state = json.load(f)
        self._state_file = state_file

    def _save_state(self):
        self._state_file.write_text(json.dumps(self._state, indent=2))

    def _save_attachment(self, att: Dict) -> Optional[AttachmentRecord]:
        title = att.get("title") or att.get("filename") or "attachment"
        ext = Path(title).suffix.lower()
        if ext not in SUPPORTED_ATTACHMENT_TYPES:
            return None

        download_url = self.confluence.get_attachment_download_url(att)
        if not download_url:
            return None

        try:
            content = self.confluence.session.get_binary(download_url)
        except Exception as e:
            logger.error("Failed to download %s: %s", title, e)
            return None

        checksum = compute_checksum(content)
        safe_title = slugify(Path(title).stem) + ext
        att_path = self.attachments_dir / f"{checksum[:8]}_{safe_title}"

        if not att_path.exists():
            att_path.write_bytes(content)

        rel_path = str(att_path.relative_to(self.output_dir))
        is_image = ext in SUPPORTED_IMAGE_TYPES

        return AttachmentRecord(
            filename=title,
            local_path=rel_path,
            checksum=checksum,
            media_type="image" if is_image else "document",
            mime_type=att.get("mediaType") or att.get("mimeType", ""),
            size_bytes=len(content),
            created=att.get("metadata", {}).get("createdDate") or att.get("createdAt", ""),
        )

    def _html_to_markdown(self, html: str, attachment_map: Dict[str, str]) -> str:
        soup = BeautifulSoup(html, "html.parser")

        for ac_img in soup.find_all("ac:image"):
            ri_att = ac_img.find("ri:attachment")
            ri_url = ac_img.find("ri:url")
            img_tag = soup.new_tag("img")
            if ri_att:
                fname = ri_att.get("ri:filename", "")
                local = attachment_map.get(fname, f"attachments/{fname}")
                img_tag["src"] = local
                img_tag["alt"] = fname
            elif ri_url:
                img_tag["src"] = ri_url.get("ri:value", "")
                img_tag["alt"] = "image"
            ac_img.replace_with(img_tag)

        for img in soup.find_all("img"):
            src = img.get("src", "")
            fname = img.get("data-linked-resource-default-alias") or os.path.basename(
                urlparse(src).path
            )
            if fname in attachment_map:
                img["src"] = attachment_map[fname]

        for tag in soup.find_all(class_=re.compile(r"(toc|breadcrumb|footer|header|nav)")):
            tag.decompose()

        converter = ConfluenceMarkdownConverter(
            attachment_map=attachment_map,
            heading_style="ATX",
            bullets="-",
            code_language=True,
        )
        md = converter.convert_soup(soup)
        md = re.sub(r"\n{3,}", "\n\n", md).strip()
        return md

    @staticmethod
    def _yaml_front_matter(meta: Dict) -> str:
        safe = {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in meta.items()
            if isinstance(v, (str, int, float, bool, list, type(None)))
        }
        return "---\n" + yaml.dump(safe, allow_unicode=True, default_flow_style=False) + "---\n\n"

    def _write_page_document(
        self,
        page_meta: Dict,
        markdown_body: str,
        attachments_meta: List[AttachmentRecord],
    ) -> Dict:
        page_id = page_meta["page_id"]
        slug = slugify(page_meta["title"])
        page_dir = self.pages_dir / f"{page_id}_{slug}"
        page_dir.mkdir(exist_ok=True)

        front = {
            "source": "confluence",
            "page_id": page_id,
            "space_key": page_meta.get("space_key", ""),
            "space_name": page_meta.get("space_name", ""),
            "title": page_meta["title"],
            "url": page_meta.get("url", ""),
            "ancestors": page_meta.get("ancestors", []),
            "labels": page_meta.get("labels", []),
            "author": page_meta.get("author", ""),
            "created_at": page_meta.get("created_at", ""),
            "updated_at": page_meta.get("updated_at", ""),
            "version": page_meta.get("version", 1),
            "has_attachments": bool(attachments_meta),
            "attachment_count": len(attachments_meta),
            "image_count": sum(1 for a in attachments_meta if a.is_image),
        }
        md_content = self._yaml_front_matter(front) + markdown_body
        md_path = page_dir / "content.md"
        md_path.write_text(md_content, encoding="utf-8")

        doc_record = {
            **front,
            "content_markdown_path": str(md_path.relative_to(self.output_dir)),
            "attachments": [a.to_dict() for a in attachments_meta],
            "word_count": len(markdown_body.split()),
            "char_count": len(markdown_body),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        json_path = page_dir / "metadata.json"
        json_path.write_text(json.dumps(doc_record, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "type": "confluence_page",
            "page_id": page_id,
            "title": page_meta["title"],
            "md_path": str(md_path.relative_to(self.output_dir)),
            "json_path": str(json_path.relative_to(self.output_dir)),
        }

    def export_space(self, space: Dict) -> int:
        space_id = space.get("id") or space.get("key")
        space_key = space.get("key", space_id)
        space_name = space.get("name", space_key)
        logger.info("Exporting space: %s (%s)", space_name, space_key)

        modified_since = None
        if self.config.get("incremental"):
            modified_since = self._state.get(space_key)

        pages = list(
            tqdm(
                self.confluence.get_pages_in_space(space_id, modified_since),
                desc=f"Pages in {space_key}",
                unit="page",
            )
        )
        count = 0
        for page in tqdm(pages, desc="Exporting pages", unit="page"):
            try:
                self._export_page(page, space_key, space_name)
                count += 1
            except Exception as e:
                logger.error("Failed to export page %s: %s", page.get("id"), e)
                self._result.errors += 1

        self._state[space_key] = datetime.now(timezone.utc).isoformat()
        self._result.confluence_pages += count
        return count

    def _export_page(self, page: Dict, space_key: str, space_name: str):
        page_id = page.get("id")
        title = page.get("title", f"page_{page_id}")
        page_url = page.get("_links", {}).get("webui", "")
        if page_url and not page_url.startswith("http"):
            page_url = self.confluence.session.base_url + "/wiki" + page_url

        storage_body, view_body, full_data = self.confluence.get_page_body(page_id)

        version = full_data.get("version", {})
        author = (
            version.get("by", {}).get("displayName", "")
            or full_data.get("history", {}).get("createdBy", {}).get("displayName", "")
        )
        created_at = full_data.get("history", {}).get("createdDate", "")
        updated_at = version.get("when", "")
        version_num = version.get("number", 1)
        ancestors = [a.get("title", "") for a in full_data.get("ancestors", [])]
        labels = self.confluence.get_page_labels(page_id)

        raw_attachments = self.confluence.get_attachments(page_id)
        attachment_map: Dict[str, str] = {}
        attachments_meta: List[AttachmentRecord] = []

        for att in raw_attachments:
            meta = self._save_attachment(att)
            if meta:
                attachments_meta.append(meta)
                attachment_map[meta.filename] = meta.local_path
                fname = os.path.basename(meta.filename)
                attachment_map[fname] = meta.local_path

        html_source = view_body or storage_body
        markdown_body = self._html_to_markdown(html_source, attachment_map)

        images = [a for a in attachments_meta if a.is_image]
        if images:
            img_section = "\n\n## Attached Images\n\n"
            for img in images:
                img_section += f"![{img.filename}]({img.local_path})\n\n"
            markdown_body += img_section

        page_meta = {
            "page_id": page_id,
            "title": title,
            "url": page_url,
            "space_key": space_key,
            "space_name": space_name,
            "ancestors": ancestors,
            "labels": labels,
            "author": author,
            "created_at": created_at,
            "updated_at": updated_at,
            "version": version_num,
        }
        entry = self._write_page_document(page_meta, markdown_body, attachments_meta)
        self._manifest.append(entry)

    def export_jira(self, jql: str, fields: List[str] = None):
        if not self.jira:
            logger.warning("Jira not configured.")
            return
        default_fields = [
            "summary", "description", "status", "priority", "assignee",
            "reporter", "labels", "components", "created", "updated",
            "issuetype", "project", "comment",
        ]
        fields = fields or default_fields
        logger.info("Exporting Jira issues: %s", jql)
        issues = list(tqdm(self.jira.search_issues(jql, fields), desc="Jira issues", unit="issue"))
        for issue in issues:
            self._export_jira_issue(issue)
            self._result.jira_issues += 1

    def _export_jira_issue(self, issue: Dict):
        key = issue.get("key", "UNKNOWN")
        f = issue.get("fields", {})
        summary = f.get("summary", "")
        description = f.get("description") or ""

        md_lines = [
            f"# {key}: {summary}", "",
            f"**Type:** {f.get('issuetype', {}).get('name', '')}",
            f"**Status:** {f.get('status', {}).get('name', '')}",
            f"**Priority:** {(f.get('priority') or {}).get('name', '')}",
            f"**Assignee:** {(f.get('assignee') or {}).get('displayName', 'Unassigned')}",
            f"**Reporter:** {(f.get('reporter') or {}).get('displayName', '')}",
            f"**Project:** {f.get('project', {}).get('name', '')}",
            f"**Created:** {f.get('created', '')}",
            f"**Updated:** {f.get('updated', '')}",
            f"**Labels:** {', '.join(f.get('labels', []))}",
            "", "## Description", "", str(description), "",
        ]

        comments = (f.get("comment") or {}).get("comments", [])
        if comments:
            md_lines.append("## Comments")
            md_lines.append("")
            for c in comments:
                author = (c.get("author") or {}).get("displayName", "Unknown")
                created = c.get("created", "")
                body = c.get("body") or ""
                if isinstance(body, dict):
                    body = _adf_to_text(body)
                md_lines.append(f"**{author}** ({created}):")
                md_lines.append(body)
                md_lines.append("")

        markdown_body = "\n".join(md_lines)
        meta = {
            "source": "jira",
            "issue_key": key,
            "summary": summary,
            "issue_type": f.get("issuetype", {}).get("name", ""),
            "status": f.get("status", {}).get("name", ""),
            "priority": (f.get("priority") or {}).get("name", ""),
            "assignee": (f.get("assignee") or {}).get("displayName", ""),
            "reporter": (f.get("reporter") or {}).get("displayName", ""),
            "project": f.get("project", {}).get("key", ""),
            "labels": f.get("labels", []),
            "created_at": f.get("created", ""),
            "updated_at": f.get("updated", ""),
        }

        front_matter = "---\n" + yaml.dump(meta, allow_unicode=True) + "---\n\n"
        full_md = front_matter + markdown_body
        out_path = self.jira_dir / f"{key}.md"
        out_path.write_text(full_md, encoding="utf-8")
        self._manifest.append({
            "type": "jira_issue",
            "issue_key": key,
            "summary": summary,
            "md_path": str(out_path.relative_to(self.output_dir)),
        })

    def write_manifest(self):
        manifest_path = self.output_dir / "manifest.json"
        manifest_data = {
            "schema_version": "2.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_documents": len(self._manifest),
            "confluence_pages": self._result.confluence_pages,
            "jira_issues": self._result.jira_issues,
            "errors": self._result.errors,
            "documents": self._manifest,
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Manifest written: %d documents", len(self._manifest))

    def run(self):
        cfg = self.config
        space_keys = cfg.get("spaces", [])
        spaces = self.confluence.get_spaces(space_keys) if space_keys else self.confluence.get_spaces()

        total = 0
        for space in spaces:
            total += self.export_space(space)

        if cfg.get("jira") and cfg["jira"].get("jql"):
            self.export_jira(jql=cfg["jira"]["jql"], fields=cfg["jira"].get("fields"))

        self.write_manifest()
        self._save_state()
        logger.info("Export complete. Total Confluence pages: %d", total)


# ---------------------------------------------------------------------------
# ADF → plain text
# ---------------------------------------------------------------------------
def _adf_to_text(adf_node: Any, depth: int = 0) -> str:
    if isinstance(adf_node, str):
        return adf_node
    if not isinstance(adf_node, dict):
        return ""
    node_type = adf_node.get("type", "")
    text = adf_node.get("text", "")
    content = adf_node.get("content", [])
    result = text
    for child in content:
        result += _adf_to_text(child, depth + 1)
    if node_type in ("paragraph", "heading", "bulletList", "orderedList", "listItem"):
        result += "\n"
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXAMPLE_CONFIG = """
base_url: \"https://yourcompany.atlassian.net\"
is_cloud: true
auth:
  type: \"token\"
  email: \"you@company.com\"
  token: \"YOUR_API_TOKEN\"
spaces:
  - \"ENG\"
incremental: true
output_dir: \"./rag_corpus\"
"""


def build_arg_parser():
    p = argparse.ArgumentParser(description="Export Atlassian Confluence/Jira to a structured RAG corpus.")
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--version", action="version", version="atlassian-rag-exporter 2.0.0")
    p.add_argument("--print-example-config", action="store_true")
    p.add_argument("--spaces", nargs="+")
    p.add_argument("--output-dir")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.print_example_config:
        print(EXAMPLE_CONFIG)
        return 0

    if args.config is None:
        parser.print_usage()
        return 2

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Config file not found: %s", args.config)
        return 1

    if args.spaces:
        config["spaces"] = args.spaces
    if args.output_dir:
        config["output_dir"] = args.output_dir

    try:
        exporter = RAGExporter(config)
        exporter.run()
    except (ValueError, KeyError) as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
