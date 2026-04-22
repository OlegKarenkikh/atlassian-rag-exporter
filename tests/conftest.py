"""conftest.py — Shared fixtures and factories for the test suite."""
from __future__ import annotations

import io
import json
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import atlassian_rag_exporter as m


# ---------------------------------------------------------------------------
# Binary factories
# ---------------------------------------------------------------------------


def make_png(width: int = 4, height: int = 4, color: tuple = (255, 0, 0)) -> bytes:
    """Create a minimal valid PNG file in memory."""
    def chunk(name: bytes, data: bytes) -> bytes:
        c = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)

    r, g, b = color
    raw = b""
    for _ in range(height):
        raw += b"\x00" + bytes([r, g, b] * width)
    compressed = zlib.compress(raw)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    return png


def make_pdf(text: str = "Test PDF") -> bytes:
    """Create a minimal valid PDF byte string."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n9\n%%EOF\n"
    )


def make_docx() -> bytes:
    """Return a minimal DOCX (zip) byte string."""
    buf = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml",
                    '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_config(tmp_path):
    return {
        "base_url": "https://test.atlassian.net",
        "is_cloud": True,
        "auth": {"type": "token", "email": "test@example.com", "token": "test-token"},
        "output_dir": str(tmp_path / "rag_corpus"),
        "spaces": ["TEST"],
        "incremental": False,
    }


@pytest.fixture()
def tmp_out(tmp_path):
    d = tmp_path / "rag_corpus"
    d.mkdir()
    return d


@pytest.fixture()
def exporter(base_config, tmp_out):
    base_config["output_dir"] = str(tmp_out)
    exp = m.RAGExporter.__new__(m.RAGExporter)
    exp.config = base_config
    exp.output_dir = tmp_out
    exp.attachments_dir = tmp_out / "attachments"
    exp.pages_dir = tmp_out / "pages"
    exp.jira_dir = tmp_out / "jira"
    exp._manifest = []
    exp._state = {}
    exp._state_file = tmp_out / ".sync_state.json"
    exp._result = m.ExportResult()
    for d in [exp.attachments_dir, exp.pages_dir, exp.jira_dir]:
        d.mkdir(parents=True, exist_ok=True)

    session_mock = MagicMock(spec=m.AtlassianSession)
    session_mock.base_url = "https://test.atlassian.net"
    confluence_mock = MagicMock(spec=m.ConfluenceClient)
    confluence_mock.session = session_mock
    exp.confluence = confluence_mock
    exp.jira = None
    return exp
