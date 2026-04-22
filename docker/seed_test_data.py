#!/usr/bin/env python3
"""
seed_test_data.py
=================
Seeds a local Confluence instance (from docker-compose) with test spaces,
pages, and image attachments for validating the RAG exporter.

Usage:
  python seed_test_data.py --base-url http://localhost:8090 \\
                           --email admin@example.com \\
                           --token YOUR_API_TOKEN
"""
import argparse
import io
import json
import struct
import zlib
import requests
from pathlib import Path


def make_minimal_png(width: int = 100, height: int = 100, color=(255, 0, 0)) -> bytes:
    """Generate a minimal valid PNG in memory (no Pillow dependency)."""
    def png_chunk(name: bytes, data: bytes) -> bytes:
        chunk = name + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

    raw = b""
    for _ in range(height):
        raw += b"\x00"  # filter byte
        for _ in range(width):
            raw += bytes(color) + b"\xff"  # RGBA

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw)

    sig = b"\x89PNG\r\n\x1a\n"
    return sig + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", idat) + png_chunk(b"IEND", b"")


class ConfluenceSeed:
    def __init__(self, base_url: str, email: str, token: str):
        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        self.s.auth = (email, token)
        self.s.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    def create_space(self, key: str, name: str) -> dict:
        r = self.s.post(f"{self.base}/wiki/rest/api/space",
                        json={"key": key, "name": name,
                              "description": {"plain": {"value": f"Test space {name}", "representation": "plain"}}})
        if r.status_code == 200:
            print(f"  Space exists: {key}")
            return r.json()
        r.raise_for_status()
        data = r.json()
        print(f"  Created space: {key} ({data.get('id')})")
        return data

    def create_page(self, space_key: str, title: str, body: str, parent_id: str = None) -> str:
        payload = {
            "type": "page",
            "title": title,
            "space": {"key": space_key},
            "body": {"storage": {"value": body, "representation": "storage"}},
        }
        if parent_id:
            payload["ancestors"] = [{"id": parent_id}]
        r = self.s.post(f"{self.base}/wiki/rest/api/content", json=payload)
        r.raise_for_status()
        page_id = r.json()["id"]
        print(f"    Created page: '{title}' (id={page_id})")
        return page_id

    def attach_image(self, page_id: str, filename: str, png_bytes: bytes):
        url = f"{self.base}/wiki/rest/api/content/{page_id}/child/attachment"
        headers = {"X-Atlassian-Token": "no-check"}
        files = {"file": (filename, io.BytesIO(png_bytes), "image/png")}
        r = self.s.post(url, files=files, headers=headers,
                        headers={**self.s.headers, "X-Atlassian-Token": "no-check",
                                 "Content-Type": None})
        # Re-do without json Content-Type
        sess = requests.Session()
        sess.auth = self.s.auth
        r = sess.post(url, files={"file": (filename, io.BytesIO(png_bytes), "image/png")},
                      headers={"X-Atlassian-Token": "no-check", "Accept": "application/json"})
        if r.ok:
            print(f"      Attached image: {filename} to page {page_id}")
        else:
            print(f"      WARN: attach failed ({r.status_code}): {r.text[:200]}")

    def seed(self):
        print("\n== Seeding test data ==")

        # Space 1: Engineering docs
        sp = self.create_space("TEST", "RAG Test Space")

        # Root page
        root_id = self.create_page("TEST", "RAG Test Home",
            "<h1>RAG Test Home</h1><p>This space is used for testing the Atlassian RAG Exporter.</p>")

        # Page with text + table
        p1 = self.create_page("TEST", "API Design Guidelines",
            """<h2>Overview</h2>
            <p>All REST APIs <strong>must</strong> follow these guidelines.</p>
            <table><tbody>
              <tr><th>Method</th><th>Use case</th></tr>
              <tr><td>GET</td><td>Read-only retrieval</td></tr>
              <tr><td>POST</td><td>Create resource</td></tr>
              <tr><td>PUT</td><td>Full update</td></tr>
              <tr><td>PATCH</td><td>Partial update</td></tr>
              <tr><td>DELETE</td><td>Remove resource</td></tr>
            </tbody></table>""", parent_id=root_id)

        # Attach a red test PNG to this page
        self.attach_image(p1, "api-diagram.png", make_minimal_png(200, 200, (200, 50, 50)))

        # Page with code blocks
        p2 = self.create_page("TEST", "Authentication Flow",
            """<h2>OAuth 2.0 Flow</h2>
            <p>The service uses OAuth 2.0 with PKCE.</p>
            <ac:structured-macro ac:name='code'>
              <ac:parameter ac:name='language'>python</ac:parameter>
              <ac:plain-text-body><![CDATA[
import requests
response = requests.get('/api/v1/me', headers={'Authorization': 'Bearer TOKEN'})
]]></ac:plain-text-body>
            </ac:structured-macro>""", parent_id=root_id)

        self.attach_image(p2, "oauth-sequence.png", make_minimal_png(300, 200, (50, 100, 200)))
        self.attach_image(p2, "token-flow.png", make_minimal_png(300, 300, (50, 200, 100)))

        # Page with multiple images inline
        p3 = self.create_page("TEST", "System Architecture",
            """<h2>Architecture Overview</h2>
            <p>The system consists of three main components:</p>
            <ul>
              <li><strong>API Gateway</strong> — handles routing and auth</li>
              <li><strong>Core Service</strong> — business logic</li>
              <li><strong>Data Layer</strong> — PostgreSQL + Redis</li>
            </ul>
            <ac:image><ri:attachment ri:filename='arch-overview.png'/></ac:image>
            <p>For detailed component diagram, see below.</p>
            <ac:image><ri:attachment ri:filename='component-diagram.png'/></ac:image>""",
            parent_id=root_id)

        self.attach_image(p3, "arch-overview.png", make_minimal_png(400, 300, (150, 150, 50)))
        self.attach_image(p3, "component-diagram.png", make_minimal_png(400, 400, (50, 150, 150)))

        print("\n== Seed complete ==")
        print(f"   Space: TEST | Root page id: {root_id}")
        print(f"   Pages created: 3 (with {2+2+2} image attachments total)")
        print(f"\n   Now run the exporter:")
        print(f"   python atlassian_rag_exporter.py --config config.yaml")


def main():
    p = argparse.ArgumentParser(description="Seed local Confluence with test data for RAG exporter.")
    p.add_argument("--base-url", default="http://localhost:8090")
    p.add_argument("--email", required=True)
    p.add_argument("--token", required=True)
    args = p.parse_args()
    ConfluenceSeed(args.base_url, args.email, args.token).seed()


if __name__ == "__main__":
    main()
