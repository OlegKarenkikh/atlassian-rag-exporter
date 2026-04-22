"""
auth_providers.py — Extended authentication providers for Atlassian RAG exporter.

Supported auth types:
  token      : email + API token (Cloud Basic Auth)
  pat        : Personal Access Token (Server/DC Bearer)
  oauth2     : OAuth 2.0 Bearer (pre-obtained, with auto-refresh)
  basic      : Local username + password (Server/DC)
  sso_cookie : Pre-obtained SSO session cookies (JSESSIONID, crowd.token_key, etc.)
  openid     : OpenID Connect Authorization Code + PKCE (interactive browser flow)
  sso_openid : Corporate SSO via OIDC client_credentials (non-interactive, service account)
  kerberos   : Kerberos/NTLM (requires requests-kerberos)
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from typing import Dict, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger("atlassian_rag_exporter.auth")


class AuthProvider:
    """Base class: apply authentication to a requests.Session."""

    name: str = "base"

    def apply(self, session: requests.Session, base_url: str) -> None:
        raise NotImplementedError

    def refresh(self, session: requests.Session, base_url: str) -> bool:
        return False


# ---------------------------------------------------------------------------
class TokenAuth(AuthProvider):
    name = "token"

    def __init__(self, email: str, token: str) -> None:
        self.email = email
        self.token = token

    def apply(self, session: requests.Session, base_url: str) -> None:
        session.auth = (self.email, self.token)


# ---------------------------------------------------------------------------
class PATAuth(AuthProvider):
    name = "pat"

    def __init__(self, token: str) -> None:
        self.token = token

    def apply(self, session: requests.Session, base_url: str) -> None:
        session.headers["Authorization"] = f"Bearer {self.token}"


# ---------------------------------------------------------------------------
class BasicAuth(AuthProvider):
    """Local username + password for Confluence/Jira Server/DC."""

    name = "basic"

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password

    def apply(self, session: requests.Session, base_url: str) -> None:
        session.auth = (self.username, self.password)


# ---------------------------------------------------------------------------
class SSOCookieAuth(AuthProvider):
    """
    Pre-obtained SSO session cookies.
    Common names: JSESSIONID, crowd.token_key, seraph.confluence, ATL_TOKEN.
    """

    name = "sso_cookie"

    def __init__(self, cookies: Dict[str, str]) -> None:
        self.cookies = cookies

    def apply(self, session: requests.Session, base_url: str) -> None:
        for name, value in self.cookies.items():
            session.cookies.set(name, value)
        logger.info("SSO cookies applied: %s", list(self.cookies.keys()))


# ---------------------------------------------------------------------------
class OAuth2Auth(AuthProvider):
    """OAuth2 Bearer with optional auto-refresh via refresh_token."""

    name = "oauth2"

    def __init__(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        token_endpoint: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        expires_at: Optional[float] = None,
    ) -> None:
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_endpoint = token_endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.expires_at = expires_at

    def apply(self, session: requests.Session, base_url: str) -> None:
        session.headers["Authorization"] = f"Bearer {self.access_token}"

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at - 60

    def refresh(self, session: requests.Session, base_url: str) -> bool:
        if not (self.refresh_token and self.token_endpoint and self.client_id):
            return False
        resp = requests.post(
            self.token_endpoint,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret or "",
            },
            timeout=30,
        )
        if resp.ok:
            data = resp.json()
            self.access_token = data["access_token"]
            self.refresh_token = data.get("refresh_token", self.refresh_token)
            self.expires_at = time.time() + data.get("expires_in", 3600)
            session.headers["Authorization"] = f"Bearer {self.access_token}"
            logger.info("OAuth2 token refreshed")
            return True
        logger.warning("OAuth2 refresh failed: %s", resp.text[:200])
        return False


# ---------------------------------------------------------------------------
class OpenIDAuth(AuthProvider):
    """
    OIDC Authorization Code + PKCE flow (interactive — opens browser).
    Starts a local HTTP server on redirect_port to capture the auth code.
    """

    name = "openid"

    def __init__(
        self,
        issuer_url: str,
        client_id: str,
        client_secret: Optional[str] = None,
        scope: str = "openid profile email",
        redirect_port: int = 9988,
    ) -> None:
        self.issuer_url = issuer_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.redirect_port = redirect_port
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._expires_at: Optional[float] = None

    def _discover(self) -> Dict:
        resp = requests.get(f"{self.issuer_url}/.well-known/openid-configuration", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _run_local_server(self, auth_url: str) -> str:
        import socket
        import webbrowser

        code_holder: Dict = {}

        class _Handler:
            def __init__(self, conn: socket.socket) -> None:
                data = conn.recv(4096).decode(errors="replace")
                first_line = data.split("\r\n")[0]
                path = first_line.split(" ")[1] if " " in first_line else "/"
                from urllib.parse import parse_qs
                from urllib.parse import urlparse as _up

                qs = parse_qs(_up(path).query)
                if "code" in qs:
                    code_holder["code"] = qs["code"][0]
                body = b"<html><body><h2>Auth successful. Close this window.</h2></body></html>"
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                conn.close()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.redirect_port))
        srv.listen(1)
        srv.settimeout(120)
        logger.info("Opening browser: %s", auth_url)
        webbrowser.open(auth_url)
        conn, _ = srv.accept()
        _Handler(conn)
        srv.close()
        if "code" not in code_holder:
            raise RuntimeError("OIDC: no authorization code received")
        return code_holder["code"]

    def apply(self, session: requests.Session, base_url: str) -> None:
        meta = self._discover()
        redirect_uri = f"http://127.0.0.1:{self.redirect_port}/callback"
        state = secrets.token_urlsafe(16)
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": self.scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        auth_url = meta["authorization_endpoint"] + "?" + urlencode(params)
        code = self._run_local_server(auth_url)
        token_data: Dict = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
        }
        if self.client_secret:
            token_data["client_secret"] = self.client_secret
        resp = requests.post(meta["token_endpoint"], data=token_data, timeout=30)
        resp.raise_for_status()
        tokens = resp.json()
        self._access_token = tokens["access_token"]
        self._refresh_token = tokens.get("refresh_token")
        self._expires_at = time.time() + tokens.get("expires_in", 3600)
        session.headers["Authorization"] = f"Bearer {self._access_token}"
        logger.info("OIDC login successful")

    @property
    def is_expired(self) -> bool:
        if self._expires_at is None:
            return False
        return time.time() >= self._expires_at - 60

    def refresh(self, session: requests.Session, base_url: str) -> bool:
        if not self._refresh_token:
            return False
        meta = self._discover()
        resp = requests.post(
            meta["token_endpoint"],
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret or "",
            },
            timeout=30,
        )
        if resp.ok:
            tokens = resp.json()
            self._access_token = tokens["access_token"]
            self._refresh_token = tokens.get("refresh_token", self._refresh_token)
            self._expires_at = time.time() + tokens.get("expires_in", 3600)
            session.headers["Authorization"] = f"Bearer {self._access_token}"
            return True
        return False


# ---------------------------------------------------------------------------
class SSOOpenIDAuth(AuthProvider):
    """
    OIDC client_credentials flow (non-interactive, service account).
    Compatible with Okta, Azure AD, Keycloak, Ping, etc.
    """

    name = "sso_openid"

    def __init__(
        self,
        token_endpoint: str,
        client_id: str,
        client_secret: str,
        scope: str = "openid",
        audience: Optional[str] = None,
    ) -> None:
        self.token_endpoint = token_endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.audience = audience
        self._access_token: Optional[str] = None
        self._expires_at: Optional[float] = None

    def _fetch_token(self) -> str:
        payload: Dict = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.scope,
        }
        if self.audience:
            payload["audience"] = self.audience
        resp = requests.post(self.token_endpoint, data=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        self._expires_at = time.time() + data.get("expires_in", 3600)
        return data["access_token"]

    def apply(self, session: requests.Session, base_url: str) -> None:
        self._access_token = self._fetch_token()
        session.headers["Authorization"] = f"Bearer {self._access_token}"
        logger.info("SSO OIDC token obtained via client_credentials")

    @property
    def is_expired(self) -> bool:
        if self._expires_at is None:
            return False
        return time.time() >= self._expires_at - 60

    def refresh(self, session: requests.Session, base_url: str) -> bool:
        try:
            self._access_token = self._fetch_token()
            session.headers["Authorization"] = f"Bearer {self._access_token}"
            return True
        except Exception as exc:
            logger.warning("SSO token refresh failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
class KerberosAuth(AuthProvider):
    """Kerberos/NTLM for Windows domain environments."""

    name = "kerberos"

    def __init__(self, mutual_authentication: str = "REQUIRED") -> None:
        self.mutual_authentication = mutual_authentication

    def apply(self, session: requests.Session, base_url: str) -> None:
        try:
            from requests_kerberos import HTTPKerberosAuth  # type: ignore[import]

            session.auth = HTTPKerberosAuth(mutual_authentication=self.mutual_authentication)
        except ImportError:
            raise ImportError(
                "requests-kerberos is required for Kerberos auth. "
                "Install: pip install requests-kerberos"
            )


# ---------------------------------------------------------------------------
AUTH_REGISTRY: Dict[str, type] = {
    "token": TokenAuth,
    "pat": PATAuth,
    "oauth2": OAuth2Auth,
    "basic": BasicAuth,
    "sso_cookie": SSOCookieAuth,
    "openid": OpenIDAuth,
    "sso_openid": SSOOpenIDAuth,
    "kerberos": KerberosAuth,
}


def build_auth_provider(auth_cfg: Dict) -> AuthProvider:
    """Instantiate an AuthProvider from a config dict (type + kwargs)."""
    auth_type = auth_cfg.get("type", "token")
    cls = AUTH_REGISTRY.get(auth_type)
    if cls is None:
        raise ValueError(f"Unknown auth type: {auth_type!r}. Supported: {', '.join(AUTH_REGISTRY)}")
    cfg = {k: v for k, v in auth_cfg.items() if k != "type"}
    return cls(**cfg)  # type: ignore[arg-type]
