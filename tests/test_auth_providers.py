"""Tests for auth_providers module."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from auth_providers import (
    BasicAuth,
    KerberosAuth,
    OAuth2Auth,
    PATAuth,
    SSOCookieAuth,
    SSOOpenIDAuth,
    TokenAuth,
    build_auth_provider,
)


class TestTokenAuth:
    def test_apply_sets_basic_auth(self):
        p = TokenAuth(email="u@e.com", token="tok")
        session = MagicMock()
        p.apply(session, "https://example.com")
        assert session.auth == ("u@e.com", "tok")

    def test_build_from_config(self):
        p = build_auth_provider({"type": "token", "email": "a@b.com", "token": "t"})
        assert isinstance(p, TokenAuth)


class TestPATAuth:
    def test_apply_sets_bearer(self):
        p = PATAuth(token="my-pat")
        session = requests.Session()
        p.apply(session, "https://example.com")
        assert session.headers["Authorization"] == "Bearer my-pat"

    def test_build_from_config(self):
        p = build_auth_provider({"type": "pat", "token": "p"})
        assert isinstance(p, PATAuth)


class TestBasicAuth:
    def test_apply_sets_auth_tuple(self):
        p = BasicAuth(username="admin", password="secret")
        session = MagicMock()
        p.apply(session, "https://example.com")
        assert session.auth == ("admin", "secret")

    def test_build_from_config(self):
        p = build_auth_provider({"type": "basic", "username": "u", "password": "p"})
        assert isinstance(p, BasicAuth)

    def test_full_round_trip(self):
        p = build_auth_provider({"type": "basic", "username": "root", "password": "toor"})
        session = MagicMock()
        p.apply(session, "https://x.com")
        assert session.auth == ("root", "toor")


class TestSSOCookieAuth:
    def test_apply_sets_cookies(self):
        p = SSOCookieAuth(cookies={"JSESSIONID": "abc123", "crowd.token_key": "xyz"})
        session = requests.Session()
        p.apply(session, "https://example.com")
        assert session.cookies.get("JSESSIONID") == "abc123"
        assert session.cookies.get("crowd.token_key") == "xyz"

    def test_build_from_config(self):
        p = build_auth_provider({"type": "sso_cookie", "cookies": {"K": "V"}})
        assert isinstance(p, SSOCookieAuth)

    def test_multiple_cookies(self):
        cookies = {"A": "1", "B": "2", "C": "3"}
        p = SSOCookieAuth(cookies=cookies)
        session = requests.Session()
        p.apply(session, "https://x.com")
        for k, v in cookies.items():
            assert session.cookies.get(k) == v


class TestOAuth2Auth:
    def test_apply_sets_bearer(self):
        p = OAuth2Auth(access_token="access-tok")
        session = requests.Session()
        p.apply(session, "https://example.com")
        assert session.headers["Authorization"] == "Bearer access-tok"

    def test_is_expired_false_when_no_expiry(self):
        p = OAuth2Auth(access_token="t")
        assert p.is_expired is False

    def test_is_expired_true_when_past(self):
        p = OAuth2Auth(access_token="t", expires_at=time.time() - 100)
        assert p.is_expired is True

    def test_is_expired_false_when_future(self):
        p = OAuth2Auth(access_token="t", expires_at=time.time() + 3600)
        assert p.is_expired is False

    def test_refresh_updates_token(self):
        p = OAuth2Auth(
            access_token="old",
            refresh_token="ref",
            token_endpoint="https://auth.example.com/token",
            client_id="cid",
        )
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"access_token": "new", "expires_in": 3600}
        session = requests.Session()
        with patch("requests.post", return_value=mock_resp):
            result = p.refresh(session, "https://example.com")
        assert result is True
        assert p.access_token == "new"
        assert session.headers["Authorization"] == "Bearer new"

    def test_refresh_returns_false_without_refresh_token(self):
        p = OAuth2Auth(access_token="t")
        assert p.refresh(MagicMock(), "https://example.com") is False

    def test_refresh_failed_response(self):
        p = OAuth2Auth(
            access_token="t",
            refresh_token="r",
            token_endpoint="https://t",
            client_id="c",
        )
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.text = "error"
        session = requests.Session()
        with patch("requests.post", return_value=mock_resp):
            result = p.refresh(session, "https://x.com")
        assert result is False

    def test_build_from_config(self):
        p = build_auth_provider({"type": "oauth2", "access_token": "tok"})
        assert isinstance(p, OAuth2Auth)


class TestSSOOpenIDAuth:
    def test_apply_fetches_token(self):
        p = SSOOpenIDAuth(
            token_endpoint="https://sso.example.com/token",
            client_id="cid",
            client_secret="secret",
        )
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"access_token": "cc-token", "expires_in": 300}
        mock_resp.raise_for_status = MagicMock()
        session = requests.Session()
        with patch("requests.post", return_value=mock_resp):
            p.apply(session, "https://example.com")
        assert session.headers["Authorization"] == "Bearer cc-token"

    def test_is_expired(self):
        p = SSOOpenIDAuth(token_endpoint="https://t", client_id="c", client_secret="s")
        p._expires_at = time.time() - 10
        assert p.is_expired is True

    def test_is_not_expired(self):
        p = SSOOpenIDAuth(token_endpoint="https://t", client_id="c", client_secret="s")
        p._expires_at = time.time() + 3600
        assert p.is_expired is False

    def test_is_expired_none(self):
        p = SSOOpenIDAuth(token_endpoint="https://t", client_id="c", client_secret="s")
        assert p.is_expired is False

    def test_refresh_fetches_new_token(self):
        p = SSOOpenIDAuth(
            token_endpoint="https://sso.example.com/token",
            client_id="cid",
            client_secret="secret",
        )
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"access_token": "refreshed", "expires_in": 3600}
        mock_resp.raise_for_status = MagicMock()
        session = requests.Session()
        with patch("requests.post", return_value=mock_resp):
            result = p.refresh(session, "https://example.com")
        assert result is True
        assert session.headers["Authorization"] == "Bearer refreshed"

    def test_refresh_exception_returns_false(self):
        p = SSOOpenIDAuth(token_endpoint="https://t", client_id="c", client_secret="s")
        session = requests.Session()
        with patch("requests.post", side_effect=Exception("network error")):
            result = p.refresh(session, "https://x.com")
        assert result is False

    def test_build_from_config(self):
        p = build_auth_provider(
            {
                "type": "sso_openid",
                "token_endpoint": "https://t",
                "client_id": "c",
                "client_secret": "s",
            }
        )
        assert isinstance(p, SSOOpenIDAuth)


class TestKerberosAuth:
    def test_raises_import_error_without_package(self):
        p = KerberosAuth()
        session = MagicMock()
        with patch.dict("sys.modules", {"requests_kerberos": None}):
            with pytest.raises(ImportError, match="requests-kerberos"):
                p.apply(session, "https://example.com")

    def test_build_from_config(self):
        p = build_auth_provider({"type": "kerberos"})
        assert isinstance(p, KerberosAuth)


class TestBuildAuthProvider:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown auth type"):
            build_auth_provider({"type": "magic_unicorn"})

    def test_defaults_to_token(self):
        p = build_auth_provider({"type": "token", "email": "e@e.com", "token": "t"})
        assert isinstance(p, TokenAuth)

    def test_base_refresh_returns_false(self):
        from auth_providers import AuthProvider

        class _Dummy(AuthProvider):
            def apply(self, s, u):
                pass

        d = _Dummy()
        assert d.refresh(MagicMock(), "https://x") is False


# ---------------------------------------------------------------------------
# OpenIDAuth — unit tests (mocked network + no real browser)
# ---------------------------------------------------------------------------
class TestOpenIDAuth:
    def _provider(self):
        from auth_providers import OpenIDAuth

        return OpenIDAuth(
            issuer_url="https://sso.example.com/realms/co",
            client_id="cli",
            client_secret="sec",
            redirect_port=19988,
        )

    def test_is_expired_false_on_init(self):
        p = self._provider()
        assert p.is_expired is False

    def test_is_expired_true(self):
        p = self._provider()
        p._expires_at = time.time() - 100
        assert p.is_expired is True

    def test_is_expired_false_future(self):
        p = self._provider()
        p._expires_at = time.time() + 3600
        assert p.is_expired is False

    def test_discover(self):
        p = self._provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "authorization_endpoint": "https://sso.example.com/auth",
            "token_endpoint": "https://sso.example.com/token",
        }
        with patch("requests.get", return_value=mock_resp):
            meta = p._discover()
        assert "authorization_endpoint" in meta

    def test_refresh_with_refresh_token(self):
        p = self._provider()
        p._refresh_token = "ref"
        mock_meta = MagicMock()
        mock_meta.raise_for_status = MagicMock()
        mock_meta.json.return_value = {
            "authorization_endpoint": "https://sso.example.com/auth",
            "token_endpoint": "https://sso.example.com/token",
        }
        mock_token = MagicMock()
        mock_token.ok = True
        mock_token.json.return_value = {"access_token": "new-oidc", "expires_in": 3600}
        session = requests.Session()
        with (
            patch("requests.get", return_value=mock_meta),
            patch("requests.post", return_value=mock_token),
        ):
            result = p.refresh(session, "https://example.com")
        assert result is True
        assert session.headers["Authorization"] == "Bearer new-oidc"

    def test_refresh_without_refresh_token_returns_false(self):
        p = self._provider()
        assert p.refresh(requests.Session(), "https://x.com") is False

    def test_apply_mocked(self):
        """Full apply flow with mocked browser and HTTP."""

        p = self._provider()

        mock_meta_resp = MagicMock()
        mock_meta_resp.raise_for_status = MagicMock()
        mock_meta_resp.json.return_value = {
            "authorization_endpoint": "https://sso.example.com/auth",
            "token_endpoint": "https://sso.example.com/token",
        }
        mock_token_resp = MagicMock()
        mock_token_resp.raise_for_status = MagicMock()
        mock_token_resp.json.return_value = {
            "access_token": "oidc-access",
            "refresh_token": "oidc-refresh",
            "expires_in": 3600,
        }

        # Mock _run_local_server to return a fake code
        with (
            patch("requests.get", return_value=mock_meta_resp),
            patch("requests.post", return_value=mock_token_resp),
            patch.object(p, "_run_local_server", return_value="auth_code_xyz"),
        ):
            session = requests.Session()
            p.apply(session, "https://example.com")

        assert session.headers["Authorization"] == "Bearer oidc-access"
        assert p._access_token == "oidc-access"
        assert p._refresh_token == "oidc-refresh"

    def test_build_from_config(self):
        from auth_providers import OpenIDAuth

        p = build_auth_provider(
            {
                "type": "openid",
                "issuer_url": "https://sso.example.com",
                "client_id": "my-app",
            }
        )
        assert isinstance(p, OpenIDAuth)


# ---------------------------------------------------------------------------
# KerberosAuth — successful apply (mocked package)
# ---------------------------------------------------------------------------
class TestKerberosAuthSuccess:
    def test_apply_with_mocked_kerberos(self):
        from auth_providers import KerberosAuth

        p = KerberosAuth(mutual_authentication="OPTIONAL")
        mock_kerberos_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.HTTPKerberosAuth = mock_kerberos_cls
        session = MagicMock()
        with patch.dict("sys.modules", {"requests_kerberos": mock_module}):
            p.apply(session, "https://x.com")
        mock_kerberos_cls.assert_called_once_with(mutual_authentication="OPTIONAL")
        assert session.auth == mock_kerberos_cls.return_value
