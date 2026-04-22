"""test_cli.py — Tests for CLI argument parser and main() function."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import atlassian_rag_exporter as m


class TestBuildArgParser:
    def test_version(self, capsys):
        p = m.build_arg_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--version"])
        assert exc.value.code == 0

    def test_print_example_config(self, capsys):
        ret = m.main(["--print-example-config"])
        out, _ = capsys.readouterr()
        assert ret == 0
        assert "base_url" in out
        assert "auth" in out

    def test_missing_config_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            m.main([])
        assert exc_info.value.code != 0

    def test_missing_config_file(self, tmp_path, capsys):
        ret = m.main(["--config", str(tmp_path / "nonexistent.yaml")])
        assert ret != 0

    def test_invalid_auth_type_in_config(self, tmp_path):
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "base_url: https://x.atlassian.net\n"
            "is_cloud: true\n"
            "auth:\n"
            "  type: magic\n"
            "  token: abc\n"
            "output_dir: /tmp/out\n"
        )
        ret = m.main(["--config", str(config_file)])
        assert ret != 0

    def test_config_spaces_override(self, tmp_path):
        config_file = tmp_path / "cfg.yaml"
        out = tmp_path / "out"
        out.mkdir()
        config_file.write_text(
            f"base_url: https://x.atlassian.net\n"
            f"is_cloud: true\n"
            f"auth:\n"
            f"  type: token\n"
            f"  email: a@b.com\n"
            f"  token: tok\n"
            f"output_dir: {out}\n"
            f"spaces:\n"
            f"  - OLD\n"
        )
        with pytest.raises(SystemExit):
            m.main(["--config", str(config_file), "--spaces", "ENG", "DOCS"])
