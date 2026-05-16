"""Tests for Cycle 22 per-op page emission (OptionDoc.op_page).

Verifies that:
- ``write_all`` emits ``l4/family/ridge.md`` with Function-signature + Parameters sections.
- ``write_all`` emits ``l5/point_metrics/theil_u1.md`` with the same key sections.
- Per-op pages link back to the standalone callable (mf.functions.*).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from macroforecast.scaffold import render_encyclopedia


@pytest.fixture(scope="module")
def enc_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("encyclopedia")
    render_encyclopedia.write_all(d)
    return d


class TestRidgeOpPage:
    def test_ridge_page_exists(self, enc_dir: Path) -> None:
        page = enc_dir / "l4" / "family" / "ridge.md"
        assert page.exists(), f"ridge.md not found at {page}"

    def test_ridge_page_has_function_signature(self, enc_dir: Path) -> None:
        body = (enc_dir / "l4" / "family" / "ridge.md").read_text(encoding="utf-8")
        assert "## Function signature" in body

    def test_ridge_page_has_parameters(self, enc_dir: Path) -> None:
        body = (enc_dir / "l4" / "family" / "ridge.md").read_text(encoding="utf-8")
        assert "## Parameters" in body

    def test_ridge_page_links_to_mf_functions(self, enc_dir: Path) -> None:
        body = (enc_dir / "l4" / "family" / "ridge.md").read_text(encoding="utf-8")
        assert "mf.functions.ridge_fit" in body


class TestTheilU1OpPage:
    def test_theil_u1_page_exists(self, enc_dir: Path) -> None:
        page = enc_dir / "l5" / "point_metrics" / "theil_u1.md"
        assert page.exists(), f"theil_u1.md not found at {page}"

    def test_theil_u1_page_has_function_signature(self, enc_dir: Path) -> None:
        body = (enc_dir / "l5" / "point_metrics" / "theil_u1.md").read_text(encoding="utf-8")
        assert "## Function signature" in body

    def test_theil_u1_page_has_parameters(self, enc_dir: Path) -> None:
        body = (enc_dir / "l5" / "point_metrics" / "theil_u1.md").read_text(encoding="utf-8")
        assert "## Parameters" in body

    def test_theil_u1_page_links_to_mf_functions(self, enc_dir: Path) -> None:
        body = (enc_dir / "l5" / "point_metrics" / "theil_u1.md").read_text(encoding="utf-8")
        assert "mf.functions.theil_u1" in body
