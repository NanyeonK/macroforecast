from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tools.docgen as docgen
from tools.docgen import renderer


def test_tools_docgen_imports_cleanly() -> None:
    assert "write_all" in docgen.__all__
    assert docgen.collect_pages()


def test_docgen_uses_source_declared_module_exports(monkeypatch) -> None:
    import macroforecast.model_selection as model_selection

    page = Path("model_selection.md")
    baseline = docgen.collect_pages()[page]
    monkeypatch.setattr(
        model_selection,
        "__all__",
        [*model_selection.__all__, "OptionalRuntimeOnly"],
    )
    monkeypatch.setattr(
        model_selection,
        "OptionalRuntimeOnly",
        lambda: None,
        raising=False,
    )

    assert docgen.collect_pages()[page] == baseline


def test_docgen_module_pages_have_source_declared_exports() -> None:
    missing = [
        page.module
        for page in renderer.MODULE_PAGES
        if renderer._source_declared_all(renderer._module(page)) is None
    ]

    assert not missing


def test_docgen_check_passes_on_committed_reference_tree() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tools.docgen", "--check", "docs/reference"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_docgen_check_fails_on_perturbed_page(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    docgen.write_all(reference)
    page = reference / "index.md"
    page.write_text(page.read_text(encoding="utf-8") + "\nperturbation\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "tools.docgen", "--check", str(reference)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "changed: index.md" in result.stderr
