from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tools.docgen as docgen


def test_tools_docgen_imports_cleanly() -> None:
    assert "write_all" in docgen.__all__
    assert docgen.collect_pages()


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
