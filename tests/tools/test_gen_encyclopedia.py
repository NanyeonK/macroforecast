"""Smoke tests for tools/gen_encyclopedia_docs.py.

Tests use subprocess.run to invoke the tool as a script, avoiding import-time
side effects from the ops registry bootstrapping.

Test functions:
    test_gen_help_exits_zero              -- --help returns exit code 0
    test_gen_dry_run_layer_all            -- --layer all --dry-run exits zero
    test_gen_dry_run_layer_l3             -- --layer L3 --dry-run produces l3/op/ paths
    test_gen_idempotent                   -- two runs with same --review-date are identical
    test_gen_l3_op_count                  -- L3 generates exactly 36 .md files
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Locate the tool relative to this test file
_REPO_ROOT = Path(__file__).parent.parent.parent
_TOOL = str(_REPO_ROOT / "tools" / "gen_encyclopedia_docs.py")

# Fixed review date for deterministic/idempotent tests
_REVIEW_DATE = "2026-05-21"


def _run(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run the tool with the given arguments, capturing output."""
    return subprocess.run(
        [sys.executable, _TOOL, *args],
        capture_output=True,
        text=True,
        cwd=cwd or str(_REPO_ROOT),
    )


def test_gen_help_exits_zero() -> None:
    """--help should exit 0 and produce non-trivial output."""
    result = _run("--help")
    assert result.returncode == 0, f"--help exited {result.returncode}"
    assert len(result.stdout) > 50, "Expected non-trivial --help output"
    assert "gen_encyclopedia" in result.stdout.lower() or "layer" in result.stdout.lower()


def test_gen_dry_run_layer_all() -> None:
    """--layer all --dry-run should exit 0 and produce stdout output."""
    result = _run("--layer", "all", "--dry-run", "--review-date", _REVIEW_DATE)
    assert result.returncode == 0, (
        f"--layer all --dry-run exited {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Should produce some output lines (at least one page path)
    assert len(result.stdout.strip()) > 0, "Expected stdout output for --dry-run"


def test_gen_dry_run_layer_l3() -> None:
    """--layer L3 --dry-run should produce at least one l3/op/ path in stdout."""
    result = _run("--layer", "L3", "--dry-run", "--review-date", _REVIEW_DATE)
    assert result.returncode == 0, (
        f"--layer L3 --dry-run exited {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Expect at least one line containing "l3/op/"
    assert "l3/op/" in result.stdout, (
        "Expected at least one l3/op/ path in dry-run output.\n"
        f"stdout: {result.stdout[:500]}"
    )


def test_gen_idempotent(tmp_path: Path) -> None:
    """Two runs with the same --review-date should produce byte-identical output.

    The second run should log no 'UPDATED' lines (all files unchanged).
    """
    out_dir = tmp_path / "enc"
    out_dir.mkdir()

    # First run
    r1 = _run(
        "--layer", "L3",
        "--out", str(out_dir),
        "--review-date", _REVIEW_DATE,
        "--force",
    )
    assert r1.returncode == 0, f"First run failed.\nstdout: {r1.stdout}\nstderr: {r1.stderr}"

    # Second run — same args
    r2 = _run(
        "--layer", "L3",
        "--out", str(out_dir),
        "--review-date", _REVIEW_DATE,
    )
    assert r2.returncode == 0, f"Second run failed.\nstdout: {r2.stdout}\nstderr: {r2.stderr}"

    # The second run should not log any "UPDATED" lines
    updated_lines = [
        line for line in r2.stdout.splitlines()
        if line.startswith("UPDATED ")
    ]
    assert len(updated_lines) == 0, (
        f"Second run should not update files when content is identical.\n"
        f"Updated: {updated_lines}"
    )


def test_gen_l3_op_count(tmp_path: Path) -> None:
    """--layer L3 should generate exactly 36 .md files under l3/op/.

    This matches the cycle 41/42 baseline count of 36 pages in
    docs/encyclopedia/l3/op/.
    """
    out_dir = tmp_path / "enc"
    out_dir.mkdir()

    result = _run(
        "--layer", "L3",
        "--out", str(out_dir),
        "--review-date", _REVIEW_DATE,
        "--force",
    )
    assert result.returncode == 0, (
        f"L3 generation failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Count generated .md files under l3/op/
    l3_op_dir = out_dir / "l3" / "op"
    assert l3_op_dir.is_dir(), (
        f"Expected l3/op/ directory to exist under {out_dir}"
    )
    md_files = sorted(l3_op_dir.glob("*.md"))
    count = len(md_files)
    assert count == 36, (
        f"Expected 36 .md files in l3/op/, got {count}.\n"
        f"Files: {[f.name for f in md_files]}"
    )


def test_gen_no_third_party_imports() -> None:
    """gen_encyclopedia_docs.py must not import any third-party package at module level.

    Uses ast.parse to walk the tool file's top-level Import and ImportFrom nodes.
    Any imported name that is not a stdlib module or macroforecast is a failure.
    Third-party imports are allowed inside function bodies (e.g., lazy imports for
    ops registry bootstrapping), but not at module scope.
    """
    import ast
    import sys

    tool_path = _REPO_ROOT / "tools" / "gen_encyclopedia_docs.py"
    source = tool_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(tool_path))

    # Collect module-level import names only (direct children of Module body)
    imported_names: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.append(node.module.split(".")[0])

    # Classify each name
    stdlib_names = sys.stdlib_module_names  # available in Python 3.10+
    third_party: list[str] = []
    for name in imported_names:
        if name in stdlib_names:
            continue
        if name.startswith("macroforecast"):
            continue
        if name == "__future__":
            continue
        if name.startswith("_"):
            continue
        third_party.append(name)

    assert not third_party, (
        f"gen_encyclopedia_docs.py has module-level third-party imports: {third_party}. "
        "All tool-level imports must be stdlib-only."
    )
