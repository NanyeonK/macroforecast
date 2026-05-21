"""Smoke tests for tools/audit_docs_vs_code.py.

Tests use subprocess.run to invoke the tool as a script (to avoid ops-registry
import-time side effects) plus a synthetic fixture directory.

Test functions:
    test_audit_help_exits_zero                        -- --help returns exit code 0
    test_audit_clean_fixture                          -- known-good ref resolves PASS
    test_audit_drift_fixture                          -- bad ref + version resolves DRIFT
    test_audit_json_schema                            -- JSON report has required keys
    test_audit_standalone_docs_false_positive_rate    -- <5% false positives on cycle-41 surface
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Locate the tool relative to this test file
_REPO_ROOT = Path(__file__).parent.parent.parent
_TOOL = str(_REPO_ROOT / "tools" / "audit_docs_vs_code.py")


def _run(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run the audit tool with the given arguments, capturing output."""
    return subprocess.run(
        [sys.executable, _TOOL, *args],
        capture_output=True,
        text=True,
        cwd=cwd or str(_REPO_ROOT),
    )


def test_audit_help_exits_zero() -> None:
    """--help should exit 0."""
    result = _run("--help")
    assert result.returncode == 0, f"--help exited {result.returncode}"
    assert len(result.stdout) > 50, "Expected non-trivial --help output"


def test_audit_clean_fixture(tmp_path: Path) -> None:
    """A synthetic .md with a known-good mf.functions.rmse reference should PASS."""
    # Write a synthetic fixture file
    fixture_dir = tmp_path / "fixture_clean"
    fixture_dir.mkdir()
    md_file = fixture_dir / "clean.md"
    md_file.write_text(
        "# Clean fixture\n\n"
        "Use `mf.functions.rmse` to compute root mean squared error.\n"
        "\n"
        "```python\n"
        "result = mf.functions.rmse(y_true, y_pred)\n"
        "```\n",
        encoding="utf-8",
    )

    out_json = tmp_path / "clean_report.json"
    result = _run(
        "--root", str(fixture_dir),
        "--out", str(out_json),
    )
    assert result.returncode == 0, (
        f"Audit exited {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    # Find the standalone_callable entry for mf.functions.rmse
    rmse_entries = [
        e for e in report["entries"]
        if e["token_class"] == "standalone_callable"
        and "rmse" in e["token"]
    ]
    assert len(rmse_entries) >= 1, (
        f"Expected at least one mf.functions.rmse entry.\n"
        f"Entries: {report['entries']}"
    )
    for entry in rmse_entries:
        assert entry["verdict"] == "PASS", (
            f"Expected PASS for mf.functions.rmse, got {entry['verdict']}: {entry['evidence']}"
        )


def test_audit_drift_fixture(tmp_path: Path) -> None:
    """A synthetic .md with a non-existent function should cause DRIFT and exit 1."""
    fixture_dir = tmp_path / "fixture_drift"
    fixture_dir.mkdir()
    md_file = fixture_dir / "drift.md"
    md_file.write_text(
        "# Drift fixture\n\n"
        "Use `mf.functions.does_not_exist` — this function does not exist.\n"
        "\n"
        "Current version is v0.0.0 — this version does not match.\n",
        encoding="utf-8",
    )

    out_json = tmp_path / "drift_report.json"
    result = _run(
        "--root", str(fixture_dir),
        "--out", str(out_json),
        "--fail-on-drift",
    )
    # With --fail-on-drift and DRIFT tokens, exit code should be 1
    assert result.returncode == 1, (
        f"Expected exit code 1 for drift fixture, got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Verify the report was written and contains DRIFT entries
    report = json.loads(out_json.read_text(encoding="utf-8"))
    drift_entries = [e for e in report["entries"] if e["verdict"] == "DRIFT"]
    assert len(drift_entries) >= 1, (
        f"Expected at least one DRIFT entry.\nAll entries: {report['entries']}"
    )


def test_audit_json_schema(tmp_path: Path) -> None:
    """The JSON report must have the required keys and summary.drift >= 1."""
    # Write a drift fixture
    fixture_dir = tmp_path / "fixture_schema"
    fixture_dir.mkdir()
    md_file = fixture_dir / "schema_test.md"
    md_file.write_text(
        "# Schema test\n\n"
        "Use `mf.functions.totally_fake_function_xyz123` here.\n",
        encoding="utf-8",
    )

    out_json = tmp_path / "schema_report.json"
    result = _run(
        "--root", str(fixture_dir),
        "--out", str(out_json),
    )
    # Exit code 0 without --fail-on-drift even with drift
    assert result.returncode == 0, (
        f"Audit exited {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))

    # Verify required top-level keys
    assert "generated_at" in report, "Missing 'generated_at' key"
    assert "summary" in report, "Missing 'summary' key"
    assert "entries" in report, "Missing 'entries' key"

    # Verify summary sub-keys
    summary = report["summary"]
    for key in ("files_scanned", "total_tokens", "pass", "drift", "unresolvable"):
        assert key in summary, f"Missing 'summary.{key}' key"

    # At least 1 DRIFT (the fake function reference)
    assert summary["drift"] >= 1, (
        f"Expected summary.drift >= 1, got {summary['drift']}"
    )

    # generated_at should be an ISO 8601 timestamp ending with Z
    assert report["generated_at"].endswith("Z"), (
        f"generated_at should end with Z: {report['generated_at']}"
    )

    # entries should be a list
    assert isinstance(report["entries"], list), "entries should be a list"


@pytest.mark.integration
def test_audit_standalone_docs_false_positive_rate() -> None:
    """Audit on docs/standalone_functions/ should have <5% DRIFT rate.

    This is a live integration test against the cycle-41 aligned doc surface.
    It requires docs/standalone_functions/ to exist and be populated.

    Run separately with: pytest -m integration tests/tools/test_audit_docs.py
    """
    standalone_dir = _REPO_ROOT / "docs" / "standalone_functions"
    if not standalone_dir.is_dir():
        pytest.skip("docs/standalone_functions/ not found — skipping integration test")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_json = Path(f.name)

    try:
        result = _run(
            "--root", str(standalone_dir),
            "--out", str(out_json),
        )
        assert result.returncode == 0, (
            f"Audit exited {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        report = json.loads(out_json.read_text(encoding="utf-8"))
        summary = report["summary"]
        total = summary["total_tokens"]
        drift = summary["drift"]

        if total == 0:
            pytest.skip("No tokens found in standalone_functions/ — skipping rate check")

        false_positive_rate = drift / total
        assert false_positive_rate < 0.05, (
            f"False-positive rate {false_positive_rate:.1%} exceeds 5% threshold.\n"
            f"Total tokens: {total}, DRIFT: {drift}\n"
            f"DRIFT entries: {[e for e in report['entries'] if e['verdict'] == 'DRIFT'][:10]}"
        )
    finally:
        out_json.unlink(missing_ok=True)


def test_audit_no_third_party_imports() -> None:
    """audit_docs_vs_code.py must not import any third-party package at module level.

    Uses ast.parse to walk the tool file's top-level Import and ImportFrom nodes.
    Any imported name that is not a stdlib module or macroforecast is a failure.
    """
    import ast
    import sys

    tool_path = _REPO_ROOT / "tools" / "audit_docs_vs_code.py"
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
        f"audit_docs_vs_code.py has module-level third-party imports: {third_party}. "
        "All tool-level imports must be stdlib-only."
    )
