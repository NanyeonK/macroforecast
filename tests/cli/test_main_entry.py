"""Tests for the macroforecast CLI entry point (python -m macroforecast)."""
from __future__ import annotations

import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "macroforecast", *args],
        capture_output=True,
        text=True,
    )


def test_help_exits_zero():
    result = _run("--help")
    assert result.returncode == 0
    assert "run" in result.stdout or "run" in result.stderr


def test_run_subcommand_accessible():
    result = _run("run", "--help")
    assert result.returncode == 0


def test_replicate_subcommand_accessible():
    result = _run("replicate", "--help")
    assert result.returncode == 0


def test_validate_subcommand_accessible():
    result = _run("validate", "--help")
    assert result.returncode == 0


def test_no_subcommand_prints_help_exits_zero():
    result = _run()
    assert result.returncode == 0


def test_run_missing_recipe_exits_nonzero():
    result = _run("run", "nonexistent_recipe_xyz.yaml")
    assert result.returncode != 0
