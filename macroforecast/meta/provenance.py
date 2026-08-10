"""Environment, git and dependency provenance.

Probes the environment; it has nothing to do with writing artifacts, which is why
it no longer lives in ``output``. ``pipeline.run`` needs the same block for the run
manifest and until 2026-08-09 imported it from ``output`` -- the second of two known
layering exceptions.

``output.collect_provenance`` re-exports this, so the public name is unchanged.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from macroforecast import __version__


def collect_provenance(
    *,
    cwd: str | Path | None = None,
    fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Collect lightweight package, Python, platform, and git provenance."""

    root = Path(cwd or Path.cwd())
    provenance = {
        "macroforecast_version": __version__,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(root),
        "git": _git_provenance(root),
        "packages": {
            package: _package_version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels")
        },
    }
    if fields is None:
        return provenance
    return {field: provenance[field] for field in fields if field in provenance}


def _git_provenance(cwd: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(cwd),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(run_git("status", "--porcelain")),
    }


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
