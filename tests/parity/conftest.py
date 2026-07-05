"""Shared fixtures/helpers for ``tests/parity/`` (WP-V1 R-parity harness).

Bridge decision (documented here, see also ``tests/parity/README.md``):
the deleted C59 suite used ``rpy2``. On this host (R 4.3.3 at ``/usr/bin/R``),
``rpy2`` 3.6.7 (the only build pip resolves) fails to import in *both* API and
ABI mode: ``undefined symbol: R_getVar`` / ``R_ParentEnv`` in
``/usr/lib/R/lib/libR.so``, even with ``R_HOME``/``LD_LIBRARY_PATH`` set
correctly. This is an rpy2-cffi/R-ABI mismatch, not a missing-library
problem, so pinning a different rpy2 version was not attempted (not in
scope for this WP; flagged for future infra work). We use a
**subprocess-Rscript bridge** instead: every parity test writes small CSVs
to a temp dir, runs a short R script synchronously via ``Rscript`` with a
bounded timeout, and parses a ``key=value`` results file the script writes
back. This has no ABI dependency and is easier to audit (the exact R call
is plain text in the test file).

Every test/module in this directory that needs R must call
``require_r("pkg1", "pkg2", ...)`` (or bare ``require_r()`` for just
Rscript) at the top of the test function, or use the ``r_available``
autouse-free helper directly. The whole directory is otherwise collected
normally -- only individual tests skip, so ``pytest tests/parity/`` degrades
gracefully to all-skipped rather than an error when R/packages are absent.

Run with: ``pytest tests/parity/ -m rparity``
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

RSCRIPT_BIN = shutil.which("Rscript") or "/usr/bin/Rscript"
R_TIMEOUT_SECONDS = 180

# R packages compile into a user library outside the default (root-owned,
# read-only) site-library on this host; see README.md "Setup" section.
_R_LIBS_USER_DEFAULT = str(
    Path.home() / "R" / "x86_64-pc-linux-gnu-library" / "4.3"
)


def _r_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("R_LIBS_USER", _R_LIBS_USER_DEFAULT)
    return env


def r_available() -> bool:
    return Path(RSCRIPT_BIN).exists()


def r_package_available(package: str) -> bool:
    if not r_available():
        return False
    try:
        proc = subprocess.run(
            [RSCRIPT_BIN, "-e", f'cat(requireNamespace("{package}", quietly=TRUE))'],
            capture_output=True,
            text=True,
            timeout=30,
            env=_r_env(),
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.stdout.strip() == "TRUE"


def require_r(*packages: str) -> None:
    """Skip the calling test if R (or any of ``packages``) is unavailable.

    Bare ``require_r()`` only checks that ``Rscript`` itself resolves.
    """
    if not r_available():
        pytest.skip(f"Rscript not found at {RSCRIPT_BIN!r}; R parity tests skipped")
    missing = [p for p in packages if not r_package_available(p)]
    if missing:
        pytest.skip(f"R package(s) not installed/loadable: {', '.join(missing)}")


_PREAMBLE_TEMPLATE = '''\
OUT_PATH <- "{out_path}"
.results_con <- file(OUT_PATH, open = "w")
emit <- function(name, value) {{
  formatted <- vapply(value, function(v) sprintf("%.17g", v), character(1))
  cat(sprintf("%s=%s\\n", name, paste(formatted, collapse = ",")), file = .results_con)
}}
emit_str <- function(name, value) {{
  cat(sprintf("%s=%s\\n", name, paste(value, collapse = ",")), file = .results_con)
}}
'''


def run_rscript(script_body: str, *, timeout: int = R_TIMEOUT_SECONDS) -> dict[str, str]:
    """Run an R script body against a fresh temp dir; return its emitted results.

    The body may call two predefined helpers to report results:
      - ``emit(name, numeric_vector)``   -> written as ``name=v1,v2,...`` with
        ``%.17g`` formatting (full double round-trip precision).
      - ``emit_str(name, character_vec)`` -> written as ``name=s1,s2,...``.

    Raises ``RuntimeError`` (with full stdout/stderr) on a non-zero exit so
    R errors surface as loud test failures rather than silent empty results.
    """
    with tempfile.TemporaryDirectory(prefix="mf_rparity_") as tmpdir:
        tmp = Path(tmpdir)
        out_path = tmp / "out.txt"
        script_path = tmp / "script.R"
        script_path.write_text(
            _PREAMBLE_TEMPLATE.format(out_path=str(out_path)) + "\n" + script_body + "\nclose(.results_con)\n"
        )
        proc = subprocess.run(
            [RSCRIPT_BIN, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmpdir,
            env=_r_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Rscript exited {proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}\n--- script ---\n{script_path.read_text()}"
            )
        results: dict[str, str] = {}
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                results[key] = value
        return results


def parse_float(value: str) -> float:
    if value.strip().upper() == "NA":
        return float("nan")
    return float(value)


def parse_float_list(value: str) -> list[float]:
    return [parse_float(v) for v in value.split(",")]


def parse_str_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def write_csv(path: Path, columns: dict[str, Any]) -> None:
    """Write a dict-of-sequences to ``path`` as a headered CSV (no index)."""
    import pandas as pd

    pd.DataFrame(columns).to_csv(path, index=False)
