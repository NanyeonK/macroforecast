"""Tutorial CI smoke test.

Extracts Python code blocks from each tutorial markdown and executes them
in a subprocess with synthetic data. Detects API drift between documented
examples and the installed package.

Blocks containing ``# doctest: +SKIP`` are excluded.
Blocks containing graduation-only markers (placeholder mf_recipes.run() calls
with non-existent file paths or inline placeholders) are also excluded -- these
are illustrative and cannot run in CI without a full recipe YAML.
All blocks in a single tutorial are concatenated into one script to preserve
inter-block state (e.g., model definition before use in a loop).

Timeout: 60 seconds per tutorial.

C67 changes (2026-05-23):
- Graduation blocks excluded via _is_graduation_block() helper (tutorials 01-03
  graduation sections use placeholder recipes / file paths that are not runnable).
- Tutorial 03 prepends Tutorial 02 synthetic data so that y and X are defined
  when the custom-class OOS loop runs (tutorial 03 reuses tutorial 02 namespace).
- Tutorial 03 block referencing _LinearARModel (private) is excluded.
- ``from __future__ import annotations`` hoisted to script top when present in
  any block, because Python requires future imports at the start of the file.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# Locate tutorial files relative to this test file.
# tests/docs/test_tutorial_smoke.py -> parents[2] -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = REPO_ROOT / "docs" / "tutorial"

# Regex matches fenced python blocks: ```python\n...\n```
_BLOCK_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)

# Blocks containing this marker are excluded from execution.
_SKIP_MARKER = "# doctest: +SKIP"

# 60 seconds per tutorial -- generous for synthetic-data-only runs.
_TIMEOUT_SECONDS = 60

# Patterns that identify graduation-only (illustrative) blocks not runnable in CI.
# These blocks call mf_recipes.run() with placeholder paths or partial YAML.
_GRADUATION_MARKERS = [
    "mf_recipes.run(",         # recipe runner with placeholder path/YAML
    "mf.recipes.run(",         # alternate alias
    'output_directory=',       # blocks that write artifacts (graduation blocks)
]

# Patterns that identify blocks referencing private implementation classes.
_PRIVATE_CLASS_MARKERS = [
    "_LinearARModel",          # private base class not in public API
]

# Future import that must appear at the very top of the combined script.
_FUTURE_IMPORT = "from __future__ import annotations\n"


def _is_graduation_block(block: str) -> bool:
    """Return True if the block is a graduation snippet not runnable in CI.

    Graduation blocks typically call mf_recipes.run() with placeholder YAML
    strings or file paths that do not exist on disk. They are illustrative and
    intentionally cannot run in the test environment.
    """
    return any(marker in block for marker in _GRADUATION_MARKERS)


def _is_private_class_block(block: str) -> bool:
    """Return True if the block references private implementation classes."""
    return any(marker in block for marker in _PRIVATE_CLASS_MARKERS)


def _extract_blocks_from_text(text: str) -> list[str]:
    """Extract Python fenced code blocks from markdown text, filtering exclusions.

    Finds every fenced ```python ... ``` block in ``text``, then filters out
    any block that contains the ``_SKIP_MARKER``, is a graduation snippet, or
    references private classes not in the public API.
    """
    blocks = _BLOCK_PATTERN.findall(text)
    filtered = []
    for b in blocks:
        if _SKIP_MARKER in b:
            continue
        if _is_graduation_block(b):
            continue
        if _is_private_class_block(b):
            continue
        filtered.append(b)
    return filtered


def _extract_blocks(path: Path) -> list[str]:
    """Read file at ``path`` and extract Python fenced code blocks.

    Delegates to :func:`_extract_blocks_from_text` after reading the file
    contents; preserves backward compatibility for all existing callers.
    """
    return _extract_blocks_from_text(path.read_text(encoding="utf-8"))


def _build_script(blocks: list[str], source_name: str, prepend: str | None = None) -> str:
    """Concatenate code blocks into a single runnable Python script.

    Blocks are joined with a comment separator so that error tracebacks
    can be attributed back to the original block boundary. A header
    comment records the source markdown filename.

    ``from __future__ import annotations`` is hoisted to the top of the
    combined script if any block contains it, because Python requires future
    imports to appear at the very beginning of a file.

    If ``prepend`` is given, it is inserted after any future imports and before
    the tutorial blocks.
    """
    # Hoist __future__ imports to file top; remove them from individual blocks.
    needs_future = any(_FUTURE_IMPORT.strip() in block for block in blocks)
    cleaned_blocks = [block.replace(_FUTURE_IMPORT.strip() + "\n", "").replace(_FUTURE_IMPORT.strip(), "") for block in blocks]

    separator = "\n# --- next tutorial block ---\n"
    body = separator.join(cleaned_blocks)

    parts = [f"# Auto-extracted from {source_name}\n"]
    if needs_future:
        parts.append(_FUTURE_IMPORT)
    if prepend:
        parts.append(prepend)
    parts.append(body)
    return "".join(parts)


# Synthetic data preamble for Tutorial 03.
# Tutorial 03 references ``y`` and ``X`` from Tutorial 02's synthetic panel.
# The smoke test runs each tutorial independently, so we prepend the data
# generation code from Tutorial 02 as a fixture block.
_TUTORIAL02_DATA_PREAMBLE = textwrap.dedent("""\
    # --- tutorial02 shared-namespace preamble (for tutorial 03 dependency) ---
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed=0)
    n = 300
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    X = pd.DataFrame(
        rng.standard_normal((n, 5)),
        index=dates,
        columns=["ip_growth", "unemp_diff", "cpi_growth", "ffr_diff", "spread"],
    )
    beta = np.array([0.4, -0.3, 0.2, -0.1, 0.15])
    eps = rng.normal(scale=0.5, size=n)
    y_vals = X.values @ beta + eps
    for t in range(1, n):
        y_vals[t] += 0.3 * y_vals[t - 1]
    y = pd.Series(y_vals, index=dates, name="gdp_growth")
    # --- end preamble ---
""")


@pytest.mark.slow
@pytest.mark.parametrize(
    "tutorial_file,prepend",
    [
        (TUTORIAL_DIR / "01_first_forecast.md", None),
        (TUTORIAL_DIR / "02_full_study.md", None),
        (TUTORIAL_DIR / "03_custom_model.md", _TUTORIAL02_DATA_PREAMBLE),
    ],
    ids=["tutorial_01", "tutorial_02", "tutorial_03"],
)
def test_tutorial_smoke(tutorial_file: Path, prepend: str | None, tmp_path: Path) -> None:
    """Execute all Python code blocks from tutorial_file and assert no errors.

    Each tutorial's non-skipped, non-graduation blocks are concatenated into a
    single script and executed in a fresh subprocess. Concatenation preserves
    inter-block state (model definition before use, variable sharing).

    Tutorial 03 prepends the Tutorial 02 synthetic data generation because
    tutorial 03's OOS loop block references ``y`` and ``X`` defined there.

    Graduation blocks (those calling mf_recipes.run() with placeholder paths
    or partial YAML) are excluded from execution -- they are illustrative and
    require a real recipe file or complete inline YAML to run.
    """
    # Step 1: Extract all executable blocks from the tutorial markdown.
    blocks = _extract_blocks(tutorial_file)
    assert blocks, f"No extractable Python blocks found in {tutorial_file.name}"

    # Step 2: Build one concatenated script from all blocks.
    script = _build_script(blocks, tutorial_file.name, prepend=prepend)

    # Step 3: Redirect tutorial output directories to tmp_path.
    script = script.replace(
        '"./tutorial_output/',
        f'"{tmp_path}/',
    )

    # Step 4: Write the script to a temporary file and execute it.
    with tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(script)
        tmp_path_script = tmp.name

    # Build subprocess environment: inherit the current environment and ensure
    # PYTHONPATH includes REPO_ROOT so the local macroforecast package is
    # importable even when the editable install points to a stale source path.
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else str(REPO_ROOT)
    )

    try:
        result = subprocess.run(
            [sys.executable, tmp_path_script],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            env=env,
        )
    finally:
        # Always clean up the temp script regardless of outcome.
        os.unlink(tmp_path_script)

    # Step 5: Assert the script exited cleanly (return code 0).
    assert result.returncode == 0, (
        f"Tutorial {tutorial_file.name} smoke test failed.\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}"
    )
