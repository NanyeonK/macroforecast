"""Tutorial CI smoke test.

Extracts Python code blocks from each tutorial markdown and executes them
in a subprocess with synthetic data. Detects API drift between documented
examples and the installed package.

Blocks containing ``# doctest: +SKIP`` are excluded.
All blocks in a single tutorial are concatenated into one script to preserve
inter-block state (e.g., model registration before mf.run()).

Timeout: 60 seconds per tutorial.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
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

# 60 seconds per tutorial — generous for synthetic-data-only runs.
_TIMEOUT_SECONDS = 60


def _extract_blocks(path: Path) -> list[str]:
    """Return all non-skipped Python code blocks from a markdown file.

    Reads the markdown at ``path``, finds every fenced ```python ... ```
    block, then filters out any block that contains the ``_SKIP_MARKER``
    string anywhere in its text.
    """
    content = path.read_text(encoding="utf-8")
    blocks = _BLOCK_PATTERN.findall(content)
    # Exclude any block the tutorial author explicitly opted out of running.
    return [b for b in blocks if _SKIP_MARKER not in b]


def _build_script(blocks: list[str], source_name: str) -> str:
    """Concatenate code blocks into a single runnable Python script.

    Blocks are joined with a comment separator so that error tracebacks
    can be attributed back to the original block boundary. A header
    comment records the source markdown filename.
    """
    separator = "\n# --- next tutorial block ---\n"
    body = separator.join(blocks)
    return f"# Auto-extracted from {source_name}\n" + body


@pytest.mark.slow
@pytest.mark.parametrize(
    "tutorial_file",
    [
        TUTORIAL_DIR / "01_first_forecast.md",
        TUTORIAL_DIR / "02_full_study.md",
        TUTORIAL_DIR / "03_custom_model.md",
    ],
    ids=["tutorial_01", "tutorial_02", "tutorial_03"],
)
def test_tutorial_smoke(tutorial_file: Path, tmp_path: Path) -> None:
    """Execute all Python code blocks from tutorial_file and assert no errors.

    Each tutorial's non-skipped blocks are concatenated into a single script
    and executed in a fresh subprocess. Concatenation is critical for tutorials
    that register custom models in one block and reference them in a later block
    (e.g., Tutorial 03's ``@mf.register_model`` before ``mf.run()``).

    Output directories inside the script are redirected to ``tmp_path`` so the
    test does not write into the project directory during CI.
    """
    # Step 1: Extract all executable blocks from the tutorial markdown.
    blocks = _extract_blocks(tutorial_file)
    assert blocks, f"No extractable Python blocks found in {tutorial_file.name}"

    # Step 2: Build one concatenated script from all blocks.
    script = _build_script(blocks, tutorial_file.name)

    # Step 3: Redirect tutorial output directories to tmp_path.
    # Tutorial code uses "./tutorial_output/..." paths; replace with a
    # tmp_path subdirectory so nothing is written into the project tree.
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

    try:
        result = subprocess.run(
            ["python", tmp_path_script],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
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
