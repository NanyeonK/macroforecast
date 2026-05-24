"""C68 Per-Algorithm How-To Validation — Independent Tester.

Validates seven new how-to files plus the reorganized index. Does NOT read
spec.md or implementation.md (pipeline isolation). Drives validation purely
from test-spec.md behavioral contracts.

Tests:
  T1     - mf.recipes.run only in graduation/See-also (or absent) for per-algo
  T2     - zero DAG occurrences across 8 files
  T3     - syntax validity for all python fenced blocks
  T4     - runnable per-algo concatenated blocks execute without error
           (compare_midas_variants block executed in subprocess with timeout)
  T5     - each per-algo how-to has "See also" + paper citation
  T6     - advanced_recipes uses mf.register_model (NOT mf.recipes.register_step)
  T_imp  - every "from macroforecast.X import Y" resolves at runtime
  T_chan - zero em-dashes in prose paragraphs (style lint)
"""
from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOWTO_DIR = REPO_ROOT / "docs" / "how_to"

PER_ALGO_FILES = [
    HOWTO_DIR / "forecast_volatility_realized_garch.md",
    HOWTO_DIR / "feature_selection_boruta.md",
    HOWTO_DIR / "bayesian_var_minnesota.md",
    HOWTO_DIR / "chow_lin_disaggregation.md",
    HOWTO_DIR / "irf_pesaran_shin_girf.md",
    HOWTO_DIR / "compare_midas_variants.md",
]
ADV_RECIPES = HOWTO_DIR / "advanced_recipes.md"
INDEX_MD = HOWTO_DIR / "index.md"
ALL_FILES = PER_ALGO_FILES + [ADV_RECIPES, INDEX_MD]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def _strip_fenced_code(text: str) -> str:
    return re.sub(r"```[\s\S]*?```", "", text)


# ---------------------------------------------------------------------------
# T1 - Recipe orchestration only in advanced_recipes / See-also
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", PER_ALGO_FILES, ids=[p.stem for p in PER_ALGO_FILES])
def test_T1_recipe_run_absent_in_per_algo(path: Path) -> None:
    """Per-algorithm how-tos must not call mf.recipes.run / mf.run / mf_recipes.run.

    Recipe orchestration belongs in advanced_recipes.md only. The per-algo
    how-tos demonstrate the standalone class API.
    """
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"mf\.recipes\.run\(|mf_recipes\.run\(|mf\.run\(", text)
    assert not matches, (
        f"{path.name}: found recipe orchestration calls in per-algo how-to: "
        f"{matches}"
    )


# ---------------------------------------------------------------------------
# T2 - Zero DAG occurrences across 8 files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_FILES, ids=[p.stem for p in ALL_FILES])
def test_T2_no_dag_jargon(path: Path) -> None:
    """No 'DAG' jargon (case-insensitive whole word) in any C68 doc file."""
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"\bDAG\b", text, re.IGNORECASE)
    # advanced_recipes uses "12-layer pipeline (L0 through L8)" instead.
    # index.md has the 3 toctrees; no DAG either.
    # Note: 'DAG' inside identifier 'feature_engineering_dag' is allowed
    # because that is a YAML recipe key. Strip those before counting.
    text_stripped = re.sub(r"feature_engineering_dag", "", text)
    matches_strict = re.findall(r"\bDAG\b", text_stripped, re.IGNORECASE)
    assert not matches_strict, (
        f"{path.name}: found DAG jargon: {matches_strict}"
    )


# ---------------------------------------------------------------------------
# T3 - Syntax validity of all python fenced blocks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_FILES, ids=[p.stem for p in ALL_FILES])
def test_T3_python_blocks_compile(path: Path) -> None:
    """Every ```python fenced block must parse via compile()."""
    blocks = _extract_python_blocks(path)
    if not blocks:
        pytest.skip(f"{path.name}: no python blocks to check")
    errors: list[str] = []
    for i, block in enumerate(blocks):
        try:
            compile(block, f"<{path.name}:block-{i}>", "exec")
        except SyntaxError as exc:
            errors.append(f"Block {i + 1}: {exc}")
    assert not errors, (
        f"{path.name}: syntax errors:\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# T4 - Runnable concatenated blocks execute (per-algo, key files only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", PER_ALGO_FILES, ids=[p.stem for p in PER_ALGO_FILES])
def test_T4_concatenated_blocks_execute(path: Path, tmp_path: Path) -> None:
    """Concatenate all python blocks for a per-algo file and execute in subprocess.

    A 60-second wall-clock timeout per file. The script runs in a fresh
    interpreter so name leakage between files is impossible.
    """
    blocks = _extract_python_blocks(path)
    assert blocks, f"{path.name}: no python blocks found"
    script_body = "\n\n".join(blocks)
    script = tmp_path / f"snippet_{path.stem}.py"
    script.write_text(script_body, encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{path.name}: snippet exceeded 60s timeout")
    assert result.returncode == 0, (
        f"{path.name}: concatenated snippet failed (rc={result.returncode}).\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# T5 - See also section + paper citation
# ---------------------------------------------------------------------------

REQUIRED_PAPERS = {
    "forecast_volatility_realized_garch.md": ["Hansen", "Huang", "Shek"],
    "feature_selection_boruta.md": ["Kursa", "Rudnicki"],
    "bayesian_var_minnesota.md": ["Litterman"],
    "chow_lin_disaggregation.md": ["Chow", "Lin"],
    "irf_pesaran_shin_girf.md": ["Pesaran", "Shin"],
    "compare_midas_variants.md": ["Foroni"],  # at least one MIDAS paper
}


@pytest.mark.parametrize("path", PER_ALGO_FILES, ids=[p.stem for p in PER_ALGO_FILES])
def test_T5_see_also_and_paper(path: Path) -> None:
    """Each per-algo how-to ends with a 'See also' section and required paper."""
    text = path.read_text(encoding="utf-8")
    assert re.search(r"^##\s+See also\s*$", text, re.MULTILINE), (
        f"{path.name}: missing '## See also' section"
    )
    required = REQUIRED_PAPERS[path.name]
    missing = [author for author in required if author not in text]
    assert not missing, (
        f"{path.name}: required paper authors missing: {missing}"
    )


# ---------------------------------------------------------------------------
# T6 - advanced_recipes uses mf.register_model (NOT mf.recipes.register_step)
# ---------------------------------------------------------------------------

def test_T6_advanced_recipes_register_model() -> None:
    """advanced_recipes.md must demonstrate mf.register_model and must NOT
    invent a mf.recipes.register_step API.
    """
    text = ADV_RECIPES.read_text(encoding="utf-8")
    assert "mf.register_model" in text, (
        "advanced_recipes.md: missing mf.register_model demonstration"
    )
    assert "mf.recipes.register_step" not in text, (
        "advanced_recipes.md: contains invented API mf.recipes.register_step"
    )


# ---------------------------------------------------------------------------
# T_imp - every "from macroforecast.X import Y" resolves at runtime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", PER_ALGO_FILES + [ADV_RECIPES], ids=[p.stem for p in PER_ALGO_FILES + [ADV_RECIPES]])
def test_T_imports_resolve(path: Path) -> None:
    """Every import statement of the form 'from macroforecast.X import Y, Z'
    must resolve at runtime."""
    text = path.read_text(encoding="utf-8")
    # Single-line imports
    single = re.findall(r"^from\s+(macroforecast\.[a-zA-Z0-9_.]+)\s+import\s+([^\n]+)$", text, re.MULTILINE)
    # Multi-line parenthesized imports
    multi = re.findall(r"^from\s+(macroforecast\.[a-zA-Z0-9_.]+)\s+import\s+\(([^)]+)\)", text, re.MULTILINE)

    pairs: list[tuple[str, str]] = []
    for mod, names in single:
        # Skip multi-line parenthesized form (handled below)
        if names.strip() == "(":
            continue
        for name in names.split(","):
            n = name.strip()
            if n and n != "(":
                pairs.append((mod, n))
    for mod, names_block in multi:
        for name in re.split(r"[,\s]+", names_block):
            n = name.strip()
            if n:
                pairs.append((mod, n))

    failures: list[str] = []
    for mod, name in pairs:
        try:
            m = importlib.import_module(mod)
        except Exception as exc:
            failures.append(f"import {mod} failed: {exc}")
            continue
        if not hasattr(m, name):
            failures.append(f"{mod}.{name} does not exist")
    assert not failures, (
        f"{path.name}: import resolution failures:\n" + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# T_chan - zero em-dashes in prose (style lint, hard fail per Chan rules)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_FILES, ids=[p.stem for p in ALL_FILES])
def test_T_chan_no_em_dash_in_prose(path: Path) -> None:
    """Chan style: no em-dash (U+2014) in prose paragraphs (outside fenced code)."""
    text = path.read_text(encoding="utf-8")
    cleaned = _strip_fenced_code(text)
    assert "—" not in cleaned, (
        f"{path.name}: contains em-dash in prose"
    )
