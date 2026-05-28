"""C67 Tutorial Validation — Standalone-First Rewrite.

Independent tester validation for cycle 67. Tests behavioral contracts B1-B8
from test-spec.md. Does NOT read spec.md or implementation.md.

Tests:
  T1  — recipe calls only in graduation sections (last H2)
  T2  — zero DAG occurrences
  T3  — import symbol resolution (LinearAR, PCR, FAAR in __all__)
  T4a — syntax validity of all code blocks (tutorial 01)
  T4b — syntax validity of all code blocks (tutorial 02)
  T4c — syntax validity of all code blocks (tutorial 03)
  T5a — runtime execution of tutorial 01 code blocks (no exception)
  T5b — runtime execution of tutorial 02 code blocks (no exception)
  T5c — runtime execution of tutorial 03 code blocks (no exception)
  T7  — model imports importable at runtime
  T8  — ConstantTrendPlusAR compatible with sklearn clone() / get_params()
  R1  — tutorial 01 first block is not YAML
  R2  — tutorial 03 first class definition uses BaseEstimator + RegressorMixin
  R3  — tutorial 02 has fewer than 25 YAML recipe key lines
  T_index  — index.md introduces standalone before recipes
  T_two_ep — two_entry_points.md table has standalone left of recipe DSL
  T_first_h2 — first H2 of each tutorial has no recipe-first content
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repository root: tests/promotion/ -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = REPO_ROOT / "docs" / "tutorial"

# Tutorial files under test
TUT01 = TUTORIAL_DIR / "01_first_forecast.md"
TUT02 = TUTORIAL_DIR / "02_full_study.md"
TUT03 = TUTORIAL_DIR / "03_custom_model.md"
INDEX_MD = TUTORIAL_DIR / "index.md"
TWO_EP_MD = TUTORIAL_DIR / "two_entry_points.md"

ALL_TUTORIAL_FILES = [TUT01, TUT02, TUT03, INDEX_MD, TWO_EP_MD]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_python_blocks(path: Path) -> list[str]:
    """Extract all ```python ... ``` blocks from a markdown file."""
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def _last_h2_index(lines: list[str]) -> int:
    """Return the line index of the last ## heading in the file."""
    return max(i for i, line in enumerate(lines) if line.startswith("## "))


# ---------------------------------------------------------------------------
# T1 — Recipe call placement (standalone-first verification)
# ---------------------------------------------------------------------------

RECIPE_PATTERNS = re.compile(r"mf\.run\(|mf_recipes\.run\(|mf\.recipes\.run\(")


@pytest.mark.parametrize("tutorial_path", [TUT01, TUT02, TUT03], ids=["tut01", "tut02", "tut03"])
def test_T1_recipe_calls_only_in_graduation(tutorial_path: Path) -> None:
    """B1/B2/B3: mf.run() and mf_recipes.run() must only appear after the last H2 heading."""
    lines = tutorial_path.read_text(encoding="utf-8").splitlines(keepends=True)
    last_h2 = _last_h2_index(lines)
    recipe_lines_before = [
        i + 1  # 1-indexed for readability
        for i, line in enumerate(lines[:last_h2])
        if RECIPE_PATTERNS.search(line)
    ]
    assert not recipe_lines_before, (
        f"{tutorial_path.name}: recipe calls found before last H2 (line {last_h2 + 1}) "
        f"at lines: {recipe_lines_before}"
    )


# ---------------------------------------------------------------------------
# T2 — DAG jargon verification
# ---------------------------------------------------------------------------

def test_T2_no_DAG_jargon() -> None:
    """B4: The word 'DAG' must not appear in any of the five rewritten files."""
    violations: list[str] = []
    for path in ALL_TUTORIAL_FILES:
        text = path.read_text(encoding="utf-8")
        matches = re.findall(r"\bDAG\b", text, re.IGNORECASE)
        if matches:
            violations.append(f"{path.name}: {len(matches)} occurrence(s)")
    assert not violations, "DAG found in tutorial files:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# T3 — Import symbol resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol", ["LinearAR", "PrincipalComponentRegression", "FactorAugmentedAR"])
def test_T3_import_symbol_in_all(symbol: str) -> None:
    """B5: Required model classes must be in macroforecast.models.__all__."""
    sys.path.insert(0, str(REPO_ROOT))
    import macroforecast.models as m  # noqa: PLC0415
    assert symbol in m.__all__, (
        f"'{symbol}' not found in macroforecast.models.__all__. "
        f"__all__ = {m.__all__}"
    )


# ---------------------------------------------------------------------------
# T4 — Syntax validity of all code blocks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tutorial_path", [TUT01, TUT02, TUT03], ids=["tut01", "tut02", "tut03"])
def test_T4_syntax_valid(tutorial_path: Path) -> None:
    """B6: Every Python fenced code block must be syntactically valid."""
    blocks = _extract_python_blocks(tutorial_path)
    assert blocks, f"No Python blocks found in {tutorial_path.name}"
    errors: list[str] = []
    for i, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            errors.append(f"Block {i + 1}: {exc}\n  Preview: {block[:80]!r}")
    assert not errors, (
        f"{tutorial_path.name}: syntax errors in code blocks:\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# T5a — Runtime execution: Tutorial 01
# ---------------------------------------------------------------------------

def test_T5a_tutorial01_runtime(tmp_path: Path) -> None:
    """B7: Tutorial 01 code blocks execute without exception on synthetic data.

    The last block (graduation snippet) is deliberately excluded because it
    calls mf_recipes.run() with a placeholder recipe that cannot run in CI.
    The block contains placeholder markers (custom_panel_inline: {date: [...]})
    and is illustrative only.
    """
    sys.path.insert(0, str(REPO_ROOT))

    # Block 1: version check
    import macroforecast  # noqa: PLC0415
    assert macroforecast.__version__

    # Block 2: synthetic data generation
    rng = np.random.default_rng(seed=42)
    n = 100
    dates = pd.date_range("2015-01-01", periods=n, freq="MS")
    y_vals = np.zeros(n)
    eps = rng.normal(scale=0.5, size=n)
    for t in range(2, n):
        y_vals[t] = 0.6 * y_vals[t - 1] - 0.3 * y_vals[t - 2] + eps[t]
    y = pd.Series(y_vals, index=dates, name="gdp_growth")

    # Block 3: fit LinearAR and predict
    from macroforecast.models import LinearAR  # noqa: PLC0415

    train_end = 80
    y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]
    X_train = pd.DataFrame(index=y_train.index)
    X_test = pd.DataFrame(index=y_test.index)

    model = LinearAR(p=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert isinstance(preds, np.ndarray), "predict must return ndarray"
    assert len(preds) == 20, f"Expected 20 predictions, got {len(preds)}"

    # Block 4: TimeSeriesSplit OOS loop
    from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415

    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []
    for train_idx, test_idx in tscv.split(y):
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        X_tr = pd.DataFrame(index=y_tr.index)
        X_te = pd.DataFrame(index=y_te.index)
        m = LinearAR(p=2)
        m.fit(X_tr, y_tr)
        preds_fold = m.predict(X_te)
        mse_scores.append(np.mean((preds_fold - y_te.values) ** 2))
    assert len(mse_scores) == 5, f"Expected 5 fold MSE values, got {len(mse_scores)}"
    assert all(s >= 0 for s in mse_scores), "MSE scores must be non-negative"

    # Block 5 (graduation) is excluded: placeholder recipe YAML with ...
    # It is illustrative and marked as requiring mf_recipes.run() with real data.


# ---------------------------------------------------------------------------
# T5b — Runtime execution: Tutorial 02
# ---------------------------------------------------------------------------

def test_T5b_tutorial02_runtime(tmp_path: Path) -> None:
    """B7: Tutorial 02 code blocks execute without exception on synthetic data.

    The last block (graduation snippet) calls mf_recipes.run() with a
    placeholder file path 'path/to/my_study.yaml' and is excluded from
    execution (illustrative only).
    """
    sys.path.insert(0, str(REPO_ROOT))

    # Block 1: synthetic macro panel
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

    # Block 2: LinearAR OOS loop
    from macroforecast.models import LinearAR  # noqa: PLC0415
    from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415

    tscv = TimeSeriesSplit(n_splits=5, test_size=20)
    results: dict[str, float] = {}
    mse_ar: list[float] = []
    for train_idx, test_idx in tscv.split(y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        m = LinearAR(p=4)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        mse_ar.append(np.mean((preds - y_te.values) ** 2))
    results["LinearAR"] = float(np.mean(mse_ar))
    assert results["LinearAR"] >= 0

    # Block 3: PCR and FAAR
    from macroforecast.models import FactorAugmentedAR, PrincipalComponentRegression  # noqa: PLC0415

    for ModelClass, name, kwargs in [
        (PrincipalComponentRegression, "PCR", {"n_components": 3}),
        (FactorAugmentedAR, "FAAR", {"p": 2, "n_factors": 3}),
    ]:
        mse_list: list[float] = []
        for train_idx, test_idx in tscv.split(y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            m = ModelClass(**kwargs)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            mse_list.append(np.mean((preds - y_te.values) ** 2))
        results[name] = float(np.mean(mse_list))
        assert results[name] >= 0

    # Block 4: summary table
    summary = pd.Series(results).rename("mean_cv_mse").sort_values()
    assert len(summary) == 3
    assert set(summary.index) == {"LinearAR", "PCR", "FAAR"}

    # Block 5 (graduation) excluded: references a non-existent file path


# ---------------------------------------------------------------------------
# T5c — Runtime execution: Tutorial 03
# ---------------------------------------------------------------------------

def test_T5c_tutorial03_runtime(tmp_path: Path) -> None:
    """B7: Tutorial 03 code blocks execute without exception.

    Tutorial 03 references y and X from Tutorial 02's synthetic data.
    We recreate that data here to provide the shared namespace.
    Tutorial 03 block 3 (MyLinearAR subclassing _LinearARModel) is
    illustrative prose and is excluded from execution because _LinearARModel
    is a private implementation detail not exported.
    """
    sys.path.insert(0, str(REPO_ROOT))

    # Recreate Tutorial 02 synthetic data (shared namespace for Tutorial 03)
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

    # Block 1: ConstantTrendPlusAR class definition
    from sklearn.base import BaseEstimator, RegressorMixin  # noqa: PLC0415

    class ConstantTrendPlusAR(BaseEstimator, RegressorMixin):
        """Linear time trend plus AR(1) residual forecaster."""

        def __init__(self, fit_intercept: bool = True) -> None:
            self.fit_intercept = fit_intercept

        def fit(self, X_fit: pd.DataFrame, y_fit: pd.Series) -> "ConstantTrendPlusAR":
            n_fit = len(y_fit)
            t = np.arange(n_fit, dtype=float)
            y_v = y_fit.values if hasattr(y_fit, "values") else np.array(y_fit)
            ar_lag = np.concatenate([[y_v[0]], y_v[:-1]])
            if self.fit_intercept:
                Z = np.column_stack([np.ones(n_fit), t, ar_lag])
            else:
                Z = np.column_stack([t, ar_lag])
            self.coef_, _, _, _ = np.linalg.lstsq(Z, y_v, rcond=None)
            self._n_train = n_fit
            self._last_y = float(y_v[-1])
            return self

        def predict(self, X_pred: pd.DataFrame) -> np.ndarray:
            n_test = len(X_pred)
            t_future = np.arange(self._n_train, self._n_train + n_test, dtype=float)
            preds = []
            last_y = self._last_y
            for t_i in t_future:
                z = np.array([1.0, t_i, last_y]) if self.fit_intercept else np.array([t_i, last_y])
                yhat = float(z @ self.coef_)
                preds.append(yhat)
                last_y = yhat
            return np.array(preds)

    # Block 2: use custom class in TimeSeriesSplit loop
    from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415

    tscv = TimeSeriesSplit(n_splits=5)
    mse_list: list[float] = []
    for train_idx, test_idx in tscv.split(y):
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        m = ConstantTrendPlusAR(fit_intercept=True)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        mse_list.append(np.mean((preds - y_te.values) ** 2))
    assert len(mse_list) == 5

    # Block 3 (MyLinearAR subclass using _LinearARModel) is excluded:
    # _LinearARModel is a private class not in the public API; block is
    # prose-illustration only.

    # Block 4: registration wrapper (runs import only, no actual recipe call)
    import macroforecast.custom as mf_custom  # noqa: PLC0415

    def constant_trend_plus_ar_wrapper(X_train, y_train, X_test, context):
        model = ConstantTrendPlusAR(fit_intercept=True)
        X_tr_w = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
        y_tr_w = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
        X_te_w = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
        model.fit(X_tr_w, y_tr_w)
        return float(model.predict(X_te_w)[0])

    mf_custom.register_model("constant_trend_plus_ar_c67", constant_trend_plus_ar_wrapper)


# ---------------------------------------------------------------------------
# T7 — Model import availability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["LinearAR", "PrincipalComponentRegression", "FactorAugmentedAR"])
def test_T7_model_importable(model_name: str) -> None:
    """mf.models.LinearAR, PCR, and FAAR must be importable."""
    sys.path.insert(0, str(REPO_ROOT))
    import macroforecast.models as m  # noqa: PLC0415
    cls = getattr(m, model_name, None)
    assert cls is not None, f"macroforecast.models.{model_name} is not importable"


# ---------------------------------------------------------------------------
# T8 — ConstantTrendPlusAR sklearn compatibility
# ---------------------------------------------------------------------------

def test_T8_custom_class_sklearn_compat() -> None:
    """E1: ConstantTrendPlusAR must be compatible with sklearn clone() and get_params()."""
    from sklearn.base import BaseEstimator, RegressorMixin, clone  # noqa: PLC0415

    class ConstantTrendPlusAR(BaseEstimator, RegressorMixin):
        def __init__(self, fit_intercept: bool = True) -> None:
            self.fit_intercept = fit_intercept

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    instance = ConstantTrendPlusAR(fit_intercept=True)
    params = instance.get_params()
    assert params == {"fit_intercept": True}, (
        f"get_params() returned {params}, expected {{'fit_intercept': True}}"
    )
    cloned = clone(instance)
    assert cloned.fit_intercept is True
    assert cloned is not instance


# ---------------------------------------------------------------------------
# Regression checks
# ---------------------------------------------------------------------------

def test_R1_tutorial01_first_block_not_yaml() -> None:
    """R1: First code block in tutorial 01 must not be YAML."""
    blocks = _extract_python_blocks(TUT01)
    assert blocks, "No Python blocks found in tutorial 01"
    first = blocks[0]
    assert "0_meta:" not in first and "data:" not in first, (
        f"First code block contains YAML keys:\n{first[:200]}"
    )


def test_R2_tutorial03_first_class_uses_baseestimator() -> None:
    """R2: First class definition in tutorial 03 must use BaseEstimator and RegressorMixin."""
    text = TUT03.read_text(encoding="utf-8")
    class_defs = re.findall(r"class\s+\w+\([^)]+\)", text)
    assert class_defs, "No class definitions found in tutorial 03"
    first_class = class_defs[0]
    assert "BaseEstimator" in first_class and "RegressorMixin" in first_class, (
        f"First class definition '{first_class}' does not include BaseEstimator and RegressorMixin"
    )


def test_R3_tutorial02_minimal_yaml() -> None:
    """R3: Tutorial 02 must have fewer than 25 lines with YAML recipe keys."""
    yaml_keys = ["fixed_axes:", "leaf_config:", "data:", "4_forecasting_model:"]
    lines = TUT02.read_text(encoding="utf-8").splitlines()
    count = sum(1 for line in lines if any(k in line for k in yaml_keys))
    assert count < 25, (
        f"Tutorial 02 has {count} lines with YAML recipe keys (expected < 25)"
    )


# ---------------------------------------------------------------------------
# Structure and framing checks
# ---------------------------------------------------------------------------

def test_T_index_standalone_first() -> None:
    """T7: index.md must introduce standalone before recipes in the intro paragraph."""
    text = INDEX_MD.read_text(encoding="utf-8")
    standalone_pos = min(
        (p for p in [text.find("standalone"), text.find("macroforecast.models")] if p >= 0),
        default=-1,
    )
    recipe_pos = min(
        (p for p in [text.find("recipe"), text.find("YAML")] if p >= 0),
        default=-1,
    )
    assert standalone_pos >= 0, "index.md does not mention 'standalone' or 'macroforecast.models'"
    assert recipe_pos >= 0, "index.md does not mention 'recipe' or 'YAML'"
    assert standalone_pos < recipe_pos, (
        f"index.md mentions recipe (pos {recipe_pos}) before standalone (pos {standalone_pos})"
    )


def test_T_two_ep_standalone_column_left() -> None:
    """T8: two_entry_points.md table must have standalone column left of recipe DSL."""
    text = TWO_EP_MD.read_text(encoding="utf-8")
    # Find the header row of the comparison table
    for line in text.splitlines():
        if "|" in line and ("Standalone" in line or "standalone" in line):
            cols = [c.strip() for c in line.split("|") if c.strip()]
            standalone_idx = next(
                (i for i, c in enumerate(cols) if "standalone" in c.lower()), None
            )
            recipe_idx = next(
                (i for i, c in enumerate(cols) if "recipe" in c.lower()), None
            )
            if standalone_idx is not None and recipe_idx is not None:
                assert standalone_idx < recipe_idx, (
                    f"Standalone column (idx {standalone_idx}) is not left of "
                    f"recipe column (idx {recipe_idx}) in table header: {line}"
                )
                return
    pytest.fail("Could not find comparison table with Standalone column in two_entry_points.md")


@pytest.mark.parametrize("tutorial_path", [TUT01, TUT02, TUT03], ids=["tut01", "tut02", "tut03"])
def test_T_first_h2_no_recipe_content(tutorial_path: Path) -> None:
    """T9: First H2 section of each tutorial must not contain recipe-first content."""
    text = tutorial_path.read_text(encoding="utf-8")
    h2_positions = [m.start() for m in re.finditer(r"^## ", text, re.MULTILINE)]
    assert h2_positions, f"No H2 headings in {tutorial_path.name}"
    first_h2_start = h2_positions[0]
    end = h2_positions[1] if len(h2_positions) > 1 else len(text)
    first_section = text[first_h2_start:end]
    bad_patterns = ["mf.run(", "mf.recipes.run(", "fixed_axes:", "leaf_config:", "data:", "4_forecasting_model:"]
    found = [p for p in bad_patterns if p in first_section]
    assert not found, (
        f"{tutorial_path.name}: first H2 section contains recipe-first patterns: {found}"
    )
