"""PR4: MCS naming clarification + SPA benchmark guard tests.

Covers:
1. Docstring must explicitly say "single-step" (not claim iterative elimination).
2. UserWarning emitted and NaN returned when spa_benchmark_model is missing.
3. Valid p-value in [0, 1] returned when spa_benchmark_model is explicitly provided.
4. StepM iterative elimination still works correctly (regression).
"""
from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pandas as pd

from macroforecast.core.runtime import _mcs_from_per_origin_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_panel(
    model_ids: list[str],
    loss_means: list[float],
    n_obs: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for mid, mu in zip(model_ids, loss_means):
        losses = rng.standard_normal(n_obs) ** 2 + mu
        for i, loss in enumerate(losses):
            records.append(
                {
                    "target": "y",
                    "horizon": 1,
                    "origin": i,
                    "model_id": mid,
                    "squared_error": float(loss),
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Test 1: MCS docstring accuracy
# ---------------------------------------------------------------------------

def test_mcs_docstring_says_single_step():
    """Docstring must explicitly state that mcs_inclusion uses the single-step
    max-test, not the full iterative Hansen-Lunde-Nason MCS elimination."""
    docstring = inspect.getdoc(_mcs_from_per_origin_panel)
    assert docstring is not None, "Function must have a docstring"

    doc_lower = docstring.lower()
    assert "single-step" in doc_lower, (
        "Docstring must state 'single-step' to clarify this is not full "
        "iterative Hansen MCS elimination."
    )
    # Must either not mention elimination at all, or qualify it with 'not'
    if "elimination" in doc_lower:
        # Ensure the mention of elimination is preceded by negation context
        assert "not" in doc_lower, (
            "If 'elimination' appears in the docstring, it must be accompanied "
            "by a negation ('not') to avoid false claims of iterative MCS."
        )


# ---------------------------------------------------------------------------
# Test 2: SPA warns when benchmark is missing
# ---------------------------------------------------------------------------

def test_spa_warns_without_benchmark():
    """UserWarning must be emitted and spa_p_values must be NaN when
    spa_benchmark_model is not in sub."""
    panel = _build_panel(
        model_ids=["model_a", "model_b", "model_c"],
        loss_means=[1.0, 1.0, 1.0],
        n_obs=20,
        seed=42,
    )
    sub = {"mcs_alpha": 0.10, "bootstrap_n_replications": 100}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _mcs_from_per_origin_panel(panel, sub)

    spa_warnings = [
        x for x in caught
        if issubclass(x.category, UserWarning)
        and (
            "spa" in str(x.message).lower()
            or "benchmark" in str(x.message).lower()
        )
    ]
    assert len(spa_warnings) >= 1, (
        "Expected at least one UserWarning about missing spa_benchmark_model, "
        f"got {len(caught)} warnings: {[str(w.message) for w in caught]}"
    )

    # All spa_p_values must be NaN when benchmark unspecified
    for key, val in result["spa_p_values"].items():
        assert math.isnan(val), (
            f"spa_p_values[{key}] should be NaN when benchmark unspecified, got {val}"
        )

    # reality_check_p_values must also be NaN
    for key, val in result["reality_check_p_values"].items():
        assert math.isnan(val), (
            f"reality_check_p_values[{key}] should be NaN when benchmark unspecified, "
            f"got {val}"
        )


# ---------------------------------------------------------------------------
# Test 3: SPA accepts explicit benchmark
# ---------------------------------------------------------------------------

def test_spa_accepts_explicit_benchmark():
    """When spa_benchmark_model is provided, spa_p_values should be real
    numbers in [0, 1] and no benchmark-related warning should be emitted."""
    panel = _build_panel(
        model_ids=["ar_benchmark", "ridge", "lasso"],
        loss_means=[0.5, 1.0, 1.5],   # ar_benchmark has lowest loss
        n_obs=40,
        seed=0,
    )
    sub = {
        "mcs_alpha": 0.10,
        "bootstrap_n_replications": 200,
        "spa_benchmark_model": "ar_benchmark",
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _mcs_from_per_origin_panel(panel, sub)

    benchmark_warnings = [
        x for x in caught
        if issubclass(x.category, UserWarning)
        and "benchmark" in str(x.message).lower()
    ]
    assert len(benchmark_warnings) == 0, (
        "No benchmark warning expected when spa_benchmark_model is explicitly "
        f"provided, got: {[str(w.message) for w in benchmark_warnings]}"
    )

    for key, val in result["spa_p_values"].items():
        assert not math.isnan(val), (
            f"spa_p_values[{key}] should not be NaN when benchmark is provided"
        )
        assert 0.0 <= val <= 1.0, (
            f"spa_p_values[{key}] = {val} is out of [0, 1]"
        )


# ---------------------------------------------------------------------------
# Test 4: StepM iterative elimination regression
# ---------------------------------------------------------------------------

def test_mcs_stepm_uses_iterative_elimination():
    """StepM (Romano-Wolf) iterative step-down should reject the clearly worst
    model. This is a regression check to confirm the StepM path is unaffected
    by the SPA benchmark guard."""
    rng = np.random.default_rng(42)
    n_obs = 50
    model_specs = [("good", 0.5), ("ok", 1.0), ("terrible", 5.0)]
    records = []
    for mid, base in model_specs:
        for t in range(n_obs):
            records.append(
                {
                    "target": "y",
                    "horizon": 1,
                    "origin": t,
                    "model_id": mid,
                    "squared_error": float(abs(rng.standard_normal()) * base),
                }
            )
    panel = pd.DataFrame(records)

    result = _mcs_from_per_origin_panel(
        panel,
        {
            "mcs_alpha": 0.10,
            "bootstrap_n_replications": 500,
            "spa_benchmark_model": "good",
        },
    )

    key = ("y", 1, 0.10)
    rejected = result["stepm_rejected"][key]
    assert "terrible" in rejected, (
        f"Expected 'terrible' in stepm_rejected, got {rejected}"
    )
    assert "good" not in rejected, (
        "The best model 'good' should never be rejected by StepM."
    )
