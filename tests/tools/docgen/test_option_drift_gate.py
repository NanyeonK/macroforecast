"""Option drift gate test — audit-only flavor (PR5 / 2026-05-27).

Purpose
-------
Assert that the known drift counts from the 2026-05-27 encyclopedia option sync
audit do not grow.  This test does NOT enforce zero drift (that would block
forward progress); instead it asserts that the drift count is at or below the
baseline captured in the audit.  Any new drift discovered after this commit
will cause the test to fail, signalling that a follow-up docs PR is needed.

Drift is measured per axis/op-group.  The baseline was established by the
audit script in docs/_audit/encyclopedia-option-sync-2026-05-27.md.

Scope
-----
Only axes/groups that have per-option sub-page directories are checked.
Axes that only have a single ``axes/{axis}.md`` page are excluded.

The L6 equal_predictive_test and nested_test DOCS_ONLY items are excluded from
the gate because the code intentionally uses AxisSpec with ``options=()`` for
L6 (runtime-validated); those docs pages are correct.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# ─── Repo paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parents[3]
ENC_ROOT = REPO_ROOT / "docs" / "reference" / "encyclopedia"


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _docs_option_pages(layer: str, axis: str) -> set[str]:
    """Return the set of option file stems for docs/{layer}/{axis}/*.md"""
    path = ENC_ROOT / layer / axis
    if not path.is_dir():
        return set()
    return {f.stem for f in path.iterdir() if f.suffix == ".md" and f.name != "index.md"}


def _code_options_from_spec(layer_module: str, spec_var: str, axis_name: str) -> set[str]:
    """Import spec and return option values for one axis."""
    import importlib
    mod = importlib.import_module(layer_module)
    spec = getattr(mod, spec_var)
    for _sub_id, axes_dict in spec.axes.items():
        if axis_name in axes_dict:
            return {o.value for o in axes_dict[axis_name].options}
    return set()


# ─── Baseline drift table ─────────────────────────────────────────────────────
# Each entry: (layer_id, axis_or_group, code_only_baseline, docs_only_baseline)
# Baselines represent the MAXIMUM allowed drift as of 2026-05-27.
# If drift grows beyond baseline, the test fails.
# L6 equal_predictive_test / nested_test are excluded (intentional AxisSpec design).

_DRIFT_BASELINES: list[tuple[str, str, int, int]] = [
    # (layer, axis, max_code_only, max_docs_only)
    ("l2", "frame_edge_policy",           1, 0),  # keep_unbalanced
    ("l2", "imputation_policy",           1, 0),  # none_propagate
    ("l2", "monthly_to_quarterly_policy", 2, 0),  # quarterly_endpoint, quarterly_sum
    ("l2", "outlier_policy",              1, 0),  # none
    ("l2", "quarterly_to_monthly_policy", 3, 0),  # chow_lin, linear_interpolation, step_forward
    ("l2", "transform_policy",            2, 0),  # custom_tcode, no_transform
    ("l3", "op",                          1, 0),  # regime_indicator
    ("l4", "model",                       4, 0),  # bagging, decision_tree, macroeconomic_random_forest, quantile_regression_forest
    ("l5", "density_metrics",             2, 0),  # crps, log_score
    ("l7", "op",                         26, 0),  # 26 ops without pages (intentional FUTURE/non-priority)
]


# ─── L3/L4/L7 ops from code ────────────────────────────────────────────────────

def _l2_code_options(axis: str) -> set[str]:
    return _code_options_from_spec(
        "macroforecast.layers.l2_preprocessing.schema", "L2_LAYER_SPEC", axis
    )


def _l3_code_ops() -> set[str]:
    """L3 ops: all ops with l3 in scope, minus internal helpers."""
    # Import both universal (diff/log/lag etc.) and l3_features (all L3-specific ops)
    from macroforecast.core.ops import registry
    import macroforecast.core.ops.universal  # noqa: F401
    import macroforecast.layers.l3_features.ops  # noqa: F401
    _INTERNAL = {
        "kernel", "polynomial", "u_midas", "midas", "varimax_rotation",
        "target_construction", "l3_feature_bundle", "l3_metadata_build",
        "nystroem_features", "level",
    }
    all_ops = registry.list_ops()
    result = set()
    for name, op in all_ops.items():
        scope = op.layer_scope
        in_l3 = scope == "l3" or (isinstance(scope, (list, tuple)) and "l3" in scope)
        if in_l3 and name not in _INTERNAL:
            result.add(name)
    return result


def _l4_code_models() -> set[str]:
    from macroforecast.layers.l4_models.ops import OPERATIONAL_MODELS, FUTURE_MODELS
    return set(OPERATIONAL_MODELS) | set(FUTURE_MODELS)


def _l5_code_options(axis: str) -> set[str]:
    return _code_options_from_spec(
        "macroforecast.layers.l5_evaluation.schema", "L5_LAYER_SPEC", axis
    )


def _l7_code_ops() -> set[str]:
    """L7 ops from registry (layer_scope includes l7)."""
    from macroforecast.core.ops import registry
    import macroforecast.layers.l7_interpretation.schema  # noqa: F401
    all_ops = registry.list_ops()
    result = set()
    for name, op in all_ops.items():
        scope = op.layer_scope
        in_l7 = scope == "l7" or (isinstance(scope, (list, tuple)) and "l7" in scope)
        if in_l7:
            result.add(name)
    return result


def _get_code_options(layer: str, axis: str) -> set[str]:
    """Dispatch to the right getter."""
    if layer == "l2":
        return _l2_code_options(axis)
    if layer == "l3" and axis == "op":
        return _l3_code_ops()
    if layer == "l4" and axis == "model":
        return _l4_code_models()
    if layer == "l5":
        return _l5_code_options(axis)
    if layer == "l7" and axis == "op":
        return _l7_code_ops()
    raise ValueError(f"No getter for {layer}.{axis}")


# ─── Parametrized test ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "layer,axis,max_code_only,max_docs_only",
    _DRIFT_BASELINES,
    ids=[f"{l}.{a}" for (l, a, _, _) in _DRIFT_BASELINES],
)
def test_option_drift_does_not_exceed_baseline(
    layer: str, axis: str, max_code_only: int, max_docs_only: int
) -> None:
    """Drift count must not exceed the 2026-05-27 baseline.

    CODE_ONLY means a valid option in code has no docs sub-page.
    DOCS_ONLY means a docs sub-page exists for an option not in code.

    Either direction of new drift (growing CODE_ONLY or new DOCS_ONLY) fails
    this test.  Reducing drift (fixing docs or removing deprecated options)
    is always safe and will naturally lower the count below the baseline.
    """
    code_opts = _get_code_options(layer, axis)
    docs_opts = _docs_option_pages(layer, axis)

    code_only = code_opts - docs_opts
    docs_only = docs_opts - code_opts

    assert len(code_only) <= max_code_only, (
        f"[{layer}.{axis}] CODE_ONLY drift grew from baseline {max_code_only} "
        f"to {len(code_only)}.\n"
        f"New options without docs pages: {sorted(code_only - _KNOWN_CODE_ONLY.get((layer, axis), set()))}\n"
        f"Add encyclopedia sub-pages or update the baseline in this test."
    )
    assert len(docs_only) <= max_docs_only, (
        f"[{layer}.{axis}] DOCS_ONLY drift grew from baseline {max_docs_only} "
        f"to {len(docs_only)}.\n"
        f"Stale pages: {sorted(docs_only)}\n"
        f"Remove the stale pages or update the baseline in this test."
    )


# Known CODE_ONLY sets at baseline (for informative failure messages only).
_KNOWN_CODE_ONLY: dict[tuple[str, str], set[str]] = {
    ("l2", "frame_edge_policy"):           {"keep_unbalanced"},
    ("l2", "imputation_policy"):           {"none_propagate"},
    ("l2", "monthly_to_quarterly_policy"): {"quarterly_endpoint", "quarterly_sum"},
    ("l2", "outlier_policy"):              {"none"},
    ("l2", "quarterly_to_monthly_policy"): {"chow_lin", "linear_interpolation", "step_forward"},
    ("l2", "transform_policy"):            {"custom_tcode", "no_transform"},
    ("l3", "op"):                          {"regime_indicator"},
    ("l4", "model"):                       {"bagging", "decision_tree", "macroeconomic_random_forest", "quantile_regression_forest"},
    ("l5", "density_metrics"):             {"crps", "log_score"},
    ("l7", "op"):                          {
        "attention_weights", "bootstrap_jackknife", "bvar_pip", "cumulative_r2_contribution",
        "deep_lift", "dual_decomposition", "fevd", "forecast_decomposition",
        "friedman_h_interaction", "gradient_shap", "group_aggregate", "historical_decomposition",
        "integrated_gradients", "lasso_inclusion_frequency", "lineage_attribution", "lofo",
        "mrf_gtvp", "orthogonalised_irf", "oshapley_vi", "pbsv", "rolling_recompute",
        "saliency_map", "shap_deep", "shap_interaction", "shap_kernel", "transformation_attribution",
    },
}
