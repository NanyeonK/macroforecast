"""Dispatcher for the 8-axis statistical-test layer (Phase 2).

Translates a per-axis spec into calls into the existing per-test functions
that live in ``macrocast.execution.build``. Per-axis errors are surfaced
in the result dict rather than raising, so one broken test does not abort
the whole study.

See plans/phases/phase_02_stat_test_split.md section 4.2.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from macrocast.registry.naming import canonical_axis_value


STAT_TEST_AXIS_NAMES: tuple[str, ...] = (
    "equal_predictive",
    "nested",
    "cpa_instability",
    "multiple_model",
    "density_interval",
    "direction",
    "residual_diagnostics",
)

META_AXIS_NAMES: tuple[str, ...] = (
    "test_scope",
)

AXIS_NAMES: tuple[str, ...] = STAT_TEST_AXIS_NAMES + META_AXIS_NAMES

DEFAULT_STAT_TEST_SPEC: dict[str, str] = {
    "equal_predictive": "none",
    "nested": "none",
    "cpa_instability": "none",
    "multiple_model": "none",
    "density_interval": "none",
    "direction": "none",
    "residual_diagnostics": "none",
    "test_scope": "per_target",
    "dependence_correction": "none",
    "overlap_handling": "allow_overlap",
}


def canonicalize_stat_test_spec(raw_spec: dict[str, Any] | None) -> dict[str, str]:
    """Return the Layer 6 split-axis spec with defaults."""

    spec = dict(DEFAULT_STAT_TEST_SPEC)
    if raw_spec:
        for key in spec:
            if key in raw_spec:
                spec[key] = canonical_axis_value(key, str(raw_spec[key]))
    return spec


def active_stat_test_axes(raw_spec: dict[str, Any] | None) -> dict[str, str]:
    """Return selected executable test-family axes, excluding meta controls."""

    spec = canonicalize_stat_test_spec(raw_spec)
    return {
        axis: spec[axis]
        for axis in STAT_TEST_AXIS_NAMES
        if spec.get(axis, "none") != "none"
    }


def _build_handlers(
    predictions: pd.DataFrame, dependence_correction: str
) -> dict[str, dict[str, Callable[[], dict[str, Any]]]]:
    from macrocast.execution.build import (
        _compute_arch_lm_test,
        _compute_bias_test,
        _compute_binomial_hit_test,
        _compute_cpa_test,
        _compute_cw_test,
        _compute_diagnostics_bundle,
        _compute_dm_hln_test,
        _compute_dm_modified_test,
        _compute_dm_test,
        _compute_enc_new_test,
        _compute_ljung_box_test,
        _compute_mcs_test,
        _compute_mincer_zarnowitz,
        _compute_mse_f_test,
        _compute_mse_t_test,
        _compute_pesaran_timmermann,
        _compute_autocorrelation_of_errors,
        _compute_bootstrap_best_model,
        _compute_berkowitz,
        _compute_chow_break_forecast,
        _compute_christoffersen_conditional,
        _compute_christoffersen_independence,
        _compute_christoffersen_unconditional,
        _compute_cusum_on_loss,
        _compute_fluctuation_test,
        _compute_interval_coverage,
        _compute_kupiec,
        _compute_pit_uniformity,
        _compute_forecast_encompassing_nested,
        _compute_mcnemar_direction,
        _compute_roc_comparison,
        _compute_stepwise_mcs,
        _compute_paired_t_on_loss_diff,
        _compute_reality_check,
        _compute_serial_dependence_loss_diff,
        _compute_rolling_dm_test,
        _compute_rossi_test,
        _compute_spa,
        _compute_wilcoxon_signed_rank,
    )

    block_bootstrap = dependence_correction == "block_bootstrap"

    return {
        "equal_predictive": {
            "dm":                    lambda: _compute_dm_test(predictions),
            "dm_hln":                lambda: _compute_dm_hln_test(predictions, dependence_correction=dependence_correction),
            "dm_modified":           lambda: _compute_dm_modified_test(predictions, dependence_correction=dependence_correction),
            "paired_t_on_loss_diff": lambda: _compute_paired_t_on_loss_diff(predictions),
            "wilcoxon_signed_rank":  lambda: _compute_wilcoxon_signed_rank(predictions),
        },
        "nested": {
            "cw":                            lambda: _compute_cw_test(predictions),
            "enc_new":                       lambda: _compute_enc_new_test(predictions, dependence_correction=dependence_correction),
            "mse_f":                         lambda: _compute_mse_f_test(predictions),
            "mse_t":                         lambda: _compute_mse_t_test(predictions, dependence_correction=dependence_correction),
            "forecast_encompassing_nested":  lambda: _compute_forecast_encompassing_nested(predictions),
        },
        "cpa_instability": {
            "cpa":                lambda: _compute_cpa_test(predictions, dependence_correction=dependence_correction),
            "rossi":              lambda: _compute_rossi_test(predictions),
            "rolling_dm":         lambda: _compute_rolling_dm_test(predictions),
            "cusum_on_loss":      lambda: _compute_cusum_on_loss(predictions),
            "fluctuation_test":   lambda: _compute_fluctuation_test(predictions),
            "chow_break_forecast": lambda: _compute_chow_break_forecast(predictions),
        },
        "multiple_model": {
            "reality_check":         lambda: _compute_reality_check(predictions, block_bootstrap=block_bootstrap),
            "spa":                   lambda: _compute_spa(predictions, block_bootstrap=block_bootstrap),
            "mcs":                   lambda: _compute_mcs_test(predictions, block_bootstrap=block_bootstrap),
            "stepwise_mcs":          lambda: _compute_stepwise_mcs(predictions),
            "bootstrap_best_model":  lambda: _compute_bootstrap_best_model(predictions),
        },
        "direction": {
            "pesaran_timmermann": lambda: _compute_pesaran_timmermann(predictions),
            "binomial_hit":       lambda: _compute_binomial_hit_test(predictions),
            "mcnemar":            lambda: _compute_mcnemar_direction(predictions),
            "roc_comparison":     lambda: _compute_roc_comparison(predictions),
        },
        "residual_diagnostics": {
            "mincer_zarnowitz":              lambda: _compute_mincer_zarnowitz(predictions),
            "ljung_box":                     lambda: _compute_ljung_box_test(predictions),
            "arch_lm":                       lambda: _compute_arch_lm_test(predictions),
            "bias_test":                     lambda: _compute_bias_test(predictions),
            "full_residual_diagnostics":     lambda: _compute_diagnostics_bundle(predictions),
            "autocorrelation_of_errors":     lambda: _compute_autocorrelation_of_errors(predictions),
            "serial_dependence_loss_diff":   lambda: _compute_serial_dependence_loss_diff(predictions),
        },
        "density_interval": {
            "pit_uniformity":              lambda: _compute_pit_uniformity(predictions),
            "berkowitz":                   lambda: _compute_berkowitz(predictions),
            "kupiec":                      lambda: _compute_kupiec(predictions),
            "christoffersen_unconditional": lambda: _compute_christoffersen_unconditional(predictions),
            "christoffersen_independence":  lambda: _compute_christoffersen_independence(predictions),
            "christoffersen_conditional":   lambda: _compute_christoffersen_conditional(predictions),
            "interval_coverage":            lambda: _compute_interval_coverage(predictions),
        },
        # test_scope is a meta-control axis (not a test); recorded via spec
        "test_scope": {},
    }


def _as_spec(raw_spec: dict[str, Any] | None) -> dict[str, str]:
    return canonicalize_stat_test_spec(raw_spec)


def dispatch_stat_tests(
    *,
    predictions: pd.DataFrame,
    stat_test_spec: dict[str, Any] | None,
    dependence_correction: str,
) -> dict[str, dict[str, Any]]:
    """Run the requested stat tests per axis; return a per-axis result dict.

    Args:
        predictions: long-form predictions table (see
            macrocast.execution.build._build_predictions).
        stat_test_spec: dict of layer-6 axis selections. Missing or
            ``"none"`` axes are skipped.
        dependence_correction: HAC / Newey-West / bootstrap window identifier
            passed through to the individual test functions.

    Returns:
        Dict keyed by axis name. Each value is the test's native result dict
        (keys include ``stat_test``, ``statistic``, ``p_value``, ``n``) plus
        an ``axis`` key. Failed tests surface as
        ``{"error": str, "exc_type": str, "stat_test": value, "axis": axis}``.
        Skipped axes do not appear in the output.
    """

    spec = _as_spec(stat_test_spec)
    active_axes = active_stat_test_axes(spec)
    if not active_axes:
        raw_keys = set(stat_test_spec or {})
        if raw_keys == {"test_scope"}:
            return {"test_scope": {"axis": "test_scope", "scope": spec["test_scope"]}}
        return {}

    handlers = _build_handlers(predictions, dependence_correction)
    out: dict[str, dict[str, Any]] = {}

    for axis, value in active_axes.items():
        axis_handlers = handlers.get(axis, {})
        if value not in axis_handlers:
            out[axis] = {
                "axis": axis,
                "stat_test": value,
                "error": f"stat test {value!r} on axis {axis!r} is not "
                         f"operational in v0.4 (see Phase 10 section 10.8)",
                "exc_type": "NotImplementedError",
            }
            continue
        try:
            payload = dict(axis_handlers[value]())
            payload["axis"] = axis
            out[axis] = payload
        except Exception as exc:
            out[axis] = {
                "axis": axis,
                "stat_test": value,
                "error": str(exc),
                "exc_type": type(exc).__name__,
            }

    return out
