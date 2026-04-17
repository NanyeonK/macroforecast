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


AXIS_NAMES: tuple[str, ...] = (
    "equal_predictive",
    "nested",
    "cpa_instability",
    "multiple_model",
    "density_interval",
    "direction",
    "residual_diagnostics",
    "test_scope",
)


LEGACY_TO_NEW: dict[str, tuple[str, str]] = {
    # equal_predictive (3 operational)
    "dm":          ("equal_predictive", "dm"),
    "dm_hln":      ("equal_predictive", "dm_hln"),
    "dm_modified": ("equal_predictive", "dm_modified"),
    # nested (4 operational)
    "cw":      ("nested", "cw"),
    "enc_new": ("nested", "enc_new"),
    "mse_f":   ("nested", "mse_f"),
    "mse_t":   ("nested", "mse_t"),
    # cpa_instability (3 operational)
    "cpa":        ("cpa_instability", "cpa"),
    "rossi":      ("cpa_instability", "rossi"),
    "rolling_dm": ("cpa_instability", "rolling_dm"),
    # multiple_model (3 operational)
    "reality_check": ("multiple_model", "reality_check"),
    "spa":           ("multiple_model", "spa"),
    "mcs":           ("multiple_model", "mcs"),
    # direction (2 operational)
    "pesaran_timmermann": ("direction", "pesaran_timmermann"),
    "binomial_hit":       ("direction", "binomial_hit"),
    # residual_diagnostics (5 operational)
    "mincer_zarnowitz": ("residual_diagnostics", "mincer_zarnowitz"),
    "ljung_box":        ("residual_diagnostics", "ljung_box"),
    "arch_lm":          ("residual_diagnostics", "arch_lm"),
    "bias_test":        ("residual_diagnostics", "bias_test"),
    "diagnostics_full": ("residual_diagnostics", "diagnostics_full"),
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
        _compute_reality_check,
        _compute_rolling_dm_test,
        _compute_rossi_test,
        _compute_spa,
    )

    block_bootstrap = dependence_correction == "block_bootstrap"

    return {
        "equal_predictive": {
            "dm":          lambda: _compute_dm_test(predictions),
            "dm_hln":      lambda: _compute_dm_hln_test(predictions, dependence_correction=dependence_correction),
            "dm_modified": lambda: _compute_dm_modified_test(predictions, dependence_correction=dependence_correction),
        },
        "nested": {
            "cw":      lambda: _compute_cw_test(predictions),
            "enc_new": lambda: _compute_enc_new_test(predictions, dependence_correction=dependence_correction),
            "mse_f":   lambda: _compute_mse_f_test(predictions),
            "mse_t":   lambda: _compute_mse_t_test(predictions, dependence_correction=dependence_correction),
        },
        "cpa_instability": {
            "cpa":        lambda: _compute_cpa_test(predictions, dependence_correction=dependence_correction),
            "rossi":      lambda: _compute_rossi_test(predictions),
            "rolling_dm": lambda: _compute_rolling_dm_test(predictions),
        },
        "multiple_model": {
            "reality_check": lambda: _compute_reality_check(predictions, block_bootstrap=block_bootstrap),
            "spa":           lambda: _compute_spa(predictions, block_bootstrap=block_bootstrap),
            "mcs":           lambda: _compute_mcs_test(predictions, block_bootstrap=block_bootstrap),
        },
        "direction": {
            "pesaran_timmermann": lambda: _compute_pesaran_timmermann(predictions),
            "binomial_hit":       lambda: _compute_binomial_hit_test(predictions),
        },
        "residual_diagnostics": {
            "mincer_zarnowitz": lambda: _compute_mincer_zarnowitz(predictions),
            "ljung_box":        lambda: _compute_ljung_box_test(predictions),
            "arch_lm":          lambda: _compute_arch_lm_test(predictions),
            "bias_test":        lambda: _compute_bias_test(predictions),
            "diagnostics_full": lambda: _compute_diagnostics_bundle(predictions),
        },
        # density_interval: all values status=planned in Phase 2; see Phase 10 section 10.8
        "density_interval": {},
        # test_scope is a meta-control axis (not a test); recorded via spec
        "test_scope": {},
    }


def _as_spec(raw_spec: dict[str, Any] | None) -> dict[str, str]:
    if raw_spec is None:
        return {}
    spec: dict[str, str] = {}
    for axis in AXIS_NAMES:
        value = raw_spec.get(axis)
        if value is None:
            continue
        spec[axis] = str(value)
    legacy = raw_spec.get("stat_test")
    if legacy is not None and legacy != "none":
        legacy_str = str(legacy)
        if legacy_str in LEGACY_TO_NEW:
            axis, value = LEGACY_TO_NEW[legacy_str]
            spec.setdefault(axis, value)
    return spec


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
            ``"none"`` axes are skipped. Legacy ``stat_test`` key is
            auto-routed via LEGACY_TO_NEW for safety.
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
    if not spec:
        return {}

    handlers = _build_handlers(predictions, dependence_correction)
    out: dict[str, dict[str, Any]] = {}

    for axis in AXIS_NAMES:
        value = spec.get(axis)
        if value is None or value == "none":
            continue
        if axis == "test_scope":
            # Meta-control axis: record the scope selection but do not run
            # a test implementation (scope influences orchestration only).
            out[axis] = {"axis": axis, "scope": value}
            continue
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
