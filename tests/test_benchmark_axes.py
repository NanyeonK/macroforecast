"""Phase 4 - registry sanity tests for benchmark_family/window/scope."""
from __future__ import annotations

from macrocast.registry.data.benchmark_family import AXIS_DEFINITION as BF
from macrocast.registry.evaluation.benchmark_window import AXIS_DEFINITION as BW
from macrocast.registry.evaluation.benchmark_scope import AXIS_DEFINITION as BS


def _by_id(definition):
    out = dict()
    for e in definition.entries:
        out[e.id] = e.status
    return out


def test_benchmark_family_layer_is_training():
    assert BF.layer == "3_training"


def test_benchmark_window_layer_is_evaluation():
    assert BW.layer == "4_evaluation"


def test_benchmark_scope_layer_is_evaluation():
    assert BS.layer == "4_evaluation"


def test_benchmark_family_operational_set():
    statuses = _by_id(BF)
    expected_operational = (
        "historical_mean",
        "zero_change",
        "autoregressive_bic",
        "custom_benchmark",
        "rolling_mean",
        "autoregressive_fixed_lag",
        "autoregressive_diffusion_index",
        "expert_benchmark",
        "factor_model_benchmark",
        "benchmark_suite",
        "paper_specific_benchmark",
        "survey_forecast",
    )
    for name in expected_operational:
        assert statuses.get(name) == "operational", f"{name} expected operational, got {statuses.get(name)}"


def test_benchmark_window_operational_set():
    statuses = _by_id(BW)
    for name in ("full_oos", "rolling", "expanding"):
        assert statuses.get(name) == "operational", f"{name} got {statuses.get(name)}"


def test_benchmark_scope_operational_set():
    statuses = _by_id(BS)
    for name in ("all_targets_horizons", "per_target_horizon"):
        assert statuses.get(name) == "operational", f"{name} got {statuses.get(name)}"
