"""Issues #200 / #201 / #205 -- density tests + quantile path + 14 L7 figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macrocast.core import figures
from macrocast.core.runtime import (
    _density_interval_battery,
    _emit_quantile_intervals,
    _resolve_forecast_object,
)


# ---------------------------------------------------------------------------
# #201 quantile path
# ---------------------------------------------------------------------------

def test_resolve_forecast_object_picks_quantile_when_set():
    nodes = [{"op": "fit_model", "params": {"forecast_object": "quantile"}}]
    assert _resolve_forecast_object(nodes) == "quantile"


def test_resolve_forecast_object_defaults_to_point():
    nodes = [{"op": "fit_model", "params": {}}]
    assert _resolve_forecast_object(nodes) == "point"


def test_emit_quantile_intervals_produces_one_value_per_level():
    forecasts = {("m", "y", 1, "2018-01-01"): 5.0, ("m", "y", 1, "2018-02-01"): 6.0}
    nodes = [
        {
            "op": "fit_model",
            "params": {
                "forecast_object": "quantile",
                "quantile_levels": [0.1, 0.5, 0.9],
                "forecast_residual_std": 1.0,
            },
        }
    ]
    intervals = _emit_quantile_intervals(forecasts, nodes)
    assert len(intervals) == 6  # 2 origins × 3 levels
    # Median quantile == point forecast.
    for (model, target, horizon, origin, q), value in intervals.items():
        if q == 0.5:
            assert value == forecasts[(model, target, horizon, origin)]


# ---------------------------------------------------------------------------
# #200 density tests battery
# ---------------------------------------------------------------------------

def test_density_battery_does_not_reject_uniform_pit():
    rng = np.random.default_rng(0)
    pit = rng.uniform(size=200)
    result = _density_interval_battery(pit, alpha=0.05)
    assert result["ks"]["p_value"] > 0.01  # truly uniform PIT
    assert "berkowitz" in result and "kupiec_pof" in result and "christoffersen_independence" in result


def test_density_battery_rejects_off_uniform_pit():
    # Concentrated PIT distribution -- KS should reject.
    pit = np.linspace(0.6, 0.9, 200)
    result = _density_interval_battery(pit, alpha=0.05)
    assert result["ks"]["reject"] is True


# ---------------------------------------------------------------------------
# #205 figure renderers
# ---------------------------------------------------------------------------

def _imp_table(n: int = 5):
    return pd.DataFrame({"feature": [f"f{i}" for i in range(n)], "importance": np.random.RandomState(0).uniform(size=n)})


def test_beeswarm_writes_pdf(tmp_path):
    out = figures.render_beeswarm(_imp_table(), output_path=tmp_path / "beeswarm.pdf", title="Bees")
    assert out.exists()


def test_force_plot_writes_pdf(tmp_path):
    out = figures.render_force_plot(_imp_table(), output_path=tmp_path / "force.pdf")
    assert out.exists()


def test_shap_dependence_scatter_writes_pdf(tmp_path):
    out = figures.render_shap_dependence_scatter(_imp_table(), output_path=tmp_path / "dep.pdf")
    assert out.exists()


def test_ale_line_writes_pdf(tmp_path):
    table = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "importance": [0.5, 0.3],
            "ale_function": [
                [{"bin_center": 0.0, "ale": -0.1}, {"bin_center": 1.0, "ale": 0.2}],
                [{"bin_center": 0.0, "ale": 0.0}, {"bin_center": 1.0, "ale": 0.1}],
            ],
        }
    )
    out = figures.render_ale_line(table, output_path=tmp_path / "ale.pdf")
    assert out.exists()


def test_attribution_heatmap_writes_pdf(tmp_path):
    table = pd.DataFrame(
        {"feature": ["a", "b", "c"], "method_x": [1.0, 0.5, 0.2], "method_y": [0.7, 0.4, 0.1]}
    ).set_index("feature")
    out = figures.render_attribution_heatmap(table, output_path=tmp_path / "attr_hm.pdf")
    assert out.exists()


def test_shapley_waterfall_writes_pdf(tmp_path):
    table = pd.DataFrame({"pipeline": ["a", "b", "c"], "contribution": [0.5, -0.2, 0.3]})
    out = figures.render_shapley_waterfall(table, output_path=tmp_path / "wf.pdf")
    assert out.exists()


def test_feature_heatmap_over_time_writes_pdf(tmp_path):
    table = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "importance": [0.5, 0.3],
            "coefficient_path": [[0.1, 0.2, 0.3], [0.5, 0.4, 0.3]],
        }
    )
    out = figures.render_feature_heatmap_over_time(table, output_path=tmp_path / "ftime.pdf")
    assert out.exists()


def test_remaining_bar_renderers_write_pdf(tmp_path):
    table = _imp_table()
    for name in [
        "render_lasso_path_inclusion_order",
        "render_pip_bar",
        "render_historical_decomp_stacked_bar",
        "render_irf_with_confidence_band",
        "render_bar_grouped_by_pipeline",
        "render_importance_by_horizon_bar",
    ]:
        renderer = getattr(figures, name)
        out = renderer(table, output_path=tmp_path / f"{name}.pdf")
        assert out.exists(), name


def test_inclusion_heatmap_writes_pdf(tmp_path):
    # Heatmap renderer expects feature-indexed numeric columns; supply a
    # (model × feature) inclusion frequency layout matching the lasso
    # path inclusion data shape.
    table = pd.DataFrame(
        {"feature": ["a", "b", "c"], "lambda_0.1": [1.0, 0.0, 1.0], "lambda_1.0": [0.5, 0.0, 0.8]}
    ).set_index("feature")
    out = figures.render_inclusion_heatmap(table, output_path=tmp_path / "incl_hm.pdf")
    assert out.exists()
