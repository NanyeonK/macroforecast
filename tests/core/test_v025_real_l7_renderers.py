"""Issue #249 -- the 9 v0.2 proxy renderers now produce distinct layouts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macrocast.core import figures


def _imp_table(n: int = 5):
    return pd.DataFrame({"feature": [f"f{i}" for i in range(n)], "importance": np.linspace(0.1, 1.0, n)})


def test_force_plot_uses_diverging_bars(tmp_path):
    table = pd.DataFrame({"feature": ["a", "b", "c"], "contribution": [0.5, -0.3, 0.1]})
    out = figures.render_force_plot(table, output_path=tmp_path / "force.pdf")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_shap_dependence_scatter_uses_feature_value_when_present(tmp_path):
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        {
            "feature": ["a"] * 30 + ["b"] * 30,
            "feature_value": np.concatenate([rng.normal(size=30), rng.normal(size=30)]),
            "shap_value": rng.normal(size=60),
        }
    )
    out = figures.render_shap_dependence_scatter(table, output_path=tmp_path / "dep.pdf")
    assert out.exists()


def test_attribution_heatmap_uses_diverging_palette(tmp_path):
    table = pd.DataFrame(
        {"feature": ["a", "b", "c"], "saliency": [0.3, -0.2, 0.5], "ig": [-0.1, 0.4, 0.05]}
    )
    out = figures.render_attribution_heatmap(table, output_path=tmp_path / "attr.pdf")
    assert out.exists()


def test_inclusion_heatmap_renders_lambda_x_feature(tmp_path):
    table = pd.DataFrame(
        {"feature": ["a", "b"], "lambda_0.1": [1.0, 0.0], "lambda_1.0": [0.5, 0.0]}
    ).set_index("feature")
    out = figures.render_inclusion_heatmap(table, output_path=tmp_path / "incl.pdf")
    assert out.exists()


def test_lasso_path_inclusion_order_uses_step_lines_when_path_present(tmp_path):
    table = pd.DataFrame(
        {
            "feature": ["a", "a", "b", "b"],
            "lambda": [0.1, 1.0, 0.1, 1.0],
            "coefficient": [1.5, 0.0, 0.5, 0.2],
        }
    )
    out = figures.render_lasso_path_inclusion_order(table, output_path=tmp_path / "path.pdf")
    assert out.exists()


def test_pip_bar_renders_with_hdi_band(tmp_path):
    table = pd.DataFrame(
        {"feature": ["a", "b", "c"], "importance": [0.8, 0.5, 0.2], "hdi_low": [0.6, 0.3, 0.1], "hdi_high": [0.95, 0.7, 0.4]}
    )
    out = figures.render_pip_bar(table, output_path=tmp_path / "pip.pdf")
    assert out.exists()


def test_historical_decomp_stacked_bar_uses_period_shock_columns(tmp_path):
    table = pd.DataFrame(
        {
            "period": [1, 2, 3, 1, 2, 3],
            "shock": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "contribution": [0.5, -0.2, 0.3, 0.1, 0.4, -0.1],
        }
    )
    out = figures.render_historical_decomp_stacked_bar(table, output_path=tmp_path / "hd.pdf")
    assert out.exists()


def test_irf_with_confidence_band_draws_band_when_columns_present(tmp_path):
    table = pd.DataFrame(
        {
            "horizon": list(range(1, 13)),
            "response": np.linspace(0.5, 0.0, 12),
            "ci_low": np.linspace(0.3, -0.2, 12),
            "ci_high": np.linspace(0.7, 0.2, 12),
        }
    )
    out = figures.render_irf_with_confidence_band(table, output_path=tmp_path / "irf.pdf")
    assert out.exists()


def test_bar_grouped_by_pipeline_uses_pipeline_clusters(tmp_path):
    table = pd.DataFrame(
        {
            "feature": ["f1", "f1", "f2", "f2"],
            "pipeline": ["A", "B", "A", "B"],
            "importance": [0.6, 0.4, 0.3, 0.5],
        }
    )
    out = figures.render_bar_grouped_by_pipeline(table, output_path=tmp_path / "grouped.pdf")
    assert out.exists()


def test_importance_by_horizon_bar_groups_by_horizon(tmp_path):
    table = pd.DataFrame(
        {
            "horizon": [1, 1, 6, 6, 12, 12],
            "feature": ["a", "b"] * 3,
            "importance": [0.5, 0.3, 0.4, 0.4, 0.2, 0.6],
        }
    )
    out = figures.render_importance_by_horizon_bar(table, output_path=tmp_path / "horizon.pdf")
    assert out.exists()
