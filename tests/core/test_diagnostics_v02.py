"""Issues #211 / #212 / #213 / #214 -- diagnostic visualisations + multi-format export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.figures import (
    render_factor_timeseries,
    render_fitted_vs_actual,
    render_rolling_loss,
    render_scree_plot,
)


# ---------------------------------------------------------------------------
# #211 / #212 figure renderers
# ---------------------------------------------------------------------------

def test_scree_plot_writes_pdf(tmp_path):
    out = render_scree_plot([4.5, 2.1, 1.2, 0.8, 0.4], output_path=tmp_path / "scree.pdf", title="Scree")
    assert out.exists()
    assert out.stat().st_size > 0


def test_factor_timeseries_writes_pdf(tmp_path):
    idx = pd.date_range("2010-01-01", periods=24, freq="MS")
    factors = pd.DataFrame(np.random.RandomState(0).normal(size=(24, 2)), index=idx, columns=["F1", "F2"])
    out = render_factor_timeseries(factors, output_path=tmp_path / "factor_ts.pdf", title="Factors")
    assert out.exists()


def test_fitted_vs_actual_writes_pdf(tmp_path):
    rng = np.random.default_rng(0)
    actual = pd.Series(rng.normal(size=30))
    fitted = actual + rng.normal(scale=0.1, size=30)
    out = render_fitted_vs_actual(fitted, actual, output_path=tmp_path / "fva.pdf", title="Fit")
    assert out.exists()


def test_rolling_loss_writes_pdf(tmp_path):
    losses = pd.Series([0.5, 0.45, 0.4, 0.38, 0.36], index=range(5))
    out = render_rolling_loss(losses, output_path=tmp_path / "loss.pdf", title="Rolling loss")
    assert out.exists()


# ---------------------------------------------------------------------------
# #213 L2.5 enrichments
# ---------------------------------------------------------------------------

def test_l2_5_delta_matrix_when_axis_set():
    import macroforecast

    recipe = """
    0_meta:
      fixed_axes:
        failure_policy: fail_fast
    1_data:
      fixed_axes:
        custom_source_policy: custom_panel_only
        frequency: monthly
        horizon_set: custom_list
      leaf_config:
        target: y
        target_horizons: [1]
        custom_panel_inline:
          date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01, 2018-07-01, 2018-08-01]
          y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
          x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    2_preprocessing:
      fixed_axes: {transform_policy: no_transform, outlier_policy: none, imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}
    2_5_pre_post_preprocessing:
      enabled: true
      fixed_axes:
        correlation_shift: delta_matrix
    """
    result = macroforecast.core.runtime.execute_l1_l2(recipe)
    diag = result.artifacts.get("l2_5_diagnostic_v1")
    assert diag is not None
    assert diag.enabled
    assert "delta_matrix" in diag.metadata
