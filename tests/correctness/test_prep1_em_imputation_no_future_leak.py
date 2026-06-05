"""Regression tests for PREP-1 and PREP-2.

Under the default 'origin_available' preprocessing scope, EM-factor imputation
(PREP-1) and IQR outlier statistics (PREP-2) must be estimated using only rows
available at the forecast origin. The runner appends the h-step target
realization row (a post-origin row) to the apply panel; that future row must not
change the imputed/outlier-flagged feature values of any origin-available row.

The runner passes the origin-available labels via transform(..., available=...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macroforecast.preprocessing.specs import preprocess_spec


def _panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 49
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    f = rng.normal(size=(n, 2))
    load = rng.normal(size=(2, 6))
    panel = pd.DataFrame(f @ load + 0.1 * rng.normal(size=(n, 6)),
                         columns=[f"s{i}" for i in range(6)], index=idx)
    panel.iloc[10, 0] = np.nan
    panel.iloc[20, 2] = np.nan
    panel.iloc[30, 4] = np.nan
    return panel


def _transform_available(spec, offset: float) -> pd.DataFrame:
    panel = _panel().copy()
    available_idx = panel.index[:48]   # rows <= origin
    future_idx = panel.index[48:]      # post-origin target realization row
    panel.loc[future_idx] = panel.loc[future_idx] + offset
    fit_panel = panel.loc[available_idx]
    fitted = spec.fit(fit_panel, policy="origin_available")
    out = fitted.transform(panel, history=fit_panel, available=available_idx)
    return out.panel.reindex(available_idx)


def test_prep1_em_imputation_ignores_future_row():
    spec = preprocess_spec(
        transform="none", outliers="none", impute="em_factor",
        em_n_factors=2, frame="keep",
    )
    base = _transform_available(spec, offset=0.0)
    perturbed = _transform_available(spec, offset=4.0)
    pd.testing.assert_frame_equal(base, perturbed, rtol=1e-9, atol=1e-9)


def test_prep2_iqr_outlier_stats_ignore_future_row():
    spec = preprocess_spec(
        transform="none", outliers="iqr", iqr_threshold=3.0,
        outlier_action="flag_as_nan", impute="none", frame="keep",
    )
    base = _transform_available(spec, offset=0.0)
    perturbed = _transform_available(spec, offset=2.5)
    # Outlier flags (-> NaN) on available rows must not depend on the future row.
    pd.testing.assert_frame_equal(
        base.isna(), perturbed.isna(), check_dtype=False
    )
