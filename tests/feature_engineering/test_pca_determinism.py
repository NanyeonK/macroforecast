"""Factor extraction must be reproducible run to run.

For a macro-shaped panel (hundreds of rows and roughly a hundred predictors,
a handful of factors) sklearn's PCA 'auto' solver selects the randomized SVD,
which is non-deterministic when no seed is fixed. A reproducibility-focused
forecasting package must return identical factors regardless of the global RNG
state, so this is a regression guard. The panel shape here mirrors FRED-MD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def _macro_panel(n: int = 706, p: int = 126, seed: int = 7) -> pd.DataFrame:
    idx = pd.date_range("1959-01-01", periods=n, freq="MS")
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.randn(n, p), index=idx, columns=[f"x{i}" for i in range(p)])
    df.index.name = "date"
    return df


def test_pca_features_deterministic_despite_global_rng() -> None:
    panel = _macro_panel()  # 706 x 126 -> 'auto' selects randomized SVD
    cols = list(panel.columns)

    first = mf.feature_engineering.pca_features(
        panel, columns=cols, n_components=8,
        fit_policy="full_sample", warn_full_sample=False,
    )
    # perturb the global RNG between calls to expose an unseeded randomized solver
    np.random.seed(12345)
    _ = np.random.randn(10_000)

    second = mf.feature_engineering.pca_features(
        panel, columns=cols, n_components=8,
        fit_policy="full_sample", warn_full_sample=False,
    )
    pd.testing.assert_frame_equal(first, second)
