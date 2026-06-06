"""Ragged-start FRED-MD panels: em_factor and PCA drop all-missing columns, not raise."""
import numpy as np
import pandas as pd
import pytest

from macroforecast.preprocessing.clean import em_factor_impute_clean


def test_em_factor_drops_all_missing_column():
    idx = pd.date_range("2000-01-31", periods=80, freq="ME")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"a": rng.standard_normal(80), "b": rng.standard_normal(80),
         "late": [np.nan] * 80, "d": rng.standard_normal(80)},
        index=idx,
    )
    df.iloc[3:6, 0] = np.nan  # ragged interior missing
    out = em_factor_impute_clean(df)
    assert out["a"].notna().all()        # ragged interior imputed
    assert out["late"].isna().all()      # all-missing column left as NaN, no raise


def test_pca_step_handles_ragged_predictor():
    import macroforecast as mf

    n = 100
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(1)
    data = {f"x{i}": rng.standard_normal(n).cumsum() for i in range(6)}
    data["y"] = 0.5 * data["x0"] + rng.standard_normal(n) * 0.2
    # a late-starting predictor: NaN for the first 40 months
    late = rng.standard_normal(n).cumsum()
    late[:40] = np.nan
    data["x_late"] = late
    frame = pd.DataFrame(data, index=idx)
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    feats = mf.feature_engineering.feature_spec(
        target="y", predictors=[c for c in frame.columns if c != "y"], lags=None, target_lags=(0, 1),
        feature_steps=[mf.feature_engineering.pca_step(
            name="F", columns=[c for c in frame.columns if c != "y"], n_components=3, scale=True)],
    )
    # fit on the early window where x_late is all-NaN -> must not raise
    builder = feats.fit(frame.iloc[:30])
    fset = builder.transform(frame.iloc[:30])
    assert not fset.X.empty
