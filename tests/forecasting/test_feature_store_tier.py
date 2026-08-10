"""The feature-fit cache needs an on-disk tier, because parallel workers are
separate processes.

`_fitted_feature_builder_for_origin` shared a fitted PCA/MARX/SIR builder across
arms through an in-memory dict. Under `n_jobs > 1` each cell runs in its own
process, so that dict is always empty and every (arm x horizon) cell refits the
same transform -- the redundancy issue #448 measured. `preprocessing_stage.py`
already solved the same problem with a second, on-disk tier; this is that tier
for features.

The store key is content-addressed on the fit panel itself, not just on row
positions: a directory outlives a run, and two datasets can occupy the same
positions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting.feature_stage import (
    _feature_store_key,
    _fit_panel_fingerprint,
)
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline


def _bundle(seed: int = 0, n: int = 120) -> mf.DataBundle:
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)}, index=idx)
    panel["y"] = 0.5 * panel["x0"] + rng.normal(size=n) * 0.3
    return mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})


def _spec(bundle, cache_dir, n_jobs: int = 1):
    window = mf.window.from_cutoffs(
        test_start=bundle.panel.index[90], horizon=1, embargo=0,
        val_method="expanding", val_min_train_size=12,
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        predictors=[f"x{i}" for i in range(6)],
        lags=0,
        target_lags=None,
        feature_steps=[
            mf.feature_engineering.pca_step(
                name="pc", input="panel", n_components=2, fit_policy="expanding"
            )
        ],
    )
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec("y", transform="level")],
        horizons=[1],
        window=window,
        # Two arms sharing one feature spec: the case the cache exists for.
        arms=[
            Arm("OLS", model="ols", features=features, is_benchmark=True),
            Arm("RIDGE", model="ridge", features=features),
        ],
        evaluation=EvalSpec(benchmark="OLS", metrics=("rmse",), tests=()),
        preprocessing_cache_dir=str(cache_dir) if cache_dir else None,
        save_models=False,
        n_jobs=n_jobs,
    )


def _forecasts(spec) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pipeline(spec).forecasts.sort_values(["contender", "date"]).reset_index(drop=True)


def test_the_store_does_not_change_any_forecast(tmp_path) -> None:
    """The whole point: a cache must be invisible in the numbers."""
    bundle = _bundle()
    without = _forecasts(_spec(bundle, None))
    with_store = _forecasts(_spec(bundle, tmp_path / "cache"))
    assert list(without["contender"]) == list(with_store["contender"])
    np.testing.assert_array_equal(
        without["prediction"].to_numpy(dtype=float),
        with_store["prediction"].to_numpy(dtype=float),
    )


def test_a_second_run_reuses_the_stored_fits(tmp_path, monkeypatch) -> None:
    """A fresh process would find the fits already on disk; count the fits."""
    bundle = _bundle()
    cache = tmp_path / "cache"
    _forecasts(_spec(bundle, cache))  # populate

    from macroforecast.feature_engineering import FeatureSpec

    calls = {"n": 0}
    original = FeatureSpec.fit

    def counting_fit(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FeatureSpec, "fit", counting_fit)
    # in-memory cache disabled the way a parallel worker sees it
    second = _forecasts(_spec(bundle, cache))
    assert calls["n"] == 0, (
        f"a warm store should serve every feature fit; {calls['n']} refits happened"
    )
    assert not second.empty


def test_a_different_dataset_does_not_get_served_a_stored_fit(tmp_path) -> None:
    """Row positions alone would collide here; the content fingerprint must not."""
    cache = tmp_path / "cache"
    a, b = _bundle(seed=0), _bundle(seed=99)
    assert a.panel.shape == b.panel.shape, "the two panels must be position-identical"
    first = _forecasts(_spec(a, cache))
    second = _forecasts(_spec(b, cache))
    assert not np.array_equal(
        first["prediction"].to_numpy(dtype=float),
        second["prediction"].to_numpy(dtype=float),
    ), "the second dataset was served the first dataset's fit"

    # and the same data, run twice, IS identical
    again = _forecasts(_spec(b, cache))
    np.testing.assert_array_equal(
        second["prediction"].to_numpy(dtype=float),
        again["prediction"].to_numpy(dtype=float),
    )


def test_the_fingerprint_separates_content_at_equal_shape() -> None:
    idx = pd.date_range("1990-01-31", periods=20, freq="ME")
    one = pd.DataFrame({"a": np.arange(20.0)}, index=idx)
    two = pd.DataFrame({"a": np.arange(20.0) + 1.0}, index=idx)
    assert _fit_panel_fingerprint(one) != _fit_panel_fingerprint(two)
    assert _fit_panel_fingerprint(one) == _fit_panel_fingerprint(one.copy())


def test_the_store_key_depends_on_both_parts() -> None:
    key = ("features", "digestA", ("fit", 0, 100))
    a = _feature_store_key(key, fit_panel_fingerprint="fpA")
    assert a != _feature_store_key(key, fit_panel_fingerprint="fpB")
    assert a != _feature_store_key(
        ("features", "digestB", ("fit", 0, 100)), fit_panel_fingerprint="fpA"
    )
    assert a == _feature_store_key(key, fit_panel_fingerprint="fpA")
