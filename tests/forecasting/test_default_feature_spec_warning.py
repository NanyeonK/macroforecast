from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _panel(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x,
            "x1": x,
            "x2": np.sin(np.arange(n) / 3.0),
        },
        index=idx,
    )


def _window() -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def _default_feature_spec_warnings(recwarn) -> list:
    return [
        record
        for record in recwarn.list
        if issubclass(record.category, UserWarning)
        and "implicit default feature spec" in str(record.message)
    ]


def test_supervised_model_with_no_features_warns() -> None:
    """A supervised model (random_forest) with features=None silently gets the
    implicit default FeatureSpec -- worth a UserWarning so an "AR vs RF" horse
    race does not quietly run RF on an unintended feature set.
    """

    with pytest.warns(UserWarning, match="implicit default feature spec"):
        mf.forecasting.run(_panel(), "random_forest", window=_window(), target="y")


def test_supervised_model_with_explicit_features_does_not_warn(recwarn) -> None:
    features = mf.feature_engineering.feature_spec(
        target="y", horizon=1, predictors=["x1", "x2"], lags=(0, 1),
    )
    mf.forecasting.run(_panel(), "random_forest", window=_window(), features=features)

    assert not _default_feature_spec_warnings(recwarn)


def test_ar_model_with_no_features_does_not_warn(recwarn) -> None:
    # "ar" is ``input_kind == "supervised"`` too, but target-lags-only is its
    # documented/intended default construction (see the ar/far carve-out in
    # ``macroforecast.forecasting.policy_config``), not an accidental panel drop.
    mf.forecasting.run(_panel(), "ar", window=_window(), target="y")

    assert not _default_feature_spec_warnings(recwarn)


def test_far_model_with_no_features_does_not_warn(recwarn) -> None:
    mf.forecasting.run(_panel(), "far", window=_window(), target="y")

    assert not _default_feature_spec_warnings(recwarn)


def test_panel_input_model_with_no_features_does_not_warn(recwarn) -> None:
    # A panel-input model (e.g. "var") never resolves a FeatureSpec at all --
    # it consumes the panel directly and is routed to the panel runner before
    # ``_feature_spec_for_policy`` is ever reached.
    n = 60
    idx = pd.date_range("2000-01-31", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({f"s{i}": np.cumsum(rng.normal(size=n)) for i in range(3)}, index=idx)
    panel["y"] = 0.5 * panel["s0"] + rng.normal(size=n)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    win = mf.window.from_cutoffs(
        test_start="2003-01-01", test_end="2003-06-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )

    mf.forecasting.run(
        bundle, "var", window=win, features=None, target="y",
        horizons=[3], forecast_policy="direct_average",
    )

    assert not _default_feature_spec_warnings(recwarn)
