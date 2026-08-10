"""Custom preprocessing steps declare whether they aggregate, and are held to it.

Before #449 a step was one callable. Under `policy="fit_window"` that callable
was re-executed on each apply window -- a window that contains the rows after the
forecast origin -- so any statistic it computed there read the future. The
package warned and ran anyway.

Now a step says which kind it is:

- `row_local=True`: each output row depends only on the matching input row, so
  re-running on a longer frame cannot change a row that was already there.
- `fit_func` / `transform_func`: whatever is derived from the sample is derived
  once, on the estimation window, and applied downstream without recomputation.

A bare `func` that declares neither is refused under `fit_window`, and still
accepted under `origin_available` where the sample is already restricted.

One thing here is worth knowing before reading the assertions. **OLS with an
intercept is invariant to any affine transform of X.** A leaky centering or
rescaling step changes the fitted coefficients and leaves the prediction
untouched -- the leak is real, but the model absorbs it. So the leaky step below
clips at sample quantiles, which is not affine. Whether a leak *reaches* a
forecast depends on the model it is fed to, which is most of why this was easy
to miss.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

N = 80
TEST_AT = 60
NUMERIC = ["x0", "x1", "x2"]


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #

def _winsorize_unrestricted(panel: pd.DataFrame, **_: object) -> pd.DataFrame:
    """Clip at quantiles of whatever frame it is handed -- the shape #449 refuses."""
    out = panel.copy()
    cols = out.select_dtypes("number").columns
    out[cols] = out[cols].clip(lower=out[cols].quantile(0.05),
                               upper=out[cols].quantile(0.95), axis=1)
    return out


def _fit_bounds(panel: pd.DataFrame, **_: object) -> dict[str, pd.Series]:
    """Derive the bounds ONCE, on the estimation window."""
    cols = panel.select_dtypes("number").columns
    return {"lo": panel[cols].quantile(0.05), "hi": panel[cols].quantile(0.95)}


def _apply_bounds(panel: pd.DataFrame, *, state=None, **_: object) -> pd.DataFrame:
    """Apply fitted bounds. Computes nothing from the frame it receives."""
    out = panel.copy()
    if not state:
        return out
    cols = [c for c in state["lo"].index if c in out.columns]
    out[cols] = out[cols].clip(lower=state["lo"][cols], upper=state["hi"][cols], axis=1)
    return out


def _double(panel: pd.DataFrame, **_: object) -> pd.DataFrame:
    out = panel.copy()
    cols = out.select_dtypes("number").columns
    out[cols] = out[cols] * 2.0
    return out


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _panel(future_fill: float | None = None) -> pd.DataFrame:
    idx = pd.date_range("1990-01-31", periods=N, freq="ME", name="date")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({c: rng.normal(size=N) for c in NUMERIC}, index=idx)
    frame["y"] = 0.5 * frame["x0"] + rng.normal(size=N) * 0.3
    if future_fill is not None:
        frame.loc[frame.index[TEST_AT + 1 :], NUMERIC] = future_fill
    return frame


def _first_forecast(frame: pd.DataFrame, step, *, policy: str = "fit_window") -> float:
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec("y", transform="level")],
        horizons=[1],
        window=mf.window.from_cutoffs(
            test_start=frame.index[TEST_AT], horizon=1, embargo=0,
            val_method="expanding", val_min_train_size=12,
        ),
        arms=[Arm("OLS", model="ols", is_benchmark=True,
                  features=mf.feature_engineering.feature_spec(
                      target="y", predictors=NUMERIC, lags=0, target_lags=1))],
        preprocessing=mf.preprocessing.preprocess_spec(
            custom_steps=[step], impute="mean", transform="none"
        ),
        preprocessing_policy=mf.window.stage_policy(policy),
        evaluation=EvalSpec(benchmark="OLS", metrics=("rmse",), tests=()),
        save_models=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecasts = run_pipeline(spec).forecasts
    return float(forecasts.sort_values("date")["prediction"].iloc[0])


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #

def test_an_undeclared_aggregating_step_is_refused_under_fit_window() -> None:
    """It used to warn and run. The message has to be usable, so check it names
    all three ways out."""
    step = mf.preprocessing.custom_preprocess_step("winsorize", _winsorize_unrestricted)
    with pytest.raises(ValueError) as excinfo:
        mf.preprocessing.preprocess_spec(
            custom_steps=[step], impute="mean", transform="none"
        ).fit(_panel(), policy="fit_window")
    message = str(excinfo.value)
    assert "winsorize" in message
    for way_out in ("row_local=True", "fit_func", "origin_available"):
        assert way_out in message, f"the error does not mention {way_out}"


def test_a_stateful_step_does_not_see_rows_after_the_origin() -> None:
    """The point of the whole change.

    Same winsorizing, split into fit/transform: the bounds come from the
    estimation window and are applied unchanged, so replacing every future
    predictor with 1e6 must leave the first forecast exactly where it was.
    """
    step = mf.preprocessing.custom_preprocess_step(
        "winsorize", fit_func=_fit_bounds, transform_func=_apply_bounds
    )
    base = _first_forecast(_panel(), step)
    poisoned = _first_forecast(_panel(future_fill=1e6), step)
    assert base == pytest.approx(poisoned, rel=0, abs=1e-10), (
        f"the first forecast moved by {poisoned - base:+.6g} when only rows "
        f"AFTER its origin changed"
    )


def test_a_row_local_step_still_works_when_declared() -> None:
    """Declaring row_local must not make an already-safe step harder to write."""
    step = mf.preprocessing.custom_preprocess_step("double", _double, row_local=True)
    base = _first_forecast(_panel(), step)
    poisoned = _first_forecast(_panel(future_fill=1e6), step)
    assert base == pytest.approx(poisoned, rel=0, abs=1e-10)


def test_origin_available_still_accepts_a_bare_func() -> None:
    """Nothing is taken away from the policy that was already restricted."""
    step = mf.preprocessing.custom_preprocess_step("winsorize", _winsorize_unrestricted)
    value = _first_forecast(_panel(), step, policy="origin_available")
    assert np.isfinite(value)


def test_the_fitted_state_is_derived_once_and_only_from_the_fit_panel() -> None:
    """Directly, without a pipeline: fit_func is called on the estimation window
    and transform_func never recomputes."""
    calls: list[int] = []

    def counting_fit(panel: pd.DataFrame, **kwargs: object):
        calls.append(len(panel))
        return _fit_bounds(panel, **kwargs)

    frame = _panel()
    train = frame.iloc[:TEST_AT]
    step = mf.preprocessing.custom_preprocess_step(
        "winsorize", fit_func=counting_fit, transform_func=_apply_bounds
    )
    fitted = mf.preprocessing.preprocess_spec(
        custom_steps=[step], impute="mean", transform="none"
    ).fit(train, policy="fit_window")

    assert calls == [len(train)], (
        f"fit_func should run once on the {len(train)}-row estimation window; got {calls}"
    )
    assert "winsorize" in fitted.custom_step_states

    fitted.transform(frame, history=train, policy="fit_window")
    assert calls == [len(train)], (
        f"transform must reuse the fitted state, not refit it; calls={calls}"
    )


def test_the_builder_refuses_incoherent_combinations() -> None:
    f = _double
    with pytest.raises(ValueError, match="not both"):
        mf.preprocessing.custom_preprocess_step("x", f, fit_func=f, transform_func=f)
    with pytest.raises(ValueError, match="requires func or transform_func"):
        mf.preprocessing.custom_preprocess_step("x")
    with pytest.raises(TypeError, match="must be callable"):
        mf.preprocessing.custom_preprocess_step("x", transform_func="not callable")
