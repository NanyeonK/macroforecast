"""Fail-fast validation and failure-surfacing contracts for pipeline wiring."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, CombinationContender, EvalSpec, TargetSpec, pipeline_spec, run_pipeline
from macroforecast.pipeline.evaluate import evaluate


def _bundle(n: int = 72):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame({"y": 1.0 + x, "x1": x}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _features():
    return mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x1"],
        lags=1,
        target_lags=(0, 1),
    )


def _window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def _spec(**over):
    kw = dict(
        data=_bundle(),
        targets=["y"],
        horizons=[1],
        window=_window(),
        arms=[
            Arm("AR", model="ar", features=_features()),
            Arm("OLS", model="ols", features=_features()),
        ],
        evaluation=EvalSpec(benchmark="AR"),
        save_models=False,
    )
    kw.update(over)
    return pipeline_spec(**kw)


def _exploding_fit(X, y):
    raise RuntimeError("boom from custom model")


def test_unknown_model_string_raises_at_spec_build_with_suggestion():
    with pytest.raises(ValueError, match=r"Unknown model 'olz'.*Did you mean 'ols'"):
        _spec(
            arms=[Arm("BAD", model="olz", features=_features())],
            evaluation=EvalSpec(benchmark="BAD"),
        )


def test_bare_callable_model_raises_at_spec_build_with_custom_model_pointer():
    def fit_ridge(X, y):
        return None

    with pytest.raises(ValueError, match=r"bare callable model 'fit_ridge'.*custom_model"):
        _spec(
            arms=[Arm("raw", model=fit_ridge, features=_features())],
            evaluation=EvalSpec(benchmark="raw"),
        )


def test_unknown_metric_and_combination_names_raise_at_spec_build():
    with pytest.raises(ValueError, match=r"Unknown metric 'rmsse'"):
        _spec(evaluation=EvalSpec(benchmark="AR", metrics=("rmsse",)))

    with pytest.raises(ValueError, match=r"unsupported method 'not_a_method'"):
        _spec(combinations=[CombinationContender("POOL", method="not_a_method")])


def test_combination_over_unknown_contender_raises_at_spec_build():
    with pytest.raises(ValueError, match=r"unknown contender name"):
        _spec(combinations=[CombinationContender("POOL", method="mean", over=("AR", "TYPO"))])


def test_partial_per_arm_mapping_raises_but_shared_mapping_still_works():
    with pytest.raises(ValueError, match=r"params looks like a per-arm mapping"):
        mf.pipeline.model_arms(["ar", "ols"], params={"ar": {"p": 2}, "typo": {"p": 3}})

    arms = mf.pipeline.model_arms(["ar", "ols"], params={"alpha": 1.0})
    assert arms[0].params == {"alpha": 1.0}
    assert arms[1].params == {"alpha": 1.0}


def test_failed_cell_warns_and_reporting_warns_on_failed_cells():
    boom = mf.models.custom_model("boom", _exploding_fit)
    spec = _spec(
        arms=[
            Arm("AR", model="ar", features=_features()),
            Arm("BOOM", model=boom, features=_features()),
            Arm("OLS", model="ols", features=_features()),
        ],
        evaluation=EvalSpec(benchmark="AR"),
    )

    with pytest.warns(RuntimeWarning) as caught:
        report = run_pipeline(spec)

    assert any("pipeline cell failed" in str(w.message) for w in caught)
    assert {cell["arm"] for cell in report.failed_cells} == {"BOOM"}
    with pytest.warns(RuntimeWarning, match="failed_cells=1"):
        mf.reporting.paper_accuracy_table(report)


def test_evaluate_warns_when_master_frame_carries_failed_cells():
    master = pd.DataFrame()
    master.attrs["macroforecast_failed_cells"] = [
        {"target": "y", "arm": "BOOM", "horizons": [1], "error": "RuntimeError: boom"}
    ]

    with pytest.warns(RuntimeWarning, match="failed_cells=1"):
        result = evaluate(master, _spec())

    assert result["accuracy"].empty


def test_unpicklable_custom_model_fails_before_parallel_dispatch():
    def local_fit(X, y):
        return None

    custom = mf.models.custom_model("local_custom", local_fit)
    spec = _spec(
        arms=[Arm("LOCAL", model=custom, features=_features())],
        evaluation=EvalSpec(benchmark="LOCAL"),
        n_jobs=2,
    )

    with pytest.raises(ValueError, match=r"unpicklable model.*module-level def.*n_jobs=1"):
        run_pipeline(spec)


def test_evaluation_error_returns_partial_report_with_master_frame(monkeypatch):
    eval_mod = importlib.import_module("macroforecast.pipeline.evaluate")

    def fail_accuracy(master, spec):
        raise ValueError("bad evaluation config")

    monkeypatch.setattr(eval_mod, "accuracy_table", fail_accuracy)
    with pytest.warns(RuntimeWarning, match="partial PipelineReport"):
        report = run_pipeline(_spec())

    assert report.evaluation_error == "ValueError: bad evaluation config"
    assert not report.forecasts.empty
    assert report.accuracy.empty
    assert report.leakage_audit["evaluation_error"] == report.evaluation_error


def _evaluation_master() -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    rows = []
    for i, date in enumerate(dates):
        actual = float(i)
        rows.append({
            "target": "y",
            "horizon": 1,
            "origin": i,
            "date": date,
            "contender": "AR",
            "prediction": actual,
            "actual": actual,
        })
        rows.append({
            "target": "y",
            "horizon": 1,
            "origin": i,
            "date": date,
            "contender": "OLS",
            "prediction": actual + 1.0,
            "actual": actual,
        })
    return pd.DataFrame(rows)


def _eval_spec_for_tests(tests: tuple[str, ...]):
    return SimpleNamespace(
        combinations=(),
        seed=0,
        arms=[],
        evaluation=SimpleNamespace(
            benchmark="AR",
            metrics=("rmse",),
            tests=tests,
            test_options={},
            cw_for_nested=True,
            mcs_alpha=0.10,
            loss=None,
        ),
    )


def test_degraded_dm_row_keeps_reason_and_warns(monkeypatch):
    import macroforecast.tests as test_mod

    def fail_dm(*args, **kwargs):
        raise ValueError("dm unstable")

    monkeypatch.setattr(test_mod, "dm_test", fail_dm)
    with pytest.warns(RuntimeWarning, match="degraded"):
        result = evaluate(_evaluation_master(), _eval_spec_for_tests(("dm",)))

    sig = result["significance"]
    assert sig.loc[0, "status"] == "degraded"
    assert "dm failed: ValueError: dm unstable" in sig.loc[0, "reason"]
    assert np.isnan(sig.loc[0, "dm_p"])


def test_degraded_mcs_cell_emits_nan_rows_with_reason(monkeypatch):
    import macroforecast.tests as test_mod

    def fail_mcs(*args, **kwargs):
        raise ValueError("mcs unstable")

    monkeypatch.setattr(test_mod, "model_confidence_set", fail_mcs)
    with pytest.warns(RuntimeWarning, match="degraded"):
        result = evaluate(_evaluation_master(), _eval_spec_for_tests(("mcs",)))

    mcs = result["mcs"]
    assert set(mcs["contender"]) == {"AR", "OLS"}
    assert set(mcs["status"]) == {"degraded"}
    assert mcs["in_mcs"].isna().all()
    assert all("mcs failed: ValueError: mcs unstable" in reason for reason in mcs["reason"])


def test_recursive_custom_model_with_exogenous_features_warns_at_spec_build():
    custom = mf.models.custom_model("mean_custom", lambda X, y: None)

    with pytest.warns(UserWarning, match="custom supervised model with exogenous features"):
        pipeline_spec(
            data=_bundle(),
            targets=[TargetSpec("y", transform="level", policy="recursive")],
            horizons=[1],
            window=_window(),
            arms=[Arm("CUSTOM", model=custom, features=_features())],
            evaluation=EvalSpec(benchmark="CUSTOM"),
            save_models=False,
        )
