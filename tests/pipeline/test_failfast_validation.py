"""Fail-fast validation and failure-surfacing contracts for pipeline wiring."""

from __future__ import annotations

import dataclasses as _dc
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, CombinationContender, EvalSpec, TargetSpec, pipeline_spec, run_pipeline
from macroforecast.pipeline.evaluate import evaluate
from macroforecast.pipeline.run import _validate_parallel_picklable


def _picklable_metric(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


class _ExplodingReduce:
    def __reduce__(self):
        raise RuntimeError("boom from reduce")


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


def test_parallel_preflight_rejects_unpicklable_arm_params():
    def local_callback():
        return None

    spec = _spec(
        arms=[
            Arm(
                "AR",
                model="ar",
                features=_features(),
                params={"callback": local_callback},
            )
        ],
        evaluation=EvalSpec(benchmark="AR"),
        n_jobs=2,
    )

    with pytest.raises(ValueError, match=r"arm 'AR'.*unpicklable params.*n_jobs=1"):
        _validate_parallel_picklable(spec)


@pytest.mark.parametrize("field", ["metrics", "loss"])
def test_parallel_preflight_rejects_unpicklable_evaluation_callable(field):
    def local_callable(*_args):
        return 0.0

    evaluation = (
        EvalSpec(benchmark="AR", metrics=("rmse", local_callable))
        if field == "metrics"
        else EvalSpec(benchmark="AR", loss=local_callable)
    )
    spec = _spec(evaluation=evaluation, n_jobs=2)

    with pytest.raises(ValueError, match=rf"evaluation\.{field}.*n_jobs=1"):
        _validate_parallel_picklable(spec)


def test_parallel_preflight_payload_backstop_catches_other_spec_state():
    def local_callback():
        return None

    spec = _dc.replace(
        _spec(n_jobs=2),
        provenance={"callback": local_callback},
    )

    with pytest.raises(ValueError, match=r"pipeline spec payload \(excluding data\)"):
        _validate_parallel_picklable(spec)


def test_parallel_preflight_serial_bypass_and_picklable_callables():
    def local_callback():
        return None

    serial = _dc.replace(
        _spec(n_jobs=2),
        arms=(
            Arm(
                "AR",
                model="ar",
                features=_features(),
                params={"callback": local_callback},
            ),
        ),
        evaluation=EvalSpec(benchmark="AR"),
        n_jobs=1,
    )
    _validate_parallel_picklable(serial)

    parallel = _spec(
        evaluation=EvalSpec(
            benchmark="AR",
            metrics=("rmse", _picklable_metric),
            loss=_picklable_metric,
        ),
        n_jobs=2,
    )
    _validate_parallel_picklable(parallel)


def test_parallel_preflight_wraps_user_reduce_exception():
    spec = _spec(
        arms=[
            Arm(
                "AR",
                model="ar",
                features=_features(),
                params={"bad": _ExplodingReduce()},
            )
        ],
        evaluation=EvalSpec(benchmark="AR"),
        n_jobs=2,
    )

    with pytest.raises(ValueError, match="RuntimeError: boom from reduce") as caught:
        _validate_parallel_picklable(spec)
    assert isinstance(caught.value.__cause__, RuntimeError)


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


# --------------------------------------------------------------------------- #
# Uniqueness of the labels a run is keyed by: contender names, horizons, and
# resolved target names. Each of these used to build cleanly and then go wrong
# quietly -- a dropped contender, a repeated cell, two targets under one label.
# --------------------------------------------------------------------------- #


def _two_target_bundle(n: int = 72):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame({"y": 1.0 + x, "z": 2.0 - x, "x1": x}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "z": 1, "x1": 1})


def _master_frame():
    dates = pd.to_datetime(["2010-01-31", "2010-02-28", "2010-03-31"])
    actual = np.array([1.0, 1.1, 1.2])
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "arm": name, "model": name, "contender": name,
                    "target": "y", "horizon": 1, "origin": np.arange(len(dates)),
                    "date": dates, "prediction": actual + shift, "actual": actual,
                }
            )
            for name, shift in (("AR", 0.0), ("OLS", 0.2))
        ],
        ignore_index=True,
    )


def test_combination_named_after_an_arm_raises_at_spec_build():
    # Used to build, then hit apply_combinations' idempotence guard and vanish, so
    # the ARM's own forecasts were scored under the combination's label.
    with pytest.raises(
        ValueError, match=r"combination name\(s\) \['AR'\] duplicate an arm contender name"
    ):
        _spec(combinations=[CombinationContender("AR", method="mean")])


def test_duplicate_combination_names_raise_at_spec_build():
    # Both used to run: the guard reads a snapshot taken before the loop, so two
    # combinations sharing a name appended two sets of rows under one label.
    with pytest.raises(ValueError, match=r"combination name\(s\) \['POOL'\] are not unique"):
        _spec(
            combinations=[
                CombinationContender("POOL", method="mean"),
                CombinationContender("POOL", method="median"),
            ]
        )


def test_unique_combination_names_build_and_apply_combinations_stays_idempotent():
    # Build-time config validation and run-time idempotence are separate contracts;
    # tightening the first must not change the second.
    from macroforecast.pipeline.evaluate import apply_combinations

    spec = _spec(
        combinations=[
            CombinationContender("POOL", method="mean"),
            CombinationContender("MEDIAN", method="median"),
        ]
    )
    assert [c.name for c in spec.combinations] == ["POOL", "MEDIAN"]

    once = apply_combinations(_master_frame(), spec)
    assert set(once["contender"]) == {"AR", "OLS", "POOL", "MEDIAN"}

    twice = apply_combinations(once, spec)
    assert len(twice) == len(once)
    assert twice["contender"].value_counts().to_dict() == once["contender"].value_counts().to_dict()


def test_repeated_horizons_raise_after_normalization():
    for horizons in ([1, 1], [1, 1.0]):
        # 1 and 1.0 are one horizon: the check runs on the normalized integers, so it
        # does not matter how the caller spelled it.
        with pytest.raises(ValueError, match=r"horizons must be unique; \[1\]"):
            _spec(horizons=horizons)


def test_distinct_horizons_keep_the_order_they_were_given():
    spec = _spec(horizons=[3, 1])
    assert spec.horizons == (3, 1)


def test_repeated_target_names_raise_however_they_are_declared():
    bundle = _two_target_bundle()
    duplicates = (
        ["y", "y"],
        # Different forecast objects, one public name -- every table is keyed by the
        # name alone, so these would be reported as a single target.
        [
            TargetSpec("y", transform="level", policy="direct"),
            TargetSpec("y", transform="change", policy="direct_average"),
        ],
        ["y", TargetSpec("y", transform="level", policy="direct")],
    )
    for targets in duplicates:
        with pytest.raises(ValueError, match=r"target names must be unique; \['y'\]"):
            _spec(data=bundle, targets=targets)


def test_distinct_targets_keep_the_order_they_were_given():
    spec = _spec(data=_two_target_bundle(), targets=["z", "y"])
    assert [t.name for t in spec.targets] == ["z", "y"]
