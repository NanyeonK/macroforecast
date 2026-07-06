from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline
from macroforecast.pipeline.result_store import result_cell_identity
from macroforecast.pipeline.run import _data_identity


FIT_COUNTS: dict[str, int] = {}


class _ConstantFit:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _recording_fit(X, y, *, offset: float = 0.0):
    FIT_COUNTS["recording"] = FIT_COUNTS.get("recording", 0) + 1
    return _ConstantFit(float(np.nanmean(np.asarray(y, dtype=float))) + float(offset))


def _custom_fit(X, y):
    FIT_COUNTS["custom"] = FIT_COUNTS.get("custom", 0) + 1
    return _ConstantFit(float(np.nanmean(np.asarray(y, dtype=float))))


def _bundle(n: int = 72, *, bump: float = 0.0):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05,
            "x1": x,
        },
        index=idx,
    )
    frame.iloc[-1, frame.columns.get_loc("x1")] += bump
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=30),
        val=mf.window.val_last_block(size=10),
        test=mf.window.test_origins(horizon=1, step=8),
    )


def _features(*, lags=(1,)):
    return mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x1"],
        lags=lags,
        target_lags=(0, 1),
    )


def _recording_model():
    _recording_fit.__mf_digest__ = "recording-fit-v1"
    return mf.models.custom_model("recording_mean", _recording_fit)


def _spec(tmp_path, *, arms=None, features=None, preprocessing=None, data=None):
    feats = _features() if features is None else features
    return pipeline_spec(
        data=_bundle() if data is None else data,
        targets=["y"],
        horizons=[1],
        window=_window(),
        arms=arms
        if arms is not None
        else [
            Arm("A", model=_recording_model(), features=feats, params={"offset": 0.0}),
            Arm("B", model=_recording_model(), features=feats, params={"offset": 0.1}),
        ],
        evaluation=EvalSpec(benchmark="A", metrics=("rmse", "relative_mse")),
        save_models=False,
        preprocessing=preprocessing,
        result_store=tmp_path / "results",
    )


def _frame_sort(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["target", "horizon", "contender", "origin", "date"] if c in frame.columns]
    return frame.sort_values(cols).reset_index(drop=True)


def _assert_report_frames_equal(left, right) -> None:
    for name in ("forecasts", "accuracy", "significance", "mcs", "density", "calibration"):
        a = getattr(left, name)
        b = getattr(right, name)
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(_frame_sort(a), _frame_sort(b))


def test_result_store_round_trip_reuses_cells_and_preserves_report_frames(tmp_path):
    FIT_COUNTS.clear()
    first = run_pipeline(_spec(tmp_path))
    assert first.provenance["result_store"]["n_computed"] == 2
    assert FIT_COUNTS["recording"] > 0

    FIT_COUNTS.clear()
    second = run_pipeline(_spec(tmp_path))

    assert second.provenance["result_store"]["n_reused"] == 2
    assert second.provenance["result_store"]["n_computed"] == 0
    assert FIT_COUNTS.get("recording", 0) == 0
    _assert_report_frames_equal(first, second)


def test_result_store_incremental_horse_race_reuses_existing_arms(tmp_path):
    first = run_pipeline(_spec(tmp_path))
    third = Arm("C", model=_recording_model(), features=_features(), params={"offset": 0.2})
    second = run_pipeline(_spec(tmp_path, arms=[*_spec(tmp_path).arms, third]))

    assert second.provenance["result_store"]["n_reused"] == 2
    assert second.provenance["result_store"]["n_computed"] == 1
    assert set(second.accuracy["contender"]) == {"A", "B", "C"}
    for contender in ("A", "B"):
        a = first.accuracy[first.accuracy["contender"] == contender].reset_index(drop=True)
        b = second.accuracy[second.accuracy["contender"] == contender].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_result_store_digest_sensitivity_oracle(tmp_path):
    base = _spec(tmp_path)
    identity = result_cell_identity(
        base,
        base.arms[0],
        base.targets[0],
        horizon=1,
        data_identity=_data_identity(base.data),
    )

    changed_param = _spec(
        tmp_path,
        arms=[Arm("A", model=_recording_model(), features=_features(), params={"offset": 9.0})],
    )
    changed_features = _spec(tmp_path, features=_features(lags=(1, 2)))
    changed_prep = _spec(
        tmp_path,
        preprocessing=mf.preprocessing.preprocess_spec(standardize="zscore"),
    )
    changed_data = _spec(tmp_path, data=_bundle(bump=1.0))

    variants = [changed_param, changed_features, changed_prep, changed_data]
    digests = [
        result_cell_identity(
            spec,
            spec.arms[0],
            spec.targets[0],
            horizon=1,
            data_identity=_data_identity(spec.data),
        ).digest
        for spec in variants
    ]
    assert all(digest != identity.digest for digest in digests)

    class Source:
        kind = "toy_vintage"

        def resolve(self, origin_date):
            return _bundle()

        def available_vintages(self):
            return ("2000-01",)

    vintage_spec = pipeline_spec(
        data=mf.data.VintagePanelSpec(
            Source(),
            pd.date_range("2000-01-31", periods=3, freq="ME"),
        ),
        targets=[TargetSpec("y", transform="level", policy="direct")],
        horizons=[1],
        window=_window(),
        arms=[base.arms[0]],
        evaluation=EvalSpec(benchmark="A", metrics=("rmse",)),
        save_models=False,
        result_store=tmp_path / "results",
    )
    vintage_identity = result_cell_identity(
        vintage_spec,
        vintage_spec.arms[0],
        vintage_spec.targets[0],
        horizon=1,
        data_identity=_data_identity(base.data),
    )
    assert vintage_identity.digest != identity.digest


def test_result_store_custom_callable_requires_digest_opt_in(tmp_path):
    if hasattr(_custom_fit, "__mf_digest__"):
        delattr(_custom_fit, "__mf_digest__")
    custom = mf.models.custom_model("custom_mean", _custom_fit)
    arms = [Arm("A", model=custom, features=_features())]

    FIT_COUNTS.clear()
    first = run_pipeline(_spec(tmp_path, arms=arms))
    second = run_pipeline(_spec(tmp_path, arms=arms))
    assert first.provenance["result_store"]["n_undigestible"] == 1
    assert second.provenance["result_store"]["n_undigestible"] == 1
    assert first.provenance["result_store"]["n_reused"] == 0
    assert second.provenance["result_store"]["n_reused"] == 0
    assert FIT_COUNTS.get("custom", 0) > 0
    assert not list((tmp_path / "results" / "cells").glob("*.json"))

    _custom_fit.__mf_digest__ = "custom-v1"
    FIT_COUNTS.clear()
    run_pipeline(_spec(tmp_path, arms=arms))
    reused = run_pipeline(_spec(tmp_path, arms=arms))
    assert reused.provenance["result_store"]["n_reused"] == 1
    assert FIT_COUNTS.get("custom", 0) > 0

    _custom_fit.__mf_digest__ = "custom-v2"
    missed = run_pipeline(_spec(tmp_path, arms=arms))
    assert missed.provenance["result_store"]["n_computed"] == 1


def test_result_store_version_mismatch_warns_once_and_reuses(tmp_path):
    run_pipeline(_spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())]))
    manifest_path = next((tmp_path / "results" / "cells").glob("*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["macroforecast_version"] = "0.0-old"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.warns(UserWarning, match="different macroforecast version") as caught:
        report = run_pipeline(
            _spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())])
        )

    assert len(caught) == 1
    assert report.provenance["result_store"]["n_reused"] == 1
    assert report.provenance["result_store"]["version_mismatches"][0]["store_version"] == "0.0-old"


def test_result_store_corrupt_manifest_is_miss_not_crash(tmp_path):
    run_pipeline(_spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())]))
    manifest_path = next((tmp_path / "results" / "cells").glob("*.json"))
    manifest_path.write_text("{")

    report = run_pipeline(_spec(tmp_path, arms=[Arm("A", model=_recording_model(), features=_features())]))

    assert report.provenance["result_store"]["n_reused"] == 0
    assert report.provenance["result_store"]["n_computed"] == 1


def test_result_store_summary_and_purge(tmp_path):
    report = run_pipeline(_spec(tmp_path))
    store = tmp_path / "results"

    summary = mf.pipeline.result_store_summary(store)
    assert set(summary["arm"]) == {"A", "B"}
    assert set(summary["horizon"]) == {1}
    assert set(summary["n_rows"]) == {len(report.forecasts) // 2}

    digest = str(summary.iloc[0]["digest"])
    assert mf.pipeline.purge_result_store(store, digests=[digest]) == 1
    assert digest not in set(mf.pipeline.result_store_summary(store)["digest"])
