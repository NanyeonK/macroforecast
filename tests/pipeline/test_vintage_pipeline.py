from __future__ import annotations

import hashlib
import json

import pandas as pd
import pandas.testing as pdt
import numpy as np

import macroforecast as mf
import macroforecast.pipeline.run as run_mod
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline


def _latest_key(keys, origin):
    origin = pd.Timestamp(origin)
    return max(key for key in keys if key <= origin)


class _Source:
    kind = "pipeline_synthetic"

    def __init__(self, periods: int = 9) -> None:
        self.reference = pd.date_range("2000-01-31", periods=periods, freq="ME", name="date")
        base = pd.DataFrame(
            {"A": [float(i) for i in range(1, periods + 1)]},
            index=self.reference,
        )
        self.bundles = {
            self.reference[i]: mf.data.DataBundle(
                base.iloc[:i].copy(),
                {"dataset": "synthetic", "frequency": "monthly", "vintage": f"v{i}"},
            )
            for i in range(2, len(self.reference))
        }

    def available_vintages(self):
        return list(self.bundles)

    def resolve(self, origin_date):
        origin = pd.Timestamp(origin_date)
        keys = [key for key in self.bundles if key <= origin]
        if not keys:
            raise mf.data.VintageUnavailableError("no vintage available")
        return self.bundles[max(keys)]


class _RevisionSource(_Source):
    def __init__(self) -> None:
        super().__init__(periods=16)
        for key, bundle in list(self.bundles.items()):
            panel = bundle.panel.copy()
            for pos in range(5, 13):
                if key >= self.reference[pos + 2] and self.reference[pos] in panel.index:
                    panel.loc[self.reference[pos], "A"] = float((pos + 1) * 10)
            self.bundles[key] = mf.data.DataBundle(panel, dict(bundle.metadata))


def _constant_model(value: float, name: str) -> mf.models.ModelSpec:
    class Fit:
        def predict(self, X: pd.DataFrame) -> pd.Series:
            return pd.Series(float(value), index=X.index)

    def fit(X: pd.DataFrame, y: pd.Series):
        return Fit()

    return mf.models.custom_model(name, fit)


def _spec(
    *,
    n_jobs=1,
    horizons=(1,),
    preprocessing_cache_dir=None,
    preprocessing=False,
    checkpoint_dir=None,
):
    source = _Source()
    data = mf.data.VintagePanelSpec(source, source.reference)
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=3),
        val=mf.window.val_last_block(size=1),
        test=mf.window.test_origins(
            first_origin=source.reference[4],
            last_origin=source.reference[6],
            horizon=1,
        ),
    )
    features = mf.feature_engineering.feature_spec(
        target="A",
        predictors=[],
        lags=None,
        target_lags=(1,),
    )
    prep = (
        mf.preprocessing.preprocess_spec(
            transform="none",
            impute="none",
            standardize="zscore",
        )
        if preprocessing
        else None
    )
    return pipeline_spec(
        data=data,
        targets=[mf.pipeline.TargetSpec("A", transform="level")],
        horizons=horizons,
        window=window,
        arms=[Arm("AR", model="ols", features=features)],
        evaluation=EvalSpec(benchmark="AR", tests=[]),
        preprocessing=prep,
        save_models=False,
        n_jobs=n_jobs,
        preprocessing_cache_dir=preprocessing_cache_dir,
        checkpoint_dir=checkpoint_dir,
    )


def _scoring_spec(actuals_vintage: str):
    source = _RevisionSource()
    data = mf.data.VintagePanelSpec(
        source,
        source.reference,
        actuals_vintage=actuals_vintage,
    )
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=3),
        val=mf.window.val_last_block(size=1),
        test=mf.window.test_origins(
            first_origin=source.reference[4],
            last_origin=source.reference[11],
            horizon=1,
        ),
    )
    features = mf.feature_engineering.feature_spec(
        target="A",
        predictors=[],
        lags=None,
        target_lags=(1,),
    )
    return pipeline_spec(
        data=data,
        targets=[mf.pipeline.TargetSpec("A", transform="level")],
        horizons=[1],
        window=window,
        arms=[
            Arm("ZERO", model=_constant_model(0.0, "vintage_zero"), features=features),
            Arm("TEN", model=_constant_model(10.0, "vintage_ten"), features=features),
        ],
        evaluation=EvalSpec(
            benchmark="ZERO",
            metrics=("relative_mse",),
            tests=("dm",),
        ),
        save_models=False,
    )


def test_first_release_actuals_change_pipeline_scores() -> None:
    latest = run_pipeline(_scoring_spec("latest"))
    first = run_pipeline(_scoring_spec("first_release"))

    latest_actuals = (
        latest.forecasts[latest.forecasts["contender"] == "ZERO"]
        .sort_values("date")["actual"]
        .to_numpy(float)
    )
    first_actuals = (
        first.forecasts[first.forecasts["contender"] == "ZERO"]
        .sort_values("date")["actual"]
        .to_numpy(float)
    )
    np.testing.assert_allclose(latest_actuals, [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
    np.testing.assert_allclose(first_actuals, [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])

    latest_rel = float(latest.accuracy[latest.accuracy["contender"] == "TEN"]["relative_mse"].iloc[0])
    first_rel = float(first.accuracy[first.accuracy["contender"] == "TEN"]["relative_mse"].iloc[0])
    assert latest_rel != first_rel

    latest_dm = float(latest.significance[latest.significance["contender"] == "TEN"]["dm_stat"].iloc[0])
    first_dm = float(first.significance[first.significance["contender"] == "TEN"]["dm_stat"].iloc[0])
    assert latest_dm != first_dm


def _custom_shape_sources():
    reference = pd.date_range("2000-01-31", periods=9, freq="ME", name="date")
    base = pd.DataFrame(
        {"A": [float(i) for i in range(1, 10)]},
        index=reference,
    )
    frames = {
        reference[i]: base.iloc[:i].copy()
        for i in range(2, len(reference))
    }

    def vintage_label(key) -> str:
        resolved = _latest_key(frames, key)
        return f"v{reference.get_loc(resolved)}"

    def callable_source(origin_date):
        return frames[_latest_key(frames, origin_date)]

    long = pd.concat(
        [
            frame.reset_index().assign(vintage=key)
            for key, frame in frames.items()
        ],
        ignore_index=True,
    )
    return reference, (
        mf.data.custom_vintages(callable_source, vintage_id=vintage_label),
        mf.data.custom_vintages(frames, vintage_id=vintage_label),
        mf.data.custom_vintages(
            long,
            vintage_column="vintage",
            date_column="date",
            vintage_id=vintage_label,
        ),
    )


def _custom_shape_spec(source, reference):
    data = mf.data.VintagePanelSpec(source, reference)
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=3),
        val=mf.window.val_last_block(size=1),
        test=mf.window.test_origins(
            first_origin=reference[4],
            last_origin=reference[6],
            horizon=1,
        ),
    )
    features = mf.feature_engineering.feature_spec(
        target="A",
        predictors=[],
        lags=None,
        target_lags=(1,),
    )
    return pipeline_spec(
        data=data,
        targets=[mf.pipeline.TargetSpec("A", transform="level")],
        horizons=[1],
        window=window,
        arms=[Arm("AR", model="ols", features=features)],
        evaluation=EvalSpec(benchmark="AR", tests=[]),
        save_models=False,
    )


def test_custom_vintage_shapes_produce_identical_pipeline_output() -> None:
    reference, sources = _custom_shape_sources()
    frames = [
        run_pipeline(_custom_shape_spec(source, reference))
        .forecasts.sort_values(["origin", "date", "model"])
        .reset_index(drop=True)
        for source in sources
    ]

    columns = [
        "target",
        "horizon",
        "origin",
        "origin_pos",
        "date",
        "model",
        "prediction",
        "actual",
        "vintage_id",
        "actuals_vintage_id",
    ]
    assert not frames[0].empty
    pdt.assert_frame_equal(frames[0][columns], frames[1][columns])
    pdt.assert_frame_equal(frames[0][columns], frames[2][columns])


def test_vintage_pipeline_parallel_equals_serial() -> None:
    serial = (
        run_pipeline(_spec(n_jobs=1, horizons=(1, 2)))
        .forecasts.sort_values(["horizon", "origin", "date", "model"])
        .reset_index(drop=True)
    )
    parallel = (
        run_pipeline(_spec(n_jobs=2, horizons=(1, 2)))
        .forecasts.sort_values(["horizon", "origin", "date", "model"])
        .reset_index(drop=True)
    )

    assert not serial.empty
    assert not parallel.empty
    pdt.assert_frame_equal(
        serial[
            [
                "target",
                "horizon",
                "origin",
                "origin_pos",
                "date",
                "model",
                "prediction",
                "actual",
                "vintage_id",
                "actuals_vintage_id",
            ]
        ],
        parallel[
            [
                "target",
                "horizon",
                "origin",
                "origin_pos",
                "date",
                "model",
                "prediction",
                "actual",
                "vintage_id",
                "actuals_vintage_id",
            ]
        ],
    )


def test_vintage_parallel_uses_vintage_namespaced_preprocessing_store(tmp_path) -> None:
    report = run_pipeline(
        _spec(
            n_jobs=2,
            horizons=(1, 2),
            preprocessing=True,
            preprocessing_cache_dir=str(tmp_path),
        )
    )

    assert not report.forecasts.empty
    # Three test origins share the same per-origin vintage across the two
    # horizon cells. The explicit store should therefore contain one fitted
    # preprocessor per origin/vintage, not one per origin/horizon.
    assert len(list(tmp_path.glob("*.pkl"))) == 3


def test_vintage_pipeline_report_carries_audit_and_provenance() -> None:
    spec = _spec()

    assert mf.pipeline.is_vintage_aware(spec) is True
    report = run_pipeline(spec)

    vintage = report.provenance["vintage_source"]
    assert vintage["kind"] == "pipeline_synthetic"
    assert vintage["actuals_vintage"] == "latest"
    assert vintage["reference_calendar"]["n_origins"] == 9
    assert vintage["origin_vintage_map"] == {
        "2000-05-31T00:00:00": "v4",
        "2000-06-30T00:00:00": "v5",
        "2000-07-31T00:00:00": "v6",
    }
    assert report.leakage_audit["vintage_boundary_audit"]["vintage_boundary_ok"] is True
    assert {"vintage_id", "actuals_vintage_id"}.issubset(report.forecasts.columns)


def _origin_vintage_map(n: int) -> dict[str, str]:
    dates = pd.date_range("2000-01-31", periods=n, freq="ME")
    return {date.isoformat(): f"v{pos}" for pos, date in enumerate(dates)}


def test_vintage_origin_map_stays_inline_at_size_gate(tmp_path) -> None:
    origin_map = _origin_vintage_map(500)
    spec = _spec(checkpoint_dir=str(tmp_path))

    vintage = run_mod._merge_vintage_sources(
        spec,
        [{"kind": "synthetic", "actuals_vintage": "latest", "origin_vintage_map": origin_map}],
    )

    assert vintage["origin_vintage_map"] == dict(sorted(origin_map.items()))
    assert not (tmp_path / "vintage_map.json").exists()


def test_vintage_origin_map_writes_sidecar_above_size_gate(tmp_path) -> None:
    origin_map = _origin_vintage_map(501)
    spec = _spec(checkpoint_dir=str(tmp_path))

    vintage = run_mod._merge_vintage_sources(
        spec,
        [{"kind": "synthetic", "actuals_vintage": "latest", "origin_vintage_map": origin_map}],
    )

    sidecar = vintage["origin_vintage_map"]
    path = tmp_path / "vintage_map.json"
    assert set(sidecar) == {"path", "sha256", "n_origins"}
    assert sidecar["path"] == str(path)
    assert sidecar["n_origins"] == 501
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == sidecar["sha256"]
    assert json.loads(payload.decode("utf-8")) == dict(sorted(origin_map.items()))
