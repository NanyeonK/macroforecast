from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline


class _Source:
    kind = "pipeline_synthetic"

    def __init__(self) -> None:
        self.reference = pd.date_range("2000-01-31", periods=9, freq="ME", name="date")
        base = pd.DataFrame(
            {"A": [float(i) for i in range(1, 10)]},
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


def _spec(*, n_jobs=1, horizons=(1,), preprocessing_cache_dir=None, preprocessing=False):
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
    )


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
