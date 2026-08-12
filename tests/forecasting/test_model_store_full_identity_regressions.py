"""Public regressions for complete stored-fit ownership identity."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline


class _PanelFit:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=panel.index, name="prediction")


def _fit_panel(bundle: mf.data.DataBundle, *, target: str) -> _PanelFit:
    return _PanelFit(float(bundle.panel[target].dropna().iloc[-1]))


class _VintageSource:
    kind = "stored_fit_identity_vintage"

    def __init__(self, panel: pd.DataFrame) -> None:
        self.panel = panel

    def available_vintages(self):
        return list(self.panel.index[24:])

    def resolve(self, origin_date):
        origin = pd.Timestamp(origin_date)
        panel = self.panel.loc[self.panel.index < origin].copy()
        if panel.empty:
            raise mf.data.VintageUnavailableError("no vintage available")
        return mf.data.DataBundle(
            panel,
            {
                "dataset": "stored_fit_identity_vintage",
                "frequency": "monthly",
                "vintage": origin.isoformat(),
            },
        )


def _bundle() -> mf.data.DataBundle:
    index = pd.date_range("2000-01-01", periods=72, freq="MS", name="date")
    x = np.sin(np.arange(len(index)) / 4.0)
    panel = pd.DataFrame(
        {
            "y1": 1.0 + 0.5 * x,
            "y2": -2.0 + 1.5 * x,
            "x": x,
            "x2": np.cos(np.arange(len(index)) / 6.0),
        },
        index=index,
    )
    return mf.data.custom_dataset(
        panel,
        transform_codes={column: 1 for column in panel.columns},
    )


def _window(index: pd.DatetimeIndex, *, retrain_every=1):
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(
            min_size=24,
            retrain_every=retrain_every,
        ),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(
            first_origin=index[48],
            last_origin=index[48],
            horizon=1,
        ),
    )


def _features(target: str, predictor: str = "x"):
    return mf.feature_engineering.feature_spec(
        target=target,
        predictors=[predictor],
        lags=(0,),
    )


def _arm(
    name: str,
    *,
    target: str = "y1",
    predictor: str = "x",
    alpha: float = 0.1,
    benchmark: bool = False,
) -> Arm:
    return Arm(
        name,
        model="ridge",
        features=_features(target, predictor),
        params={"ridge": {"alpha": alpha}},
        model_selection={"ridge": None},
        is_benchmark=benchmark,
    )


def _pipeline_report(
    bundle: mf.data.DataBundle,
    *,
    targets: list[TargetSpec],
    arms: list[Arm],
    model_store: str | Path,
):
    return run_pipeline(
        pipeline_spec(
            data=bundle,
            targets=targets,
            horizons=[1],
            window=_window(bundle.panel.index),
            arms=arms,
            evaluation=EvalSpec(
                benchmark=arms[0].name,
                metrics=("rmse",),
                tests=(),
            ),
            save_models=True,
            model_store=model_store,
            n_jobs=1,
        )
    )


def _stored_for(frame: pd.DataFrame, column: str, value: str) -> dict:
    rows = frame.loc[(frame[column] == value) & frame["stored_model"].notna()]
    assert not rows.empty
    return rows["stored_model"].iloc[0]


def _sidecar(stored: dict) -> dict:
    return json.loads(Path(stored["metadata_path"]).read_text(encoding="utf-8"))


def _canonical_components_digest(sidecar: dict) -> str:
    components = sidecar["fit_identity"]["components"]
    canonical = json.dumps(
        components,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def test_same_model_arms_have_distinct_self_verifying_fits(tmp_path: Path) -> None:
    bundle = _bundle()
    report = _pipeline_report(
        bundle,
        targets=[TargetSpec("y1", transform="level", policy="direct")],
        arms=[
            _arm("RIDGE_LOW", alpha=0.1, benchmark=True),
            _arm("RIDGE_HIGH", alpha=10.0),
        ],
        model_store=tmp_path,
    )
    low = _stored_for(report.forecasts, "arm", "RIDGE_LOW")
    high = _stored_for(report.forecasts, "arm", "RIDGE_HIGH")

    assert low["model_path"] != high["model_path"]
    for stored, arm, alpha in (
        (low, "RIDGE_LOW", 0.1),
        (high, "RIDGE_HIGH", 10.0),
    ):
        sidecar = _sidecar(stored)
        identity = sidecar["fit_identity"]
        assert sidecar["arm"] == arm
        assert sidecar["target"] == "y1"
        assert sidecar["params"]["alpha"] == alpha
        assert identity["namespace"] == "macroforecast/model-store/fit-identity"
        assert identity["version"] == 1
        assert identity["digest_length"] == 16
        assert identity["complete"] is True
        assert identity["digest"] == _canonical_components_digest(sidecar)
        assert f"__f{identity['digest']}" in str(stored["model_path"])
        mf.models.load_fit(stored["model_path"])


def test_same_arm_targets_have_distinct_owned_fits(tmp_path: Path) -> None:
    bundle = _bundle()
    report = _pipeline_report(
        bundle,
        targets=[
            TargetSpec("y1", transform="level", policy="direct"),
            TargetSpec("y2", transform="level", policy="direct"),
        ],
        arms=[_arm("RIDGE", benchmark=True)],
        model_store=tmp_path,
    )
    first = _stored_for(report.forecasts, "target", "y1")
    second = _stored_for(report.forecasts, "target", "y2")

    assert first["model_path"] != second["model_path"]
    assert _sidecar(first)["target"] == "y1"
    assert _sidecar(second)["target"] == "y2"


def test_direct_runs_separate_effective_params_and_repeat_idempotently(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    kwargs = dict(
        data=bundle,
        model="ridge",
        window=_window(bundle.panel.index),
        target="y1",
        features=_features("y1"),
        model_selection={"ridge": None},
        save_models=True,
        model_store=tmp_path,
    )

    low = mf.forecasting.run(
        **kwargs,
        params={"ridge": {"alpha": 0.1}},
    ).to_frame()["stored_model"].dropna().iloc[0]
    file_count = len(list(tmp_path.glob("*/*")))
    repeated = mf.forecasting.run(
        **kwargs,
        params={"ridge": {"alpha": 0.1}},
    ).to_frame()["stored_model"].dropna().iloc[0]
    assert repeated["model_path"] == low["model_path"]
    assert len(list(tmp_path.glob("*/*"))) == file_count

    high = mf.forecasting.run(
        **kwargs,
        params={"ridge": {"alpha": 10.0}},
    ).to_frame()["stored_model"].dropna().iloc[0]
    assert high["model_path"] != low["model_path"]


def test_same_arm_name_across_runs_separates_feature_specs(tmp_path: Path) -> None:
    bundle = _bundle()
    paths: list[str] = []
    for predictor in ("x", "x2"):
        report = _pipeline_report(
            bundle,
            targets=[TargetSpec("y1", transform="level", policy="direct")],
            arms=[
                _arm(
                    "RIDGE",
                    predictor=predictor,
                    benchmark=True,
                )
            ],
            model_store=tmp_path,
        )
        paths.append(_stored_for(report.forecasts, "arm", "RIDGE")["model_path"])
    assert len(set(paths)) == 2


def test_ordinary_panel_store_identity_owns_arm_target_and_policy(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    model = mf.models.custom_model(
        "stored_identity_panel",
        _fit_panel,
        default_params={"target": None},
        input_kind="panel",
        mf_digest="stored-identity-panel/v1",
    )
    report = mf.forecasting.run(
        bundle,
        model,
        window=_window(bundle.panel.index),
        target="y1",
        features=None,
        model_selection=None,
        save_models=True,
        model_store=tmp_path,
    )
    stored = report.to_frame()["stored_model"].dropna().iloc[0]
    sidecar = _sidecar(stored)
    components = sidecar["fit_identity"]["components"]["run"]

    assert "__f" in str(stored["metadata_path"])
    assert sidecar["target"] == "y1"
    assert components["forecast_policy"] == "direct"


@pytest.mark.parametrize("panel_model", [False, True])
def test_vintage_routes_include_vintage_owned_identity(
    tmp_path: Path,
    panel_model: bool,
) -> None:
    bundle = _bundle()
    vintages = mf.data.VintagePanelSpec(
        _VintageSource(bundle.panel),
        bundle.panel.index,
    )
    if panel_model:
        model = mf.models.custom_model(
            "stored_identity_vintage_panel",
            _fit_panel,
            default_params={"target": None},
            input_kind="panel",
            mf_digest="stored-identity-vintage-panel/v1",
        )
        features = None
        selection = None
        params = None
    else:
        model = "ridge"
        features = mf.feature_engineering.feature_spec(
            target="y1",
            predictors=[],
            lags=None,
            target_lags=(1,),
        )
        selection = {"ridge": None}
        params = {"ridge": {"alpha": 0.1}}

    report = mf.forecasting.run(
        vintages,
        model,
        window=_window(bundle.panel.index),
        target="y1",
        features=features,
        model_selection=selection,
        params=params,
        save_models=True,
        model_store=tmp_path,
    )
    stored = report.to_frame()["stored_model"].dropna().iloc[0]
    sidecar = _sidecar(stored)

    assert "__f" in str(stored["metadata_path"])
    assert sidecar["target"] == "y1"
    assert sidecar["fit_identity"]["components"]["fit"]["vintage_id"] is not None


def test_feature_set_route_separates_structurally_distinct_predictors(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    paths: list[str] = []
    for predictor in ("x", "x2"):
        feature_set = mf.feature_engineering.FeatureSet(
            X=bundle.panel[[predictor]],
            y=bundle.panel[["y1"]],
            metadata={},
            feature_metadata=pd.DataFrame(),
            target_metadata=pd.DataFrame(),
            target="y1",
            targets=("y1",),
            horizons=(1,),
            predictors=(predictor,),
        )
        stored = mf.forecasting.run(
            feature_set,
            "ridge",
            window=_window(bundle.panel.index),
            params={"ridge": {"alpha": 0.1}},
            model_selection={"ridge": None},
            save_models=True,
            model_store=tmp_path,
        ).to_frame()["stored_model"].dropna().iloc[0]
        paths.append(stored["model_path"])

    assert len(set(paths)) == 2


@pytest.mark.parametrize(
    "policy",
    ["direct_average", "path_average", "recursive"],
)
def test_multistep_policy_routes_record_owned_fit_coordinates(
    tmp_path: Path,
    policy: str,
) -> None:
    bundle = _bundle()
    features = (
        mf.feature_engineering.feature_spec(
            target="y1",
            predictors=[],
            lags=None,
            target_lags=(0, 1),
        )
        if policy == "recursive"
        else _features("y1")
    )
    row = mf.forecasting.run(
        bundle,
        "ridge",
        window=_window(bundle.panel.index),
        target="y1",
        features=features,
        horizon=3,
        forecast_policy=policy,
        params={"ridge": {"alpha": 0.1}},
        model_selection={"ridge": None},
        save_models=True,
        model_store=tmp_path,
    ).to_frame().iloc[0]

    stored = row["stored_model"]
    fits = list(stored["steps"].values()) if policy == "path_average" else [stored]
    assert fits
    assert len({fit["model_path"] for fit in fits}) == len(fits)
    for fit in fits:
        sidecar = _sidecar(fit)
        identity = sidecar["fit_identity"]
        assert identity["components"]["run"]["forecast_policy"] == policy
        assert identity["components"]["fit"]["target_key"] is not None
        assert f"__f{identity['digest']}" in fit["model_path"]


def test_custom_model_implementation_marker_separates_same_named_models(
    tmp_path: Path,
) -> None:
    bundle = _bundle()

    def fit_low(X, y, **params):
        return mf.models.ridge(X, y, **params)

    def fit_high(X, y, **params):
        return mf.models.ridge(X, y, **params)

    paths: list[str] = []
    for fit_func, marker in ((fit_low, "implementation-a"), (fit_high, "implementation-b")):
        model = mf.models.custom_model(
            "same_named_custom",
            fit_func,
            default_params={"alpha": 0.1},
            mf_digest=marker,
        )
        stored = mf.forecasting.run(
            bundle,
            model,
            window=_window(bundle.panel.index),
            target="y1",
            features=_features("y1"),
            model_selection={"same_named_custom": None},
            save_models=True,
            model_store=tmp_path,
        ).to_frame()["stored_model"].dropna().iloc[0]
        sidecar = _sidecar(stored)
        assert sidecar["fit_identity"]["complete"] is True
        paths.append(stored["model_path"])

    assert len(set(paths)) == 2


def test_unmarked_custom_model_reports_incomplete_fit_identity(tmp_path: Path) -> None:
    bundle = _bundle()

    def fit_custom(X, y, **params):
        return mf.models.ridge(X, y, **params)

    model = mf.models.custom_model(
        "opaque_custom",
        fit_custom,
        default_params={"alpha": 0.1},
    )
    with pytest.warns(UserWarning, match="fit identity contains opaque"):
        stored = mf.forecasting.run(
            bundle,
            model,
            window=_window(bundle.panel.index),
            target="y1",
            features=_features("y1"),
            model_selection={"opaque_custom": None},
            save_models=True,
            model_store=tmp_path,
        ).to_frame()["stored_model"].dropna().iloc[0]

    identity = _sidecar(stored)["fit_identity"]
    assert identity["complete"] is False
    assert "fit_identity.fit.model_implementation" in identity["opaque_fields"]


@pytest.mark.parametrize("cadence_owner", ["window", "stage_policy"])
def test_date_offset_cadences_have_complete_distinct_fit_identities(
    tmp_path: Path,
    cadence_owner: str,
) -> None:
    bundle = _bundle()
    paths: list[str] = []
    for cadence in ("3MS", "6MS"):
        window = _window(
            bundle.panel.index,
            retrain_every=cadence if cadence_owner == "window" else 1,
        )
        feature_policy = mf.window.stage_policy(
            "fit_window",
            update=cadence if cadence_owner == "stage_policy" else "every_origin",
        )
        stored = mf.forecasting.run(
            bundle,
            "ridge",
            window=window,
            target="y1",
            features=_features("y1"),
            feature_policy=feature_policy,
            params={"ridge": {"alpha": 0.1}},
            model_selection={"ridge": None},
            save_models=True,
            model_store=tmp_path,
        ).to_frame()["stored_model"].dropna().iloc[0]
        identity = _sidecar(stored)["fit_identity"]
        assert identity["complete"] is True
        assert identity["opaque_fields"] == []
        paths.append(stored["model_path"])

    assert len(set(paths)) == 2


def test_legacy_and_new_identity_stems_purge_together(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "ridge"
    legacy_dir.mkdir()
    legacy_model = legacy_dir / "origin_0_h1_20000101_direct_h1.pkl"
    legacy_sidecar = legacy_model.with_suffix(".json")
    legacy_model.write_bytes(b"legacy")
    legacy_sidecar.write_text(
        json.dumps({"model_path": str(legacy_model)}),
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory() as separate:
        bundle = _bundle()
        stored = mf.forecasting.run(
            bundle,
            "ridge",
            window=_window(bundle.panel.index),
            target="y1",
            features=_features("y1"),
            params={"ridge": {"alpha": 0.1}},
            model_selection={"ridge": None},
            save_models=True,
            model_store=separate,
        ).to_frame()["stored_model"].dropna().iloc[0]
        new_model = Path(stored["model_path"])
        new_sidecar = Path(stored["metadata_path"])
        target_model = legacy_dir / new_model.name
        target_sidecar = legacy_dir / new_sidecar.name
        target_model.write_bytes(new_model.read_bytes())
        target_sidecar.write_text(
            new_sidecar.read_text(encoding="utf-8").replace(
                str(new_model), str(target_model)
            ).replace(str(new_sidecar), str(target_sidecar)),
            encoding="utf-8",
        )

    assert mf.pipeline.purge_model_store(tmp_path, aliases=["ridge"]) == 2
    assert not list(legacy_dir.glob("*"))
