from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, rescore, run_pipeline
from macroforecast.pipeline.selection_history import _parse_cell_dir


def _selection_history_spec(checkpoint_dir, *, selection_history: bool = True):
    idx = pd.date_range("2000-01-01", periods=48, freq="MS", name="date")
    rng = np.random.default_rng(7)
    t = np.arange(len(idx), dtype=float)
    panel = pd.DataFrame(
        {
            "y": np.r_[0.0, t[:-1]] + rng.normal(scale=0.01, size=len(idx)),
            "x1": t,
            "x2": np.tile([1.0, -1.0], len(idx) // 2),
        },
        index=idx,
    )
    bundle = mf.data.custom_dataset(panel, transform_codes={column: 1 for column in panel.columns})
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.predictor_screen(
                method="t_stat",
                top_k=1,
                min_k=1,
            )
        ],
        drop_missing=False,
    )
    window = mf.window.from_cutoffs(
        test_start="2003-01-01",
        test_end="2003-03-01",
        mode="expanding",
        val_method="last_block",
        retrain_every=1,
    )
    return pipeline_spec(
        data=bundle,
        targets=["y"],
        horizons=[1],
        window=window,
        arms=[
            Arm(
                "RIDGE",
                model="ridge",
                features=features,
                params={"alpha": 0.2},
                model_selection={"ridge": None},
            )
        ],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        checkpoint_dir=None if checkpoint_dir is None else str(checkpoint_dir),
        selection_history=selection_history,
        save_models=False,
    )


def _sort_history(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["arm", "horizon", "origin_pos", "kind", "name", "value"]
    return frame.sort_values(columns).reset_index(drop=True)


def test_selection_history_records_exact_feature_rows_and_params(tmp_path) -> None:
    ckpt = tmp_path / "ckpt"
    report = run_pipeline(_selection_history_spec(ckpt))

    sidecars = sorted((ckpt / "y__RIDGE" / "h1").glob("origin_*_selection.jsonl"))
    assert len(sidecars) == 3

    history = mf.pipeline.selection_history(report)
    feature_rows = history[history["kind"] == "feature"].sort_values("origin_pos")
    assert feature_rows[["arm", "horizon", "name", "value", "method"]].to_dict("records") == [
        {"arm": "RIDGE", "horizon": 1, "name": "x1", "value": 1, "method": "t_stat"},
        {"arm": "RIDGE", "horizon": 1, "name": "x1", "value": 1, "method": "t_stat"},
        {"arm": "RIDGE", "horizon": 1, "name": "x1", "value": 1, "method": "t_stat"},
    ]
    param_rows = history[(history["kind"] == "param") & (history["name"] == "alpha")]
    assert set(param_rows["value"]) == {0.2}
    assert mf.selection_history(report).equals(history)
    assert mf.selection_frequency_table is mf.pipeline.selection_frequency_table


def test_selection_history_default_writes_no_sidecars(tmp_path) -> None:
    ckpt = tmp_path / "ckpt"
    run_pipeline(_selection_history_spec(ckpt, selection_history=False))

    assert not list(ckpt.rglob("*_selection.jsonl"))
    assert mf.pipeline.selection_history(ckpt).empty


def test_selection_history_requires_checkpoint_dir() -> None:
    with pytest.raises(ValueError, match="requires checkpoint_dir"):
        _selection_history_spec(None, selection_history=True)


def test_selection_history_survives_rescore(tmp_path) -> None:
    ckpt = tmp_path / "ckpt"
    spec = _selection_history_spec(ckpt)
    live = run_pipeline(spec)
    rescored = rescore(ckpt, spec)

    pd.testing.assert_frame_equal(
        _sort_history(mf.pipeline.selection_history(live)),
        _sort_history(mf.pipeline.selection_history(rescored)),
    )


def test_selection_frequency_table_counts_origin_frequency(tmp_path) -> None:
    ckpt = tmp_path / "ckpt"
    report = run_pipeline(_selection_history_spec(ckpt))

    table = mf.pipeline.selection_frequency_table(
        report,
        by=("arm", "horizon", "kind", "name"),
    )
    x1 = table[(table["kind"] == "feature") & (table["name"] == "x1")].iloc[0]
    assert x1["n_selected"] == 3
    assert x1["n_origins"] == 3
    assert x1["frequency"] == pytest.approx(1.0)


def _write_history_sidecar(h_dir, *, target=None, arm=None) -> None:
    h_dir.mkdir(parents=True)
    (h_dir / "origin_0.parquet").touch()
    record = {
        "horizon": 1,
        "origin": 0,
        "origin_pos": 0,
        "kind": "feature",
        "name": "x1",
        "value": 1,
    }
    if target is not None:
        record["target"] = target
    if arm is not None:
        record["arm"] = arm
    (h_dir / "origin_0_selection.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RIDGE", (None, "RIDGE")),
        ("Y__RIDGE", ("Y", "RIDGE")),
        ("GDP__DEF__AR", (None, None)),
        ("a___b", (None, None)),
        ("a____b", (None, None)),
    ],
)
def test_parse_cell_dir_fails_closed_on_ambiguous_separator(name, expected) -> None:
    assert _parse_cell_dir(name) == expected


def test_path_loader_preserves_ambiguous_sidecar_identity(tmp_path) -> None:
    _write_history_sidecar(
        tmp_path / "GDP__DEF__AR" / "h1", target="GDP__DEF", arm="AR"
    )

    history = mf.pipeline.selection_history(tmp_path)

    assert history[["target", "arm"]].iloc[0].to_dict() == {
        "target": "GDP__DEF",
        "arm": "AR",
    }
    assert history.loc[0, "horizon"] == 1


def test_path_loader_keeps_ambiguous_missing_identity_unknown(tmp_path) -> None:
    _write_history_sidecar(tmp_path / "GDP__DEF__AR" / "h3")

    history = mf.pipeline.selection_history(tmp_path)

    assert pd.isna(history.loc[0, "target"])
    assert pd.isna(history.loc[0, "arm"])
    assert history.loc[0, "horizon"] == 3


def test_path_loader_fills_only_missing_sidecar_identity(tmp_path) -> None:
    _write_history_sidecar(tmp_path / "GDP__AR" / "h1", target="RAW_GDP")

    history = mf.pipeline.selection_history(tmp_path)

    assert history.loc[0, "target"] == "RAW_GDP"
    assert history.loc[0, "arm"] == "AR"
