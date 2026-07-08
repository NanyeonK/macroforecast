from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, rescore, run_pipeline


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
