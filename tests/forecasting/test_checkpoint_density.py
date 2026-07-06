"""Checkpoint schema extension for the density pipeline (Phase 1).

Before this change ``forecasting/checkpoint.py``'s ``LEAN_FORECAST_COLUMNS``
was point-only: a checkpointed/resumed run silently dropped
``variance_prediction``/``quantile_predictions`` for any origin recovered from
disk, and ``pipeline/rescore.py`` (which reconstructs its ENTIRE master frame
from the checkpoint) was therefore ALWAYS point-only regardless of what the
live run emitted. These tests cover the extended schema: ``variance_prediction``
is now a fixed lean column (a plain float); quantile predictions are stored as
wide ``q_<pct>`` columns (parquet needs scalar columns) and reconstructed back
into the SAME ``{level: value}`` mapping representation on load. OLD
checkpoints (written before this column existed) must still load fine and
behave as point-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting import checkpoint as ckpt


def teardown_function() -> None:
    mf.meta.reset_config()


def _panel(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.1 * np.sin(np.arange(n) / 2.0),
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


def _features() -> mf.feature_engineering.FeatureSpec:
    return mf.feature_engineering.feature_spec(target="y", target_lags=[1, 2])


def _cell_horizon_dir(cell: Path) -> Path:
    return cell / "h1"  # horizon=1 (direct policy default) in every test below


class _FakeVarianceFit:
    """Deterministic fit exposing ``predict_variance`` -- same shape as the
    ``x_variance_model`` fixture in ``tests/forecasting/test_forecasting.py``.
    """

    def predict(self, X):
        return np.full(len(X), 2.0)

    def predict_variance(self, X):
        return np.full(len(X), 0.5)


def _fake_variance_model(X, y):
    return _FakeVarianceFit()


def _variance_model_spec():
    return mf.models.ModelSpec(name="fake_variance", family="test", fit_func=_fake_variance_model)


# --------------------------------------------------------------------------- #
# 1. round trip: variance_prediction survives write + load, on-disk and in the
#    reconstructed frame.
# --------------------------------------------------------------------------- #

def test_checkpoint_roundtrip_preserves_variance_prediction(tmp_path: Path) -> None:
    cell = tmp_path / "cell"
    result = mf.forecasting.run(
        _panel(), _variance_model_spec(), window=_window(), features=_features(),
        save_models=False, checkpoint_path=cell,
    )
    table = result.to_frame()
    assert set(table["variance_prediction"].dropna()) == {0.5}

    hdir = _cell_horizon_dir(cell)
    files = sorted(hdir.glob("origin_*.parquet"))
    assert files, "expected per-origin parquet files"
    for path in files:
        frame = pd.read_parquet(path)
        assert "variance_prediction" in frame.columns
        assert set(frame["variance_prediction"].dropna()) == {0.5}

    loaded = ckpt.load_checkpoint_frame(hdir)
    assert set(loaded["variance_prediction"].dropna()) == {0.5}


def test_checkpoint_roundtrip_preserves_quantile_predictions(tmp_path: Path) -> None:
    cell = tmp_path / "cell"
    result = mf.forecasting.run(
        _panel(), "quantile_regression_forest", window=_window(), features=_features(),
        params={"quantile_regression_forest": {
            "n_estimators": 10, "random_state": 0, "quantile_levels": (0.1, 0.5, 0.9),
        }},
        model_selection={"quantile_regression_forest": None},
        save_models=False, checkpoint_path=cell,
    )
    table = result.to_frame()
    live_quantiles = table["quantile_predictions"].dropna()
    assert not live_quantiles.empty
    assert set(live_quantiles.iloc[0]) == {"0.1", "0.5", "0.9"}

    hdir = _cell_horizon_dir(cell)
    files = sorted(hdir.glob("origin_*.parquet"))
    assert files
    for path in files:
        frame = pd.read_parquet(path)
        # wide scalar columns, not a dict column, on disk.
        assert {"q_10", "q_50", "q_90"} <= set(frame.columns)
        for column in ("q_10", "q_50", "q_90"):
            assert frame[column].dropna().map(np.isfinite).all()

    loaded = ckpt.load_checkpoint_frame(hdir)
    assert "quantile_predictions" in loaded.columns
    reconstructed = loaded["quantile_predictions"].dropna()
    assert not reconstructed.empty
    assert set(reconstructed.iloc[0]) == {"0.1", "0.5", "0.9"}

    # values match the live (non-checkpointed) representation exactly, keyed
    # by origin_pos + date (the checkpoint's own row identity).
    live_indexed = table.dropna(subset=["quantile_predictions"]).set_index(["origin_pos", "date"])
    loaded_indexed = loaded.dropna(subset=["quantile_predictions"]).set_index(["origin_pos", "date"])
    common = live_indexed.index.intersection(loaded_indexed.index)
    assert len(common) > 0
    for key in common:
        live_q = live_indexed.loc[key, "quantile_predictions"]
        loaded_q = loaded_indexed.loc[key, "quantile_predictions"]
        for level in ("0.1", "0.5", "0.9"):
            assert np.isclose(live_q[level], loaded_q[level])


def test_checkpoint_files_stay_scalar_only_with_quantile_wide_columns(tmp_path: Path) -> None:
    """The module's own contract (scalar-only parquet columns) must hold even
    once wide q_<pct> columns are added."""
    cell = tmp_path / "cell"
    mf.forecasting.run(
        _panel(), "quantile_regression_forest", window=_window(), features=_features(),
        params={"quantile_regression_forest": {"n_estimators": 10, "random_state": 0}},
        model_selection={"quantile_regression_forest": None},
        save_models=False, checkpoint_path=cell,
    )
    hdir = _cell_horizon_dir(cell)
    for path in sorted(hdir.glob("origin_*.parquet")):
        frame = pd.read_parquet(path)
        for column in frame.columns:
            for value in frame[column].tolist():
                assert not isinstance(value, (dict, list, tuple, set)), (
                    f"non-scalar value {value!r} in column {column!r}"
                )


# --------------------------------------------------------------------------- #
# 2. resume: a checkpoint-loaded (not recomputed) origin's quantile_predictions
#    matches what a full, uninterrupted run produced for that same origin.
# --------------------------------------------------------------------------- #

def test_resume_preserves_quantile_predictions_for_checkpoint_loaded_origin(
    tmp_path: Path, monkeypatch
) -> None:
    cell = tmp_path / "cell"
    run_kwargs = dict(
        window=_window(), features=_features(),
        params={"quantile_regression_forest": {
            "n_estimators": 10, "random_state": 0, "quantile_levels": (0.05, 0.5, 0.95),
        }},
        model_selection={"quantile_regression_forest": None},
        save_models=False,
    )
    full = mf.forecasting.run(_panel(), "quantile_regression_forest", checkpoint_path=cell, **run_kwargs)
    full_frame = full.to_frame().set_index("origin_pos")

    hdir = _cell_horizon_dir(cell)
    positions = sorted(ckpt.completed_origin_positions(hdir))
    assert len(positions) >= 2
    last_pos = positions[-1]
    (hdir / f"origin_{last_pos}.parquet").unlink()

    import macroforecast.forecasting.runner as runner

    computed: list[int] = []
    original = runner._fit_predict_origin

    def _spy(item, *args, **kwargs):
        computed.append(int(item["row"].get("origin_pos")))
        return original(item, *args, **kwargs)

    monkeypatch.setattr(runner, "_fit_predict_origin", _spy)

    resumed = mf.forecasting.run(_panel(), "quantile_regression_forest", checkpoint_path=cell, **run_kwargs)
    resumed_frame = resumed.to_frame().set_index("origin_pos")

    assert computed == [last_pos]  # only the deleted origin recomputed

    # every origin's quantile_predictions -- resumed-from-disk AND recomputed --
    # matches the uninterrupted full run's, value for value.
    for pos in positions:
        full_q = full_frame.loc[pos, "quantile_predictions"]
        resumed_q = resumed_frame.loc[pos, "quantile_predictions"]
        if isinstance(full_q, pd.Series):  # multiple rows at this origin_pos
            full_q, resumed_q = full_q.iloc[0], resumed_q.iloc[0]
        assert isinstance(resumed_q, dict)
        assert set(resumed_q) == set(full_q)
        for level, value in full_q.items():
            assert np.isclose(resumed_q[level], value)


# --------------------------------------------------------------------------- #
# 3. old-schema compat: a checkpoint directory written before this change
#    (no variance_prediction / q_<pct> columns) must still load fine and
#    behave as point-only downstream.
# --------------------------------------------------------------------------- #

_OLD_LEAN_COLUMNS = (
    "target", "horizon", "origin", "origin_pos", "date", "model",
    "prediction", "actual", "forecast_policy", "target_transform",
)


def _write_old_schema_origin(directory: Path, origin_pos: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{
        "target": "y", "horizon": 1, "origin": pd.Timestamp("2001-01-31"),
        "origin_pos": origin_pos, "date": pd.Timestamp("2001-02-28"), "model": "ols",
        "prediction": 1.0, "actual": 1.1, "forecast_policy": "direct", "target_transform": "level",
    }], columns=list(_OLD_LEAN_COLUMNS))
    frame.to_parquet(directory / f"origin_{origin_pos}.parquet", index=False)


def test_old_schema_checkpoint_loads_fine(tmp_path: Path) -> None:
    directory = tmp_path / "old_ckpt"
    _write_old_schema_origin(directory, origin_pos=0)
    _write_old_schema_origin(directory, origin_pos=1)

    loaded = ckpt.load_checkpoint_frame(directory)
    assert len(loaded) == 2
    assert set(_OLD_LEAN_COLUMNS) <= set(loaded.columns)
    # no density columns were ever written -> absent from the schema entirely
    # (not merely NaN), consistent with the whole-frame schema check density_
    # metric/calibration-test requests raise against.
    assert "variance_prediction" not in loaded.columns
    assert "quantile_predictions" not in loaded.columns


def test_old_schema_checkpoint_evaluates_point_only_by_default_and_errors_if_density_requested(
    tmp_path: Path,
) -> None:
    from macroforecast.pipeline.evaluate import evaluate

    directory = tmp_path / "old_ckpt"
    _write_old_schema_origin(directory, origin_pos=0)
    _write_old_schema_origin(directory, origin_pos=1)
    loaded = ckpt.load_checkpoint_frame(directory)
    master = loaded.assign(contender="OLS")

    class _EvalSpecDouble:
        """Duck-typed spec for the FULL evaluate() orchestrator: it needs
        ``combinations``/``arms``/``seed`` too (``apply_combinations``/
        ``significance_table``/``mcs_table`` read them), not just ``evaluation``.
        """

        def __init__(self, evaluation):
            self.evaluation = evaluation
            self.combinations = ()
            self.arms = ()
            self.seed = 0

    from macroforecast.pipeline.spec import EvalSpec

    default_spec = _EvalSpecDouble(EvalSpec(benchmark="OLS"))
    result = evaluate(master, default_spec)
    assert not result["accuracy"].empty
    assert result["density"].empty
    assert result["calibration"].empty

    density_spec = _EvalSpecDouble(EvalSpec(benchmark="OLS", metrics=("rmse", "crps")))
    with pytest.raises(ValueError, match="variance_prediction"):
        evaluate(master, density_spec)


def test_mixed_old_and_new_schema_checkpoint_dir_unions_gracefully(tmp_path: Path) -> None:
    """A directory with one legacy-schema origin file and one new-schema
    (variance-carrying) origin file must load without error; the union fills
    the legacy row's variance_prediction with NaN rather than raising.
    """
    directory = tmp_path / "mixed_ckpt"
    _write_old_schema_origin(directory, origin_pos=0)

    new_frame = pd.DataFrame([{
        "target": "y", "horizon": 1, "origin": pd.Timestamp("2001-03-31"),
        "origin_pos": 1, "date": pd.Timestamp("2001-04-30"), "model": "fake_variance",
        "prediction": 2.0, "actual": 2.2, "forecast_policy": "direct", "target_transform": "level",
        "variance_prediction": 0.5,
    }])
    new_frame.to_parquet(directory / "origin_1.parquet", index=False)

    loaded = ckpt.load_checkpoint_frame(directory)
    assert len(loaded) == 2
    assert "variance_prediction" in loaded.columns
    old_row = loaded.loc[loaded["origin_pos"] == 0, "variance_prediction"]
    new_row = loaded.loc[loaded["origin_pos"] == 1, "variance_prediction"]
    assert old_row.isna().all()
    assert (new_row == 0.5).all()
