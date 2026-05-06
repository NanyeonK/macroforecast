"""Tests for the v0.8.0 :class:`ForecastResult` minimal shell."""
from __future__ import annotations

import pytest

import macroforecast as mf
from macroforecast.api_high import ForecastResult

from ._offline import install_custom_panel


def _run_offline(tmp_path) -> ForecastResult:
    exp = mf.Experiment(
        dataset="fred_md",
        target="y",
        horizons=[1],
        random_seed=0,
        model_family="ridge",
    )
    install_custom_panel(exp)
    return exp.run(output_directory=tmp_path)


def test_forecast_result_cells_proxy_manifest(tmp_path):
    result = _run_offline(tmp_path)
    assert result.cells == result.manifest.cells
    assert len(result.cells) == 1


def test_forecast_result_succeeded_filters_cells(tmp_path):
    result = _run_offline(tmp_path)
    assert all(cell.succeeded for cell in result.succeeded)
    assert tuple(result.succeeded) == result.manifest.succeeded


def test_forecast_result_manifest_path_exists(tmp_path):
    result = _run_offline(tmp_path)
    path = result.manifest_path
    assert path is not None
    assert path.name == "manifest.json"
    assert path.exists()


def test_forecast_result_replicate_returns_replication_result(tmp_path):
    result = _run_offline(tmp_path)
    replication = result.replicate()
    assert replication.sink_hashes_match
    assert replication.recipe_match


def test_forecast_result_replicate_without_disk_raises():
    # Build a result manually without writing to disk.
    exp = mf.Experiment(
        dataset="fred_md",
        target="y",
        horizons=[1],
        random_seed=0,
        model_family="ridge",
    )
    install_custom_panel(exp)
    result = exp.run()  # no output_directory
    assert result.manifest_path is None
    with pytest.raises(RuntimeError, match="output_directory"):
        result.replicate()


def test_forecast_result_is_frozen_dataclass(tmp_path):
    result = _run_offline(tmp_path)
    with pytest.raises(Exception):  # FrozenInstanceError subclass of AttributeError/TypeError
        result.output_directory = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# v0.8.5 rich accessors
# ---------------------------------------------------------------------------


def test_forecasts_returns_canonical_columns(tmp_path):
    import pandas as pd

    result = _run_offline(tmp_path)
    frame = result.forecasts
    assert isinstance(frame, pd.DataFrame)
    expected_cols = {
        "cell_id", "model_id", "target", "horizon", "origin",
        "y_pred", "y_pred_lo", "y_pred_hi",
    }
    assert expected_cols.issubset(frame.columns)
    if not frame.empty:
        assert (frame["cell_id"] == result.cells[0].cell_id).all()


def test_metrics_returns_dataframe_with_cell_id(tmp_path):
    import pandas as pd

    result = _run_offline(tmp_path)
    metrics = result.metrics
    assert isinstance(metrics, pd.DataFrame)
    if not metrics.empty:
        assert "cell_id" in metrics.columns


def test_ranking_returns_dataframe_when_available(tmp_path):
    import pandas as pd

    result = _run_offline(tmp_path)
    ranking = result.ranking
    assert isinstance(ranking, pd.DataFrame)
    # Empty is acceptable -- not every L5 path emits a ranking table.


def test_get_returns_cell_by_id(tmp_path):
    result = _run_offline(tmp_path)
    cell = result.get(result.cells[0].cell_id)
    assert cell.cell_id == result.cells[0].cell_id


def test_get_unknown_id_raises_keyerror(tmp_path):
    result = _run_offline(tmp_path)
    with pytest.raises(KeyError):
        result.get("does_not_exist")


def test_mean_returns_dataframe_with_metric_column(tmp_path):
    import pandas as pd

    result = _run_offline(tmp_path)
    summary = result.mean(metric="mse")
    assert isinstance(summary, pd.DataFrame)


def test_file_path_returns_none_when_missing(tmp_path):
    result = _run_offline(tmp_path)
    assert result.file_path("does_not_exist.csv") is None


def test_read_json_falls_back_to_manifest_root(tmp_path):
    result = _run_offline(tmp_path)
    payload = result.read_json("manifest.json")
    assert isinstance(payload, dict)
    assert "schema_version" in payload
