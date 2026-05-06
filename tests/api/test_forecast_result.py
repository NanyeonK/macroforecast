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
