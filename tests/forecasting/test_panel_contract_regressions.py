"""Regression coverage for panel test-target and target-transform contracts."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import macroforecast as mf

_PREDICT_INPUTS: list[pd.DataFrame] = []
_SENTINEL = -999.0


class _PanelProbeFit:
    def __init__(self, target: str) -> None:
        self.target = target

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        _PREDICT_INPUTS.append(panel.copy())
        return panel[self.target].fillna(_SENTINEL).rename("prediction")


def _fit_panel_probe(
    data: Any,
    *,
    target: str = "y",
) -> _PanelProbeFit:
    del data
    return _PanelProbeFit(target)


def _model() -> mf.models.ModelSpec:
    return mf.models.custom_model(
        "panel_contract_probe",
        _fit_panel_probe,
        default_params={"target": None},
        input_kind="panel",
    )


def _panel() -> pd.DataFrame:
    index = pd.date_range("2000-01-01", periods=72, freq="MS", name="date")
    rng = np.random.default_rng(31)
    return pd.DataFrame(
        {"y": rng.normal(size=len(index)), "x": rng.normal(size=len(index))},
        index=index,
    )


def _window(index: pd.DatetimeIndex):
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(
            first_origin=index[48],
            last_origin=index[54],
            horizon=2,
            step=2,
        ),
    )


def _run(
    model: str | mf.models.ModelSpec,
    *,
    name: str,
    target_transform: str | None,
) -> pd.DataFrame:
    panel = _panel()
    params = {name: {"n_lag": 1}} if name == "var" else None
    return mf.forecasting.run(
        panel,
        model,
        window=_window(panel.index),
        target="y",
        horizon=2,
        forecast_policy="direct",
        target_transform=target_transform,
        model_selection={name: None},
        params=params,
        save_models=False,
    ).to_frame()


def test_generic_panel_predict_input_masks_only_the_target() -> None:
    _PREDICT_INPUTS.clear()
    panel = _panel()
    table = _run(_model(), name="panel_contract_probe", target_transform="level")

    assert not table.empty
    assert table["prediction"].eq(_SENTINEL).all()
    assert table["actual"].notna().all()
    assert _PREDICT_INPUTS
    for received in _PREDICT_INPUTS:
        assert list(received.columns) == list(panel.columns)
        assert received["y"].isna().all()
        pdt.assert_series_equal(received["x"], panel.loc[received.index, "x"])


@pytest.mark.parametrize(
    "target_transform",
    ["change", "growth", "log_growth", "average_change", "average_value"],
)
@pytest.mark.parametrize("model_kind", ["custom", "var"])
def test_panel_runner_refuses_transformed_targets(
    target_transform: str,
    model_kind: str,
) -> None:
    model: str | mf.models.ModelSpec
    if model_kind == "var":
        model, name = "var", "var"
    else:
        model, name = _model(), "panel_contract_probe"
    with pytest.raises(ValueError, match="panel-input models cannot produce"):
        _run(model, name=name, target_transform=target_transform)


@pytest.mark.parametrize(
    ("requested", "reported"),
    [
        (None, "level"),
        ("level", "level"),
        ("future_level", "level"),
        ("value", "level"),
        ("identity", "level"),
    ],
)
def test_panel_runner_reports_normalized_raw_value_transform(
    requested: str | None,
    reported: str,
) -> None:
    table = _run(_model(), name="panel_contract_probe", target_transform=requested)
    assert not table.empty
    assert set(table["target_transform"]) == {reported}
