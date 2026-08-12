"""Regression coverage for raw-value aliases in forecast scale views."""
from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

import macroforecast as mf


def _forecasts(transform: str | None = "value") -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2000-02-29")],
            "origin": [pd.Timestamp("2000-01-31")],
            "origin_pos": [0],
            "horizon": [1],
            "forecast_policy": ["direct"],
            "target": ["y"],
            "model": ["probe"],
            "model_spec": ["probe"],
            "prediction": [2.5],
            "actual": [3.0],
            "combined": [False],
        }
    )
    if transform is not None:
        frame["target_transform"] = transform
    return frame


@pytest.mark.parametrize(
    ("view", "expected_rows"),
    [
        ("transformed_only", 1),
        ("back_transformed_only", 1),
        ("both_overlay", 2),
    ],
)
def test_value_is_available_in_every_scale_view(view: str, expected_rows: int) -> None:
    scale = mf.forecast_analysis.forecast_scale_view(_forecasts(), view=view)

    assert len(scale) == expected_rows
    assert scale["back_transform_available"].map(bool).all()
    assert scale["prediction"].eq(2.5).all()
    assert scale["actual"].eq(3.0).all()


def test_level_and_value_have_identical_scale_semantics() -> None:
    level = mf.forecast_analysis.forecast_scale_view(_forecasts("level"))
    value = mf.forecast_analysis.forecast_scale_view(_forecasts("value"))

    pdt.assert_frame_equal(
        level.drop(columns="target_transform"),
        value.drop(columns="target_transform"),
    )


@pytest.mark.parametrize("transform", ["Value", "LEVEL"])
def test_raw_value_matching_preserves_existing_case_insensitivity(transform: str) -> None:
    scale = mf.forecast_analysis.forecast_scale_view(
        _forecasts(transform),
        view="back_transformed_only",
    )

    assert bool(scale.iloc[0]["back_transform_available"])
    assert scale.iloc[0]["prediction"] == 2.5


@pytest.mark.parametrize("transform", [" value ", "identity", "average_value"])
def test_raw_value_matching_does_not_widen_to_other_labels(transform: str) -> None:
    scale = mf.forecast_analysis.forecast_scale_view(
        _forecasts(transform),
        view="back_transformed_only",
    )

    row = scale.iloc[0]
    assert not bool(row["back_transform_available"])
    assert row["prediction"] is None
    assert row["actual"] is None


def test_value_metadata_default_is_available_without_a_transform_column() -> None:
    forecasts = _forecasts(None)
    forecasts.attrs["macroforecast_metadata"] = {
        "forecast_policy": {"target_transform": "value"}
    }

    scale = mf.forecast_analysis.forecast_scale_view(
        forecasts,
        view="back_transformed_only",
    )

    row = scale.iloc[0]
    assert bool(row["back_transform_available"])
    assert row["prediction"] == 2.5
    assert row["actual"] == 3.0


def test_value_keeps_custom_back_transform_precedence() -> None:
    scale = mf.forecast_analysis.forecast_scale_view(
        _forecasts(),
        view="back_transformed_only",
        back_transform=lambda **_: {"prediction": 25.0, "actual": 30.0},
    )

    row = scale.iloc[0]
    assert bool(row["back_transform_available"])
    assert row["prediction"] == 25.0
    assert row["actual"] == 30.0
