from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _panel() -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=8, freq="ME")
    return pd.DataFrame(
        {
            "headline": np.arange(10.0, 18.0),
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [3.0, 1.0, 2.0, 5.0, 4.0, 9.0, 6.0, 7.0],
        },
        index=idx,
    )


def test_forward_average_target_records_albacore_source() -> None:
    target = mf.feature_engineering.forward_average_target(
        _panel(),
        target="headline",
        horizon=2,
        transform="change",
    )

    assert target["headline_average_change_h2"].iloc[0] == pytest.approx(1.0)
    assert (
        target.attrs["macroforecast_metadata"]["forward_average_target"][
            "source_method"
        ]
        == "assemblage_forward_target"
    )
    meta = target.attrs["macroforecast_target_metadata"]
    assert meta.loc[0, "source_method"] == "assemblage_forward_target"


def test_rank_space_and_moving_average_changes_match_assemblage_primitives() -> None:
    panel = _panel()

    ranks = mf.feature_engineering.rank_space_features(panel, columns=["a", "b"])
    ma = mf.feature_engineering.moving_average_changes(
        panel,
        columns=["a"],
        window=2,
        method="compound_percent",
    )

    assert ranks.loc[panel.index[1], ["rank_1", "rank_2"]].tolist() == [1.0, 2.0]
    expected = ((1.0 + 1.0 / 100.0) * (1.0 + 2.0 / 100.0) - 1.0) * 100.0
    assert ma["a_ma2"].iloc[1] == pytest.approx(expected)
    assert (
        ranks.attrs["macroforecast_metadata"]["rank_space_features"][
            "source_method"
        ]
        == "assemblage_x_transformation_rank_space"
    )


def test_reference_weight_alignment_and_weighted_aggregate_are_generic() -> None:
    panel = _panel()

    weights = mf.feature_engineering.align_reference_weights(
        {"a": 2.0, "b": 1.0},
        ["a", "b"],
    )
    aggregate = mf.feature_engineering.weighted_aggregate(
        panel,
        weights,
        columns=["a", "b"],
        name="basket",
    )

    assert weights.sum() == pytest.approx(1.0)
    assert aggregate["basket"].iloc[0] == pytest.approx(5.0 / 3.0)
    assert (
        aggregate.attrs["macroforecast_metadata"]["weighted_aggregate"][
            "source_method"
        ]
        == "assemblage_weighted_core_measure"
    )
