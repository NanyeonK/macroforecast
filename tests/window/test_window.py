from __future__ import annotations

import pandas as pd
import pytest

import macroforecast as mf
from tests.selection.helpers import xy


def test_expanding_split_uses_past_data_only() -> None:
    splits = mf.window.expanding_split(12, min_train_size=6, horizon=2, step=2)

    seen = list(splits)

    assert seen
    for train_idx, val_idx in seen:
        assert train_idx.max() < val_idx.min()
        assert len(val_idx) == 2


def test_split_table_reports_label_ranges() -> None:
    X, _ = xy(12)

    table = mf.window.split_table(
        "expanding",
        len(X),
        index=X.index,
        min_train_size=6,
        horizon=2,
        step=3,
    )

    assert list(table["n_validation"]) == [2, 2]
    assert table["train_start"].iloc[0] == X.index[0]
    assert table["validation_start"].iloc[0] == X.index[6]


def test_window_spec_builders_split_and_table() -> None:
    X, _ = xy(12)
    window = mf.window.last_block(validation_size=3, embargo=1)

    splits = window.split(len(X))
    table = window.to_table(len(X), index=X.index)

    assert window.to_dict()["method"] == "last_block"
    assert len(splits) == 1
    assert len(splits[0][0]) == 8
    assert table["validation_start"].iloc[0] == X.index[9]


def test_splitters_validate_edge_cases_and_aliases() -> None:
    holdout = mf.window.make_splitter("holdout", 12, validation_size=3)
    expanding = mf.window.make_splitter(
        "time_series_split",
        12,
        min_train_size=6,
        horizon=2,
    )

    assert len(holdout) == 1
    assert expanding
    with pytest.raises(ValueError, match="validation_ratio"):
        next(mf.window.last_block_split(12, validation_ratio=1.0))
    with pytest.raises(ValueError, match="validation_size"):
        next(mf.window.last_block_split(12, validation_size=12))
    with pytest.raises(ValueError, match="no training observations"):
        next(mf.window.last_block_split(5, validation_size=2, embargo=3))
    with pytest.raises(ValueError, match="embargo must be non-negative"):
        next(mf.window.last_block_split(12, validation_size=3, embargo=-1))
    with pytest.raises(ValueError, match="embargo must be non-negative"):
        next(mf.window.poos_split(12, validation_size=3, embargo=-1))
    with pytest.raises(ValueError, match="embargo must be non-negative"):
        next(mf.window.expanding_split(12, min_train_size=6, embargo=-1))
    with pytest.raises(ValueError, match="embargo must be non-negative"):
        next(mf.window.rolling_blocks_split(12, n_blocks=2, embargo=-1))
    with pytest.raises(ValueError, match="embargo must be non-negative"):
        next(mf.window.blocked_kfold_split(12, n_splits=3, embargo=-1))
    with pytest.raises(ValueError, match="index length"):
        mf.window.split_table("last_block", 12, index=pd.RangeIndex(11))
    with pytest.raises(ValueError, match="n_splits"):
        next(mf.window.blocked_kfold_split(12, n_splits=1))
    with pytest.raises(ValueError, match="n_splits must be smaller than or equal to n_samples"):
        next(mf.window.blocked_kfold_split(4, n_splits=5))
    with pytest.raises(ValueError, match="no valid chronological folds"):
        mf.window.make_splitter("blocked_kfold", 4, n_splits=4, embargo=4)
