from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.model_selection.splitters import resolve_validation_splitter
from tests.model_selection.helpers import first_prediction, score_model, xy


def test_explicit_folds_within_fold_expanding_oracle() -> None:
    splitter = mf.model_selection.explicit_folds(
        [8, 13, 19, 24],
        within_fold="expanding",
    )

    splits, name, metadata = resolve_validation_splitter(pd.RangeIndex(24), splitter)

    expected = [
        (list(range(pos)), [pos])
        for pos in range(8, 24)
    ]
    assert name == "explicit_folds"
    assert metadata["temporal_order"] is True
    assert [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in splits
    ] == expected


def test_explicit_folds_accept_date_label_boundaries() -> None:
    index = pd.date_range("2000-01-31", periods=24, freq="ME")
    splitter = mf.model_selection.explicit_folds(
        [index[7], index[12], index[18], index[23]],
    )

    splits, _, _ = resolve_validation_splitter(index, splitter)

    assert [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in splits
    ] == [
        (list(range(8)), list(range(8, 13))),
        (list(range(13)), list(range(13, 19))),
        (list(range(19)), list(range(19, 24))),
    ]


def test_recursive_threefold_preset_composes_expanding_folds() -> None:
    splits, name, metadata = resolve_validation_splitter(
        pd.RangeIndex(24),
        mf.model_selection.recursive_threefold(),
    )

    assert name == "recursive_threefold"
    assert metadata["temporal_order"] is True
    assert [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in splits
    ] == [
        (list(range(6)), list(range(6, 12))),
        (list(range(12)), list(range(12, 18))),
        (list(range(18)), list(range(18, 24))),
    ]


def test_named_splitter_presets_match_window_splitters_byte_for_byte() -> None:
    index = pd.RangeIndex(30)

    poos, _, _ = resolve_validation_splitter(
        index,
        mf.model_selection.validation_splitter("poos", validation_size=5),
    )
    kfold, _, _ = resolve_validation_splitter(
        index,
        mf.model_selection.validation_splitter("kfold", n_splits=5),
    )

    expected_poos = mf.window.make_splitter("poos", 30, validation_size=5)
    expected_kfold = mf.window.make_splitter("blocked_kfold", 30, n_splits=5)
    assert [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in poos
    ] == [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in expected_poos
    ]
    assert [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in kfold
    ] == [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in expected_kfold
    ]


def test_select_params_uses_search_validation_splitter_metadata() -> None:
    X, y = xy()
    search = mf.model_selection.grid(
        {"alpha": [0.1, 1.0]},
        validation_splitter=mf.model_selection.explicit_folds([20, 30, 40]),
    )

    result = mf.model_selection.select_params(mf.models.ridge, X, y, search)

    assert result.window == "explicit_folds"
    assert result.metadata["split_source"] == "validation_splitter"
    assert result.metadata["validation_splitter"]["method"] == "explicit_folds"
    assert result.metadata["split_summary"] == [
        {
            "split": 0,
            "n_train": 20,
            "n_validation": 10,
            "train_start_pos": 0,
            "train_end_pos": 19,
            "validation_start_pos": 20,
            "validation_end_pos": 29,
        },
        {
            "split": 1,
            "n_train": 30,
            "n_validation": 10,
            "train_start_pos": 0,
            "train_end_pos": 29,
            "validation_start_pos": 30,
            "validation_end_pos": 39,
        },
    ]


def test_two_parameter_grid_is_scored_jointly() -> None:
    X, y = xy()
    losses = {
        (0, 0): 0.0,
        (0, 1): 100.0,
        (1, 0): 1.0,
        (1, 1): 2.0,
    }

    def interaction_model(
        X: pd.DataFrame,
        y: pd.Series,
        *,
        a: int,
        b: int,
    ) -> object:
        return score_model(X, y, score_value=losses[(a, b)])

    result = mf.model_selection.select_params(
        interaction_model,
        X,
        y,
        mf.model_selection.grid({"a": [0, 1], "b": [0, 1]}),
        window=mf.window.last_block(validation_size=6),
        metric=first_prediction,
    )

    trials = result.trials.loc[result.trials["status"] == "ok"]
    marginal_a = trials.groupby("a")["score"].mean().idxmin()
    marginal_b = trials.groupby("b")["score"].mean().idxmin()
    assert len(trials) == 4
    assert result.best_params == {"a": 0, "b": 0}
    assert {"a": int(marginal_a), "b": int(marginal_b)} == {"a": 1, "b": 0}
    assert result.best_params != {"a": int(marginal_a), "b": int(marginal_b)}
