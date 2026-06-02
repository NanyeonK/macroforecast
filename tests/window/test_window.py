from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from tests.model_selection.helpers import xy


def test_expanding_split_uses_past_data_only() -> None:
    splits = mf.window.expanding_split(12, min_train_size=6, horizon=2, step=2)

    seen = list(splits)

    assert seen
    for train_idx, val_idx in seen:
        assert train_idx.max() < val_idx.min()
        assert len(val_idx) == 2


def test_stage_policy_normalizes_scope_and_update() -> None:
    policy = mf.window.stage_policy(
        "fixed",
        reference_start="2000-01-31",
        reference_end="2019-12-31",
        update="12ME",
    )

    assert isinstance(policy, mf.window.StagePolicy)
    assert policy.to_dict()["scope"] == "fixed_reference"
    assert policy.to_dict()["update"] == "12ME"
    assert mf.window.resolve_stage_policy("train").scope == "fit_window"


def test_stage_policy_resolves_index_and_panel_rows() -> None:
    X, _ = xy(24)
    window = mf.window.spec(
        estimation=mf.window.estimation_rolling(size=8),
        val=mf.window.val_last_block(size=2),
        test=mf.window.test_origins(first_origin=X.index[12], horizon=1),
    )
    item = next(window.iter_origins(X.index))

    available = mf.window.stage_index(X.index, item, mf.window.stage_policy("origin_available"))
    fit = mf.window.stage_index(X.index, item, mf.window.stage_policy("fit_window"))
    reference = mf.window.stage_panel(
        X,
        None,
        mf.window.stage_policy(
            "fixed_reference",
            reference_start=X.index[2],
            reference_end=X.index[5],
        ),
    )

    assert list(available) == list(X.index[4:12])
    assert list(fit) == list(X.index[4:12])
    assert list(reference.index) == list(X.index[2:6])


def test_custom_stage_policy_rejects_missing_or_duplicate_labels() -> None:
    X, _ = xy(12)
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=4),
        val=mf.window.val_last_block(size=2),
        test=mf.window.test_origins(first_origin=X.index[6], horizon=1),
    )
    item = next(window.iter_origins(X.index))

    def missing_label(index, *, item, policy):
        return [index[0], pd.Timestamp("2099-01-31")]

    def duplicate_label(index, *, item, policy):
        return [index[0], index[0]]

    with pytest.raises(ValueError, match="outside the supplied index"):
        mf.window.stage_index(
            X.index,
            item,
            mf.window.custom_stage_policy(missing_label),
        )
    with pytest.raises(ValueError, match="duplicate labels"):
        mf.window.stage_index(
            X.index,
            item,
            mf.window.custom_stage_policy(duplicate_label),
        )


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


def test_direct_window_spec_method_still_drives_val_split() -> None:
    window = mf.window.WindowSpec(method="last_block", validation_size=3)

    splits = window.split(12)

    assert window.to_dict()["method"] == "last_block"
    assert window.to_dict()["val"]["method"] == "last_block"
    assert len(splits) == 1
    assert list(splits[0][1]) == [9, 10, 11]


def test_component_window_builds_test_origin_table() -> None:
    X, _ = xy(24)
    window = mf.window.spec(
        estimation=mf.window.estimation_rolling(size=8, embargo=1, retrain_every=2),
        val=mf.window.val_expanding(min_train_size=8, horizon=3, step=2),
        test=mf.window.test_origins(
            first_origin=X.index[12],
            last_origin=X.index[16],
            horizon=3,
            step=2,
        ),
        alignment=mf.window.alignment_drop_incomplete(),
    )

    origins = window.origins(X.index)
    mask = window.test_mask(X.index)

    assert list(origins["origin"]) == [X.index[12], X.index[14], X.index[16]]
    assert list(origins["n_estimation"]) == [8, 8, 8]
    assert origins["estimation_end"].iloc[0] == X.index[10]
    assert origins["fit_end"].iloc[1] == X.index[10]
    assert origins["estimation_end"].iloc[1] == X.index[12]
    assert origins["test_end"].iloc[0] == X.index[14]
    assert list(origins["retrain"]) == [True, False, True]
    assert mask.loc[X.index[12]] == np.True_
    assert mask.loc[X.index[13]] == np.True_
    assert mask.loc[X.index[19]] == np.False_
    assert window.to_dict()["estimation"]["mode"] == "rolling"
    assert window.to_dict()["val"]["method"] == "expanding"
    assert window.to_dict()["test"]["horizon"] == 3


def test_calendar_step_builds_date_offset_test_origins() -> None:
    X, _ = xy(30)
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=6),
        val=mf.window.val_last_block(size=2),
        test=mf.window.test_origins(
            first_origin=X.index[8],
            last_origin=X.index[14],
            horizon=1,
            step="2ME",
        ),
    )
    offset_window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=6),
        val=mf.window.val_last_block(size=2),
        test=mf.window.test_origins(
            first_origin=X.index[8],
            last_origin=X.index[17],
            horizon=1,
            step=pd.offsets.MonthEnd(3),
        ),
    )

    origins = window.origins(X.index)
    offset_origins = offset_window.origins(X.index)

    assert list(origins["origin_pos"]) == [8, 10, 12, 14]
    assert list(offset_origins["origin_pos"]) == [8, 11, 14, 17]
    assert window.to_dict()["test"]["step"] == "2ME"
    assert offset_window.to_dict()["test"]["step"] == "3ME"
    assert window.validate(X.index)["ok"] is True


def test_calendar_retrain_and_retune_cadences() -> None:
    X, _ = xy(30)
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(
            min_size=6,
            embargo=1,
            retrain_every="6ME",
        ),
        val=mf.window.val_last_block(
            size=2,
            retune_every="3ME",
            retune_on_retrain=True,
        ),
        test=mf.window.test_origins(
            first_origin=X.index[12],
            last_origin=X.index[24],
            horizon=1,
            step="1ME",
        ),
    )
    free_retune = mf.window.spec(
        estimation=mf.window.estimation_expanding(
            min_size=6,
            embargo=1,
            retrain_every="6ME",
        ),
        val=mf.window.val_last_block(
            size=2,
            retune_every="3ME",
            retune_on_retrain=False,
        ),
        test=mf.window.test_origins(
            first_origin=X.index[12],
            last_origin=X.index[24],
            horizon=1,
            step="1ME",
        ),
    )

    plan = window.plan(X.index)
    free_plan = free_retune.plan(X.index)

    assert list(plan.loc[plan["retrain"], "origin_pos"]) == [12, 18, 24]
    assert list(plan.loc[plan["retune"], "origin_pos"]) == [12, 18, 24]
    assert list(free_plan.loc[free_plan["retune"], "origin_pos"]) == [
        12,
        15,
        18,
        21,
        24,
    ]
    assert window.to_dict()["estimation"]["retrain_every"] == "6ME"
    assert window.to_dict()["val"]["retune_every"] == "3ME"
    assert window.validate(X.index)["ok"] is True


def test_irregular_calendar_step_and_cadence_use_next_available_label() -> None:
    full_index = pd.date_range("2000-01-31", periods=18, freq="ME")
    idx = full_index.delete([7, 10, 14])
    X = pd.DataFrame({"x": np.arange(len(idx), dtype=float)}, index=idx)
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(
            min_size=3,
            retrain_every="6ME",
        ),
        val=mf.window.val_last_block(
            size=2,
            retune_every="3ME",
            retune_on_retrain=False,
        ),
        test=mf.window.test_origins(
            first_origin=idx[4],
            last_origin=idx[-1],
            horizon=1,
            step="3ME",
        ),
    )

    plan = window.plan(X.index)

    assert list(plan["origin"]) == [idx[4], idx[7], idx[9], idx[12]]
    assert list(plan["origin_pos"]) == [4, 7, 9, 12]
    assert list(plan.loc[plan["retrain"], "origin_pos"]) == [4, 9]
    assert list(plan.loc[plan["retune"], "origin_pos"]) == [4, 7, 9, 12]
    assert list(plan["test_step"]) == ["3ME", "3ME", "3ME", "3ME"]
    assert list(plan["retrain_cadence"]) == ["6ME", "6ME", "6ME", "6ME"]
    assert list(plan["retune_cadence"]) == ["3ME", "3ME", "3ME", "3ME"]


def test_window_plan_adds_retune_metadata_and_origin_splits() -> None:
    X, _ = xy(24)
    window = mf.window.spec(
        estimation=mf.window.estimation_rolling(size=10, embargo=1, retrain_every=2),
        val=mf.window.val_last_block(size=3, retune_every=2),
        test=mf.window.test_origins(
            first_origin=X.index[12],
            last_origin=X.index[16],
            horizon=2,
            step=2,
        ),
    )

    plan = window.plan(X.index)
    splits = window.val_splits_for_origin(X.index, X.index[12])
    iterated = list(window.iter_origins(X.index))

    assert list(plan["origin"]) == [X.index[12], X.index[14], X.index[16]]
    assert list(plan["retune"]) == [True, False, True]
    assert list(plan["retune_group"]) == [0, 0, 1]
    assert list(plan["retrain_cadence"]) == [2, 2, 2]
    assert list(plan["retune_cadence"]) == [2, 2, 2]
    assert list(plan["retune_on_retrain"]) == [True, True, True]
    assert list(plan["reuse_params"]) == [True, True, True]
    assert list(plan["n_val_splits"]) == [1, 0, 1]
    assert list(plan["n_selection"]) == [10, 10, 10]
    assert plan["selection_start"].iloc[0] == X.index[1]
    assert plan["selection_end"].iloc[0] == X.index[10]
    assert plan["selection_start"].iloc[1] == X.index[1]
    assert plan["selection_start"].iloc[2] == X.index[5]
    assert plan["val_start"].iloc[0] == X.index[8]
    assert plan["val_end"].iloc[0] == X.index[10]
    assert list(splits[0][1]) == [8, 9, 10]
    assert list(iterated[0]["test_idx"]) == [12, 13]
    assert len(iterated[0]["val_splits"]) == 1
    assert iterated[1]["val_splits"] == []
    assert window.to_dict()["val"]["retune_on_retrain"] is True
    assert window.to_dict()["val"]["reuse_params"] is True


def test_retune_policy_can_follow_or_ignore_retrain_cadence() -> None:
    X, _ = xy(24)
    common = {
        "estimation": mf.window.estimation_rolling(
            size=10,
            embargo=1,
            retrain_every=2,
        ),
        "test": mf.window.test_origins(
            first_origin=X.index[12],
            last_origin=X.index[16],
            horizon=2,
            step=2,
        ),
    }
    on_retrain = mf.window.spec(
        **common,
        val=mf.window.val_last_block(size=3, retune_every=1),
    )
    independent = mf.window.spec(
        **common,
        val=mf.window.val_last_block(
            size=3,
            retune_every=1,
            retune_on_retrain=False,
        ),
    )

    report = on_retrain.validate(X.index)

    assert report["ok"] is True
    assert any("fit window lags" in warning for warning in report["warnings"])
    assert list(on_retrain.plan(X.index)["retune"]) == [True, False, True]
    assert list(independent.plan(X.index)["retune"]) == [True, True, True]


def test_validate_requires_reusable_parameters_when_retune_is_skipped() -> None:
    X, _ = xy(24)
    window = mf.window.spec(
        estimation=mf.window.estimation_rolling(size=10, embargo=1),
        val=mf.window.val_last_block(size=3, retune_every=2, reuse_params=False),
        test=mf.window.test_origins(
            first_origin=X.index[12],
            last_origin=X.index[16],
            horizon=2,
            step=2,
        ),
    )

    report = window.validate(X.index)

    assert report["ok"] is False
    assert any("reuse_params" in error for error in report["errors"])


def test_iter_slices_returns_aligned_model_ready_objects() -> None:
    X, y = xy(24)
    window = mf.window.spec(
        estimation=mf.window.estimation_rolling(size=10, embargo=1, retrain_every=2),
        val=mf.window.val_last_block(size=3, retune_every=2),
        test=mf.window.test_origins(
            first_origin=X.index[12],
            last_origin=X.index[16],
            horizon=2,
            step=2,
        ),
    )

    first, second, third = list(window.iter_slices(X, y))

    assert len(first["X_estimation"]) == first["row"]["n_estimation"]
    assert len(first["X_fit"]) == first["row"]["n_fit"]
    assert len(first["X_test"]) == first["row"]["n_test"]
    assert len(first["y_test"]) == 2
    assert len(first["val_splits"]) == 1
    assert second["val_splits"] == []
    assert third["X_fit"].index[0] == X.index[5]


def test_iter_slices_supports_retrain_retune_runner_loop() -> None:
    X, y = xy(28)
    window = mf.window.spec(
        estimation=mf.window.estimation_rolling(size=10, embargo=1, retrain_every=2),
        val=mf.window.val_last_block(size=3, retune_every=2),
        test=mf.window.test_origins(
            first_origin=X.index[12],
            last_origin=X.index[18],
            horizon=2,
            step=2,
        ),
    )
    fit_value: float | None = None
    selected_param: int | None = None
    fit_origins: list[int] = []
    retune_origins: list[int] = []
    predictions: list[pd.Series] = []

    for item in window.iter_slices(X, y):
        row = item["row"]
        origin_pos = int(row["origin_pos"])
        if row["retune"]:
            retune_origins.append(origin_pos)
            selected_param = len(item["val_splits"])
            assert selected_param > 0
            for _, val_idx in item["val_splits"]:
                assert int(val_idx.max()) < origin_pos
        if row["retrain"]:
            fit_origins.append(origin_pos)
            assert selected_param is not None
            fit_value = float(item["y_fit"].mean())
        assert fit_value is not None
        pred = pd.Series(fit_value, index=item["X_test"].index, name="prediction")
        assert len(pred) == int(row["n_test"])
        predictions.append(pred)

    assert fit_origins == [12, 16]
    assert retune_origins == [12, 16]
    assert len(predictions) == 4


def test_window_validate_and_from_cutoffs() -> None:
    X, _ = xy(36)
    window = mf.window.from_cutoffs(
        test_start=X.index[18],
        test_end=X.index[22],
        estimation_start=X.index[0],
        mode="rolling",
        estimation_size=12,
        val_method="last_block",
        val_size=3,
        horizon=2,
        retune_every=2,
    )

    report = window.validate(X.index)

    assert mf.from_cutoffs is mf.window.from_cutoffs
    assert report["ok"] is True
    assert report["n_origins"] == 5
    assert report["n_retune"] == 3
    assert window.to_dict()["metadata"]["from_cutoffs"] is True


def test_from_cutoffs_supports_calendar_cadence() -> None:
    X, _ = xy(36)
    from_cutoffs = mf.window.from_cutoffs(
        test_start=X.index[12],
        test_end=X.index[24],
        estimation_start=X.index[0],
        mode="rolling",
        estimation_size=10,
        val_method="last_block",
        val_size=2,
        horizon=1,
        step="1ME",
        retrain_every="6ME",
        retune_every="6ME",
    )
    direct = mf.window.spec(
        estimation=mf.window.estimation_rolling(
            start=X.index[0],
            size=10,
            retrain_every="6ME",
        ),
        val=mf.window.val_last_block(size=2, retune_every="6ME"),
        test=mf.window.test_origins(
            first_origin=X.index[12],
            last_origin=X.index[24],
            horizon=1,
            step="1ME",
        ),
        metadata={"from_cutoffs": True},
    )

    from_plan = from_cutoffs.plan(X.index)
    direct_plan = direct.plan(X.index)

    pd.testing.assert_frame_equal(
        from_plan.loc[
            :,
            [
                "origin",
                "origin_pos",
                "retrain",
                "retune",
                "test_step",
                "retrain_cadence",
                "retune_cadence",
            ],
        ],
        direct_plan.loc[
            :,
            [
                "origin",
                "origin_pos",
                "retrain",
                "retune",
                "test_step",
                "retrain_cadence",
                "retune_cadence",
            ],
        ],
    )
    assert list(from_plan.loc[from_plan["retrain"], "origin_pos"]) == [12, 18, 24]
    assert list(from_plan.loc[from_plan["retune"], "origin_pos"]) == [12, 18, 24]
    assert from_cutoffs.to_dict()["metadata"]["from_cutoffs"] is True


def test_window_validate_reports_invalid_inner_validation() -> None:
    X, _ = xy(12)
    window = mf.window.spec(
        estimation=mf.window.estimation_rolling(size=4),
        val=mf.window.val_last_block(size=4),
        test=mf.window.test_origins(first_origin=X.index[6], last_origin=X.index[8]),
    )

    report = window.validate(X.index)

    assert report["ok"] is False
    assert "validation_size" in report["errors"][0]


def test_window_alignment_drops_incomplete_rows() -> None:
    X, y = xy(8)
    X = X.copy()
    y = y.copy()
    X.iloc[1, 0] = np.nan
    y.iloc[3] = np.nan
    window = mf.window.spec(alignment=mf.window.alignment_drop_incomplete())

    aligned_X, aligned_y = window.align(X, y)

    assert len(aligned_X) == 6
    assert aligned_X.index.equals(aligned_y.index)
    assert X.index[1] not in aligned_X.index
    assert y.index[3] not in aligned_y.index


def test_window_alignment_can_keep_missing_rows() -> None:
    X, y = xy(8)
    X = X.copy()
    X.iloc[1, 0] = np.nan
    y.iloc[3] = np.nan
    window = mf.window.spec(alignment=mf.window.alignment_keep_missing())

    aligned_X, aligned_y = window.align(X, y)

    assert len(aligned_X) == len(X) - 1
    assert aligned_X.isna().any().any()
    assert aligned_X.index.equals(aligned_y.index)
    assert y.index[3] not in aligned_y.index


def test_component_windows_validate_time_frame_inputs() -> None:
    X, _ = xy(8)
    window = mf.window.spec()

    with pytest.raises(ValueError, match="monotonic increasing"):
        window.origins(X.index[::-1])
    with pytest.raises(ValueError, match="size"):
        mf.window.estimation_rolling(size=0)
    with pytest.raises(ValueError, match="estimation_size"):
        mf.window.from_cutoffs(test_start=X.index[4], mode="rolling")
    with pytest.raises(ValueError, match="horizon"):
        mf.window.test_origins(horizon=0)
    with pytest.raises(ValueError, match="retune_every"):
        mf.window.val_last_block(retune_every=0)
    with pytest.raises(ValueError, match="retrain_every offset string"):
        mf.window.estimation_expanding(retrain_every="")
    with pytest.raises(ValueError, match="retune_every offset string"):
        mf.window.val_last_block(retune_every="")
    with pytest.raises(ValueError, match="step offset string"):
        mf.window.test_origins(step="")
    with pytest.raises(ValueError, match="join"):
        mf.window.alignment_drop_incomplete(join="bad")
    with pytest.raises(TypeError, match="DatetimeIndex"):
        mf.window.spec(test=mf.window.test_origins(step="2ME")).origins(
            pd.RangeIndex(12),
        )
    with pytest.raises(TypeError, match="retrain_every"):
        mf.window.spec(
            estimation=mf.window.estimation_expanding(retrain_every="1ME"),
        ).origins(pd.RangeIndex(12))
    with pytest.raises(TypeError, match="retune_every"):
        mf.window.spec(
            val=mf.window.val_last_block(retune_every="1ME"),
        ).plan(pd.RangeIndex(12))


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


def test_public_window_exports_use_train_val_test_names() -> None:
    assert mf.EstimationWindow is mf.window.EstimationWindow
    assert mf.ValWindow is mf.window.ValWindow
    assert mf.TestWindow is mf.window.TestWindow
    assert mf.test_origins is mf.window.test_origins
    assert mf.estimation_rolling is mf.window.estimation_rolling
    assert mf.val_expanding is mf.window.val_expanding
    assert not hasattr(mf, "TrainWindow")
    assert not hasattr(mf, "train_rolling")
    assert not hasattr(mf, "ForecastWindow")
    assert not hasattr(mf, "forecast_origins")
    assert not hasattr(mf.window, "TrainWindow")
    assert not hasattr(mf.window, "train_rolling")
    assert not hasattr(mf.window, "ForecastWindow")
    assert not hasattr(mf.window, "forecast_origins")
