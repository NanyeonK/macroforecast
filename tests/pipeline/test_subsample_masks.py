from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import EvalSpec, SubsampleWindow, evaluate

eval_mod = importlib.import_module("macroforecast.pipeline.evaluate")


def _spec(evaluation: EvalSpec) -> SimpleNamespace:
    return SimpleNamespace(evaluation=evaluation, combinations=(), arms=(), seed=42)


def _master(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for origin, date in enumerate(dates):
        actual = 1.0 + 0.05 * origin + 0.1 * np.sin(origin / 5.0)
        rows.extend(
            [
                {
                    "target": "y",
                    "horizon": 1,
                    "origin": origin,
                    "date": date,
                    "contender": "AR",
                    "prediction": actual + 0.30,
                    "actual": actual,
                },
                {
                    "target": "y",
                    "horizon": 1,
                    "origin": origin,
                    "date": date,
                    "contender": "OLS",
                    "prediction": actual + 0.08 + 0.01 * (origin % 3),
                    "actual": actual,
                },
            ]
        )
    return pd.DataFrame(rows)


def _fake_fred_bundle(series_id: str, index: pd.DatetimeIndex, values: list[int]) -> mf.data.DataBundle:
    panel = pd.DataFrame({series_id: values}, index=index)
    panel.index.name = "date"
    return mf.data.DataBundle(
        panel=panel,
        metadata={
            "dataset": "fred_series",
            "series_id": series_id,
            "artifact": {
                "source_url": f"https://example.test/{series_id}.csv",
                "local_path": f"/tmp/{series_id}.csv",
                "file_sha256": f"sha-{series_id}",
                "cache_hit": True,
            },
        },
    )


def test_user_mask_splits_rows_by_intersection_and_reaches_all_eval_tables() -> None:
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    state = pd.Series([idx % 4 != 0 for idx in range(len(dates))], index=dates)
    window = SubsampleWindow(
        start="2019-01-01",
        end="2022-12-01",
        exclude=(("2020-01-01", "2020-06-01"),),
        mask=state,
    )
    evaluation = EvalSpec(
        benchmark="AR",
        metrics=("rmse",),
        tests=("dm", "gw", "mcs"),
        subsamples={"state": window},
    )

    res = evaluate(_master(dates), _spec(evaluation))

    expected_dates = [
        date
        for date in dates
        if date >= pd.Timestamp("2019-01-01")
        and date <= pd.Timestamp("2022-12-01")
        and not (pd.Timestamp("2020-01-01") <= date <= pd.Timestamp("2020-06-01"))
        and bool(state.loc[date])
    ]
    row = res["accuracy"].loc[
        (res["accuracy"]["subsample"] == "state")
        & (res["accuracy"]["contender"] == "OLS")
    ].iloc[0]
    assert int(row["n_common"]) == len(expected_dates)
    assert set(res["significance"]["subsample"]) == {"state"}
    assert "gw" in set(res["significance"]["test"].dropna())
    assert set(res["mcs"]["subsample"]) == {"state"}

    provenance = res["forecasts"].attrs["macroforecast_subsample_provenance"]
    summary = provenance["state"]["mask_summary"]
    assert provenance["state"]["mask_source"] == "user_series"
    assert summary["n_obs"] == len(dates)
    assert summary["n_true"] == int(state.sum())
    assert isinstance(summary["sha256"], str)


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (pd.Series([2], index=[pd.Timestamp("2020-01-01")]), "boolean or 0/1"),
        (pd.Series([np.nan], index=[pd.Timestamp("2020-01-01")]), "without NaN"),
        (
            pd.Series(
                [True, False],
                index=[pd.Timestamp("2020-01-01 01:00"), pd.Timestamp("2020-01-01 12:00")],
            ),
            "duplicate normalized date",
        ),
        (pd.Series([True], index=["not-a-date"]), "parseable date"),
        (pd.Series([], dtype=bool, index=pd.DatetimeIndex([])), "at least one date"),
        ("not_a_named_mask", "must be one of"),
    ],
)
def test_subsample_mask_construction_validation(mask, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SubsampleWindow(mask=mask)


def test_named_nber_masks_fetch_and_expansion_is_complement(monkeypatch) -> None:
    dates = pd.date_range("2020-01-01", periods=72, freq="MS")

    def fake_load_fred_series(series_id: str, *, frequency=None, **_kwargs):
        assert series_id == "USREC"
        assert frequency == "monthly"
        values = [1 if idx < 36 else 0 for idx in range(len(dates))]
        return _fake_fred_bundle(series_id, dates, values)

    monkeypatch.setattr(eval_mod, "load_fred_series", fake_load_fred_series)
    evaluation = EvalSpec(
        benchmark="AR",
        metrics=("rmse",),
        tests=("dm",),
        subsamples={
            "recession": SubsampleWindow(mask="nber_recession"),
            "expansion": SubsampleWindow(mask="nber_expansion"),
        },
    )

    res = evaluate(_master(dates), _spec(evaluation))

    counts = {
        name: int(
            res["accuracy"]
            .loc[
                (res["accuracy"]["subsample"] == name)
                & (res["accuracy"]["contender"] == "OLS"),
                "n_common",
            ]
            .iloc[0]
        )
        for name in ("recession", "expansion")
    }
    assert counts == {"recession": 36, "expansion": 36}
    provenance = res["forecasts"].attrs["macroforecast_subsample_provenance"]
    assert provenance["recession"]["mask_source"] == "nber_recession"
    assert provenance["recession"]["mask_summary"]["series_id"] == "USREC"
    assert provenance["recession"]["mask_summary"]["raw_sha256"] == "sha-USREC"


def test_mask_anchor_mismatch_and_partial_coverage_are_strict() -> None:
    target_dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    month_end_mask = pd.Series(
        [True] * len(target_dates),
        index=pd.date_range("2020-01-31", periods=12, freq="ME"),
    )
    with pytest.raises(ValueError, match="month-end.*month-start.*Reindex"):
        evaluate(
            _master(target_dates),
            _spec(
                EvalSpec(
                    benchmark="AR",
                    metrics=("rmse",),
                    tests=("dm",),
                    subsamples={"bad": SubsampleWindow(mask=month_end_mask)},
                )
            ),
        )

    partial = pd.Series([True] * 11, index=target_dates[:11])
    with pytest.raises(ValueError, match="missing 1 forecast target date"):
        evaluate(
            _master(target_dates),
            _spec(
                EvalSpec(
                    benchmark="AR",
                    metrics=("rmse",),
                    tests=("dm",),
                    subsamples={"bad": SubsampleWindow(mask=partial)},
                )
            ),
        )


def test_quarter_start_targets_select_usrecq(monkeypatch) -> None:
    dates = pd.date_range("2020-01-01", periods=16, freq="QS")
    calls: list[str] = []

    def fake_load_fred_series(series_id: str, *, frequency=None, **_kwargs):
        calls.append(series_id)
        assert frequency == "quarterly"
        return _fake_fred_bundle(series_id, dates, [1] * len(dates))

    monkeypatch.setattr(eval_mod, "load_fred_series", fake_load_fred_series)
    evaluation = EvalSpec(
        benchmark="AR",
        metrics=("rmse",),
        tests=("dm",),
        subsamples={"recession": SubsampleWindow(mask="nber_recession")},
    )

    res = evaluate(_master(dates), _spec(evaluation))

    assert calls == ["USRECQ"]
    row = res["accuracy"].loc[
        (res["accuracy"]["subsample"] == "recession")
        & (res["accuracy"]["contender"] == "OLS")
    ].iloc[0]
    assert int(row["n_common"]) == len(dates)
