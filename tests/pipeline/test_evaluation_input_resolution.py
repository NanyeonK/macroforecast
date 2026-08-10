"""``resolve_evaluation_inputs`` -- the one place an evaluation loads anything.

``evaluate()`` scores a forecast frame and nothing else, so the data behind a
named subsample mask (``"nber_recession"``) is resolved before it runs. These
tests pin what that boundary owes its callers: every distinct FRED series is
fetched once per evaluation operation however many subsamples read it; the
resolved series/frequency/inversion and provenance match what the evaluator used
to derive for itself; nothing at all is loaded when no name is used; and the
strict frame-shaped errors still come from the evaluator, with their original
wording, rather than being pre-empted by a fetch.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    EvalSpec,
    SubsampleWindow,
    evaluate,
    resolve_evaluation_inputs,
)
import macroforecast.pipeline.evaluation_inputs as inputs_mod


def _spec(evaluation: EvalSpec) -> SimpleNamespace:
    return SimpleNamespace(evaluation=evaluation, combinations=(), arms=(), seed=42)


def _master(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for origin, date in enumerate(dates):
        actual = 1.0 + 0.05 * origin + 0.1 * np.sin(origin / 5.0)
        rows.extend(
            [
                {
                    "target": "y", "horizon": 1, "origin": origin, "date": date,
                    "contender": "AR", "prediction": actual + 0.30, "actual": actual,
                },
                {
                    "target": "y", "horizon": 1, "origin": origin, "date": date,
                    "contender": "OLS", "prediction": actual + 0.08 + 0.01 * (origin % 3),
                    "actual": actual,
                },
            ]
        )
    return pd.DataFrame(rows)


def _bundle(series_id: str, index: pd.DatetimeIndex, values: list[int]) -> mf.data.DataBundle:
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


def _record_loads(monkeypatch, dates: pd.DatetimeIndex, values: list[int]) -> list[tuple[str, str]]:
    """Patch the loader to record ``(series_id, frequency)`` and return the log."""
    calls: list[tuple[str, str]] = []

    def fake_load_fred_series(series_id: str, *, frequency=None, **_kwargs):
        calls.append((series_id, frequency))
        return _bundle(series_id, dates, values)

    monkeypatch.setattr(inputs_mod, "load_fred_series", fake_load_fred_series)
    return calls


def _forbid_loads(monkeypatch) -> None:
    def dead_loader(*_args, **_kwargs):
        raise AssertionError("nothing here should load a FRED series")

    monkeypatch.setattr(inputs_mod, "load_fred_series", dead_loader)


def test_recession_and_expansion_share_one_fetch_per_evaluation(monkeypatch) -> None:
    dates = pd.date_range("2020-01-01", periods=72, freq="MS")
    calls = _record_loads(monkeypatch, dates, [1 if idx < 36 else 0 for idx in range(72)])
    spec = _spec(
        EvalSpec(
            benchmark="AR",
            metrics=("rmse",),
            tests=("dm",),
            subsamples={
                "recession": SubsampleWindow(mask="nber_recession"),
                "expansion": SubsampleWindow(mask="nber_expansion"),
                # A third name for the same series: still one fetch.
                "recession_again": SubsampleWindow(mask="nber_recession"),
            },
        )
    )

    inputs = resolve_evaluation_inputs(_master(dates), spec)

    assert calls == [("USREC", "monthly")]
    assert set(inputs.subsample_masks) == {"recession", "expansion", "recession_again"}
    # One load, but each name gets its own state: expansion is the complement.
    recession = inputs.subsample_masks["recession"].state
    expansion = inputs.subsample_masks["expansion"].state
    assert bool((expansion == ~recession).all())
    assert int(recession.sum()) == 36
    assert int(expansion.sum()) == 36


def test_a_second_evaluation_resolves_again_rather_than_reusing_a_stale_load(monkeypatch):
    """Once per evaluation *operation*, not once per process: no hidden global cache."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    calls = _record_loads(monkeypatch, dates, [1] * 24)
    spec = _spec(
        EvalSpec(
            benchmark="AR",
            metrics=("rmse",),
            tests=("dm",),
            subsamples={"recession": SubsampleWindow(mask="nber_recession")},
        )
    )
    master = _master(dates)

    resolve_evaluation_inputs(master, spec)
    resolve_evaluation_inputs(master, spec)

    assert calls == [("USREC", "monthly"), ("USREC", "monthly")]


def test_quarterly_targets_resolve_the_quarterly_series(monkeypatch) -> None:
    dates = pd.date_range("2020-01-01", periods=16, freq="QS")
    calls = _record_loads(monkeypatch, dates, [1] * 16)
    spec = _spec(
        EvalSpec(
            benchmark="AR",
            metrics=("rmse",),
            tests=("dm",),
            subsamples={
                "recession": SubsampleWindow(mask="nber_recession"),
                "expansion": SubsampleWindow(mask="nber_expansion"),
            },
        )
    )

    inputs = resolve_evaluation_inputs(_master(dates), spec)

    assert calls == [("USRECQ", "quarterly")]
    summary = inputs.subsample_masks["expansion"].mask_summary
    assert summary["series_id"] == "USRECQ"
    assert summary["frequency"] == "quarterly"


def test_resolved_provenance_carries_the_artifact_fields_the_report_publishes(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    _record_loads(monkeypatch, dates, [1 if idx < 12 else 0 for idx in range(36)])
    spec = _spec(
        EvalSpec(
            benchmark="AR",
            metrics=("rmse",),
            tests=("dm",),
            subsamples={
                "recession": SubsampleWindow(mask="nber_recession"),
                "expansion": SubsampleWindow(mask="nber_expansion"),
            },
        )
    )
    master = _master(dates)

    res = evaluate(master, spec, inputs=resolve_evaluation_inputs(master, spec))

    provenance = res["forecasts"].attrs["macroforecast_subsample_provenance"]
    assert provenance["recession"]["mask_source"] == "nber_recession"
    assert provenance["expansion"]["mask_source"] == "nber_expansion"
    for name, n_true in (("recession", 12), ("expansion", 24)):
        summary = provenance[name]["mask_summary"]
        assert summary["series_id"] == "USREC"
        assert summary["frequency"] == "monthly"
        assert summary["source_url"] == "https://example.test/USREC.csv"
        assert summary["cache_path"] == "/tmp/USREC.csv"
        assert summary["raw_sha256"] == "sha-USREC"
        assert summary["cache_hit"] is True
        assert summary["n_obs"] == 36
        assert summary["n_true"] == n_true
        assert summary["first"] == "2020-01-01"
        assert summary["last"] == "2022-12-01"


@pytest.mark.parametrize(
    "subsamples",
    [
        pytest.param(None, id="no_subsamples"),
        pytest.param(
            {"early": SubsampleWindow(end="2020-12-01")}, id="date_window_only"
        ),
        pytest.param(
            {
                "state": SubsampleWindow(
                    mask=pd.Series(
                        [idx % 2 == 0 for idx in range(24)],
                        index=pd.date_range("2020-01-01", periods=24, freq="MS"),
                    )
                )
            },
            id="user_series_mask",
        ),
    ],
)
def test_evaluations_that_name_nothing_load_nothing_and_need_no_inputs(
    monkeypatch, subsamples
) -> None:
    """The common case stays exactly as pure as it always was, inputs or not."""
    _forbid_loads(monkeypatch)
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    master = _master(dates)
    spec = _spec(
        EvalSpec(benchmark="AR", metrics=("rmse",), tests=("dm",), subsamples=subsamples)
    )

    inputs = resolve_evaluation_inputs(master, spec)
    assert dict(inputs.subsample_masks) == {}

    with_inputs = evaluate(master, spec, inputs=inputs)
    without_inputs = evaluate(master, spec)
    for table in ("accuracy", "significance", "mcs", "density", "calibration"):
        pd.testing.assert_frame_equal(with_inputs[table], without_inputs[table])


def test_resolution_leaves_frame_shaped_errors_to_the_evaluator(monkeypatch) -> None:
    """A frame that cannot be scored is not worth a fetch, and keeps its own message."""
    _forbid_loads(monkeypatch)
    spec = _spec(
        EvalSpec(
            benchmark="AR",
            metrics=("rmse",),
            tests=("dm",),
            subsamples={"recession": SubsampleWindow(mask="nber_recession")},
        )
    )

    no_dates = _master(pd.date_range("2020-01-01", periods=12, freq="MS")).drop(
        columns=["date"]
    )
    assert dict(resolve_evaluation_inputs(no_dates, spec).subsample_masks) == {}
    with pytest.raises(ValueError, match="has no 'date' column"):
        evaluate(no_dates, spec, inputs=resolve_evaluation_inputs(no_dates, spec))

    bad_dates = _master(pd.date_range("2020-01-01", periods=12, freq="MS"))
    bad_dates.loc[0, "date"] = pd.NaT
    assert dict(resolve_evaluation_inputs(bad_dates, spec).subsample_masks) == {}
    with pytest.raises(ValueError, match="invalid or missing value"):
        evaluate(bad_dates, spec, inputs=resolve_evaluation_inputs(bad_dates, spec))


def test_inputs_resolved_for_a_different_mask_are_refused(monkeypatch) -> None:
    """Provenance must describe the mask that was applied, so a mismatch raises."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    _record_loads(monkeypatch, dates, [1 if idx < 12 else 0 for idx in range(24)])
    master = _master(dates)
    recession = _spec(
        EvalSpec(
            benchmark="AR",
            metrics=("rmse",),
            tests=("dm",),
            subsamples={"phase": SubsampleWindow(mask="nber_recession")},
        )
    )
    expansion = _spec(
        EvalSpec(
            benchmark="AR",
            metrics=("rmse",),
            tests=("dm",),
            subsamples={"phase": SubsampleWindow(mask="nber_expansion")},
        )
    )

    stale = resolve_evaluation_inputs(master, recession)
    with pytest.raises(ValueError, match="resolved mask passed for it is 'nber_recession'"):
        evaluate(master, expansion, inputs=stale)
