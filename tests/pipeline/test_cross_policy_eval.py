"""``evaluate_cross_policy`` scores contenders from several forecast policies
against ONE benchmark fixed to a single policy (the common-denominator convention).

This is the ergonomic wrapper for the GCLS-style recipe where the direct and the
path-average tables are both scored against the single direct FM. Without it a
user must hand-qualify the contender names by ``forecast_policy`` and re-score
through a hand-built spec; the helper does that and returns a tidy frame.
"""
import numpy as np
import pandas as pd
import pytest

from macroforecast.pipeline import evaluate_cross_policy


def _forecasts():
    """A tiny multi-policy master forecast frame.

    Two arms (FM, AR) each run under two policies (direct_average, path_average)
    for one target Y over two horizons. The benchmark is the DIRECT FM, so every
    contender -- including the path FM and both ARs -- is scored against it.
    """
    rng = np.random.default_rng(0)
    rows = []
    origins = pd.date_range("2009-01-01", periods=20, freq="MS")
    for arm in ("FM", "AR"):
        for policy in ("direct_average", "path_average"):
            for h in (1, 6):
                # distinct error scale per (arm, policy) so relative_mse != 1 off-benchmark
                scale = {"FM": 0.5, "AR": 1.0}[arm] * (1.2 if policy == "path_average" else 1.0)
                actual = rng.normal(size=len(origins))
                pred = actual + rng.normal(size=len(origins)) * scale
                rows.append(pd.DataFrame({
                    "target": "Y", "horizon": h, "origin": origins,
                    "contender": arm, "arm": arm, "forecast_policy": policy,
                    "prediction": pred, "actual": actual,
                }))
    return pd.concat(rows, ignore_index=True)


def test_cross_policy_scores_every_policy_against_one_benchmark():
    acc = evaluate_cross_policy(
        _forecasts(), benchmark="FM", benchmark_policy="direct_average",
    )
    # tidy columns: arm and forecast_policy are split back out
    assert {"arm", "forecast_policy", "relative_mse"}.issubset(acc.columns)
    # the direct FM is the benchmark and scores 1.0 against itself
    direct_fm = acc[(acc["arm"] == "FM") & (acc["forecast_policy"] == "direct_average")]
    assert (direct_fm["is_benchmark"]).all()
    for h in (1, 6):
        row = direct_fm[direct_fm["horizon"] == h].iloc[0]
        assert abs(row["relative_mse"] - 1.0) < 1e-9
    # the PATH FM is a distinct contender, NOT the benchmark, and is finite
    path_fm = acc[(acc["arm"] == "FM") & (acc["forecast_policy"] == "path_average")]
    assert not path_fm["is_benchmark"].any()
    assert np.isfinite(path_fm["relative_mse"]).all()
    # every (arm, policy, horizon) combination is present: 4 contenders x 2 horizons
    assert len(acc) == 8


def test_cross_policy_missing_benchmark_policy_raises():
    with pytest.raises(ValueError, match="not present"):
        evaluate_cross_policy(
            _forecasts(), benchmark="FM", benchmark_policy="recursive",
        )


def test_cross_policy_requires_policy_column():
    bad = _forecasts().drop(columns=["forecast_policy"])
    with pytest.raises(ValueError, match="missing column"):
        evaluate_cross_policy(bad, benchmark="FM", benchmark_policy="direct_average")


def test_cross_policy_rejects_separator_collision():
    df = _forecasts()
    df["forecast_policy"] = df["forecast_policy"].str.replace("_", "::", regex=False)
    with pytest.raises(ValueError, match="separator"):
        evaluate_cross_policy(df, benchmark="FM", benchmark_policy="direct::average")
