"""Regression tests for the evaluation-sample truncation bug.

BUG: accuracy_table enforced ONE common sample across ALL contenders
(listwise deletion). A single short-coverage contender then silently truncated
every contender's relRMSE/RMSE/R2 sample to the shortest one, and shifted the
evaluation period with no warning. This made relRMSE incomparable to an external
benchmark (e.g. a paper) that used the full sample.

FIX: pairwise sample (contender vs benchmark) for the per-contender metrics;
the all-contender common sample is kept only where a JOINT sample is required
(MCS), and ragged coverage is warned about so it is visible.
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from macroforecast.pipeline.evaluate import accuracy_table


def _spec(benchmark="FM"):
    return SimpleNamespace(evaluation=SimpleNamespace(benchmark=benchmark))


def _ragged_master(n=120, short_start=60):
    """FM and A cover all n origins; B covers only origins >= short_start."""
    rng = np.random.default_rng(0)
    actual = rng.normal(size=n)
    rows = []
    for i in range(n):
        a = float(actual[i])
        rows.append({"target": "Y", "horizon": 1, "contender": "FM",
                     "origin": i, "prediction": a + rng.normal() * 0.50, "actual": a})
        rows.append({"target": "Y", "horizon": 1, "contender": "A",
                     "origin": i, "prediction": a + rng.normal() * 0.40, "actual": a})
        if i >= short_start:
            rows.append({"target": "Y", "horizon": 1, "contender": "B",
                         "origin": i, "prediction": a + rng.normal() * 0.40, "actual": a})
    return pd.DataFrame(rows)


def test_full_coverage_contender_not_truncated_by_short_arm():
    acc = accuracy_table(_ragged_master(n=120, short_start=60), _spec())
    a = acc[acc.contender == "A"].iloc[0]
    fm = acc[acc.contender == "FM"].iloc[0]
    b = acc[acc.contender == "B"].iloc[0]
    # A and FM both cover all 120; their pairwise sample must be 120, NOT 60.
    assert int(a["n_common"]) == 120, f"A truncated to {a['n_common']} (all-arm intersection bug)"
    assert int(fm["n_common"]) == 120
    # B genuinely only has 60 origins -> its pairwise-with-FM sample is 60.
    assert int(b["n_common"]) == 60


def test_full_coverage_relrmse_matches_full_sample_computation():
    master = _ragged_master(n=120, short_start=60)
    acc = accuracy_table(master, _spec())
    # Independent full-sample relRMSE for A vs FM over all 120 origins.
    w = master.pivot_table(index="origin", columns="contender", values="prediction")
    y = master.groupby("origin")["actual"].first()
    mse_a = ((w["A"] - y) ** 2).mean()
    mse_fm = ((w["FM"] - y) ** 2).mean()
    expected = mse_a / mse_fm
    got = float(acc[acc.contender == "A"].iloc[0]["relative_mse"])
    assert abs(got - expected) < 1e-9, f"A relRMSE {got} != full-sample {expected}"


def test_ragged_coverage_emits_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        accuracy_table(_ragged_master(n=120, short_start=60), _spec())
    msgs = " ".join(str(w.message) for w in caught)
    assert "coverage" in msgs.lower() or "ragged" in msgs.lower() or "origins" in msgs.lower(), \
        f"no ragged-coverage warning emitted; got: {msgs!r}"


def test_uniform_coverage_unchanged():
    # When all contenders cover the same origins, the per-contender n is identical
    # and no warning fires (no behavioural change for the common case).
    rng = np.random.default_rng(1)
    n = 80
    actual = rng.normal(size=n)
    rows = []
    for i in range(n):
        a = float(actual[i])
        for c, s in (("FM", 0.5), ("A", 0.4), ("B", 0.45)):
            rows.append({"target": "Y", "horizon": 1, "contender": c,
                         "origin": i, "prediction": a + rng.normal() * s, "actual": a})
    acc = accuracy_table(pd.DataFrame(rows), _spec())
    assert set(acc["n_common"]) == {n}
