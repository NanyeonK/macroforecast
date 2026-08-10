"""Clark-West must be expressible for a forecast COMBINATION, not just for arms.

``significance_table`` builds its set of CW-eligible contenders by walking
``spec.arms`` and reading ``Arm.nested_in_benchmark``. ``CombinationContender``
had no such field, so a combination could never be CW-eligible: the COMB row
came back with ``cw_stat``/``cw_p`` silently NaN, with no warning saying why.

That is the wrong default for the forecast-combination literature, where CW on
the *combination* is the headline test (Welch and Goyal 2008; Rapach, Strauss
and Zhou 2010) -- and the test is licensed, because a mean of arms that each
nest the benchmark itself nests the benchmark: setting every slope to zero
recovers the benchmark forecast.

Nestedness stays an explicit declaration, exactly as it is for ``Arm``. The
package does not try to infer it from the members, because an estimated-weight
combination need not nest what its members nest.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm,
    CombinationContender,
    EvalSpec,
    TargetSpec,
    pipeline_spec,
    run_pipeline,
)


def _bundle(n: int = 156) -> mf.DataBundle:
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.4 * np.roll(x1, 1) + 0.3 * np.roll(x2, 1) + rng.normal(scale=0.5, size=n)
    y[0] = 0.0
    panel = pd.DataFrame({"Y": y, "X1": x1, "X2": x2}, index=idx)
    return mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})


def _window() -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=72),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=3),
    )


def _spec(*, nested_combination: bool | None):
    """``None`` means: do not pass the argument at all."""
    feats = mf.feature_engineering.feature_spec(target="Y", target_lags=(0, 1))
    arm_kw = dict(target="Y", lags=0, target_lags=None)
    arms = [
        Arm("HA", model="hist_mean", features=feats, is_benchmark=True),
        Arm("A1", model="ols",
            features=mf.feature_engineering.feature_spec(predictors=["X1"], **arm_kw),
            nested_in_benchmark=True),
        Arm("A2", model="ols",
            features=mf.feature_engineering.feature_spec(predictors=["X2"], **arm_kw),
            nested_in_benchmark=True),
    ]
    comb_kw = {} if nested_combination is None else {"nested_in_benchmark": nested_combination}
    return pipeline_spec(
        data=_bundle(),
        targets=[TargetSpec("Y", transform="level", policy="direct")],
        horizons=[1],
        window=_window(),
        arms=arms,
        evaluation=EvalSpec(benchmark="HA", metrics=("rmse",), tests=("dm", "cw")),
        combinations=[CombinationContender(name="COMB", method="mean",
                                           over=("A1", "A2"), **comb_kw)],
        save_models=False,
    )


def _sig(spec) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pipeline(spec).significance


def test_field_exists_and_defaults_to_false() -> None:
    assert "nested_in_benchmark" in CombinationContender.__dataclass_fields__
    assert CombinationContender(name="C", method="mean", over=("A1",)).nested_in_benchmark is False


def test_unflagged_combination_still_gets_dm_but_no_cw() -> None:
    """The pre-existing default is preserved: no CW unless asked for."""
    sig = _sig(_spec(nested_combination=None))
    row = sig[sig["contender"] == "COMB"]
    assert len(row) == 1
    assert row["dm_p"].notna().all(), "DM is always licensed"
    assert row["cw_p"].isna().all(), "CW must not appear for an undeclared combination"
    # the arms are unaffected either way
    assert sig[sig["contender"].isin({"A1", "A2"})]["cw_p"].notna().all()


def test_flagged_combination_gets_clark_west() -> None:
    sig = _sig(_spec(nested_combination=True))
    row = sig[sig["contender"] == "COMB"]
    assert len(row) == 1
    assert row["cw_stat"].notna().all(), "CW must be emitted for a declared combination"
    assert row["cw_p"].notna().all()
    p = float(row["cw_p"].iloc[0])
    assert 0.0 <= p <= 1.0


def test_flagged_combination_cw_matches_a_hand_computed_test() -> None:
    """The emitted statistic is the real CW on the combination path, not a stand-in."""
    spec = _spec(nested_combination=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = run_pipeline(spec)
    fc = rep.forecasts
    piv = fc.pivot_table(index="date", columns="contender", values="prediction")
    act = fc.drop_duplicates("date").set_index("date")["actual"].reindex(piv.index)
    ok = piv[["HA", "COMB"]].notna().all(axis=1) & act.notna()
    a = act[ok].to_numpy(float)
    fb = piv.loc[ok, "HA"].to_numpy(float)
    fcv = piv.loc[ok, "COMB"].to_numpy(float)
    want = mf.tests.clark_west_test((a - fb) ** 2, (a - fcv) ** 2, fb, fcv, horizon=1)
    got = rep.significance[rep.significance["contender"] == "COMB"]
    np.testing.assert_allclose(float(got["cw_stat"].iloc[0]),
                               float(getattr(want, "statistic", np.nan)), rtol=1e-10)


def test_silent_nan_is_warned_about() -> None:
    """The trap this fixes: CW wanted, arms nested, combination silently NaN."""
    with pytest.warns(UserWarning, match="nested_in_benchmark"):
        run_pipeline(_spec(nested_combination=None))
