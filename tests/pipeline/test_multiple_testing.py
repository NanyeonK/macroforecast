"""`EvalSpec.multiple_testing` was declared but refused: setting it raised.

Comparing N contenders against one benchmark is N tests, so an unadjusted 5%
rule expects one false winner in twenty even when nothing has any skill. The
field now works, over the family a reader actually scans at once -- the
contenders within one (target, horizon) cell.

Issue #454.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline
from macroforecast.tests import adjust_pvalues, romano_wolf_pvalues

# p-values and their adjustments as the textbook defines them, so the
# implementation is pinned to the definition rather than to its own output.
_P = [0.01, 0.04, 0.03, 0.20]
_EXPECTED = {
    "bonferroni": [0.04, 0.16, 0.12, 0.80],
    # sorted 0.01 0.03 0.04 0.20; multipliers 4 3 2 1; running max
    "holm": [0.04, 0.09, 0.09, 0.20],
    # sorted; n/i multipliers 4, 2, 4/3, 1; reverse running min
    "bh": [0.04, 0.0533333333, 0.0533333333, 0.20],
}


@pytest.mark.parametrize("method", sorted(_EXPECTED))
def test_the_closed_form_adjustments_match_their_definitions(method: str) -> None:
    np.testing.assert_allclose(
        adjust_pvalues(_P, method=method), _EXPECTED[method], rtol=0, atol=1e-9
    )


def test_a_test_that_did_not_run_is_not_part_of_the_family() -> None:
    """NaN carries through and does not inflate N for the others."""
    with_nan = adjust_pvalues([0.01, np.nan, 0.03, 0.20], method="bonferroni")
    assert np.isnan(with_nan[1])
    # three finite p-values, so the multiplier is 3 and not 4
    np.testing.assert_allclose(with_nan[[0, 2, 3]], [0.03, 0.09, 0.60], rtol=0, atol=1e-9)


def test_the_adjustments_are_ordered_by_conservatism() -> None:
    bonf = adjust_pvalues(_P, method="bonferroni")
    holm = adjust_pvalues(_P, method="holm")
    bh = adjust_pvalues(_P, method="bh")
    assert np.all(bh <= holm + 1e-12), "BH controls FDR so it cannot exceed Holm"
    assert np.all(holm <= bonf + 1e-12), "Holm is uniformly at least as powerful as Bonferroni"


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="bonferroni"):
        adjust_pvalues(_P, method="sidak")
    with pytest.raises(ValueError, match="romano_wolf_pvalues"):
        adjust_pvalues(_P, method="romano_wolf")


def test_romano_wolf_keeps_the_genuine_winner_and_rejects_the_rest() -> None:
    """Three worthless contenders and one real one, in the panel's own units."""
    rng = np.random.default_rng(0)
    n = 200
    diffs = np.column_stack(
        [rng.normal(size=n), rng.normal(size=n), rng.normal(size=n),
         rng.normal(loc=0.35, size=n)]
    )
    p = romano_wolf_pvalues(diffs, n_boot=499, block_length=3, random_state=0)
    assert p[3] < 0.05, "the contender with a genuinely positive differential should survive"
    assert np.all(p[:3] > 0.10), "the three worthless contenders should not"


def test_romano_wolf_is_no_more_conservative_than_holm() -> None:
    """Its whole point: it uses the cross-contender dependence Holm cannot see."""
    from scipy import stats

    rng = np.random.default_rng(1)
    n = 200
    common = rng.normal(size=(n, 1))
    diffs = common + 0.3 * rng.normal(size=(n, 4))  # strongly correlated contenders
    raw = np.array([
        1.0 - stats.norm.cdf(np.sqrt(n) * diffs[:, j].mean() / diffs[:, j].std(ddof=1))
        for j in range(4)
    ])
    rw = romano_wolf_pvalues(diffs, n_boot=499, block_length=3, random_state=0)
    holm = adjust_pvalues(raw, method="holm")
    assert np.all(rw <= holm + 1e-9), "Romano-Wolf should not be stricter than Holm"


def test_romano_wolf_is_deterministic() -> None:
    rng = np.random.default_rng(2)
    diffs = rng.normal(size=(120, 3))
    a = romano_wolf_pvalues(diffs, n_boot=199, random_state=7)
    b = romano_wolf_pvalues(diffs, n_boot=199, random_state=7)
    np.testing.assert_array_equal(a, b)


@lru_cache(maxsize=None)
def _fitted(multiple_testing):
    """One small pipeline per method, cached -- several tests want the same run.

    Deliberately tiny (15 origins, three contenders): this exercises the
    plumbing, and the adjustment arithmetic itself is pinned against its
    definition in the unit tests above, where it costs nothing.
    """
    n = 70
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(3)}, index=idx)
    panel["y"] = 0.5 * panel["x0"] + rng.normal(size=n) * 0.4
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    window = mf.window.from_cutoffs(
        test_start=idx[55], horizon=1, embargo=0,
        val_method="expanding", val_min_train_size=10,
    )
    features = mf.feature_engineering.feature_spec(
        target="y", predictors=[f"x{i}" for i in range(3)], lags=0, target_lags=1
    )
    arms = [Arm("AR", model="ar", is_benchmark=True)] + [
        Arm(m.upper(), model=m, features=features)
        for m in ("ols", "ridge", "lasso")
    ]
    spec = pipeline_spec(
        data=bundle, targets=[TargetSpec("y", transform="level")], horizons=[1],
        window=window, arms=arms,
        evaluation=EvalSpec(benchmark="AR", metrics=("rmse",), tests=("dm",),
                            multiple_testing=multiple_testing),
        save_models=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pipeline(spec).significance


def test_no_method_leaves_the_report_untouched() -> None:
    """The default must stay byte-identical."""
    assert not any(c.endswith("_p_adj") for c in _fitted(None).columns)


@pytest.mark.parametrize("method", ["bonferroni", "holm", "bh", "romano_wolf"])
def test_the_pipeline_emits_an_adjusted_column(method: str) -> None:
    sig = _fitted(method)
    assert "dm_p_adj" in sig.columns
    both = sig[["dm_p", "dm_p_adj"]].dropna()
    assert not both.empty, f"{method} produced no adjusted p-values"
    assert np.all(both["dm_p_adj"] >= both["dm_p"] - 1e-9), (
        "an adjusted p-value can never be smaller than the raw one"
    )
    assert np.all(both["dm_p_adj"] <= 1.0 + 1e-12)


def test_an_unknown_method_is_refused_by_the_spec() -> None:
    with pytest.raises(ValueError, match="multiple_testing"):
        _fitted("no_such_method")


def test_every_method_adjusts_the_same_family() -> None:
    """A contender whose test did not run gets no adjusted p-value, whichever
    method is configured.

    Romano-Wolf CAN compute one from the loss differentials alone, and the first
    implementation did -- producing a `dm_p_adj` next to an empty `dm_p`, which
    reads as the adjustment of something that was never tested, and letting those
    columns inflate the max statistic every other contender is charged against.
    """
    frames = {m: _fitted(m) for m in ("bonferroni", "holm", "bh", "romano_wolf")}
    reference = frames["bonferroni"]
    missing = ~np.isfinite(reference["dm_p"].to_numpy(dtype=float))
    if not missing.any():
        pytest.skip("this fixture produced a p-value for every contender")
    for method, frame in frames.items():
        adj = frame["dm_p_adj"].to_numpy(dtype=float)
        assert np.all(np.isnan(adj[missing])), (
            f"{method} adjusted a p-value for a contender that produced none"
        )
        assert np.all(np.isfinite(adj[~missing])), (
            f"{method} left a tested contender unadjusted"
        )
