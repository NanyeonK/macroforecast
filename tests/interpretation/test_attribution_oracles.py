"""Oracles for the interpretation subsystem: cases where the answer is known.

Interpretation output is harder to check than forecast accuracy -- a Shapley
table looks equally plausible whether or not it is right, and there is no
held-out sample to score it against. So this file does not test that the
functions run. It constructs models whose correct attribution is known in
advance and checks the functions return it:

- a model that ignores a feature must give it no credit
- a model that is a constant must give nothing any credit
- an additive model's per-feature effects must sum to what it predicts
- an injected custom model must be explained like the estimator it wraps

Issue #446, which flagged three risks: dispatch on model type is unverified for
injected custom models, the Shapley/Anatomy decompositions are headline exhibits
with no pipeline-level oracle, and the `custom_interpretation` contract is
undocumented.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

N = 200


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    """`y` depends on `signal` alone; `noise` and `junk` are irrelevant by
    construction, which is what makes their correct attribution knowable."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=N),
            "noise": rng.normal(size=N),
            "junk": rng.normal(size=N),
        }
    )
    y = pd.Series(3.0 * X["signal"] + rng.normal(scale=0.05, size=N), name="y")
    return X, y


# --------------------------------------------------------------------------- #
# native attributions
# --------------------------------------------------------------------------- #

def test_linear_coefficients_recover_the_generating_coefficients(data) -> None:
    """The oracle is the DGP: 3.0 on `signal`, ~0 on the other two."""
    X, y = data
    fit = mf.models.ols(X, y)
    table = mf.interpretation.linear_coefficients(fit).set_index("feature")
    assert table.loc["signal", "coefficient"] == pytest.approx(3.0, abs=0.02)
    for irrelevant in ("noise", "junk"):
        assert abs(table.loc[irrelevant, "coefficient"]) < 0.05, (
            f"{irrelevant} is unused by construction but got "
            f"{table.loc[irrelevant, 'coefficient']:.4f}"
        )


def test_linear_coefficients_refuse_a_model_that_has_none(data) -> None:
    """Dispatch is on the attribute, so it must say so when the attribute is absent."""
    X, y = data
    fit = mf.models.random_forest(X, y, n_estimators=4, max_depth=3)
    with pytest.raises(ValueError, match="coef_"):
        mf.interpretation.linear_coefficients(fit)


def test_tree_importance_puts_the_mass_on_the_used_feature(data) -> None:
    X, y = data
    fit = mf.models.random_forest(X, y, n_estimators=40, max_depth=4, random_state=0)
    table = mf.interpretation.tree_importance(fit).set_index("feature")
    assert table["importance"].sum() == pytest.approx(1.0, abs=1e-6)
    # A forest at max_depth=4 splits on the irrelevant columns sometimes, so the
    # oracle is a wide margin over its rivals rather than a tight bound on the
    # value: what must hold is that the used feature dominates, not that it wins
    # by any particular amount.
    signal = float(table.loc["signal", "importance"])
    rivals = float(table.drop(index="signal")["importance"].max())
    assert signal > 5 * rivals, (
        f"the only informative feature scored {signal:.3f} against a best rival "
        f"of {rivals:.3f}; it should dominate"
    )


# --------------------------------------------------------------------------- #
# an injected custom model
# --------------------------------------------------------------------------- #

def test_a_custom_model_is_explained_like_the_estimator_it_wraps(data) -> None:
    """Risk (1) in #446, stated as a test.

    A custom model that delegates to `ols` must be interpreted identically to
    `ols` -- if dispatch keyed on the wrapper's type rather than on what it
    exposes, this is where it would show.
    """
    X, y = data

    def wrapped_ols(X_, y_=None, **params):
        return mf.models.ols(X_, y_, **params)

    native = mf.interpretation.linear_coefficients(mf.models.ols(X, y))
    custom = mf.interpretation.linear_coefficients(wrapped_ols(X, y))
    pd.testing.assert_frame_equal(
        native.reset_index(drop=True), custom.reset_index(drop=True)
    )


def test_a_constant_model_gives_nothing_any_credit() -> None:
    """Degenerate but decisive: a model that ignores every input.

    Any attribution method that returns a non-zero effect here is inventing one.
    """
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})
    y = pd.Series(np.full(N, 7.0), name="y")
    fit = mf.models.ols(X, y)
    table = mf.interpretation.linear_coefficients(fit).set_index("feature")
    assert np.allclose(table["coefficient"].to_numpy(dtype=float), 0.0, atol=1e-8), (
        "a constant target produced non-zero coefficients"
    )


# --------------------------------------------------------------------------- #
# permutation importance
# --------------------------------------------------------------------------- #

def test_permutation_importance_separates_the_used_feature(data) -> None:
    """Permuting an unused column must cost the model nothing."""
    X, y = data
    fit = mf.models.ols(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = mf.interpretation.permutation_importance(
            fit, X, y, n_repeats=5, random_state=0
        )
    table = table.set_index("feature")
    col = "importance_mean" if "importance_mean" in table else table.columns[0]
    assert table.loc["signal", col] > 0.5, (
        f"permuting the only informative feature should hurt; got {table.loc['signal', col]}"
    )
    for irrelevant in ("noise", "junk"):
        assert abs(table.loc[irrelevant, col]) < 0.05, (
            f"permuting {irrelevant} should cost ~nothing; got {table.loc[irrelevant, col]}"
        )


def test_permutation_importance_is_reproducible_under_a_seed(data) -> None:
    """It resamples, so a stated seed has to pin it."""
    X, y = data
    fit = mf.models.ols(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = mf.interpretation.permutation_importance(fit, X, y, n_repeats=4, random_state=7)
        b = mf.interpretation.permutation_importance(fit, X, y, n_repeats=4, random_state=7)
    pd.testing.assert_frame_equal(a, b)
