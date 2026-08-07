"""K-fold validation splits are seeded, so CV selection reproduces (#513).

`random_kfold_split` seeds itself with `random_state=0`. `make_splitter` declared
`random_state: int | None = None` and forwarded that value **explicitly**, so the
callee's seeded default was never reached and every k-fold shuffle drew from OS
entropy. Same input, different folds, different selected hyperparameter.

That made every `,KF` arm of a run irreproducible. It was found in a replication,
not here: a controlled A/B on an unrelated setting produced a run-to-run spread 16x
larger than the effect being measured, and the arm it moved most (`AR,KF`) is built
with `predictors=[]` and could not be reached by the setting at all.

The tests below are written so that each one fails on the unfixed code for its own
reason, rather than four ways of noticing the same symptom.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.model_selection import SearchSpec, select_params, validation_splitter
from macroforecast.window.core import make_splitter, random_kfold_split


def _folds(splits):
    return [(tuple(np.asarray(tr).tolist()), tuple(np.asarray(va).tolist())) for tr, va in splits]


def test_make_splitter_agrees_with_the_function_it_delegates_to():
    """The defect in one line: two doors to the same shuffle disagreed."""
    direct = _folds(random_kfold_split(200, n_splits=5))
    via = _folds(make_splitter("random_kfold", 200, n_splits=5))
    assert via == direct, (
        "make_splitter's default disagrees with random_kfold_split's own default; "
        "a None forwarded from the caller overrides the callee's seed"
    )


def test_the_same_call_twice_gives_the_same_folds():
    assert _folds(make_splitter("random_kfold", 200, n_splits=5)) == _folds(
        make_splitter("random_kfold", 200, n_splits=5)
    ), "identical arguments produced different folds"


def test_none_still_means_unseeded():
    """The seed is a default, not a removal of the option.

    Someone who wants a fresh shuffle per call must still be able to ask for one,
    or the fix would have replaced one silent behaviour with another.
    """
    draws = {tuple(_folds(make_splitter("random_kfold", 400, n_splits=5, random_state=None)))
             for _ in range(6)}
    assert len(draws) > 1, "random_state=None no longer produces an unseeded shuffle"


def test_cv_selection_reproduces_end_to_end():
    """What a user actually notices: the same search picks the same model.

    Scores are compared exactly. A tolerance here would pass on the unfixed code
    whenever two candidates happened to land close together, which is most of the
    time -- the observed spread was ~3% in score, but the failure that matters is
    that the SELECTED parameter changes.
    """
    rng = np.random.default_rng(7)
    n = 300
    X = pd.DataFrame({f"l{i}": rng.normal(size=n) for i in range(6)})
    y = pd.Series(0.4 * X["l0"] + 0.2 * X["l1"] + rng.normal(size=n), name="y")

    def run():
        spec = SearchSpec(
            method="grid",
            param_grid={"alpha": (0.01, 0.1, 1.0, 3.0)},
            validation_splitter=validation_splitter("random_kfold", n_splits=5),
        )
        return select_params("ridge", X, y, search=spec)

    results = [run() for _ in range(4)]
    params = [r.best_params for r in results]
    scores = [float(r.best_score) for r in results]
    assert all(p == params[0] for p in params), f"selected parameters varied: {params}"
    assert all(s == scores[0] for s in scores), f"validation scores varied: {scores}"


def test_the_window_builder_is_seeded_too():
    """``random_kfold()`` already promised ``random_state=0``; the spec now agrees.

    The mismatch was not only in make_splitter -- ValidationSpec.random_state and the
    val_random_state builder argument also defaulted to None and forwarded it, so a
    window built through those paths was unseeded while the one built through
    ``random_kfold()`` was not.
    """
    from macroforecast.window.core import ValWindow

    spec = mf.window.random_kfold(n_splits=5)
    assert spec.val.random_state == 0
    default = ValWindow(method="random_kfold", n_splits=5)
    assert default.random_state == 0, (
        "a ValWindow built without an explicit seed is still unseeded"
    )
