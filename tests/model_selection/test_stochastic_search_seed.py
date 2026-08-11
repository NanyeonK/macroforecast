"""Stochastic search must reproduce by default, and still yield to an explicit None.

Every test here fails on the pre-fix code, where ``SearchSpec.random_state`` and the
six ``random_state``-bearing builders all defaulted to ``None`` and so seeded
``np.random.default_rng`` from OS entropy (issue #516).

The oracle is deliberately not a golden value copied from a run: ``score_model``
returns its ``score_value`` argument as the prediction and ``first_prediction``
scores it back out, so the ``score_value`` column of the trial table *is* the
sequence of RNG draws. Two searches agree exactly iff their generators agreed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import macroforecast as mf
from tests.model_selection.helpers import first_prediction, score_model, xy

N_REPEATS = 5


def _distribution() -> dict[str, Any]:
    return {"score_value": mf.model_selection.uniform(0.0, 1.0)}


def _draws(search: Any) -> list[float]:
    """Run one search and return its sampled ``score_value`` sequence."""

    X, y = xy()
    result = mf.model_selection.select_params(
        score_model,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
        metric=first_prediction,
    )
    return [float(value) for value in result.trials["score_value"]]


def _rng_reading_custom_search(
    *,
    model: Any,
    X: Any,
    y: Any,
    splits: Any,
    metric: Any,
    fixed_params: Any,
    search: Any,
    rng: np.random.Generator,
    maximize: bool,
    evaluate_candidate: Any,
    n_draws: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Custom search whose candidates come straight off the injected generator."""

    rows = [
        evaluate_candidate(
            model,
            X,
            y,
            splits,
            metric,
            fixed_params,
            {"score_value": float(rng.random())},
            trial,
        )
        for trial in range(n_draws)
    ]
    return rows, {"custom_runtime": {"evaluated": len(rows)}}


def test_directly_constructed_random_search_spec_repeats_exactly() -> None:
    """The reproducer from #516: five identical calls, nothing but the RNG free."""

    runs = [
        _draws(
            mf.model_selection.SearchSpec(
                method="random",
                param_distributions=_distribution(),
                n_iter=6,
            )
        )
        for _ in range(N_REPEATS)
    ]

    assert runs.count(runs[0]) == N_REPEATS


def test_random_search_builder_default_repeats_exactly() -> None:
    runs = [
        _draws(mf.model_selection.random_search(_distribution(), n_iter=6))
        for _ in range(N_REPEATS)
    ]

    assert runs.count(runs[0]) == N_REPEATS


def test_genetic_search_builder_default_repeats_exactly() -> None:
    runs = [
        _draws(
            mf.model_selection.genetic_search(
                _distribution(),
                population_size=4,
                generations=2,
            )
        )
        for _ in range(N_REPEATS)
    ]

    assert runs.count(runs[0]) == N_REPEATS


def test_bayesian_search_builder_default_repeats_exactly() -> None:
    runs = [
        _draws(mf.model_selection.bayesian_search(_distribution(), n_iter=6))
        for _ in range(N_REPEATS)
    ]

    assert runs.count(runs[0]) == N_REPEATS


def test_custom_search_builder_default_repeats_exactly() -> None:
    runs = [
        _draws(
            mf.model_selection.custom_search(
                "rng_reading",
                _rng_reading_custom_search,
                n_draws=4,
            )
        )
        for _ in range(N_REPEATS)
    ]

    assert runs.count(runs[0]) == N_REPEATS


def test_select_params_method_kwarg_route_repeats_exactly() -> None:
    """The kwarg route builds the spec internally; it must be seeded too."""

    X, y = xy()
    runs = []
    for _ in range(N_REPEATS):
        result = mf.model_selection.select_params(
            "random_forest",
            X,
            y,
            preset="small",
            method="random",
            n_iter=3,
            window=mf.window.last_block(validation_size=6),
        )
        runs.append(
            [
                tuple(sorted(record.items()))
                for record in result.trials.to_dict("records")
            ]
        )

    assert runs.count(runs[0]) == N_REPEATS


ENTRY_POINTS = [
    pytest.param(
        lambda **kw: mf.model_selection.SearchSpec(
            method="random", param_distributions=_distribution(), **kw
        ),
        id="SearchSpec",
    ),
    pytest.param(lambda **kw: mf.model_selection.fixed(**kw), id="fixed"),
    pytest.param(
        lambda **kw: mf.model_selection.random_search(_distribution(), **kw),
        id="random_search",
    ),
    pytest.param(
        lambda **kw: mf.model_selection.bayesian_search(_distribution(), **kw),
        id="bayesian_search",
    ),
    pytest.param(
        lambda **kw: mf.model_selection.genetic_search(_distribution(), **kw),
        id="genetic_search",
    ),
    pytest.param(
        lambda **kw: mf.model_selection.custom_search(
            "noop", _rng_reading_custom_search, **kw
        ),
        id="custom_search",
    ),
    pytest.param(
        lambda **kw: mf.model_selection.search_spec(
            "random_forest", preset="small", method="random", **kw
        ),
        id="search_spec",
    ),
]


@pytest.mark.parametrize("build", ENTRY_POINTS)
def test_entry_point_default_is_the_reproducible_seed(build: Any) -> None:
    assert build().random_state == 0


@pytest.mark.parametrize("build", ENTRY_POINTS)
def test_entry_point_passes_an_explicit_random_state_through(build: Any) -> None:
    assert build(random_state=None).random_state is None
    assert build(random_state=7).random_state == 7


def test_search_spec_metadata_records_the_default_seed() -> None:
    metadata = mf.model_selection.random_search(_distribution()).to_metadata()

    assert metadata["random_state"] == 0


def _recorded_seeds(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every seed handed to ``np.random.default_rng`` during a search."""

    seeds: list[Any] = []
    real_default_rng = np.random.default_rng

    def recording_default_rng(seed: Any = None) -> np.random.Generator:
        seeds.append(seed)
        return real_default_rng(seed)

    monkeypatch.setattr(np.random, "default_rng", recording_default_rng)
    return seeds


def test_explicit_none_reaches_the_generator_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entropy opt-out is a contract about the seed, not about two runs differing.

    Asserting that two unseeded runs produce different numbers is a coin flip that
    can pass on broken code and fail on correct code. Asserting what the search
    hands to ``default_rng`` is exact.
    """

    seeds = _recorded_seeds(monkeypatch)

    _draws(
        mf.model_selection.random_search(_distribution(), n_iter=4, random_state=None)
    )

    assert seeds[0] is None


def test_default_reaches_the_generator_as_the_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = _recorded_seeds(monkeypatch)

    _draws(mf.model_selection.random_search(_distribution(), n_iter=4))

    assert seeds[0] == 0


def test_select_params_still_accepts_a_search_with_random_state_left_none() -> None:
    """``random_state=None`` on ``select_params`` means "no override", and must stay so.

    ``_has_search_overrides`` treats a non-None override as a conflict with an
    explicit ``search=``. If the override kwarg had been re-defaulted to 0, this
    ordinary call would have started raising.
    """

    X, y = xy()

    result = mf.model_selection.select_params(
        score_model,
        X,
        y,
        mf.model_selection.random_search(_distribution(), n_iter=3),
        window=mf.window.last_block(validation_size=6),
        metric=first_prediction,
        random_state=None,
    )

    assert len(result.trials) == 3


def test_grid_search_candidates_are_unaffected_by_the_seeded_default() -> None:
    """Grid enumerates; it never draws. Seeding the spec must not perturb it."""

    X, y = xy()
    search = mf.model_selection.grid({"score_value": [0.25, 0.5, 0.75]})

    result = mf.model_selection.select_params(
        score_model,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
        metric=first_prediction,
        maximize=True,
    )

    assert list(result.trials["score_value"]) == [0.25, 0.5, 0.75]
    assert result.best_params == {"score_value": 0.75}
