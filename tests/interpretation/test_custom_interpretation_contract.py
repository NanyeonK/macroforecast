"""The `custom_interpretation` contract, pinned.

It was one sentence of docstring and no guide coverage (#446, risk 3). These
tests fix the parts a caller has to be able to rely on -- the exact call
signature, what a callable may return, and what schema comes back -- and one
part they must NOT rely on: nothing here validates that a custom method's
numbers mean anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

N = 60


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})
    y = pd.Series(2.0 * X["a"] + rng.normal(scale=0.1, size=N), name="y")
    return mf.models.ols(X, y), X, y


def test_the_callable_is_invoked_with_the_documented_signature(fitted) -> None:
    """`model` unchanged, `X` a DataFrame, `y` and `metadata` always keywords."""
    fit, X, y = fitted
    seen: dict[str, object] = {}

    def spy(model, frame, *, y=None, metadata=None, **params):
        seen.update(
            model_is_fit=model is fit,
            frame_type=type(frame).__name__,
            columns=list(frame.columns),
            y_len=None if y is None else len(y),
            metadata=dict(metadata or {}),
            params=dict(params),
        )
        return {"feature": list(frame.columns), "importance": [1.0, 2.0]}

    mf.interpretation.custom_interpretation(
        fit, X, spy, y=y, metadata={"note": "hello"}, alpha=0.5
    )
    assert seen["model_is_fit"] is True, "model must be passed through unchanged"
    assert seen["frame_type"] == "DataFrame"
    assert seen["columns"] == ["a", "b"]
    assert seen["y_len"] == N
    assert seen["metadata"] == {"note": "hello"}
    assert seen["params"] == {"alpha": 0.5}


def test_y_and_metadata_are_passed_even_when_absent(fitted) -> None:
    """A callable can rely on the keywords existing, so they are always sent."""
    fit, X, _ = fitted
    seen: dict[str, object] = {}

    def spy(model, frame, *, y=None, metadata=None, **params):
        seen["y"] = y
        seen["metadata"] = metadata
        return pd.DataFrame({"feature": list(frame.columns), "importance": [0.0, 0.0]})

    mf.interpretation.custom_interpretation(fit, X, spy)
    assert seen["y"] is None
    assert seen["metadata"] == {}


def test_a_callable_that_ignores_the_keywords_is_refused_clearly(fitted) -> None:
    """Documented consequence: `(model, X)` alone is not enough."""
    fit, X, _ = fitted

    def too_narrow(model, frame):  # no y=, no metadata=, no **kwargs
        return {"feature": list(frame.columns), "importance": [0.0, 0.0]}

    with pytest.raises(TypeError):
        mf.interpretation.custom_interpretation(fit, X, too_narrow)


@pytest.mark.parametrize(
    "returned",
    [
        pytest.param(lambda cols: pd.DataFrame({"feature": cols, "v": [1.0, 2.0]}), id="dataframe"),
        pytest.param(lambda cols: {"feature": cols, "v": [1.0, 2.0]}, id="columnar-mapping"),
        pytest.param(
            lambda cols: [{"feature": c, "v": float(i)} for i, c in enumerate(cols)],
            id="row-mappings",
        ),
    ],
)
def test_a_table_like_return_becomes_a_real_table(fitted, returned) -> None:
    """One row per feature, not one row holding lists.

    The columnar-mapping case used to produce a single row whose cells each held
    a list -- no error, and a table that looks wrong only if you print it. The
    first version of this test checked isinstance and column names, which both
    passed on the broken output, so it now checks the shape and the values.
    """
    fit, X, _ = fitted
    table = mf.interpretation.custom_interpretation(
        fit, X, lambda m, f, **k: returned(list(f.columns))
    )
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["feature", "v"]
    assert len(table) == len(X.columns), (
        f"expected one row per feature; got {len(table)} rows: {table.to_dict()}"
    )
    assert list(table["feature"]) == list(X.columns)
    assert all(np.isscalar(v) or isinstance(v, float) for v in table["v"]), (
        f"cells should hold scalars, not containers: {list(table[chr(39)+chr(118)+chr(39)])}"
    )


def test_a_mapping_of_scalars_stays_one_row(fitted) -> None:
    """The other natural shape: named metrics, which ARE one row."""
    fit, X, _ = fitted
    table = mf.interpretation.custom_interpretation(
        fit, X, lambda m, f, **k: {"r2": 0.81, "mse": 1.2}
    )
    assert len(table) == 1
    assert float(table.loc[0, "r2"]) == 0.81


def test_the_result_carries_the_same_schema_as_a_built_in(fitted) -> None:
    """The reason to use this rather than calling your own function: traceability."""
    fit, X, y = fitted

    def method(model, frame, *, y=None, metadata=None, **params):
        return {"feature": list(frame.columns), "importance": [1.0, 2.0]}

    table = mf.interpretation.custom_interpretation(
        fit, X, method, y=y, name="range", metadata={"source": "mine"}, k=3
    )
    schema = getattr(table, "attrs", {}).get("macroforecast_metadata_schema", {})
    assert schema, "no schema attached"
    assert schema.get("kind") == "custom_interpretation"
    assert schema.get("method") == "range"
    meta = schema.get("metadata", {})
    assert meta.get("n_obs") == N
    assert meta.get("has_target") is True
    assert meta.get("params") == {"k": 3}
    assert meta.get("user_metadata") == {"source": "mine"}


def test_nothing_checks_that_a_custom_method_is_correct(fitted) -> None:
    """Documented non-guarantee, pinned so it cannot be assumed away.

    A method returning obvious nonsense is accepted and schema-stamped exactly
    like a correct one. That is the deliberate design -- the package cannot know
    what a user's method means -- and a caller putting custom output in a paper
    needs to know the package is not vouching for it.
    """
    fit, X, _ = fitted

    def nonsense(model, frame, *, y=None, metadata=None, **params):
        # attributions that sum to nothing in particular, for features that
        # the model demonstrably does not weight this way
        return {"feature": list(frame.columns), "importance": [1e9, -1e9]}

    table = mf.interpretation.custom_interpretation(fit, X, nonsense)
    assert list(table["importance"]) == [1e9, -1e9], (
        "the package altered a custom result; it is supposed to pass it through"
    )
    schema = getattr(table, "attrs", {}).get("macroforecast_metadata_schema", {})
    assert schema.get("kind") == "custom_interpretation"
