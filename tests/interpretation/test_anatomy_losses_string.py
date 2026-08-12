"""How ``anatomy_pipeline(losses=...)`` reads a scalar string (F-068).

``losses`` was annotated ``Sequence[str]`` and consumed by iterating it
directly, so a scalar ``str`` -- the spelling every other named-item argument
in macroforecast accepts -- was iterated *character by character*::

    anatomy_pipeline(..., losses="rmse")
    # performance_values keys: {"r", "m", "s", "e"}

Each character was then handed to ``pbsv(loss=...)`` as if it were a loss
name, so the caller either got four meaningless decompositions or a confusing
downstream failure naming a single letter.  Nothing in the argument's own
handling refused the input, and the damage was done only after the expensive
``precompute_anatomy`` call had already run.

The contract pinned here is: a scalar ``str`` means exactly one named loss;
any other iterable is taken member-wise and unchanged, preserving empty and
duplicate sequences; ``bytes``/``bytearray`` are refused rather than decoded,
and a non-string member is refused naming its position.  Every refusal happens
*before* precompute, so a mistyped ``losses`` costs nothing.  Loss names
themselves are untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from macroforecast.interpretation import anatomy as anatomy_module
from macroforecast.interpretation.anatomy import _normalize_losses, anatomy_pipeline


# --------------------------------------------------------------------------
# Normalization: a scalar string is one named loss.
# --------------------------------------------------------------------------


def test_scalar_string_is_one_named_loss_not_four_characters() -> None:
    """The bug itself: ``"rmse"`` is one loss, never ``r``/``m``/``s``/``e``."""

    assert _normalize_losses("rmse") == ("rmse",)


def test_scalar_string_is_not_split_on_separators() -> None:
    """A scalar string is a *name*, so no character in it is a delimiter."""

    assert _normalize_losses("mean_squared_error") == ("mean_squared_error",)
    assert _normalize_losses("a,b") == ("a,b",)


def test_sequence_members_pass_through_in_order() -> None:
    """The existing sequence spelling keeps its exact meaning."""

    assert _normalize_losses(["rmse", "mae"]) == ("rmse", "mae")
    assert _normalize_losses(("mae", "rmse")) == ("mae", "rmse")


def test_empty_sequence_stays_empty() -> None:
    """An empty request is a real request for no PBSV tables, not an error."""

    assert _normalize_losses([]) == ()
    assert _normalize_losses(()) == ()


def test_duplicate_members_are_preserved_not_deduplicated() -> None:
    """Normalization does not decide that a repeated loss was a mistake."""

    assert _normalize_losses(["rmse", "rmse"]) == ("rmse", "rmse")


def test_generator_is_materialized_once_and_survives_reuse() -> None:
    """``losses`` is read twice downstream, so a one-shot iterable is fixed."""

    normalized = _normalize_losses(iter(["rmse", "mae"]))

    assert normalized == ("rmse", "mae")
    # Second read yields the same names rather than an empty result.
    assert tuple(normalized) == ("rmse", "mae")


# --------------------------------------------------------------------------
# Fail-closed refusals: bytes and non-string members.
# --------------------------------------------------------------------------


def test_bytes_are_refused_rather_than_decoded() -> None:
    """``b"rmse"`` would otherwise iterate to four integers."""

    with pytest.raises(TypeError, match="losses"):
        _normalize_losses(b"rmse")  # type: ignore[arg-type]


def test_bytearray_is_refused_like_bytes() -> None:
    with pytest.raises(TypeError, match="losses"):
        _normalize_losses(bytearray(b"rmse"))  # type: ignore[arg-type]


def test_non_string_member_is_refused_naming_its_position() -> None:
    """A silent ``str(...)`` coercion would invent a loss name like ``"2"``."""

    with pytest.raises(TypeError, match=r"losses\[1\]"):
        _normalize_losses(["rmse", 2])  # type: ignore[list-item]


def test_bytes_member_is_refused_naming_its_position() -> None:
    with pytest.raises(TypeError, match=r"losses\[0\]"):
        _normalize_losses([b"rmse"])  # type: ignore[list-item]


def test_non_iterable_is_refused() -> None:
    with pytest.raises(TypeError, match="losses"):
        _normalize_losses(3.0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Wiring: the pipeline normalizes before it pays for precompute.
# --------------------------------------------------------------------------


def _frame() -> pd.DataFrame:
    index = pd.date_range("2000-01-31", periods=12, freq="ME")
    return pd.DataFrame({"a": np.arange(12.0), "b": np.arange(12.0)[::-1]}, index=index)


def _series() -> pd.Series:
    index = pd.date_range("2000-01-31", periods=12, freq="ME")
    return pd.Series(np.arange(12.0), index=index, name="target")


def _forbid_precompute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any precompute call an unmistakable test failure."""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("precompute_anatomy ran before losses were validated")

    monkeypatch.setattr(anatomy_module, "precompute_anatomy", _boom)


def test_bytes_losses_are_refused_before_precompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is a ``TypeError`` about ``losses``, not a backend error."""

    _forbid_precompute(monkeypatch)

    with pytest.raises(TypeError, match="losses"):
        anatomy_pipeline(_frame(), _series(), "ols", window="expanding", losses=b"rmse")


def test_non_string_member_is_refused_before_precompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_precompute(monkeypatch)

    with pytest.raises(TypeError, match=r"losses\[1\]"):
        anatomy_pipeline(
            _frame(), _series(), "ols", window="expanding", losses=["rmse", 2]
        )


def _stub_backend(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the optional-backend boundary; record every ``loss`` requested.

    The ``anatomy`` backend is an optional dependency, so the only way to
    observe which loss names reach ``pbsv`` is to stand in for it here.  The
    control flow under test -- normalization, the ``pbsv`` fan-out, and the
    recorded metadata -- is the real ``anatomy_pipeline`` code.
    """

    from macroforecast.interpretation import core as core_module

    requested: list[str] = []

    monkeypatch.setattr(anatomy_module, "precompute_anatomy", lambda *a, **k: object())
    monkeypatch.setattr(
        core_module, "anatomy_explain", lambda *a, **k: pd.DataFrame({"c": [0.0]})
    )
    monkeypatch.setattr(
        core_module, "oshapley_vi", lambda *a, **k: pd.DataFrame({"c": [0.0]})
    )

    def _pbsv(anatomy: Any, *, loss: str = "rmse", **kwargs: Any) -> pd.DataFrame:
        requested.append(loss)
        return pd.DataFrame({"c": [0.0]})

    monkeypatch.setattr(core_module, "pbsv", _pbsv)
    return requested


def test_pipeline_requests_one_pbsv_for_a_scalar_string_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End of the bug: one ``pbsv`` call for ``"rmse"``, keyed by its full name."""

    requested = _stub_backend(monkeypatch)

    result = anatomy_pipeline(
        _frame(), _series(), "ols", window="expanding", losses="rmse"
    )

    assert requested == ["rmse"]
    assert list(result.performance_values) == ["rmse"]
    assert result.metadata["losses"] == ["rmse"]


def test_pipeline_still_honours_a_sequence_of_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-existing spelling is unchanged."""

    requested = _stub_backend(monkeypatch)

    result = anatomy_pipeline(
        _frame(), _series(), "ols", window="expanding", losses=["rmse", "mae"]
    )

    assert requested == ["rmse", "mae"]
    assert list(result.performance_values) == ["rmse", "mae"]
    assert result.metadata["losses"] == ["rmse", "mae"]


def test_pipeline_preserves_an_empty_losses_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No PBSV work is requested, and the metadata says so."""

    requested = _stub_backend(monkeypatch)

    result = anatomy_pipeline(_frame(), _series(), "ols", window="expanding", losses=())

    assert requested == []
    assert result.performance_values == {}
    assert result.metadata["losses"] == []


def test_pipeline_preserves_duplicate_losses_in_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicates collapse in the keyed tables but stay in the recorded request."""

    requested = _stub_backend(monkeypatch)

    result = anatomy_pipeline(
        _frame(), _series(), "ols", window="expanding", losses=["rmse", "rmse"]
    )

    assert requested == ["rmse", "rmse"]
    assert list(result.performance_values) == ["rmse"]
    assert result.metadata["losses"] == ["rmse", "rmse"]


def test_forecast_result_entry_point_refuses_bytes_before_precompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``anatomy_from_forecast_result`` inherits the same contract."""

    from macroforecast.interpretation.anatomy import anatomy_from_forecast_result

    _forbid_precompute(monkeypatch)

    with pytest.raises(TypeError, match="losses"):
        anatomy_from_forecast_result(
            object(),
            _frame(),
            _series(),
            "ols",
            window="expanding",
            losses=b"rmse",
        )
