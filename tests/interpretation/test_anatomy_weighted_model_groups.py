"""Weighted ``model_groups`` against the pinned ``anatomy`` backend (F-072).

The pinned backend (``anatomy==0.1.6``) accepts a mapping of model weights but
ignores the numbers: ``Anatomy.explain`` selects the branch with
``type(comb_set == list)``, which evaluates ``comb_set == list`` (``False``) and
then takes ``type(False)`` -- a truthy ``bool`` -- so the equal-weight branch is
taken for every group and ``weights`` is always ``np.repeat(1.0, len(...))``.
These tests pin the macroforecast-side behaviour that a requested weight is the
weight actually used, and that the combination happens on the precomputed model
output BEFORE the output transformer, which is what a nonlinear loss such as
RMSE distinguishes.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.metrics import rmse


def _has_anatomy() -> bool:
    return importlib.util.find_spec("anatomy") is not None


pytestmark = pytest.mark.skipif(
    not _has_anatomy(), reason="requires optional anatomy backend"
)


class _Linear:
    """Fixed linear predictor, so per-model forecasts are known up front."""

    def __init__(self, weights: list[float]) -> None:
        self._weights = np.asarray(weights, dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self._weights


@pytest.fixture(scope="module")
def anatomy_fixture() -> dict[str, Any]:
    """Precompute one small two-model ``Anatomy`` shared by the module."""

    import anatomy as anatomy_mod

    from macroforecast.interpretation.anatomy import anatomy_provider

    n_obs = 26
    index = pd.period_range("2000-01", periods=n_obs, freq="M").to_timestamp("M")
    rng = np.random.default_rng(20260812)
    X = pd.DataFrame(
        {"x1": rng.normal(size=n_obs), "x2": rng.normal(size=n_obs)},
        index=index,
    )
    y = pd.Series(rng.normal(size=n_obs), index=index, name="target")
    window = mf.window.from_cutoffs(
        test_start=X.index[20],
        test_end=X.index[22],
        estimation_min_size=12,
        val_method="last_block",
        val_size=3,
        horizon=1,
        step=1,
    )
    models = {
        "m1": mf.models.custom_model("f072_m1", lambda X, y, **kw: _Linear([1.0, 0.0])),
        "m2": mf.models.custom_model("f072_m2", lambda X, y, **kw: _Linear([3.0, 0.5])),
    }
    provider = anatomy_provider(X, y, models, window=window)
    np.random.seed(20260812)
    obj = anatomy_mod.Anatomy(provider=provider, n_iterations=2).precompute(n_jobs=1)

    per_model = obj.explain().sum(axis=1)
    return {
        "anatomy": obj,
        "y_true": obj.get_forecast_index(),
        "forecast": {
            "m1": per_model.xs("m1").to_numpy(),
            "m2": per_model.xs("m2").to_numpy(),
        },
        "actual": y.reindex(obj.get_forecast_index()).to_numpy(),
    }


def _row_sums(table: pd.DataFrame, model_set: str) -> np.ndarray:
    """Sum long-form contributions per index label for one model set."""

    rows = table[table["model_set"] == model_set]
    grouped = rows.groupby("index", sort=False)["contribution"].sum()
    return grouped.to_numpy()


def test_weighted_model_group_uses_requested_weights(
    anatomy_fixture: dict[str, Any],
) -> None:
    """A numeric weight mapping combines forecasts with THOSE weights."""

    obj = anatomy_fixture["anatomy"]
    f1 = anatomy_fixture["forecast"]["m1"]
    f2 = anatomy_fixture["forecast"]["m2"]

    for weights, expected in (
        ({"m1": 1.0, "m2": 0.0}, f1),
        ({"m1": 0.0, "m2": 1.0}, f2),
        ({"m1": 0.75, "m2": 0.25}, 0.75 * f1 + 0.25 * f2),
        ({"m1": 3.0, "m2": 1.0}, 0.75 * f1 + 0.25 * f2),
    ):
        table = mf.interpretation.anatomy_explain(
            obj, model_groups={"combo": weights}, metric="forecast"
        )
        np.testing.assert_allclose(_row_sums(table, "combo"), expected, atol=1e-12)


def test_weighted_group_combines_before_nonlinear_rmse_transform(
    anatomy_fixture: dict[str, Any],
) -> None:
    """PBSV efficiency holds against RMSE of the COMBINED forecast."""

    obj = anatomy_fixture["anatomy"]
    actual = anatomy_fixture["actual"]
    f1 = anatomy_fixture["forecast"]["m1"]
    f2 = anatomy_fixture["forecast"]["m2"]
    combined = 0.75 * f1 + 0.25 * f2

    table = mf.interpretation.pbsv(
        obj, model_groups={"combo": {"m1": 0.75, "m2": 0.25}}, loss="rmse"
    )
    total = float(table["contribution"].sum())
    weighted_per_model_loss = 0.75 * float(rmse(actual, f1)) + 0.25 * float(
        rmse(actual, f2)
    )

    assert total == pytest.approx(float(rmse(actual, combined)), abs=1e-10)
    # Weighting per-model loss explanations instead would give this number.
    assert abs(total - weighted_per_model_loss) > 1e-6


def test_equal_weight_mapping_matches_plain_list_group(
    anatomy_fixture: dict[str, Any],
) -> None:
    """Uniform weights reproduce the backend's own equal-weight combination."""

    obj = anatomy_fixture["anatomy"]
    listed = mf.interpretation.anatomy_explain(
        obj, model_groups={"combo": ["m1", "m2"]}, output="wide"
    )
    mapped = mf.interpretation.anatomy_explain(
        obj, model_groups={"combo": {"m1": 0.5, "m2": 0.5}}, output="wide"
    )

    pd.testing.assert_frame_equal(mapped, listed, atol=1e-12, check_exact=False)


def test_mixed_list_and_mapping_groups(anatomy_fixture: dict[str, Any]) -> None:
    """List groups keep backend results while mapping groups are weighted."""

    obj = anatomy_fixture["anatomy"]
    f1 = anatomy_fixture["forecast"]["m1"]
    f2 = anatomy_fixture["forecast"]["m2"]

    listed_alone = mf.interpretation.anatomy_explain(
        obj, model_groups={"equal": ["m1", "m2"]}, output="wide"
    )
    mixed = mf.interpretation.anatomy_explain(
        obj,
        model_groups={"equal": ["m1", "m2"], "tilted": {"m1": 0.9, "m2": 0.1}},
        output="wide",
    )

    assert list(mixed.index.get_level_values(0).unique()) == ["equal", "tilted"]
    pd.testing.assert_frame_equal(mixed.xs("equal"), listed_alone.xs("equal"))
    np.testing.assert_allclose(
        mixed.xs("tilted").sum(axis=1).to_numpy(),
        0.9 * f1 + 0.1 * f2,
        atol=1e-12,
    )


def test_weighted_group_subset_and_wide_long_agree(
    anatomy_fixture: dict[str, Any],
) -> None:
    """Subset selection and both output shapes carry the same numbers."""

    obj = anatomy_fixture["anatomy"]
    index = obj.get_forecast_index()
    subset = index[:2]
    weights = {"m1": 0.75, "m2": 0.25}
    expected = (
        0.75 * anatomy_fixture["forecast"]["m1"][:2]
        + 0.25 * anatomy_fixture["forecast"]["m2"][:2]
    )

    wide = mf.interpretation.anatomy_explain(
        obj, model_groups={"combo": weights}, explanation_subset=subset, output="wide"
    )
    long = mf.interpretation.anatomy_explain(
        obj, model_groups={"combo": weights}, explanation_subset=subset, output="long"
    )

    assert len(wide) == 2
    assert "base_contribution" in wide.columns
    np.testing.assert_allclose(wide.sum(axis=1).to_numpy(), expected, atol=1e-12)
    np.testing.assert_allclose(_row_sums(long, "combo"), expected, atol=1e-12)
    assert set(long["feature"]) == {"base_contribution", "x1", "x2"}
    assert bool(long.loc[long["feature"] == "base_contribution", "is_base"].all())
    assert long.attrs["macroforecast_metadata_schema"]["metadata"]["model_groups"] == {
        "combo": weights
    }


@pytest.mark.parametrize(
    "groups",
    [
        None,
        {"equal": ["m1", "m2"]},
        {"combo": {"m1": 0.75, "m2": 0.25}},
        {"equal": ["m1", "m2"], "tilted": {"m1": 0.9, "m2": 0.1}},
    ],
    ids=["none", "all_sequence", "weighted", "mixed"],
)
def test_nonmatching_subset_fails_clearly_on_every_dispatch_path(
    anatomy_fixture: dict[str, Any],
    groups: dict[str, Any] | None,
) -> None:
    """A non-empty subset matching no forecast date is refused before the backend.

    The backend selects rows with ``self._xy_test.index.isin(subset)``, so a
    nonmatching subset is an all-False mask that only fails later as an opaque
    assertion. Every dispatch branch must refuse it the same way instead.
    """

    obj = anatomy_fixture["anatomy"]
    nonmatching = pd.DatetimeIndex(["1900-01-31", "1900-02-28"])
    assert len(nonmatching) > 0
    assert not obj.get_forecast_index().isin(nonmatching).any()

    with pytest.raises(ValueError, match="selects no forecast dates"):
        mf.interpretation.anatomy_explain(
            obj, model_groups=groups, explanation_subset=nonmatching
        )


@pytest.mark.parametrize(
    "groups",
    [
        None,
        {"equal": ["m1", "m2"]},
        {"combo": {"m1": 0.75, "m2": 0.25}},
        {"equal": ["m1", "m2"], "tilted": {"m1": 0.9, "m2": 0.1}},
    ],
    ids=["none", "all_sequence", "weighted", "mixed"],
)
def test_partially_matching_subset_still_explains(
    anatomy_fixture: dict[str, Any],
    groups: dict[str, Any] | None,
) -> None:
    """A subset with at least one matching date keeps its existing behaviour."""

    obj = anatomy_fixture["anatomy"]
    index = obj.get_forecast_index()
    partial = pd.DatetimeIndex([index[0], pd.Timestamp("1900-01-31")])

    table = mf.interpretation.anatomy_explain(
        obj, model_groups=groups, explanation_subset=partial, output="wide"
    )

    assert len(table) > 0


def test_oshapley_vi_uses_weighted_group(anatomy_fixture: dict[str, Any]) -> None:
    """oShapley-VI aggregates the weighted contributions, not equal-weighted."""

    obj = anatomy_fixture["anatomy"]
    weights = {"m1": 1.0, "m2": 0.0}

    table = mf.interpretation.oshapley_vi(obj, model_groups={"combo": weights})
    explained = mf.interpretation.anatomy_explain(
        obj, model_groups={"combo": weights}, metric="forecast"
    )
    solo = mf.interpretation.oshapley_vi(obj, model_groups={"combo": ["m1"]})

    assert set(table["model_set"]) == {"combo"}
    assert set(table["feature"]) == {"x1", "x2"}
    assert table.attrs["macroforecast_metadata_schema"]["kind"] == "oshapley_vi"
    assert set(table["rank"]) == {1.0, 2.0}
    for feature in ("x1", "x2"):
        rows = explained[explained["feature"] == feature]["contribution"]
        expected = float(np.mean(np.abs(rows.to_numpy())))
        got = float(table.loc[table["feature"] == feature, "importance"].iloc[0])
        assert got == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(
        table.sort_values("feature")["importance"].to_numpy(),
        solo.sort_values("feature")["importance"].to_numpy(),
        atol=1e-12,
    )


def test_model_groups_none_preserves_backend_behaviour(
    anatomy_fixture: dict[str, Any],
) -> None:
    """``model_groups=None`` still delegates to the untouched backend default."""

    obj = anatomy_fixture["anatomy"]
    expected = obj.explain()
    got = mf.interpretation.anatomy_explain(obj, output="wide")

    pd.testing.assert_frame_equal(got, expected)


def test_weighted_group_does_not_mutate_backend_or_global_rng(
    anatomy_fixture: dict[str, Any],
) -> None:
    """The pinned object and the global RNG are left exactly as they were."""

    obj = anatomy_fixture["anatomy"]
    tensor_before = obj._Y.copy()
    names_before = list(obj._model_names)
    tensor_id = id(obj._Y)
    np.random.seed(7)
    state_before = np.random.get_state()

    mf.interpretation.anatomy_explain(
        obj, model_groups={"combo": {"m1": 0.4, "m2": 0.6}}
    )

    state_after = np.random.get_state()
    assert id(obj._Y) == tensor_id
    np.testing.assert_array_equal(obj._Y, tensor_before)
    assert list(obj._model_names) == names_before
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        ({"combo": {}}, "empty"),
        ({"combo": []}, "empty"),
        ({"combo": {"m1": "heavy"}}, "numeric"),
        ({"combo": {"m1": None}}, "numeric"),
        ({"combo": {"m1": float("nan"), "m2": 1.0}}, "finite"),
        ({"combo": {"m1": float("inf"), "m2": 1.0}}, "finite"),
        ({"combo": {"m1": 1.0, "m2": -1.0}}, "sum to zero"),
        ({"combo": {"m1": 0.0, "m2": 0.0}}, "sum to zero"),
        ({"combo": "m1"}, "sequence of model names"),
        ({"combo": {"m1": 1.0, "nope": 2.0}}, "not available"),
    ],
)
def test_invalid_model_groups_fail_clearly(
    anatomy_fixture: dict[str, Any],
    groups: dict[str, Any],
    match: str,
) -> None:
    """Malformed groups raise before any explanation is produced."""

    with pytest.raises(ValueError, match=match):
        mf.interpretation.anatomy_explain(
            anatomy_fixture["anatomy"], model_groups=groups
        )


def test_all_sequence_group_rejects_unavailable_model(
    anatomy_fixture: dict[str, Any],
) -> None:
    """A plain sequence group is validated too, not only a weight mapping."""

    with pytest.raises(ValueError, match="not available"):
        mf.interpretation.anatomy_explain(
            anatomy_fixture["anatomy"], model_groups={"g": ["missing"]}
        )


def test_mixed_groups_reject_unavailable_sequence_member(
    anatomy_fixture: dict[str, Any],
) -> None:
    """Validation covers every group before any group is explained."""

    with pytest.raises(ValueError, match="not available"):
        mf.interpretation.anatomy_explain(
            anatomy_fixture["anatomy"],
            model_groups={"equal": ["m1", "nope"], "tilted": {"m1": 0.9, "m2": 0.1}},
        )


def test_small_scale_weights_are_accepted_and_scale_invariant(
    anatomy_fixture: dict[str, Any],
) -> None:
    """A finite nonzero small-scale request is a legitimate ratio, not zero."""

    obj = anatomy_fixture["anatomy"]
    f1 = anatomy_fixture["forecast"]["m1"]
    f2 = anatomy_fixture["forecast"]["m2"]

    tiny = mf.interpretation.anatomy_explain(
        obj, model_groups={"combo": {"m1": 1e-20, "m2": 3e-20}}, output="wide"
    )
    unit = mf.interpretation.anatomy_explain(
        obj, model_groups={"combo": {"m1": 1.0, "m2": 3.0}}, output="wide"
    )

    pd.testing.assert_frame_equal(tiny, unit, atol=1e-12, check_exact=False)
    np.testing.assert_allclose(
        tiny.sum(axis=1).to_numpy(), 0.25 * f1 + 0.75 * f2, atol=1e-12
    )


def test_near_overflow_weights_are_accepted_and_scale_invariant(
    anatomy_fixture: dict[str, Any],
) -> None:
    """A finite near-overflow request must not sum to ``inf`` and be lost.

    Each weight below is finite, but their raw sum overflows to ``inf``, which
    would drive the combination factors to zero or ``nan``. Normalizing by the
    largest absolute weight before summing keeps the request an ordinary equal
    combination.
    """

    obj = anatomy_fixture["anatomy"]

    huge = mf.interpretation.anatomy_explain(
        obj, model_groups={"combo": {"m1": 1e308, "m2": 1e308}}, output="wide"
    )
    unit = mf.interpretation.anatomy_explain(
        obj, model_groups={"combo": {"m1": 1.0, "m2": 1.0}}, output="wide"
    )

    assert np.isfinite(huge.to_numpy()).all()
    pd.testing.assert_frame_equal(huge, unit, atol=1e-12, check_exact=False)
