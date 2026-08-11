"""Regression tests for pipeline-spec immutability and diagnostics."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, InterpretSpec, TargetSpec, pipeline_spec


def _bundle() -> object:
    index = pd.date_range("2000-01-31", periods=60, freq="ME", name="date")
    frame = pd.DataFrame(
        {"y": np.arange(60.0), "x": np.linspace(-1.0, 1.0, 60)}, index=index
    )
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x": 1})


def _interpret_spec(provenance: Mapping[str, object]) -> object:
    return pipeline_spec(
        data=_bundle(),
        targets=[TargetSpec("y", transform="level")],
        horizons=[1],
        window="dummy",
        arms=[
            Arm("AR", model="ar"),
            Arm(
                "RF",
                model="random_forest",
                interpret=InterpretSpec(methods=("shap",)),
            ),
        ],
        evaluation=EvalSpec(benchmark="AR"),
        save_models=False,
        provenance=provenance,
    )


def _guarded_spec(mode: str) -> object:
    return pipeline_spec(
        data=_bundle(),
        targets=[TargetSpec("y", transform="level", policy="direct")],
        horizons=[1],
        window="dummy",
        arms=[Arm("ARIMA", model="arima")],
        evaluation=EvalSpec(benchmark="ARIMA"),
        on_unsupported_direct=mode,
    )


def test_numeric_ndarray_is_detached_and_read_only() -> None:
    source = np.arange(4.0)
    arm = Arm("ARRAY", model="ridge", params={"weights": source})
    frozen = arm.params["weights"]

    source[0] = 99.0

    assert frozen[0] == 0.0
    assert frozen.flags.writeable is False
    with pytest.raises(ValueError):
        frozen[0] = -1.0


def test_object_ndarray_elements_are_recursively_frozen() -> None:
    nested = {"grid": [1, 2]}
    source = np.empty((1, 2), dtype=object)
    source[0, 0] = nested
    source[0, 1] = bytearray(b"ab")

    frozen = Arm("OBJECT", model="ridge", params={"values": source}).params[
        "values"
    ]
    nested["grid"].append(3)
    source[0, 1][0] = ord("z")

    assert frozen.shape == (1, 2)
    assert frozen.flags.writeable is False
    assert isinstance(frozen[0, 0], Mapping)
    assert frozen[0, 0]["grid"] == (1, 2)
    assert frozen[0, 1] == b"ab"


def test_ndarray_subclass_is_detached_without_erasing_its_identity() -> None:
    source = np.ma.array([1.0, 2.0], mask=[False, True])
    frozen = Arm("MASKED", model="ridge", params={"values": source}).params[
        "values"
    ]

    source[0] = 99.0
    source.mask[0] = True

    assert isinstance(frozen, np.ma.MaskedArray)
    assert frozen[0] == 1.0
    assert frozen.mask.tolist() == [False, True]
    assert frozen.flags.writeable is False
    assert frozen.mask.flags.writeable is False


def test_bytearray_becomes_detached_immutable_bytes() -> None:
    source = bytearray(b"ab")
    frozen = Arm("BYTES", model="ridge", params={"blob": source}).params["blob"]

    source[0] = ord("z")

    assert frozen == b"ab"
    assert isinstance(frozen, bytes)


def test_array_freeze_survives_arm_pickle_round_trip() -> None:
    arm = Arm(
        "PICKLE",
        model="ridge",
        params={"values": np.arange(3.0), "blob": bytearray(b"ab")},
        metadata={"nested": {"values": [1, 2]}},
    )

    restored = pickle.loads(pickle.dumps(arm))

    np.testing.assert_array_equal(restored.params["values"], np.arange(3.0))
    assert restored.params["values"].flags.writeable is False
    assert restored.params["blob"] == b"ab"
    assert restored.metadata["nested"]["values"] == (1, 2)


@pytest.mark.parametrize("metadata", [None, [], "metadata"])
def test_metadata_must_be_a_mapping(metadata: object) -> None:
    with pytest.raises(TypeError, match=r"Arm\.metadata must be a mapping"):
        Arm("BAD", model="ridge", metadata=metadata)  # type: ignore[arg-type]


def test_params_none_and_valid_mappings_keep_their_contract() -> None:
    no_params = Arm("NONE", model="ridge", params=None, metadata={})
    mapped = Arm("MAPPED", model="ridge", params={"grid": [1, 2]}, metadata={})

    assert no_params.params is None
    assert mapped.params["grid"] == (1, 2)
    assert isinstance(mapped.metadata, Mapping)


@pytest.mark.parametrize(
    ("existing", "expected_prefix"),
    [
        ("existing warning", ["existing warning"]),
        (["first", "second"], ["first", "second"]),
        (("first",), ["first"]),
    ],
)
def test_generated_warning_preserves_existing_warning_entries(
    existing: object, expected_prefix: list[str]
) -> None:
    spec = _interpret_spec({"warnings": existing})

    assert list(spec.provenance["warnings"][:-1]) == expected_prefix
    assert "interpretation will re-fit models" in spec.provenance["warnings"][-1]


@pytest.mark.parametrize("mode", ["warn", "reroute"])
def test_direct_policy_warning_points_at_public_caller(mode: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _guarded_spec(mode)

    guard = next(item for item in caught if "iterat" in str(item.message).lower())
    assert guard.filename == __file__
    if mode == "reroute":
        assert "Rerouting" in str(guard.message)
