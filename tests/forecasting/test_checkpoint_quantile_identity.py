"""Checkpoint quantile column identity (F-059 / FC-002).

The first density checkpoint schema stored a record's ``quantile_predictions``
mapping as wide ``q_<pct>`` columns, where ``<pct>`` was ``round(level * 100)``.
That encoding is neither injective nor lossless. Levels ``0.024`` and ``0.025``
both became ``q_02`` (the second write silently overwrote the first), ``0.975``
and ``0.976`` both became ``q_98``, and every level at or above ``0.995`` became
``q_100``, which the two-digit reader could not parse at all. A level such as
``1/3`` came back as ``0.33``.

That breaks the contract ``forecasting/runner.py::_merge_checkpoint_records``
documents: a resumed-from-checkpoint origin's ``quantile_predictions`` must
carry the SAME ``{str(level): value}`` keys a freshly-computed origin's does, so
one forecast table never mixes two representations. These tests pin the
replacement grammar -- ``qx1_`` plus the 16 lowercase hex digits of
``struct.pack(">d", level)`` -- which is exact, injective, and fixed width, and
they pin what happens to the files the old grammar already wrote: legacy
``q_01``..``q_99`` stay readable at their integer-percent value, while both
endpoints of that grid stay deliberately unread -- ``q_100`` is not invented as
``1.0``, and ``q_00`` is not fabricated as ``0.0`` the way the old reader did.
"""
from __future__ import annotations

import math
import random
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.forecasting import checkpoint as ckpt


def teardown_function() -> None:
    mf.meta.reset_config()


# The grammar under test, spelled out independently of the implementation so a
# regression in the encoder cannot quietly redefine what the tests accept.
_NEW_COLUMN_RE = re.compile(r"^qx1_[0-9a-f]{16}$")
_LEGACY_COLUMN_RE = re.compile(r"^q_(\d{2})$")


def _expected_name(level: float) -> str:
    return "qx1_" + struct.pack(">d", level).hex()


def _bits(value: float) -> bytes:
    return struct.pack(">d", float(value))


def _wide(mapping: dict) -> dict[str, float]:
    return ckpt._quantile_wide_columns({"quantile_predictions": mapping})


# --------------------------------------------------------------------------- #
# 1. The collapse this feature exists to remove.
# --------------------------------------------------------------------------- #

_COLLAPSING_LEVELS = (0.024, 0.025, 0.975, 0.976)


def test_four_nearby_levels_get_four_distinct_columns_and_exact_levels() -> None:
    """0.024/0.025 and 0.975/0.976 previously collapsed onto q_02 and q_98."""
    mapping = {level: float(index) for index, level in enumerate(_COLLAPSING_LEVELS)}

    columns = _wide(mapping)

    assert len(columns) == 4, f"levels collapsed onto {sorted(columns)}"
    assert sorted(columns) == sorted(_expected_name(lv) for lv in _COLLAPSING_LEVELS)

    recovered = ckpt._quantile_dict_from_wide(columns)
    assert recovered is not None
    assert set(recovered) == {str(lv) for lv in _COLLAPSING_LEVELS}
    for level, value in mapping.items():
        assert _bits(float(str(level))) == _bits(level)  # keys are exact, not rounded
        assert recovered[str(level)] == value


def test_column_grammar_is_fixed_width_hex_and_disjoint_from_the_legacy_family() -> None:
    for level in (*_COLLAPSING_LEVELS, 0.05, 0.5, 1 / 3, 0.9995):
        name = ckpt._quantile_column_name(level)
        assert name == _expected_name(level)
        assert _NEW_COLUMN_RE.match(name), name
        assert len(name) == 20  # 4-character prefix + 16 hex digits, always
        assert name == name.lower()
        assert _LEGACY_COLUMN_RE.match(name) is None  # never mistaken for q_<pct>


def test_seeded_random_levels_roundtrip_exactly_and_never_collide() -> None:
    """Property-style, but seeded, so a failure is reproducible from the log."""
    rng = random.Random(20260812)
    levels = [rng.random() for _ in range(500)]
    levels = [lv for lv in levels if 0.0 < lv < 1.0]

    names = {ckpt._quantile_column_name(lv) for lv in levels}
    assert len(names) == len({_bits(lv) for lv in levels}), "distinct levels collided"

    columns = _wide({lv: float(i) for i, lv in enumerate(levels)})
    recovered = ckpt._quantile_dict_from_wide(columns)
    assert recovered is not None
    for index, level in enumerate(levels):
        assert _bits(float(str(level))) == _bits(level)
        assert recovered[str(level)] == float(index)


_REPRESENTATIVE_LEVELS = (
    0.005, 0.015, 0.025, 0.05, 0.1, 1 / 3, 0.5, 0.9, 0.95, 0.975, 0.995,
)
_NEAR_BOUNDARY_LEVELS = (
    math.nextafter(0.0, 1.0),   # smallest positive subnormal
    5e-324,
    1e-300,
    math.nextafter(0.5, 1.0),
    math.nextafter(1.0, 0.0),   # 0.9999999999999999
    1.0 - 2.0**-53,
)


def test_representative_and_near_boundary_levels_roundtrip_bit_exactly() -> None:
    for level in (*_REPRESENTATIVE_LEVELS, *_NEAR_BOUNDARY_LEVELS):
        columns = _wide({level: 7.5})
        assert list(columns) == [_expected_name(level)]
        recovered = ckpt._quantile_dict_from_wide(columns)
        assert recovered == {str(level): 7.5}
        (key,) = recovered
        assert _bits(float(key)) == _bits(level), f"{key} is not bit-exact for {level!r}"


# The runner contract passes ``quantile_predictions`` around keyed by
# ``str(level)``, so the encoder must accept a numeric STRING key on equal terms
# with the float it spells. That acceptance is deliberate, not an accident of
# ``float()`` being permissive -- see ``_validated_quantile_level``.
_STRING_KEY_CASES = (
    # (key as a caller may write it, the float it denotes)
    ("0.024", 0.024),
    ("0.976", 0.976),
    ("0.025", 0.025),
    ("0.975", 0.975),
    ("0.5", 0.5),
    # Spellings NOT already in ``str(float)`` normal form, which is how the
    # decoded key is shown to be re-derived from the level itself rather than
    # echoed back from whatever the caller wrote.
    ("0.0240", 0.024),
    ("2.4e-2", 0.024),
    (" 0.976 ", 0.976),
    ("9.76E-1", 0.976),
)


def test_numeric_string_levels_are_accepted_and_agree_with_their_floats() -> None:
    """A ``str(level)``-keyed mapping encodes and decodes exactly as the
    float-keyed one does, and decoding yields the live ``str(float)`` key.
    """
    for key, level in _STRING_KEY_CASES:
        string_columns = _wide({key: 7.5})
        float_columns = _wide({level: 7.5})

        # Accepted at all: a rejected key would expand to no column.
        assert string_columns, f"string key {key!r} produced no column"
        # The same column name as the float, bit for bit, not merely a close one.
        assert list(string_columns) == [_expected_name(level)]
        assert string_columns == float_columns

        recovered = ckpt._quantile_dict_from_wide(string_columns)
        # Decodes to the LIVE str(float) key, not the caller's spelling, and
        # carries the value through unchanged.
        assert recovered == {str(level): 7.5}
        (decoded_key,) = recovered
        assert _bits(float(decoded_key)) == _bits(level)
        assert recovered == ckpt._quantile_dict_from_wide(float_columns)


def test_string_and_float_keys_for_nearby_levels_share_one_column_space() -> None:
    """The 0.024/0.976 pair is what the old encoder collapsed, so the string
    path must keep those levels distinct and interchangeable with the floats.
    """
    indexed = {level: float(index) for index, level in enumerate(_COLLAPSING_LEVELS)}
    string_columns = _wide({str(level): value for level, value in indexed.items()})

    assert len(string_columns) == 4, (
        f"string keys collapsed onto {sorted(string_columns)}"
    )
    assert string_columns == _wide(indexed)

    recovered = ckpt._quantile_dict_from_wide(string_columns)
    assert recovered == {str(level): value for level, value in indexed.items()}
    # Re-encoding an already-decoded (string-keyed) mapping is a fixed point,
    # which is what lets a resumed origin merge with a freshly-computed one.
    assert _wide(recovered) == string_columns


# --------------------------------------------------------------------------- #
# 2. Invalid input drops a column; it never raises and never invents a label.
# --------------------------------------------------------------------------- #

def test_invalid_levels_produce_no_column_and_do_not_raise() -> None:
    for bad in (
        float("inf"),      # round(inf * 100) raised an UNCAUGHT OverflowError
        float("-inf"),
        float("nan"),
        0.0,               # not strictly inside (0, 1)
        1.0,
        -0.25,
        1.5,
        "not a number",
        None,
        object(),
    ):
        assert _wide({bad: 1.0}) == {}, f"level {bad!r} produced a column"


def test_invalid_predictions_are_still_dropped_as_before() -> None:
    assert _wide({0.5: "not a number"}) == {}
    assert _wide({0.5: None}) == {}
    assert ckpt._quantile_wide_columns({"quantile_predictions": None}) == {}
    assert ckpt._quantile_wide_columns({}) == {}


def test_a_record_with_no_quantile_columns_decodes_to_none() -> None:
    assert ckpt._quantile_dict_from_wide({"prediction": 1.0, "actual": 1.1}) is None
    assert ckpt._quantile_dict_from_wide({_expected_name(0.5): None}) is None
    assert ckpt._quantile_dict_from_wide({_expected_name(0.5): float("nan")}) is None


# --------------------------------------------------------------------------- #
# 3. The legacy family stays readable, at its own (coarse) value.
# --------------------------------------------------------------------------- #

def test_legacy_q_columns_decode_to_their_integer_percent_value() -> None:
    record = {f"q_{pct:02d}": float(pct) for pct in range(1, 100)}
    recovered = ckpt._quantile_dict_from_wide(record)
    assert recovered is not None
    assert recovered == {str(pct / 100.0): float(pct) for pct in range(1, 100)}


def test_legacy_q_100_is_ignored_rather_than_read_as_one() -> None:
    """q_100 is what the old encoder emitted for every level >= 0.995. It was
    never readable, and the true level is unrecoverable, so it stays dropped --
    inventing ``1.0`` would be a fabricated quantile."""
    assert ckpt._quantile_dict_from_wide({"q_100": 3.0}) is None
    mixed = ckpt._quantile_dict_from_wide({"q_100": 3.0, "q_50": 1.0})
    assert mixed == {"0.5": 1.0}


def test_legacy_q_00_is_ignored_rather_than_fabricated_as_zero() -> None:
    """q_00 is the other endpoint: what the old encoder emitted for every level
    below 0.005. Unlike q_100 the old two-digit reader DID parse it, returning a
    ``"0.0"`` key for a level that can never be 0.0 and whose true value is
    unrecoverable. It is now dropped, so no fabricated 0.0 reaches the mapping."""
    assert ckpt._quantile_dict_from_wide({"q_00": 3.0}) is None
    mixed = ckpt._quantile_dict_from_wide({"q_00": 3.0, "q_50": 1.0})
    assert mixed == {"0.5": 1.0}
    assert "0.0" not in mixed
    # Both endpoints together still leave only the interior column behind.
    both = ckpt._quantile_dict_from_wide({"q_00": 3.0, "q_100": 4.0, "q_50": 1.0})
    assert both == {"0.5": 1.0}
    # And the level-level helper agrees, for either endpoint.
    assert ckpt._legacy_level_from_quantile_column("q_00") is None
    assert ckpt._legacy_level_from_quantile_column("q_99") == 0.99


def test_new_family_takes_wholesale_precedence_over_legacy_within_a_row() -> None:
    """A row carrying both families is a row the new writer produced next to
    stale legacy columns; the exact family wins outright, and no rounded legacy
    level leaks into the mapping."""
    record = {
        _expected_name(0.024): 1.0,
        _expected_name(0.976): 2.0,
        "q_02": 99.0,
        "q_98": 99.0,
        "q_50": 99.0,
    }
    assert ckpt._quantile_dict_from_wide(record) == {"0.024": 1.0, "0.976": 2.0}


def test_corrupt_new_family_payloads_are_ignored_on_load() -> None:
    record = {
        "qx1_zzzzzzzzzzzzzzzz": 1.0,                       # not hex
        "qx1_3fa99999999999": 1.0,                         # too short
        "qx1_7ff0000000000000": 1.0,                       # decodes to +inf
        "qx1_0000000000000000": 1.0,                       # decodes to 0.0
        "qx1_3FA999999999999A": 1.0,                       # uppercase is not the grammar
        _expected_name(0.5): 4.0,
    }
    assert ckpt._quantile_dict_from_wide(record) == {"0.5": 4.0}


def test_a_row_whose_only_new_columns_are_corrupt_falls_back_to_legacy() -> None:
    record = {"qx1_zzzzzzzzzzzzzzzz": 1.0, "q_05": 2.0}
    assert ckpt._quantile_dict_from_wide(record) == {"0.05": 2.0}


def test_no_hex_payload_ever_leaks_into_a_returned_mapping() -> None:
    columns = _wide({0.024: 1.0, 0.5: 2.0})
    recovered = ckpt._quantile_dict_from_wide({**columns, "q_95": 3.0})
    assert recovered is not None
    for key in recovered:
        assert not key.startswith("qx1_")
        float(key)  # every key parses as a plain float literal


# --------------------------------------------------------------------------- #
# 4. Whole-directory behaviour.
# --------------------------------------------------------------------------- #

_BASE_ROW = {
    "target": "y", "horizon": 1, "origin": pd.Timestamp("2001-01-31"),
    "date": pd.Timestamp("2001-02-28"), "model": "ols", "prediction": 1.0,
    "actual": 1.1, "forecast_policy": "direct", "target_transform": "level",
}


def _write_legacy_quantile_origin(directory: Path, origin_pos: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    row = {**_BASE_ROW, "origin_pos": origin_pos, "q_05": 0.5, "q_95": 1.5}
    pd.DataFrame([row]).to_parquet(directory / f"origin_{origin_pos}.parquet", index=False)


def test_a_directory_holding_both_schemas_unions_deterministically(tmp_path: Path) -> None:
    directory = tmp_path / "mixed"
    _write_legacy_quantile_origin(directory, origin_pos=0)
    ckpt.append_origin_records(
        directory,
        1,
        [{**_BASE_ROW, "origin_pos": 1,
          "quantile_predictions": {0.024: 0.4, 0.976: 1.6}}],
    )

    loaded = ckpt.load_checkpoint_frame(directory)
    again = ckpt.load_checkpoint_frame(directory)
    assert list(loaded.columns) == list(again.columns)
    assert len(loaded) == 2

    by_pos = loaded.set_index("origin_pos")["quantile_predictions"]
    assert by_pos.loc[0] == {"0.05": 0.5, "0.95": 1.5}
    assert by_pos.loc[1] == {"0.024": 0.4, "0.976": 1.6}


def test_new_origin_parquet_stays_scalar_only_with_deterministic_column_order(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "new"
    levels = (0.976, 0.024, 0.5, 0.025)  # deliberately unsorted at the source
    ckpt.append_origin_records(
        directory,
        0,
        [{**_BASE_ROW, "origin_pos": 0,
          "quantile_predictions": {lv: float(i) for i, lv in enumerate(levels)}}],
    )

    frame = pd.read_parquet(directory / "origin_0.parquet")
    for column in frame.columns:
        for value in frame[column].tolist():
            assert not isinstance(value, (dict, list, tuple, set)), (
                f"non-scalar value {value!r} in column {column!r}"
            )

    extra = [c for c in frame.columns if c not in ckpt.LEAN_FORECAST_COLUMNS]
    assert extra == sorted(_expected_name(lv) for lv in levels)


# --------------------------------------------------------------------------- #
# 5. The public path: a real resumed run, not the helpers.
# --------------------------------------------------------------------------- #

def _panel(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.1 * np.sin(np.arange(n) / 2.0),
            "x1": x,
            "x2": np.sin(np.arange(n) / 3.0),
        },
        index=idx,
    )


def _window() -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def test_resume_roundtrips_off_grid_quantile_levels_through_a_real_run(
    tmp_path: Path, monkeypatch
) -> None:
    """The end-to-end statement of the defect: with levels the old grammar
    collapsed, a resumed origin must still carry the same four keys the live
    run produced."""
    cell = tmp_path / "cell"
    run_kwargs = dict(
        window=_window(),
        features=mf.feature_engineering.feature_spec(target="y", target_lags=[1, 2]),
        params={"quantile_regression_forest": {
            "n_estimators": 10, "random_state": 0,
            "quantile_levels": _COLLAPSING_LEVELS,
        }},
        model_selection={"quantile_regression_forest": None},
        save_models=False,
    )
    full = mf.forecasting.run(
        _panel(), "quantile_regression_forest", checkpoint_path=cell, **run_kwargs
    )
    full_frame = full.to_frame().set_index("origin_pos")

    hdir = cell / "h1"
    positions = sorted(ckpt.completed_origin_positions(hdir))
    assert len(positions) >= 2
    last_pos = positions[-1]
    (hdir / f"origin_{last_pos}.parquet").unlink()

    import macroforecast.forecasting.runner as runner

    computed: list[int] = []
    original = runner._fit_predict_origin

    def _spy(item, *args, **kwargs):
        computed.append(int(item["row"].get("origin_pos")))
        return original(item, *args, **kwargs)

    monkeypatch.setattr(runner, "_fit_predict_origin", _spy)
    resumed = mf.forecasting.run(
        _panel(), "quantile_regression_forest", checkpoint_path=cell, **run_kwargs
    )
    resumed_frame = resumed.to_frame().set_index("origin_pos")

    assert computed == [last_pos]  # only the deleted origin was recomputed

    expected_keys = {str(lv) for lv in _COLLAPSING_LEVELS}
    for pos in positions:
        full_q = full_frame.loc[pos, "quantile_predictions"]
        resumed_q = resumed_frame.loc[pos, "quantile_predictions"]
        if isinstance(full_q, pd.Series):
            full_q, resumed_q = full_q.iloc[0], resumed_q.iloc[0]
        assert isinstance(resumed_q, dict)
        assert set(full_q) == expected_keys
        assert set(resumed_q) == expected_keys
        for level, value in full_q.items():
            assert np.isclose(resumed_q[level], value)
