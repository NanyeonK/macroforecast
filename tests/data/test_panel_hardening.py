from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def test_as_panel_rejects_invalid_dates_by_default() -> None:
    frame = pd.DataFrame({"date": ["2020-01-01", "not-a-date"], "x": [1.0, 2.0]})

    with pytest.raises(ValueError, match="invalid or missing date values"):
        mf.data.as_panel(frame, date="date")


def test_as_panel_can_permissively_report_dropped_dates_and_numeric_coercion() -> None:
    frame = pd.DataFrame({"date": ["2020-01-01", "bad-date", "2020-03-01"], "x": ["1.0", "bad-number", "bad-number"]})

    panel = mf.data.as_panel(frame, date="date", strict=False)

    assert list(panel.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-03-01"]
    report = panel.attrs["macroforecast_panel_report"]
    assert report["invalid_date_rows_dropped"] == 1
    assert report["numeric_coercion"]["coerced_cells"] == 1


def test_as_panel_rejects_non_numeric_values_by_default() -> None:
    frame = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "x": ["1.0", "bad-number"]})

    with pytest.raises(ValueError, match="non-numeric panel values"):
        mf.data.as_panel(frame, date="date")


def test_validate_panel_rejects_infinite_values() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2, freq="MS"), "x": [1.0, np.inf]})

    with pytest.raises(ValueError, match="infinite values"):
        mf.data.as_panel(frame, date="date")


def test_spec_predictors_all_expands_to_non_target_columns() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "target": [1.0, 2.0, 3.0],
                "x": [4.0, 5.0, 6.0],
            }
        ),
        date="date",
    )

    data_spec = mf.data.spec(panel, target="target")

    assert data_spec.predictors == ("x",)
    assert data_spec.metadata["data_spec"]["predictors"] == ["x"]


def test_spec_allows_target_only_design() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "target": [1.0, 2.0, 3.0],
                "x": [4.0, 5.0, 6.0],
            }
        ),
        date="date",
    )

    data_spec = mf.data.spec(panel, target="target", predictors=[])

    assert data_spec.predictors == ()
    assert list(data_spec.panel.columns) == ["target"]
    assert data_spec.metadata["data_spec"]["predictors"] == []


def test_spec_rejects_target_leakage_in_explicit_predictors() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "target": [1.0, 2.0, 3.0],
                "x": [4.0, 5.0, 6.0],
            }
        ),
        date="date",
    )

    with pytest.raises(ValueError, match="must not include target"):
        mf.data.spec(panel, target="target", predictors=["target", "x"])


def test_load_custom_csv_preserves_first_pass_panel_report_when_permissive(tmp_path) -> None:
    path = tmp_path / "custom.csv"
    path.write_text(
        "date,x\n"
        "2020-01-01,1.0\n"
        "bad-date,2.0\n"
        "2020-03-01,bad-number\n",
        encoding="utf-8",
    )

    bundle = mf.data.load_custom_csv(path, date="date", strict=False)

    assert bundle.metadata["panel"]["invalid_date_rows_dropped"] == 1
    assert bundle.metadata["panel"]["numeric_coercion"]["coerced_cells"] == 1


def test_as_panel_duplicate_dates_in_long_format_names_pivot_fix() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"],
            "country": ["US", "CA", "US", "CA"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="pivot to wide"):
        mf.data.as_panel(frame, date="date")


def test_load_custom_csv_custom_na_values_pass_strict_mode(tmp_path) -> None:
    path = tmp_path / "custom_na.csv"
    path.write_text(
        "date,x\n"
        "2020-01-01,1.0\n"
        "2020-02-01,MISSING\n",
        encoding="utf-8",
    )

    bundle = mf.data.load_custom_csv(path, date="date", na_values=["MISSING"])

    assert np.isnan(bundle.panel.loc[pd.Timestamp("2020-02-01"), "x"])


def test_load_custom_csv_dayfirst_dates_parse_when_requested(tmp_path) -> None:
    path = tmp_path / "dayfirst.csv"
    path.write_text(
        "date,x\n"
        "01/02/2020,1.0\n"
        "02/03/2020,2.0\n",
        encoding="utf-8",
    )

    bundle = mf.data.load_custom_csv(path, date="date", dayfirst=True)

    assert list(bundle.panel.index) == [
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-03-02"),
    ]


def test_load_custom_csv_date_format_parses_when_requested(tmp_path) -> None:
    path = tmp_path / "date_format.csv"
    path.write_text(
        "date,x\n"
        "20200131,1.0\n"
        "20200229,2.0\n",
        encoding="utf-8",
    )

    bundle = mf.data.load_custom_csv(path, date="date", date_format="%Y%m%d")

    assert list(bundle.panel.index) == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
    ]


def test_set_frequencies_marks_existing_panel_as_mixed() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "monthly": [1.0, 2.0, 3.0, 4.0],
                "quarterly": [10.0, np.nan, np.nan, 20.0],
            }
        ),
        date="date",
    )

    bundle = mf.data.set_frequencies(
        panel,
        {"monthly": "monthly", "quarterly": "quarterly"},
        metadata={"dataset": "custom_mixed"},
    )

    assert bundle.metadata["frequency"] == "mixed"
    assert bundle.metadata["native_frequency_by_column"] == {
        "monthly": "monthly",
        "quarterly": "quarterly",
    }
    assert bundle.metadata["native_frequency_counts"] == {"monthly": 1, "quarterly": 1}
    assert mf.data.panel_info(bundle)["frequency"] == "mixed"


def test_load_custom_csv_accepts_column_frequency_contract(tmp_path) -> None:
    path = tmp_path / "mixed.csv"
    path.write_text(
        "date,m,q\n"
        "2020-01-01,1,10\n"
        "2020-02-01,2,\n"
        "2020-03-01,3,\n"
        "2020-04-01,4,20\n",
        encoding="utf-8",
    )

    bundle = mf.data.load_custom_csv(
        path,
        date="date",
        frequency_by_column={"m": "m", "q": "q"},
    )

    assert bundle.metadata["frequency"] == "mixed"
    assert bundle.metadata["native_frequency_by_column"] == {"m": "monthly", "q": "quarterly"}
    assert bundle.panel["q"].isna().sum() == 2


def test_set_frequencies_rejects_incomplete_frequency_map() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=2, freq="MS"),
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
            }
        ),
        date="date",
    )

    with pytest.raises(ValueError, match="must include every panel column"):
        mf.data.set_frequencies(panel, {"x": "monthly"})


# --------------------------------------------------------------------------- #
# F-007: complex panels are rejected before anything casts them to float
# --------------------------------------------------------------------------- #

def _complex_frame() -> pd.DataFrame:
    index = pd.date_range("2000-01-31", periods=6, freq="ME", name="date")
    return pd.DataFrame(
        {"real": np.arange(6, dtype=float), "cplx": np.arange(6) + 1j * np.arange(6)},
        index=index,
    )


def test_as_panel_rejects_complex_columns() -> None:
    """Complex passes ``is_numeric_dtype`` but is not real-valued forecasting data.

    Before this, ``as_panel`` accepted it and the float cast downstream dropped the
    imaginary part, so two panels differing only in their imaginary values shared one
    content fingerprint and pandas said nothing louder than a ``ComplexWarning``.
    """
    with pytest.raises(TypeError, match="real-valued"):
        mf.data.as_panel(_complex_frame())


def test_as_panel_still_rejects_complex_when_not_strict() -> None:
    """``strict=False`` relaxes coercion, not the value domain.

    Missing dates and unparseable strings are ordinary input mess that a permissive
    load may absorb. Complex data is a different kind of thing: nothing downstream can
    represent it, so accepting it would only move the loss somewhere quieter.
    """
    with pytest.raises(TypeError, match="real-valued"):
        mf.data.as_panel(_complex_frame(), strict=False)


def test_validate_panel_rejects_complex_columns() -> None:
    """The direct call is guarded too, not just the ``as_panel`` path."""
    with pytest.raises(TypeError, match="real-valued"):
        mf.data.validate_panel(_complex_frame())


def test_custom_dataset_rejects_complex_columns() -> None:
    """The contract propagates to a downstream custom-data entry point."""
    with pytest.raises(TypeError, match="real-valued"):
        mf.data.custom_dataset(_complex_frame().reset_index())


def test_complex_rejection_names_the_offending_columns() -> None:
    """A caller has to be able to find which column to split."""
    with pytest.raises(TypeError, match="cplx"):
        mf.data.validate_panel(_complex_frame())


# --------------------------------------------------------------------------- #
# F-006: a numeric first column is not a date column
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strict", [True, False])
def test_as_panel_rejects_numeric_first_column_inference(strict: bool) -> None:
    """Without ``date=``, the first column is the date candidate — but not if it is numeric.

    ``x=[1, 2, 3]`` used to parse as three timestamps 1–3 ns after the Unix epoch, and
    ``x`` then left the panel as the index, so an ordinary predictor was silently
    consumed and the result looked like a valid canonical panel. ``strict=False`` did
    not help: this is structural ambiguity, not a coercion the caller opted to tolerate.
    """
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})

    with pytest.raises(TypeError, match="date="):
        mf.data.as_panel(frame, strict=strict)


@pytest.mark.parametrize("strict", [True, False])
def test_as_panel_rejects_numeric_string_first_column_inference(strict: bool) -> None:
    """Numbers written as strings read as numbers too, so they are refused the same way."""
    frame = pd.DataFrame({"x": ["1", "2", "3"], "y": [4.0, 5.0, 6.0]})

    with pytest.raises(TypeError, match="numeric"):
        mf.data.as_panel(frame, strict=strict)


def test_as_panel_still_infers_a_date_string_first_column() -> None:
    """The case the inference was written for keeps working."""
    frame = pd.DataFrame(
        {"when": ["2020-01-31", "2020-02-29", "2020-03-31"], "y": [1.0, 2.0, 3.0]}
    )

    panel = mf.data.as_panel(frame)

    assert isinstance(panel.index, pd.DatetimeIndex)
    assert list(panel.columns) == ["y"]


def test_as_panel_still_infers_a_datetime_dtype_first_column() -> None:
    frame = pd.DataFrame(
        {"when": pd.to_datetime(["2020-01-31", "2020-02-29"]), "y": [1.0, 2.0]}
    )

    assert isinstance(mf.data.as_panel(frame).index, pd.DatetimeIndex)


def test_explicit_date_column_is_still_the_callers_authority() -> None:
    """The guard is on inference only. Naming the column keeps the explicit parse path."""
    frame = pd.DataFrame({"t": [20200131, 20200229], "y": [1.0, 2.0]})

    panel = mf.data.as_panel(frame, date="t")

    assert isinstance(panel.index, pd.DatetimeIndex)
    assert list(panel.columns) == ["y"]


# --------------------------------------------------------------------------- #
# F-008: DataSpec normalisation is exact and deterministic
# --------------------------------------------------------------------------- #

def _spec_frame() -> pd.DataFrame:
    index = pd.date_range("2000-01-31", periods=6, freq="ME", name="date")
    return pd.DataFrame(
        {"y": np.arange(6.0), "a": np.arange(6.0), "b": np.arange(6.0)}, index=index
    )


@pytest.mark.parametrize(
    "predictors", [np.array(["a", "b"]), pd.Index(["a", "b"]), ["a", "b"], ("a", "b")]
)
def test_spec_accepts_array_like_predictor_collections(predictors) -> None:
    """``predictors == 'all'`` against an array returns an array, and ``if`` on it raises.

    So a numpy array or a pandas Index — both ordinary ways to hold a column list —
    could not be passed at all. The sentinel is tested only after confirming a string.
    """
    assert mf.data.spec(_spec_frame(), target="y", predictors=predictors).predictors == ("a", "b")


def test_spec_rejects_a_bare_non_sentinel_predictor_string() -> None:
    """``predictors='a'`` is a typo for ``['a']``, not an iterable of one-letter names."""
    with pytest.raises(ValueError, match="predictors must be 'all'"):
        mf.data.spec(_spec_frame(), target="y", predictors="a")


def test_spec_preserves_predictor_and_horizon_order() -> None:
    result = mf.data.spec(_spec_frame(), target="y", predictors=["b", "a"], horizons=[3, 1])

    assert result.predictors == ("b", "a")
    assert result.horizons == (3, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": "y", "predictors": ["a", "a"]},
        {"targets": ["y", "y"]},
        {"target": "y", "horizons": [1, 1]},
    ],
)
def test_spec_rejects_duplicate_inputs(kwargs) -> None:
    """Duplicates were absorbed — predictors de-duplicated, targets and horizons kept.

    Either way the recorded spec stopped matching what the caller wrote, so a typo
    survived into the run instead of being reported.
    """
    with pytest.raises(ValueError, match="duplicates"):
        mf.data.spec(_spec_frame(), **kwargs)


@pytest.mark.parametrize("horizons", [np.int64(2), [np.int64(1), np.int64(3)], 2, [1, 3]])
def test_spec_accepts_integral_horizons(horizons) -> None:
    """numpy integer scalars are integers in every sense that matters here."""
    assert mf.data.spec(_spec_frame(), target="y", horizons=horizons).horizons


@pytest.mark.parametrize("horizons", [1.0, 1.9, True, "2", 0, -1])
def test_spec_rejects_non_integral_or_non_positive_horizons(horizons) -> None:
    """``int(1.9)`` used to succeed, so the run forecast a horizon nobody asked for.

    ``1.0`` is rejected with ``1.9`` on purpose: accepting one and not the other would
    make the rule depend on the value rather than on the type.
    """
    with pytest.raises((TypeError, ValueError)):
        mf.data.spec(_spec_frame(), target="y", horizons=horizons)


def test_spec_rejects_non_string_column_names() -> None:
    """``str(value)`` turned anything into a name, failing later as a missing column."""
    with pytest.raises(TypeError, match="non-empty column-name strings"):
        mf.data.spec(_spec_frame(), targets=[None])


def test_spec_rejects_targets_given_as_one_string() -> None:
    """``targets='y'`` would otherwise iterate the characters."""
    with pytest.raises(TypeError, match="not a single string"):
        mf.data.spec(_spec_frame(), targets="y")


def test_spec_errors_do_not_leak_ambiguous_truth_internals() -> None:
    """A caller should read about their argument, not about numpy's truth value."""
    with pytest.raises((TypeError, ValueError)) as exc:
        mf.data.spec(_spec_frame(), target="y", horizons=1.9)

    assert "truth value" not in str(exc.value)
    assert "ambiguous" not in str(exc.value)


# --------------------------------------------------------------------------- #
# F-009: the overall label describes the output panel
# --------------------------------------------------------------------------- #

def _frequency_frame() -> pd.DataFrame:
    index = pd.date_range("2000-01-31", periods=6, freq="ME", name="date")
    return pd.DataFrame({"m": np.arange(6.0), "q": np.arange(6.0)}, index=index)


def test_mixed_output_frequencies_derive_a_mixed_overall_label() -> None:
    bundle = mf.data.set_frequencies(_frequency_frame(), {"m": "monthly", "q": "quarterly"})

    assert bundle.metadata["frequency"] == "mixed"


def test_explicit_frequency_contradicting_the_columns_is_rejected() -> None:
    """It used to be written through, so overall said ``monthly`` while the per-column
    counts said monthly AND quarterly — one metadata dict disagreeing with itself."""
    with pytest.raises(ValueError, match="contradicts"):
        mf.data.set_frequencies(
            _frequency_frame(), {"m": "monthly", "q": "quarterly"}, frequency="monthly"
        )


def test_explicit_mixed_is_checked_too() -> None:
    """``'mixed'`` was a free pass; now it has to be true of the columns as well."""
    with pytest.raises(ValueError, match="contradicts"):
        mf.data.set_frequencies(
            _frequency_frame(), {"m": "monthly", "q": "monthly"}, frequency="mixed"
        )


def test_mixed_native_with_a_homogeneous_output_map_is_monthly() -> None:
    """The valid case the invariant has to keep: aligned output, mixed source.

    A quarterly series aligned to monthly output IS monthly in the panel a consumer
    receives, which is what ``metadata['frequency']`` describes.
    """
    bundle = mf.data.set_frequencies(
        _frequency_frame(),
        {"m": "monthly", "q": "quarterly"},
        output_frequency_by_column={"m": "monthly", "q": "monthly"},
    )

    assert bundle.metadata["frequency"] == "monthly"
    assert set(bundle.metadata["native_frequency_counts"]) == {"monthly", "quarterly"}
    assert set(bundle.metadata["output_frequency_counts"]) == {"monthly"}


@pytest.mark.parametrize("frequency", [None, "unknown", "monthly"])
def test_homogeneous_columns_agree_with_none_unknown_and_the_matching_label(frequency) -> None:
    bundle = mf.data.set_frequencies(
        _frequency_frame(), {"m": "monthly", "q": "monthly"}, frequency=frequency
    )

    assert bundle.metadata["frequency"] == "monthly"


# --------------------------------------------------------------------------- #
# F-008 (follow-up): rejection messages must name the argument, not the mechanics
# --------------------------------------------------------------------------- #

#: Phrases that mean the caller is reading about Python's or numpy's internals rather
#: than about the argument they passed.
_IMPLEMENTATION_INTERNALS = ("not iterable", "iteration over", "truth value", "ambiguous")


@pytest.mark.parametrize(
    "horizons",
    [
        pytest.param(1.9, id="python-float-scalar"),
        pytest.param(np.float64(1.9), id="numpy-float-scalar"),
        pytest.param(np.bool_(True), id="numpy-bool-scalar"),
        pytest.param([np.bool_(True)], id="numpy-bool-nested"),
        pytest.param(np.array(2), id="zero-dimensional-array"),
        pytest.param(np.array(1.9), id="zero-dimensional-float-array"),
        pytest.param(object(), id="arbitrary-object"),
    ],
)
def test_horizon_rejection_messages_name_the_argument(horizons) -> None:
    """These were already rejected — by whatever Python raised on the way out.

    A scalar ``1.9`` fell through to the iterable branch and failed as "'float' object
    is not iterable"; ``np.bool_(True)`` is neither ``bool`` nor ``np.integer``, so it
    did the same; a 0-d array failed as "iteration over a 0-d array". All true about the
    implementation and useless about the argument, which is what the caller has to fix.
    """
    with pytest.raises(TypeError) as exc:
        mf.data.spec(_spec_frame(), target="y", horizons=horizons)

    message = str(exc.value)
    assert "horizons" in message, "the message must name the argument"
    for phrase in _IMPLEMENTATION_INTERNALS:
        assert phrase not in message, f"message leaks implementation detail: {phrase!r}"


def test_numpy_bool_is_a_bool_here_not_an_integer() -> None:
    """numpy makes ``np.bool_`` a subclass of neither ``bool`` nor ``np.integer``.

    So it has to be named explicitly, or it is silently outside both the bool guard and
    the integer branch.
    """
    with pytest.raises(TypeError, match="not a bool"):
        mf.data.spec(_spec_frame(), target="y", horizons=np.bool_(True))

    with pytest.raises(TypeError, match="not a bool"):
        mf.data.spec(_spec_frame(), target="y", horizons=[1, np.bool_(True)])


@pytest.mark.parametrize("horizons", [np.array(2), np.array(1.9)])
def test_zero_dimensional_array_is_rejected_with_the_fix_named(horizons) -> None:
    """Not accepted as an integer scalar: nothing in the package's user-facing input
    contracts takes a 0-d array, so it is refused — with the unwrap spelled out."""
    with pytest.raises(TypeError, match="0-dimensional array"):
        mf.data.spec(_spec_frame(), target="y", horizons=horizons)

    with pytest.raises(TypeError, match=r"\.item\(\)"):
        mf.data.spec(_spec_frame(), target="y", horizons=horizons)


@pytest.mark.parametrize("horizons", [np.array(2), np.array(1.9)])
def test_zero_dimensional_advice_never_recommends_lossy_coercion(horizons) -> None:
    """The hint must not undo the contract it is attached to.

    int(np.array(1.9)) is 1, which is exactly the silent truncation this function
    rejects 1.9 for. Advice to coerce would therefore tell the caller to reproduce
    the defect by hand, so the message says to extract the value and pass it only if it
    is already a positive integer.
    """
    with pytest.raises(TypeError) as exc:
        mf.data.spec(_spec_frame(), target="y", horizons=horizons)

    message = str(exc.value)
    assert "int(" not in message, "the message must not recommend int() coercion"
    assert ".item()" in message
    assert "already a positive integer" in message


@pytest.mark.parametrize(
    "horizons", [2, np.int64(2), [1, 3], (1, 3), np.array([1, 3]), pd.Index([1, 3])]
)
def test_valid_horizon_inputs_are_unaffected(horizons) -> None:
    """The narrowing must not cost the accepted forms, scalar or iterable."""
    assert mf.data.spec(_spec_frame(), target="y", horizons=horizons).horizons
