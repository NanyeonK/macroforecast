from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import macroforecast as mf


class _SyntheticVintageSource:
    kind = "synthetic_vintage"

    def __init__(self, bundles: dict[pd.Timestamp, mf.data.DataBundle]) -> None:
        self.bundles = dict(bundles)
        self.calls: list[pd.Timestamp] = []

    def available_vintages(self):
        return list(self.bundles)

    def resolve(self, origin_date):
        origin = pd.Timestamp(origin_date)
        keys = [key for key in self.bundles if key <= origin]
        if not keys:
            raise mf.data.VintageUnavailableError("no vintage available")
        key = max(keys)
        self.calls.append(origin)
        return self.bundles[key]


def _oracle_bundles(leaky: bool = False) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, mf.data.DataBundle]]:
    reference = pd.date_range("2000-01-31", periods=9, freq="ME", name="date")
    base_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    base_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    bundles: dict[pd.Timestamp, mf.data.DataBundle] = {}
    for i in range(2, len(reference)):
        origin = reference[i]
        stop = i + 1 if leaky else i
        a_values = base_a[:stop].copy()
        b_values = base_b[:stop].copy()
        if i >= 4:
            # The third vintage revises A for February and March; B never revises.
            a_values[1] = 20.0
            a_values[2] = 30.0
        panel = pd.DataFrame(
            {"A": a_values, "B": b_values},
            index=reference[:stop],
        )
        bundles[origin] = mf.data.DataBundle(
            panel,
            {
                "dataset": "synthetic",
                "frequency": "monthly",
                "vintage": f"v{i}",
            },
        )
    return reference, bundles


def _month_start_bundles() -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, mf.data.DataBundle]]:
    reference = pd.date_range("2000-01-01", periods=9, freq="MS", name="date")
    bundles: dict[pd.Timestamp, mf.data.DataBundle] = {}
    for i in range(2, len(reference)):
        origin = reference[i]
        panel = pd.DataFrame(
            {
                "A": [float(value) for value in range(1, i + 1)],
                "B": [float(value * 10) for value in range(1, i + 1)],
            },
            index=reference[:i],
        )
        bundles[origin] = mf.data.DataBundle(
            panel,
            {
                "dataset": "synthetic_ms",
                "frequency": "monthly",
                "vintage": f"v{i}",
            },
        )
    return reference, bundles


def _publication_lag_bundles(
    *,
    missing_forever: set[pd.Timestamp] | None = None,
) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, mf.data.DataBundle]]:
    reference = pd.date_range("2000-01-31", periods=10, freq="ME", name="date")
    missing = set(missing_forever or set())
    bundles: dict[pd.Timestamp, mf.data.DataBundle] = {}
    for i in range(2, len(reference)):
        values: list[float] = []
        for j, date in enumerate(reference[:i]):
            if date in missing or i < j + 2:
                values.append(np.nan)
            else:
                values.append(float(j + 1))
        panel = pd.DataFrame(
            {"A": values, "B": [float((j + 1) * 10) for j in range(i)]},
            index=reference[:i],
        )
        bundles[reference[i]] = mf.data.DataBundle(
            panel,
            {
                "dataset": "publication_lag",
                "frequency": "monthly",
                "vintage": f"v{i}",
            },
        )
    return reference, bundles


def _window(
    reference: pd.DatetimeIndex,
    *,
    horizon: int = 1,
    last_origin: pd.Timestamp | None = None,
) -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=3),
        val=mf.window.val_last_block(size=1),
        test=mf.window.test_origins(
            first_origin=reference[4],
            last_origin=reference[6] if last_origin is None else last_origin,
            horizon=horizon,
        ),
    )


def _features() -> mf.feature_engineering.FeatureSpec:
    return mf.feature_engineering.feature_spec(
        target="A",
        predictors=[],
        lags=None,
        target_lags=(1,),
    )


def _recording_model(sink: list[dict[str, pd.DataFrame | pd.Series]]) -> mf.models.ModelSpec:
    class Fit:
        def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
            sink.append({"X_fit": X.copy(), "y_fit": y.copy()})

        def predict(self, X: pd.DataFrame) -> pd.Series:
            sink[-1]["X_test"] = X.copy()
            return X.iloc[:, 0]

    def fit(X: pd.DataFrame, y: pd.Series):
        return Fit(X, y)

    return mf.models.custom_model("recording_vintage_oracle", fit)


def _constant_model(value: float = 1.0, *, name: str = "constant_vintage_oracle") -> mf.models.ModelSpec:
    class Fit:
        def predict(self, X: pd.DataFrame) -> pd.Series:
            return pd.Series(float(value), index=X.index)

    def fit(X: pd.DataFrame, y: pd.Series):
        return Fit()

    return mf.models.custom_model(name, fit)


def _recording_panel_model(sink: list[dict[str, pd.DataFrame]]) -> mf.models.ModelSpec:
    class Fit:
        def __init__(self, bundle: mf.data.DataBundle, *, target: str) -> None:
            panel = bundle.panel.copy()
            sink.append({"fit_panel": panel})
            self._last = float(panel[target].dropna().iloc[-1])

        def predict(self, test_panel: pd.DataFrame) -> pd.Series:
            sink[-1]["test_panel"] = test_panel.copy()
            return pd.Series(self._last, index=test_panel.index)

    def fit(bundle: mf.data.DataBundle, *, target: str):
        return Fit(bundle, target=target)

    return mf.models.custom_model(
        "recording_panel_vintage_oracle",
        fit,
        default_params={"target": None},
        input_kind="panel",
    )


def test_vintage_direct_oracle_uses_exact_real_time_fit_rows() -> None:
    reference, bundles = _oracle_bundles()
    source = _SyntheticVintageSource(bundles)
    spec = mf.data.VintagePanelSpec(source, reference)
    records: list[dict[str, pd.DataFrame | pd.Series]] = []

    result = mf.forecasting.run(
        spec,
        _recording_model(records),
        window=_window(reference),
        target="A",
        features=_features(),
        save_models=False,
    )

    table = result.to_frame()
    assert list(table["vintage_id"]) == ["v4", "v5", "v6"]
    np.testing.assert_allclose(table["prediction"].to_numpy(float), [4.0, 5.0, 6.0])
    np.testing.assert_allclose(table["actual"].to_numpy(float), [6.0, 7.0, 8.0])

    expected = [
        (
            pd.DataFrame(
                {"A_lag1": [1.0, 20.0]},
                index=pd.DatetimeIndex(["2000-02-29", "2000-03-31"], name="date"),
            ),
            pd.Series(
                [30.0, 4.0],
                index=pd.DatetimeIndex(["2000-02-29", "2000-03-31"], name="date"),
                name="A_level_h1",
            ),
            pd.DataFrame(
                {"A_lag1": [4.0]},
                index=pd.DatetimeIndex(["2000-05-31"], name="date"),
            ),
        ),
        (
            pd.DataFrame(
                {"A_lag1": [1.0, 20.0, 30.0]},
                index=pd.DatetimeIndex(
                    ["2000-02-29", "2000-03-31", "2000-04-30"],
                    name="date",
                ),
            ),
            pd.Series(
                [30.0, 4.0, 5.0],
                index=pd.DatetimeIndex(
                    ["2000-02-29", "2000-03-31", "2000-04-30"],
                    name="date",
                ),
                name="A_level_h1",
            ),
            pd.DataFrame(
                {"A_lag1": [5.0]},
                index=pd.DatetimeIndex(["2000-06-30"], name="date"),
            ),
        ),
        (
            pd.DataFrame(
                {"A_lag1": [1.0, 20.0, 30.0, 4.0]},
                index=pd.DatetimeIndex(
                    ["2000-02-29", "2000-03-31", "2000-04-30", "2000-05-31"],
                    name="date",
                ),
            ),
            pd.Series(
                [30.0, 4.0, 5.0, 6.0],
                index=pd.DatetimeIndex(
                    ["2000-02-29", "2000-03-31", "2000-04-30", "2000-05-31"],
                    name="date",
                ),
                name="A_level_h1",
            ),
            pd.DataFrame(
                {"A_lag1": [6.0]},
                index=pd.DatetimeIndex(["2000-07-31"], name="date"),
            ),
        ),
    ]
    for record, (X_fit, y_fit, X_test) in zip(records, expected, strict=True):
        pdt.assert_frame_equal(record["X_fit"], X_fit)
        pdt.assert_series_equal(record["y_fit"], y_fit)
        pdt.assert_frame_equal(record["X_test"], X_test)


def test_vintage_reference_calendar_anchor_mismatch_raises_clear_error() -> None:
    month_start_reference, bundles = _month_start_bundles()
    month_end_reference = pd.date_range(
        month_start_reference[0].to_period("M").end_time.normalize(),
        periods=len(month_start_reference),
        freq="ME",
        name="date",
    )
    spec = mf.data.VintagePanelSpec(
        _SyntheticVintageSource(bundles),
        month_end_reference,
    )

    with pytest.raises(ValueError, match="freq='MS'"):
        mf.forecasting.run(
            spec,
            _constant_model(0.0),
            window=_window(month_end_reference),
            target="A",
            features=_features(),
            save_models=False,
        )


def test_vintage_month_start_reference_runs_smoke() -> None:
    reference, bundles = _month_start_bundles()
    spec = mf.data.VintagePanelSpec(_SyntheticVintageSource(bundles), reference)

    result = mf.forecasting.run(
        spec,
        _constant_model(0.0),
        window=_window(reference),
        target="A",
        features=_features(),
        save_models=False,
    )

    assert not result.to_frame().empty
    assert result.metadata["vintage_boundary_audit"]["vintage_boundary_ok"] is True


def test_vintage_runner_validations_and_embargo_warning() -> None:
    reference, bundles = _oracle_bundles()
    source = _SyntheticVintageSource(bundles)
    spec = mf.data.VintagePanelSpec(source, reference)
    bad_window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=3, retrain_every=2),
        val=mf.window.val_last_block(size=1),
        test=mf.window.test_origins(first_origin=reference[4], last_origin=reference[4]),
    )
    with pytest.raises(ValueError, match="retrain_every=1"):
        mf.forecasting.run(
            spec,
            "ols",
            window=bad_window,
            target="A",
            features=_features(),
            save_models=False,
        )

    embargo_window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=3, embargo=1),
        val=mf.window.val_last_block(size=1),
        test=mf.window.test_origins(first_origin=reference[4], last_origin=reference[4]),
    )
    with pytest.warns(UserWarning, match="double-purge"):
        mf.forecasting.run(
            spec,
            "ols",
            window=embargo_window,
            target="A",
            features=_features(),
            save_models=False,
        )


def _actual_revision_bundles() -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, mf.data.DataBundle]]:
    reference, bundles = _oracle_bundles()
    revised: dict[pd.Timestamp, mf.data.DataBundle] = {}
    for key, bundle in bundles.items():
        panel = bundle.panel.copy()
        if key >= reference[7] and reference[5] in panel.index:
            panel.loc[reference[5], "A"] = 60.0
        if key >= reference[8] and reference[6] in panel.index:
            panel.loc[reference[6], "A"] = 70.0
        revised[key] = mf.data.DataBundle(panel, dict(bundle.metadata))
    return reference, revised


def test_vintage_first_release_actuals_are_resolved_per_target_date() -> None:
    reference, bundles = _actual_revision_bundles()
    latest = mf.data.VintagePanelSpec(_SyntheticVintageSource(bundles), reference)
    first_release = mf.data.VintagePanelSpec(
        _SyntheticVintageSource(bundles),
        reference,
        actuals_vintage="first_release",
    )

    kwargs = dict(
        model=_constant_model(0.0),
        window=_window(reference, last_origin=reference[5]),
        target="A",
        features=_features(),
        save_models=False,
    )
    latest_table = mf.forecasting.run(latest, **kwargs).to_frame()
    first_table = mf.forecasting.run(first_release, **kwargs).to_frame()

    np.testing.assert_allclose(latest_table["actual"].to_numpy(float), [60.0, 70.0])
    np.testing.assert_allclose(first_table["actual"].to_numpy(float), [6.0, 7.0])
    assert list(latest_table["actuals_vintage_id"]) == ["v8", "v8"]
    assert list(first_table["actuals_vintage_id"]) == ["v6", "v7"]
    assert latest_table["actual"].tolist() != first_table["actual"].tolist()


def test_vintage_first_release_walks_forward_across_publication_lag() -> None:
    reference, bundles = _publication_lag_bundles()
    spec = mf.data.VintagePanelSpec(
        _SyntheticVintageSource(bundles),
        reference,
        actuals_vintage="first_release",
    )
    records: list[dict[str, pd.DataFrame]] = []

    result = mf.forecasting.run(
        spec,
        _recording_panel_model(records),
        window=_window(reference, last_origin=reference[5]),
        target="A",
        features=None,
        save_models=False,
    )

    table = result.to_frame()
    np.testing.assert_allclose(table["actual"].to_numpy(float), [6.0, 7.0])
    assert list(table["actuals_vintage_id"]) == ["v7", "v8"]
    summary = result.metadata["vintage_source"]["first_release_actuals"]
    assert summary["not_found_count"] == 0
    assert summary["resolved_vintage_map"][f"A|{reference[5].isoformat()}"] == "v7"


def test_vintage_first_release_missing_forever_warns_and_records_count() -> None:
    missing_date = pd.Timestamp("2000-06-30")
    reference, bundles = _publication_lag_bundles(missing_forever={missing_date})
    spec = mf.data.VintagePanelSpec(
        _SyntheticVintageSource(bundles),
        reference,
        actuals_vintage="first_release",
    )

    with pytest.warns(UserWarning, match="first-release actuals were not found"):
        result = mf.forecasting.run(
            spec,
            _recording_panel_model([]),
            window=_window(reference, last_origin=reference[4]),
            target="A",
            features=None,
            save_models=False,
        )

    table = result.to_frame()
    assert pd.isna(table.loc[0, "actual"])
    summary = result.metadata["vintage_source"]["first_release_actuals"]
    assert summary["not_found_count"] == 1
    assert summary["not_found"] == [{"target": "A", "date": missing_date.isoformat()}]


def test_vintage_change_target_transform_warns_about_revision_footgun() -> None:
    reference, bundles = _oracle_bundles()
    spec = mf.data.VintagePanelSpec(_SyntheticVintageSource(bundles), reference)
    features = mf.feature_engineering.feature_spec(
        target="A",
        predictors=[],
        lags=None,
        target_lags=(1,),
        target_transform="change",
    )

    with pytest.warns(UserWarning, match="pre-transform the target within each vintage"):
        mf.forecasting.run(
            spec,
            _constant_model(),
            window=_window(reference),
            target="A",
            features=features,
            save_models=False,
        )


def test_vintage_recursive_policy_uses_per_origin_vintage_rows() -> None:
    reference, bundles = _oracle_bundles()
    spec = mf.data.VintagePanelSpec(_SyntheticVintageSource(bundles), reference)

    result = mf.forecasting.run(
        spec,
        _recording_model([]),
        window=_window(reference),
        target="A",
        features=_features(),
        forecast_policy="recursive",
        save_models=False,
    )

    table = result.to_frame()
    assert list(table["forecast_policy"]) == ["recursive", "recursive", "recursive"]
    assert list(table["vintage_id"]) == ["v4", "v5", "v6"]
    np.testing.assert_allclose(table["prediction"].to_numpy(float), [4.0, 5.0, 6.0])
    np.testing.assert_allclose(table["actual"].to_numpy(float), [6.0, 7.0, 8.0])


def test_vintage_path_average_policy_runs_against_resolved_vintages() -> None:
    reference, bundles = _oracle_bundles()
    spec = mf.data.VintagePanelSpec(_SyntheticVintageSource(bundles), reference)

    result = mf.forecasting.run(
        spec,
        _constant_model(2.0),
        window=_window(reference, horizon=2, last_origin=reference[5]),
        target="A",
        features=_features(),
        horizon=2,
        forecast_policy="path_average",
        save_models=False,
    )

    table = result.to_frame()
    assert list(table["forecast_policy"]) == ["path_average", "path_average"]
    assert list(table["vintage_id"]) == ["v4", "v5"]
    np.testing.assert_allclose(table["prediction"].to_numpy(float), [2.0, 2.0])


def test_vintage_panel_model_uses_origin_bundle_fit_rows_and_latest_actuals() -> None:
    reference, bundles = _oracle_bundles()
    spec = mf.data.VintagePanelSpec(_SyntheticVintageSource(bundles), reference)
    records: list[dict[str, pd.DataFrame]] = []

    result = mf.forecasting.run(
        spec,
        _recording_panel_model(records),
        window=_window(reference),
        target="A",
        features=None,
        save_models=False,
    )

    table = result.to_frame()
    assert list(table["vintage_id"]) == ["v4", "v5", "v6"]
    np.testing.assert_allclose(table["prediction"].to_numpy(float), [4.0, 5.0, 6.0])
    np.testing.assert_allclose(table["actual"].to_numpy(float), [6.0, 7.0, 8.0])

    expected_fit_indices = [
        pd.DatetimeIndex(["2000-01-31", "2000-02-29", "2000-03-31", "2000-04-30"], name="date"),
        pd.DatetimeIndex(
            ["2000-01-31", "2000-02-29", "2000-03-31", "2000-04-30", "2000-05-31"],
            name="date",
        ),
        pd.DatetimeIndex(
            [
                "2000-01-31",
                "2000-02-29",
                "2000-03-31",
                "2000-04-30",
                "2000-05-31",
                "2000-06-30",
            ],
            name="date",
        ),
    ]
    for record, expected_index in zip(records, expected_fit_indices, strict=True):
        pdt.assert_index_equal(record["fit_panel"].index, expected_index)


def test_vintage_full_panel_stage_scopes_resolve_per_origin() -> None:
    reference, bundles = _oracle_bundles()
    spec = mf.data.VintagePanelSpec(_SyntheticVintageSource(bundles), reference)

    feature_scope = mf.window.stage_policy("full_panel")
    result = mf.forecasting.run(
        spec,
        _recording_model([]),
        window=_window(reference),
        target="A",
        features=_features(),
        feature_policy=feature_scope,
        save_models=False,
    )
    assert not result.to_frame().empty

    preprocessing = mf.preprocessing.preprocess_spec(
        transform="none",
        impute="none",
        standardize="zscore",
    )
    preprocessed = mf.forecasting.run(
        spec,
        _constant_model(0.0),
        window=_window(reference),
        target="A",
        features=_features(),
        preprocessing=preprocessing,
        preprocessing_policy=mf.window.stage_policy("full_panel"),
        save_models=False,
    )
    assert not preprocessed.to_frame().empty


def test_vintage_boundary_audit_warns_and_marks_result() -> None:
    reference, bundles = _oracle_bundles(leaky=True)
    spec = mf.data.VintagePanelSpec(_SyntheticVintageSource(bundles), reference)

    with pytest.warns(RuntimeWarning, match="vintage boundary audit"):
        result = mf.forecasting.run(
            spec,
            "ols",
            window=_window(reference),
            target="A",
            features=_features(),
            save_models=False,
        )

    audit = result.metadata["vintage_boundary_audit"]
    assert audit["vintage_boundary_ok"] is False
    assert audit["n_violations"] == 3
    assert audit["violations"][0]["max_panel_date"] >= audit["violations"][0]["origin"]


def test_vintage_static_extras_are_truncated_and_boundary_audit_stays_clean() -> None:
    reference, bundles = _oracle_bundles()
    extra = pd.DataFrame(
        {"calendar_dummy": np.arange(len(reference), dtype=float)},
        index=reference,
    )
    base_source = _SyntheticVintageSource(bundles)
    source = mf.data.with_static_extras(base_source, extra, join="outer")
    origin = reference[4]
    resolved = source.resolve(origin)

    assert resolved.panel.index.max() < origin
    assert pd.Timestamp(reference[5]) not in resolved.panel.index

    base_result = mf.forecasting.run(
        mf.data.VintagePanelSpec(_SyntheticVintageSource(bundles), reference),
        _constant_model(3.0, name="constant_base"),
        window=_window(reference),
        target="A",
        features=_features(),
        save_models=False,
    )
    extra_result = mf.forecasting.run(
        mf.data.VintagePanelSpec(source, reference),
        _constant_model(3.0, name="constant_extra"),
        window=_window(reference),
        target="A",
        features=_features(),
        save_models=False,
    )

    assert extra_result.metadata["vintage_boundary_audit"]["vintage_boundary_ok"] is True
    np.testing.assert_allclose(
        extra_result.to_frame()["prediction"].to_numpy(float),
        base_result.to_frame()["prediction"].to_numpy(float),
    )


def test_vintage_cache_keys_separate_vintage_ids_and_reuse_same_id() -> None:
    reference, bundles = _oracle_bundles()
    cache: dict[object, object] = {}

    spec_a = mf.data.VintagePanelSpec(_SyntheticVintageSource(bundles), reference)
    mf.forecasting.run(
        spec_a,
        "ols",
        window=_window(reference),
        target="A",
        features=_features(),
        save_models=False,
        preprocessing_cache=cache,
    )
    after_first = len(cache)
    mf.forecasting.run(
        spec_a,
        "ols",
        window=_window(reference),
        target="A",
        features=_features(),
        save_models=False,
        preprocessing_cache=cache,
    )
    assert len(cache) == after_first

    shifted = {
        key: mf.data.DataBundle(value.panel + 100.0, {**value.metadata, "vintage": f"{value.metadata['vintage']}_other"})
        for key, value in bundles.items()
    }
    spec_b = mf.data.VintagePanelSpec(_SyntheticVintageSource(shifted), reference)
    mf.forecasting.run(
        spec_b,
        "ols",
        window=_window(reference),
        target="A",
        features=_features(),
        save_models=False,
        preprocessing_cache=cache,
    )

    assert len(cache) > after_first
    vintage_tags = {key[-1] for key in cache if isinstance(key, tuple) and "vintage" in key}
    assert {"v4", "v5", "v6", "v4_other", "v5_other", "v6_other"}.issubset(vintage_tags)
