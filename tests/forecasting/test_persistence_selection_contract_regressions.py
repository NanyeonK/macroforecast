"""Regression coverage for stored-model identity and panel selection semantics."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting.policies.base import _model_store_stem, _store_path_part


class _MarkerFit:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=X.index, name="prediction")


class _SameText:
    def __str__(self) -> str:
        return "unsafe value"


class _StaticVintageSource:
    kind = "static_test_vintage"

    def __init__(self, bundle: mf.data.DataBundle) -> None:
        self.bundle = bundle

    def available_vintages(self):
        return [self.bundle.panel.index[-1]]

    def resolve(self, origin_date):
        del origin_date
        return self.bundle


def _fit_marker_a(X: pd.DataFrame, y: pd.Series) -> _MarkerFit:
    del X, y
    return _MarkerFit("a")


def _fit_marker_b(X: pd.DataFrame, y: pd.Series) -> _MarkerFit:
    del X, y
    return _MarkerFit("b")


def _panel() -> pd.DataFrame:
    index = pd.date_range("2000-01-01", periods=72, freq="MS", name="date")
    return pd.DataFrame(
        {
            "y": 1.0 + np.sin(np.arange(len(index)) / 4.0),
            "x": np.cos(np.arange(len(index)) / 5.0),
        },
        index=index,
    )


def _window(index: pd.DatetimeIndex):
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(
            first_origin=index[48],
            last_origin=index[48],
            horizon=1,
        ),
    )


def _stored_model(
    panel: pd.DataFrame,
    *,
    name: str,
    fit_func,
    model_store: Path,
) -> dict:
    model = mf.models.ModelSpec(name=name, family="test", fit_func=fit_func)
    features = mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x"],
        lags=(0,),
    )
    table = mf.forecasting.run(
        panel,
        model,
        window=_window(panel.index),
        features=features,
        model_selection={name: None},
        save_models=True,
        model_store=model_store,
    ).to_frame()
    return table.iloc[0]["stored_model"]


def test_colliding_aliases_coexist_and_roundtrip(tmp_path: Path) -> None:
    panel = _panel()
    first = _stored_model(
        panel,
        name="model:a",
        fit_func=_fit_marker_a,
        model_store=tmp_path,
    )
    second = _stored_model(
        panel,
        name="model a",
        fit_func=_fit_marker_b,
        model_store=tmp_path,
    )

    assert first["model_path"] != second["model_path"]
    assert first["metadata_path"] != second["metadata_path"]
    assert {path.name for path in tmp_path.iterdir()} == {
        _store_path_part("model:a", kind="alias"),
        _store_path_part("model a", kind="alias"),
    }
    for stored, expected_alias, expected_marker in (
        (first, "model:a", "a"),
        (second, "model a", "b"),
    ):
        metadata = json.loads(Path(stored["metadata_path"]).read_text(encoding="utf-8"))
        assert metadata["alias"] == expected_alias
        loaded = mf.models.load_fit(stored["model_path"])
        assert loaded.marker == expected_marker


def test_safe_store_identity_keeps_historical_components() -> None:
    assert _store_path_part("plain_model", kind="alias") == "plain_model"
    assert _model_store_stem(
        {
            "origin_pos": 0,
            "horizon": 1,
            "origin": pd.Timestamp("2000-01-01"),
            "target_key": "direct_h1",
        }
    ) == "origin_0_h1_20000101_direct_h1"


def test_store_component_digest_contract_is_pinned() -> None:
    assert _store_path_part("model:a", kind="alias") == "model_a__h7480707e89b5"


def test_store_digest_payload_separates_kind_type_and_supports_surrogates() -> None:
    value = "unsafe value"
    assert _store_path_part(value, kind="alias") != _store_path_part(
        value, kind="origin"
    )
    assert _store_path_part(value, kind="alias") != _store_path_part(
        _SameText(), kind="alias"
    )
    assert "__h" in _store_path_part("\ud800", kind="alias")


def test_lossy_store_components_cannot_collide_or_forge() -> None:
    assert _store_path_part("Model", kind="alias") != _store_path_part(
        "model", kind="alias"
    )
    assert _store_path_part("x__h0123456789ab", kind="alias") != "x__h0123456789ab"
    composed = "é"
    decomposed = unicodedata.normalize("NFD", composed)
    assert _store_path_part(composed, kind="alias") != _store_path_part(
        decomposed, kind="alias"
    )
    long_component = _store_path_part("x" * 500, kind="alias")
    assert len(long_component) <= 64
    assert _store_path_part("con", kind="alias") != "con"


def test_lossy_origin_labels_have_distinct_stems() -> None:
    common = {"origin_pos": 0, "horizon": 1, "target_key": "direct_h1"}
    first = _model_store_stem({**common, "origin": "2020 Q1"})
    second = _model_store_stem({**common, "origin": "2020_Q1"})
    assert first != second


def test_legacy_sanitized_alias_directory_remains_purgeable(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "model_a"
    legacy_dir.mkdir()
    model_path = legacy_dir / "origin_0_h1_20000101_direct_h1.pkl"
    metadata_path = model_path.with_suffix(".json")
    model_path.write_bytes(b"legacy")
    metadata_path.write_text(
        json.dumps({"alias": "model:a", "model_path": str(model_path)}),
        encoding="utf-8",
    )

    assert mf.pipeline.purge_model_store(tmp_path, aliases=["model:a"]) == 0
    assert model_path.exists()
    assert metadata_path.exists()
    assert mf.pipeline.purge_model_store(tmp_path, aliases=["model_a"]) == 1
    assert not model_path.exists()
    assert not metadata_path.exists()


def test_panel_default_selection_raises_before_model_store_write(tmp_path: Path) -> None:
    panel = _panel()

    with pytest.raises(ValueError, match="requests the default"):
        mf.forecasting.run(
            panel,
            "var",
            window=_window(panel.index),
            target="y",
            features=None,
            model_selection=None,
            save_models=True,
            model_store=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_vintage_panel_default_selection_raises_before_model_store_write(
    tmp_path: Path,
) -> None:
    panel = _panel()
    bundle = mf.data.DataBundle(panel, {"dataset": "static_test_vintage"})
    vintages = mf.data.VintagePanelSpec(_StaticVintageSource(bundle), panel.index)

    with pytest.raises(ValueError, match="requests the default"):
        mf.forecasting.run(
            vintages,
            "var",
            window=_window(panel.index),
            target="y",
            features=None,
            model_selection=None,
            save_models=True,
            model_store=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_panel_explicit_search_spec_is_rejected_actionably() -> None:
    panel = _panel()
    search = mf.model_selection.SearchSpec(
        method="grid",
        param_grid={"n_lag": (1, 2)},
    )

    with pytest.raises(ValueError, match="cannot apply a SearchSpec"):
        mf.forecasting.run(
            panel,
            "var",
            window=_window(panel.index),
            target="y",
            features=None,
            model_selection=search,
            save_models=False,
        )


def test_panel_explicit_selection_optout_runs() -> None:
    panel = _panel()
    table = mf.forecasting.run(
        panel,
        "var",
        window=_window(panel.index),
        target="y",
        features=None,
        params={"var": {"n_lag": 1}},
        model_selection={"var": None},
        save_models=False,
    ).to_frame()
    assert not table.empty


def test_panel_fully_pinned_default_space_runs_without_optout() -> None:
    # The built-in VAR search space is exactly {n_lag}; pinning it exhausts the search.
    panel = _panel()
    table = mf.forecasting.run(
        panel,
        "var",
        window=_window(panel.index),
        target="y",
        features=None,
        params={"var": {"n_lag": 1}},
        model_selection=None,
        save_models=False,
    ).to_frame()
    assert not table.empty


def test_panel_model_without_default_search_space_runs() -> None:
    panel = _panel()
    model = mf.models.custom_model(
        "plain_panel_model",
        lambda data, *, target="y": _MarkerFit("plain"),
        default_params={"target": None},
        input_kind="panel",
    )
    table = mf.forecasting.run(
        panel,
        model,
        window=_window(panel.index),
        target="y",
        features=None,
        model_selection=None,
        save_models=False,
    ).to_frame()
    assert not table.empty
