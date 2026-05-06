"""Tests for the v0.8.0 ``mf.forecast(...)`` one-shot helper.

We exercise the wiring of the default recipe (L0 seed, L1 dataset/target,
L2/L3/L4/L5 defaults) by inspecting the recipe ``Experiment`` would
produce for the same args; the actual ``execute_recipe`` path is covered
in test_experiment.py via the offline custom-panel substitution.
"""
from __future__ import annotations

import pytest

import macroforecast as mf
from macroforecast.api_high import _build_default_recipe


def test_forecast_is_publicly_exported():
    assert callable(mf.forecast)


def test_forecast_default_recipe_layer0_seed_and_axes():
    b = _build_default_recipe(
        dataset="fred_md",
        target="INDPRO",
        horizons=[1, 3, 6],
        frequency=None,
        start="1980-01",
        end="2019-12",
        model_family="ar_p",
        random_seed=42,
    )
    recipe = b.build()
    l0 = recipe["0_meta"]
    assert l0["fixed_axes"]["failure_policy"] == "fail_fast"
    assert l0["fixed_axes"]["reproducibility_mode"] == "seeded_reproducible"
    assert l0["fixed_axes"]["compute_mode"] == "serial"
    assert l0["leaf_config"]["random_seed"] == 42


def test_forecast_default_recipe_layer1_target_horizons_and_window():
    b = _build_default_recipe(
        dataset="fred_md",
        target="INDPRO",
        horizons=[1, 3, 6],
        frequency=None,
        start="1980-01",
        end="2019-12",
        model_family="ar_p",
        random_seed=0,
    )
    recipe = b.build()
    l1 = recipe["1_data"]
    assert l1["fixed_axes"]["dataset"] == "fred_md"
    assert l1["fixed_axes"]["frequency"] == "monthly"
    assert l1["fixed_axes"]["horizon_set"] == "custom_list"
    assert l1["leaf_config"]["target"] == "INDPRO"
    assert l1["leaf_config"]["target_horizons"] == [1, 3, 6]
    assert l1["leaf_config"]["sample_start_date"] == "1980-01"
    assert l1["leaf_config"]["sample_end_date"] == "2019-12"
    assert l1["fixed_axes"]["sample_start_rule"] == "fixed_date"
    assert l1["fixed_axes"]["sample_end_rule"] == "fixed_date"


def test_forecast_fred_qd_sets_quarterly_frequency():
    b = _build_default_recipe(
        dataset="fred_qd",
        target="GDPC1",
        horizons=[1],
        frequency=None,
        start=None,
        end=None,
        model_family="ar_p",
        random_seed=0,
    )
    recipe = b.build()
    assert recipe["1_data"]["fixed_axes"]["frequency"] == "quarterly"


def test_forecast_fred_sd_alone_requires_explicit_frequency():
    with pytest.raises(ValueError, match="frequency"):
        _build_default_recipe(
            dataset="fred_sd",
            target="UR_CA",
            horizons=[1],
            frequency=None,
            start=None,
            end=None,
            model_family="ar_p",
            random_seed=0,
        )


def test_forecast_fred_md_rejects_conflicting_frequency():
    with pytest.raises(ValueError, match="fixes frequency"):
        _build_default_recipe(
            dataset="fred_md",
            target="INDPRO",
            horizons=[1],
            frequency="quarterly",
            start=None,
            end=None,
            model_family="ar_p",
            random_seed=0,
        )


def test_forecast_default_layer4_has_single_fit_model_with_chosen_family():
    b = _build_default_recipe(
        dataset="fred_md",
        target="INDPRO",
        horizons=[1],
        frequency=None,
        start=None,
        end=None,
        model_family="ridge",
        random_seed=0,
    )
    recipe = b.build()
    l4 = recipe["4_forecasting_model"]
    fit_nodes = [n for n in l4["nodes"] if n.get("op") == "fit_model"]
    assert len(fit_nodes) == 1
    assert fit_nodes[0]["params"]["family"] == "ridge"


def test_forecast_default_layer5_primary_metric_mse():
    b = _build_default_recipe(
        dataset="fred_md",
        target="INDPRO",
        horizons=[1],
        frequency=None,
        start=None,
        end=None,
        model_family="ar_p",
        random_seed=0,
    )
    recipe = b.build()
    assert recipe["5_evaluation"]["fixed_axes"]["primary_metric"] == "mse"


def test_forecast_rejects_empty_horizons():
    with pytest.raises(ValueError, match="horizons"):
        _build_default_recipe(
            dataset="fred_md",
            target="INDPRO",
            horizons=[],
            frequency=None,
            start=None,
            end=None,
            model_family="ar_p",
            random_seed=0,
        )
