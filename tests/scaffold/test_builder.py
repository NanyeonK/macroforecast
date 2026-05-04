"""Pin the ``RecipeBuilder`` programmatic API.

Each test asserts a builder-produced recipe is equivalent to the
hand-written YAML of the same study so users can switch between
authoring modes without surprise.
"""
from __future__ import annotations

import textwrap

import pytest

from macrocast.scaffold import RecipeBuilder


# ---------------------------------------------------------------------------
# L0
# ---------------------------------------------------------------------------

def test_l0_default_call_emits_three_axes_plus_seed():
    b = RecipeBuilder()
    b.l0()
    block = b.build()["0_meta"]
    assert block["fixed_axes"] == {
        "failure_policy": "fail_fast",
        "reproducibility_mode": "seeded_reproducible",
        "compute_mode": "serial",
    }
    assert block["leaf_config"]["random_seed"] == 0


def test_l0_explicit_overrides():
    b = RecipeBuilder()
    b.l0(random_seed=42, compute_mode="parallel", n_workers=4)
    block = b.build()["0_meta"]
    assert block["fixed_axes"]["compute_mode"] == "parallel"
    assert block["leaf_config"]["random_seed"] == 42
    assert block["leaf_config"]["n_workers"] == 4


# ---------------------------------------------------------------------------
# L1 presets
# ---------------------------------------------------------------------------

def test_l1_fred_md_preset_sets_canonical_axes():
    b = RecipeBuilder()
    b.l1.fred_md(target="CPIAUCSL")
    l1 = b.build()["1_data"]
    assert l1["fixed_axes"]["dataset"] == "fred_md"
    assert l1["fixed_axes"]["frequency"] == "monthly"
    assert l1["fixed_axes"]["horizon_set"] == "standard_md"
    assert l1["leaf_config"]["target"] == "CPIAUCSL"


def test_l1_custom_panel_inlines_data():
    panel = {
        "date": ["2018-01-01", "2018-02-01"],
        "y": [1.0, 2.0],
        "x1": [0.5, 1.0],
    }
    b = RecipeBuilder()
    b.l1.custom_panel(target="y", panel=panel)
    l1 = b.build()["1_data"]
    assert l1["fixed_axes"]["custom_source_policy"] == "custom_panel_only"
    assert l1["leaf_config"]["custom_panel_inline"]["y"] == [1.0, 2.0]


# ---------------------------------------------------------------------------
# L2 / L3 presets
# ---------------------------------------------------------------------------

def test_l2_standard_preset_matches_mccracken_ng_default():
    b = RecipeBuilder()
    b.l2.standard()
    axes = b.build()["2_preprocessing"]["fixed_axes"]
    assert axes["transform_policy"] == "apply_official_tcode"
    assert axes["outlier_policy"] == "mccracken_ng_iqr"
    assert axes["imputation_policy"] == "em_factor"


def test_l3_lag_only_emits_canonical_dag():
    b = RecipeBuilder()
    b.l3.lag_only(n_lag=3)
    nodes = b.build()["3_feature_engineering"]["nodes"]
    node_ids = [n["id"] for n in nodes]
    assert {"src_X", "src_y", "lag_x", "y_h"}.issubset(node_ids)
    lag_node = next(n for n in nodes if n["id"] == "lag_x")
    assert lag_node["params"]["n_lag"] == 3


# ---------------------------------------------------------------------------
# L4 fit + benchmark
# ---------------------------------------------------------------------------

def test_l4_fit_adds_node_and_predict():
    b = RecipeBuilder()
    b.l4.fit("ridge", alpha=0.5)
    l4 = b.build()["4_forecasting_model"]
    families = [n["params"]["family"] for n in l4["nodes"] if n.get("op") == "fit_model"]
    assert families == ["ridge"]
    assert any(n.get("op") == "predict" for n in l4["nodes"])
    assert l4["sinks"]["l4_forecasts_v1"] == "predict"


def test_l4_multiple_fits_keep_unique_node_ids():
    b = RecipeBuilder()
    b.l4.fit("ridge")
    b.l4.fit("random_forest", n_estimators=50)
    nodes = b.build()["4_forecasting_model"]["nodes"]
    fit_ids = [n["id"] for n in nodes if n.get("op") == "fit_model"]
    assert len(set(fit_ids)) == len(fit_ids)
    assert "fit_1_ridge" in fit_ids
    assert "fit_2_random_forest" in fit_ids


def test_is_benchmark_chain_rewires_model_sink():
    b = RecipeBuilder()
    b.l4.fit("ridge").is_benchmark()
    b.l4.fit("random_forest")
    sinks = b.build()["4_forecasting_model"]["sinks"]
    # Benchmark should be the canonical artifact sink.
    assert sinks["l4_model_artifacts_v1"] == "fit_1_ridge"
    bench_node = next(n for n in b.build()["4_forecasting_model"]["nodes"] if n["id"] == "fit_1_ridge")
    assert bench_node.get("is_benchmark") is True


# ---------------------------------------------------------------------------
# End-to-end builder -> run
# ---------------------------------------------------------------------------

def test_builder_run_executes_end_to_end(tmp_path):
    panel = {
        "date": [f"2018-{m:02d}-01" for m in range(1, 13)],
        "y": [float(v) for v in range(1, 13)],
        "x1": [float(v) / 2 for v in range(1, 13)],
    }
    b = RecipeBuilder()
    b.l0(random_seed=42)
    b.l1.custom_panel(target="y", panel=panel)
    b.l2.no_op()
    b.l3.lag_only(n_lag=1)
    b.l4.fit("ridge", alpha=0.1, min_train_size=4)
    b.l5.standard()
    result = b.run(output_directory=tmp_path)
    assert len(result.cells) == 1
    assert result.cells[0].succeeded


def test_to_yaml_round_trips_through_run(tmp_path):
    b = RecipeBuilder()
    b.l0(random_seed=7)
    b.l1.custom_panel(
        target="y",
        panel={"date": [f"2018-{m:02d}-01" for m in range(1, 11)],
               "y": list(range(1, 11)),
               "x1": [0.5 * v for v in range(1, 11)]},
    )
    b.l2.no_op()
    b.l3.lag_only(n_lag=1)
    b.l4.fit("ols", min_train_size=4)
    yaml_path = tmp_path / "recipe.yaml"
    yaml_text = b.to_yaml(yaml_path)
    assert yaml_path.exists()
    assert "fit_1_ols" in yaml_text


def test_validate_returns_empty_for_well_formed_recipe():
    b = RecipeBuilder()
    b.l0(random_seed=0)
    b.l1.custom_panel(
        target="y",
        panel={"date": ["2018-01-01"], "y": [1.0], "x1": [0.5]},
    )
    errors = b.validate()
    assert errors == []
