from pathlib import Path

import yaml

from macrocast import macrocast_single_run


def test_macrocast_single_run_executable_route_preview() -> None:
    out = macrocast_single_run(yaml_path="examples/recipes/model-benchmark.yaml")
    assert out["route_preview"]["route_owner"] == "single_run"
    assert out["route_preview"]["wizard_status"] == "implemented"
    assert out["compile_preview"]["execution_status"] == "executable"
    assert out["tree_context"]["fixed_design"]["dataset_adapter"] == "fred_md"
    assert out["runs_preview"]["artifact_subdir"] == out["compile_preview"]["run_spec"]["artifact_subdir"]
    assert out["manifest_preview"]["tree_context"]["route_owner"] == "single_run"
    assert "comparison_summary.json" in out["manifest_preview"]["expected_artifacts"]


def test_macrocast_single_run_sweep_runner_route_blocks_run_manifest() -> None:
    out = macrocast_single_run(yaml_path="examples/recipes/feature-builder-comparison.yaml")
    assert out["route_preview"]["route_owner"] == "single_run"
    assert out["route_preview"]["wizard_status"] == "sweep_runner_ready"
    assert out["compile_preview"]["execution_status"] == "ready_for_sweep_runner"
    assert out["tree_context"]["sweep_axes"]["feature_builder"] == ["autoreg_lagged_target", "raw_feature_panel"]
    assert out["blocked_preview_stages"] == ["runs_preview", "manifest_preview"]
    assert "execute_sweep" in out["blocked_preview_reason"]


def test_macrocast_single_run_wrapper_route_blocks_run_manifest(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "wrapper_preview",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "orchestrated_bundle"}, "leaf_config": {"wrapper_family": "benchmark_suite", "bundle_label": "fred-md-baselines"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "information_set_type": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    path = tmp_path / "wrapper_preview.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")

    out = macrocast_single_run(yaml_path=str(path))
    assert out["route_preview"]["route_owner"] == "wrapper"
    assert out["route_preview"]["wizard_status"] == "wrapper_not_supported"
    assert out["compile_preview"]["execution_status"] == "not_supported"
    assert out["compile_preview"]["wrapper_handoff"]["wrapper_family"] == "benchmark_suite"
    assert out["tree_context"]["route_owner"] == "wrapper"
    assert out["blocked_preview_stages"] == ["runs_preview", "manifest_preview"]
    assert "no executable wrapper runner contract" in out["blocked_preview_reason"]



def test_macrocast_single_run_interactive_first_step(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / "wizard.yaml"), ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=1)
    assert out["interactive"] is True
    assert out["yaml_path"].endswith("wizard.yaml")
    assert out["completed_choices"][0]["key"] == "research_design"
    assert out["completed_choices"][0]["value"] == "single_path_benchmark"
    assert out["current_choice"]["key"] == "task"
    assert out["route_preview"]["route_owner"] == "single_run"


def test_macrocast_single_run_interactive_drops_unsupported_wrapper_choice(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / "wrapper.yaml")])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=0)
    assert out["current_choice"]["key"] == "research_design"
    assert "orchestrated_bundle" not in out["current_choice"]["options"]


def test_macrocast_single_run_interactive_task_switches_next_choice(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / "multi.yaml"), "", "2"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=2)
    assert [item["key"] for item in out["completed_choices"]] == ["research_design", "task"]
    assert out["completed_choices"][1]["value"] == "multi_target_point_forecast"
    assert out["current_choice"]["key"] == "experiment_unit"



def test_macrocast_single_run_interactive_framework_follows_target(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / "framework.yaml"), "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=4)
    assert [item["key"] for item in out["completed_choices"]] == ["research_design", "task", "experiment_unit", "target"]
    assert out["current_choice"]["key"] == "framework"


def test_macrocast_single_run_interactive_custom_benchmark_requests_plugin_fields(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / "custom.yaml"), "", "", "", "", "", "4"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=6)
    assert [item["key"] for item in out["completed_choices"]] == ["research_design", "task", "experiment_unit", "target", "framework", "benchmark_family"]
    assert out["completed_choices"][-1]["value"] == "custom_benchmark"
    assert out["current_choice"]["key"] == "benchmark_plugin_path"


def test_macrocast_single_run_interactive_tcode_switch_normalizes_preprocess(monkeypatch, tmp_path: Path) -> None:
    answers = iter([
        str(tmp_path / "preprocess.yaml"),
        "",  # research_design
        "",  # task
        "",  # experiment_unit
        "",  # target
        "",  # framework
        "",  # benchmark_family
        "2", # tcode_policy -> extra_preprocess_without_tcode
    ])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=7)
    payload = yaml.safe_load(Path(out["yaml_path"]).read_text())
    preprocess = payload["path"]["2_preprocessing"]["fixed_axes"]
    assert out["completed_choices"][-1] == {"key": "tcode_policy", "value": "extra_preprocess_without_tcode"}
    assert preprocess["tcode_policy"] == "extra_preprocess_without_tcode"
    assert preprocess["preprocess_order"] == "extra_only"
    assert preprocess["preprocess_fit_scope"] == "train_only"



def test_macrocast_single_run_interactive_metric_follows_feature_builder(monkeypatch, tmp_path: Path) -> None:
    answers = iter([
        str(tmp_path / "metric.yaml"),
        "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
    ])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=15)
    assert out["completed_choices"][-1]["key"] == "primary_metric"
    assert out["current_choice"]["key"] == "manifest_mode"


def test_macrocast_single_run_interactive_stat_test_follows_manifest_mode(monkeypatch, tmp_path: Path) -> None:
    answers = iter([
        str(tmp_path / "stat.yaml"),
        "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
        "2",
    ])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=17)
    assert out["completed_choices"][-1] == {"key": "stat_test", "value": "dm"}
    assert out["current_choice"]["key"] == "importance_method"


def test_macrocast_single_run_interactive_importance_persists_yaml(monkeypatch, tmp_path: Path) -> None:
    answers = iter([
        str(tmp_path / "importance.yaml"),
        "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "2",
    ])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=18)
    payload = yaml.safe_load(Path(out["yaml_path"]).read_text())
    assert out["completed_choices"][-1] == {"key": "importance_method", "value": "minimal_importance"}
    assert payload["path"]["7_importance"]["fixed_axes"]["importance_method"] == "minimal_importance"



def test_macrocast_single_run_current_choice_exposes_planned_option_details(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / "planned.yaml"), "", "", "", "", "", "", "", "", "", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=13)
    choice = out["current_choice"]
    assert choice["key"] == "feature_builder"
    assert choice["option_details"]["factor_pca"]["status"] == "operational"


def test_macrocast_single_run_planned_feature_builder_stops_with_explicit_message(monkeypatch, tmp_path: Path) -> None:
    answers = iter([
        str(tmp_path / "factor.yaml"),
        "", "", "", "", "", "", "", "", "", "", "", "", "", "3",
    ])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=14)
    assert out["completed_choices"][-1] == {"key": "feature_builder", "value": "factor_pca"}
    assert out["route_preview"]["wizard_status"] in {"implemented", "blocked_or_nonexecutable"}
    assert out["route_preview"]["route_owner"] == "single_run"


def test_macrocast_single_run_planned_importance_option_is_labeled(monkeypatch, tmp_path: Path) -> None:
    answers = iter([
        str(tmp_path / "importance-planned.yaml"),
        "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
    ])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=17)
    choice = out["current_choice"]
    assert choice["key"] == "importance_method"
    assert choice["option_details"]["tree_shap"]["status"] == "operational"



def test_macrocast_single_run_model_path_mode_follows_preprocess_fit_scope(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / "grid.yaml"), "", "", "", "", "", "", "", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=11)
    assert out["current_choice"]["key"] == "model_path_mode"


def test_macrocast_single_run_model_grid_route_message(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / "grid.yaml"), "", "", "", "", "", "", "", "", "", "", "", "2"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=13)
    payload = yaml.safe_load(Path(out["yaml_path"]).read_text())
    assert out["completed_choices"][-1] == {"key": "model_path_mode", "value": "model_grid"}
    assert payload["path"]["3_training"]["sweep_axes"]["model_family"] == ["ar", "ridge", "lasso", "randomforest"]
    assert out["route_preview"]["wizard_status"] == "sweep_runner_ready"
    assert "Model grid" in out["route_preview"]["message"]


def test_macrocast_single_run_drops_full_sweep_choice(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / "full.yaml"), "", "", "", "", "", "", "", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=11)
    assert out["current_choice"]["key"] == "model_path_mode"
    assert "full_sweep" not in out["current_choice"]["options"]



def test_macrocast_single_run_interactive_experiment_unit_follows_task(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / "experiment-unit.yaml"), "", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = macrocast_single_run(max_steps=2)
    assert [item["key"] for item in out["completed_choices"]] == ["research_design", "task"]
    assert out["current_choice"]["key"] == "experiment_unit"
