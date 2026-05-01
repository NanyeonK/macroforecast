from macrocast.core.layers.l0 import (
    build_minimal_recipe,
    build_recipe_with_l0_only,
    execute_recipe,
    normalize_to_dag_form,
    parse_layer_yaml,
    resolve_axes,
    validate_layer,
)


def test_l0_minimal_yaml_parses():
    """Empty fixed_axes resolves to all defaults."""
    yaml_text = "0_meta:\n  fixed_axes: {}"
    layer = parse_layer_yaml(yaml_text, "l0")
    dag = normalize_to_dag_form(layer, "l0")
    resolved = resolve_axes(dag)
    assert resolved["failure_policy"] == "fail_fast"
    assert resolved["reproducibility_mode"] == "seeded_reproducible"
    assert resolved["compute_mode"] == "serial"
    assert resolved["random_seed"] == 42


def test_l0_explicit_yaml_parses():
    yaml_text = """
    0_meta:
      fixed_axes:
        failure_policy: continue_on_failure
        reproducibility_mode: seeded_reproducible
        compute_mode: parallel
      leaf_config:
        random_seed: 100
        parallel_unit: oos_dates
        n_workers: 4
    """
    layer = parse_layer_yaml(yaml_text, "l0")
    dag = normalize_to_dag_form(layer, "l0")
    resolved = resolve_axes(dag)
    assert resolved["compute_mode"] == "parallel"
    assert resolved["parallel_unit"] == "oos_dates"
    assert resolved["n_workers"] == 4


def test_l0_parallel_without_unit_fails():
    """compute_mode=parallel requires parallel_unit in leaf_config."""
    yaml_text = """
    0_meta:
      fixed_axes:
        compute_mode: parallel
    """
    layer = parse_layer_yaml(yaml_text, "l0")
    report = validate_layer(layer)
    assert report.has_hard_errors
    assert any("parallel_unit" in i.message for i in report.hard_errors)


def test_l0_exploratory_with_seed_fails():
    """exploratory mode + leaf_config.random_seed conflict."""
    yaml_text = """
    0_meta:
      fixed_axes:
        reproducibility_mode: exploratory
      leaf_config:
        random_seed: 42
    """
    layer = parse_layer_yaml(yaml_text, "l0")
    report = validate_layer(layer)
    assert report.has_hard_errors


def test_l0_no_sweep_allowed():
    """L0 axes are not sweepable."""
    yaml_text = """
    0_meta:
      fixed_axes:
        failure_policy: {sweep: [fail_fast, continue_on_failure]}
    """
    layer = parse_layer_yaml(yaml_text, "l0")
    report = validate_layer(layer)
    assert report.has_hard_errors
    assert any("sweep" in i.message.lower() for i in report.hard_errors)


def test_l0_strict_mode_does_not_exist():
    """strict reproducibility mode was rejected; only seeded/exploratory."""
    yaml_text = """
    0_meta:
      fixed_axes:
        reproducibility_mode: strict
    """
    layer = parse_layer_yaml(yaml_text, "l0")
    report = validate_layer(layer)
    assert report.has_hard_errors


def test_l0_parallel_models_subtype_does_not_exist():
    """parallel_models was rejected; use compute_mode=parallel + leaf_config.parallel_unit."""
    yaml_text = """
    0_meta:
      fixed_axes:
        compute_mode: parallel_models
    """
    layer = parse_layer_yaml(yaml_text, "l0")
    report = validate_layer(layer)
    assert report.has_hard_errors


def test_l0_derived_study_scope_recorded():
    """Manifest auto-derives study_scope from recipe shape."""
    recipe = build_minimal_recipe(targets=["CPI"], methods=["ridge"])
    manifest = execute_recipe(recipe)
    l0_record = manifest.layer_execution_log["l0"]
    assert l0_record.derived["study_scope"] == "one_target_one_method"
    assert l0_record.derived["execution_route"] == "comparison_sweep"


def test_l0_gpu_deterministic_flag():
    """gpu_deterministic is always optional, default false."""
    yaml_text = "0_meta:\n  fixed_axes: {}"
    layer = parse_layer_yaml(yaml_text, "l0")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l0"))
    assert resolved["gpu_deterministic"] is False


def test_l0_manifest_records_all_resolved():
    """Manifest records all resolved L0 axes including defaults."""
    yaml_text = "0_meta:\n  fixed_axes: {}"
    recipe = build_recipe_with_l0_only(yaml_text)
    manifest = execute_recipe(recipe)
    l0 = manifest.layer_execution_log["l0"]
    for axis in ["failure_policy", "reproducibility_mode", "compute_mode"]:
        entry = l0.resolved_axes[axis]
        assert entry.value is not None
        assert entry.source == "package_default"


def test_l0_registered_with_spec_correct_class():
    from macrocast.core.layers.l0 import L0StudySetup
    from macrocast.core.layers.registry import get_layer

    spec = get_layer("l0")
    assert spec.cls is L0StudySetup
    assert spec.produces == ("l0_meta_v1",)
    assert spec.ui_mode == "list"
    assert spec.category == "setup"


def test_l0_sink_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS

    assert "l0" in LAYER_SINKS
    assert "l0_meta_v1" in LAYER_SINKS["l0"]
