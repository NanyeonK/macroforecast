from macrocast.core.layers.registry import get_layer
from macrocast.core.ops.registry import get_op


def layer_id_to_yaml_key(layer_id: str) -> str:
    return {
        "l1_5": "1_5_data_summary",
        "l2_5": "2_5_pre_post_preprocessing",
        "l3_5": "3_5_feature_diagnostics",
        "l4_5": "4_5_generator_diagnostics",
    }[layer_id]


def _module(layer_id: str):
    if layer_id == "l1_5":
        from macrocast.core.layers import l1_5 as module
    elif layer_id == "l2_5":
        from macrocast.core.layers import l2_5 as module
    elif layer_id == "l3_5":
        from macrocast.core.layers import l3_5 as module
    else:
        from macrocast.core.layers import l4_5 as module
    return module


def test_all_4_diagnostics_default_disabled():
    for layer_id in ["l1_5", "l2_5", "l3_5", "l4_5"]:
        recipe = _module(layer_id).parse_recipe_yaml("")
        if layer_id in recipe.layers:
            assert recipe.layers[layer_id].enabled is False


def test_all_4_diagnostics_disabled_have_no_axis_nodes_or_sinks():
    for layer_id in ["l1_5", "l2_5", "l3_5", "l4_5"]:
        module = _module(layer_id)
        yaml_text = f"{layer_id_to_yaml_key(layer_id)}:\n  enabled: false"
        layer = module.parse_layer_yaml(yaml_text, layer_id)
        dag = module.normalize_to_dag_form(layer, layer_id)
        resolved = module.resolve_axes(dag)
        assert dag.nodes == {}
        assert dag.sinks == {}
        assert all(not resolved.get_active(axis) for axis in module.AXIS_NAMES)


def test_all_4_diagnostics_have_z_export_sub_layer():
    for layer_id in ["l1_5", "l2_5", "l3_5", "l4_5"]:
        layer_class = get_layer(layer_id).cls
        assert any("Z_export" in name or "Z_diagnostic" in name for name in layer_class.sub_layers.keys())


def test_all_4_diagnostics_diagnostic_format_default_pdf():
    for layer_id in ["l1_5", "l2_5", "l3_5", "l4_5"]:
        module = _module(layer_id)
        yaml_text = f"{layer_id_to_yaml_key(layer_id)}:\n  enabled: true"
        layer = module.parse_layer_yaml(yaml_text, layer_id)
        resolved = module.resolve_axes(module.normalize_to_dag_form(layer, layer_id))
        assert resolved["diagnostic_format"] == "pdf"


def test_all_4_diagnostics_attach_to_manifest_default_true():
    for layer_id in ["l1_5", "l2_5", "l3_5", "l4_5"]:
        module = _module(layer_id)
        yaml_text = f"{layer_id_to_yaml_key(layer_id)}:\n  enabled: true"
        layer = module.parse_layer_yaml(yaml_text, layer_id)
        resolved = module.resolve_axes(module.normalize_to_dag_form(layer, layer_id))
        assert resolved["attach_to_manifest"] is True


def test_all_4_diagnostics_category_diagnostic():
    for layer_id in ["l1_5", "l2_5", "l3_5", "l4_5"]:
        assert get_layer(layer_id).category == "diagnostic"


def test_all_4_diagnostics_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS

    for layer_id in ["l1_5", "l2_5", "l3_5", "l4_5"]:
        assert layer_id in LAYER_SINKS


def test_l8_saved_objects_all_diagnostics_options_present():
    op_spec = get_op("l8_saved_objects")
    saved_objects = {
        "forecasts", "forecast_intervals", "metrics", "ranking", "decomposition", "regime_metrics", "state_metrics",
        "model_artifacts", "combination_weights", "feature_metadata", "clean_panel", "raw_panel", "diagnostics_l1_5",
        "diagnostics_l2_5", "diagnostics_l3_5", "diagnostics_l4_5", "diagnostics_all", "tests", "importance",
        "transformation_attribution",
    }
    assert op_spec.layer_scope == ("l8",)
    for layer_id in ["l1_5", "l2_5", "l3_5", "l4_5"]:
        assert f"diagnostics_{layer_id}" in saved_objects
    assert "diagnostics_all" in saved_objects
