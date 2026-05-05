from macroforecast.core.layers.l1_5 import parse_recipe_yaml as parse_l1_5_recipe
from macroforecast.core.layers.registry import get_layer


def layer_id_to_yaml_key(layer_id: str) -> str:
    return {"l1_5": "1_5_data_summary", "l2_5": "2_5_pre_post_preprocessing"}[layer_id]


def _parse_layer_yaml(yaml_text: str, layer_id: str):
    if layer_id == "l1_5":
        from macroforecast.core.layers.l1_5 import parse_layer_yaml
    else:
        from macroforecast.core.layers.l2_5 import parse_layer_yaml

    return parse_layer_yaml(yaml_text, layer_id)


def _normalize(layer: dict, layer_id: str):
    if layer_id == "l1_5":
        from macroforecast.core.layers.l1_5 import normalize_to_dag_form
    else:
        from macroforecast.core.layers.l2_5 import normalize_to_dag_form

    return normalize_to_dag_form(layer, layer_id)


def _resolve(dag, layer_id: str):
    if layer_id == "l1_5":
        from macroforecast.core.layers.l1_5 import resolve_axes
    else:
        from macroforecast.core.layers.l2_5 import resolve_axes

    return resolve_axes(dag)


def test_all_diagnostics_default_disabled():
    for layer_id in ["l1_5", "l2_5"]:
        recipe = parse_l1_5_recipe("")
        if layer_id in recipe.layers:
            assert recipe.layers[layer_id].enabled is False


def test_all_diagnostics_have_z_export_sub_layer():
    for layer_id in ["l1_5", "l2_5"]:
        layer_class = get_layer(layer_id).cls
        assert any("Z_export" in sub_name for sub_name in layer_class.sub_layers.keys())


def test_diagnostic_format_default_pdf_across_all():
    for layer_id in ["l1_5", "l2_5"]:
        yaml_text = f"{layer_id_to_yaml_key(layer_id)}:\n  enabled: true"
        layer = _parse_layer_yaml(yaml_text, layer_id)
        resolved = _resolve(_normalize(layer, layer_id), layer_id)
        assert resolved["diagnostic_format"] == "pdf"


def test_diagnostic_attach_to_manifest_default_true():
    for layer_id in ["l1_5", "l2_5"]:
        yaml_text = f"{layer_id_to_yaml_key(layer_id)}:\n  enabled: true"
        layer = _parse_layer_yaml(yaml_text, layer_id)
        resolved = _resolve(_normalize(layer, layer_id), layer_id)
        assert resolved["attach_to_manifest"] is True
