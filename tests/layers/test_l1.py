from pathlib import Path

import pandas as pd

import macroforecast as mf


def test_data_block_authoring_uses_data_namespace():
    root = mf.data.data(dataset="fred_md", target="CPIAUCSL", sample_start_rule="max_balanced")
    assert root == {
        "data": {
            "fixed_axes": {"dataset": "fred_md", "sample_start_rule": "max_balanced"},
            "leaf_config": {"target": "CPIAUCSL"},
        }
    }


def test_data_block_infers_multi_target_structure():
    root = mf.data.data(dataset="fred_qd", targets=("GDPC1", "PCECC96"))
    assert root["data"]["fixed_axes"]["target_structure"] == "multi_target"
    assert root["data"]["leaf_config"]["targets"] == ["GDPC1", "PCECC96"]


def test_data_namespace_exposes_pandas_loader_and_metadata(tmp_path):
    fixture = Path("tests/fixtures/fred_md_sample.csv")
    frame = mf.data.load_fred_md(local_source=fixture, cache_root=tmp_path)

    assert isinstance(frame, pd.DataFrame)
    info = mf.data.metadata(frame)
    assert info["dataset"] == "fred_md"
    assert info["frequency"] == "monthly"
    assert "artifact" in info


def test_data_result_loader_keeps_raw_envelope(tmp_path):
    fixture = Path("tests/fixtures/fred_md_sample.csv")
    result = mf.data.load_fred_md_result(local_source=fixture, cache_root=tmp_path)

    assert result.dataset_metadata.dataset == "fred_md"
    assert isinstance(result.data, pd.DataFrame)
    assert mf.data.metadata(result)["dataset"] == "fred_md"


def test_old_schema_namespace_is_not_public():
    assert "schema" not in mf.data.__all__
    assert not hasattr(mf.data, "schema")
    assert not hasattr(mf.data, "validate_layer")


def test_internal_registry_uses_data_marker():
    from macroforecast.core.layers.registry import get_layer

    assert get_layer("l1").cls.__name__ == "Data"
