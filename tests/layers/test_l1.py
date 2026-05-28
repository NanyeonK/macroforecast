from pathlib import Path

import pandas as pd

import macroforecast as mf


def test_data_namespace_exposes_bundle_loader_and_metadata(tmp_path):
    fixture = Path("tests/fixtures/fred_md_sample.csv")
    bundle = mf.data.load_fred_md(local_source=fixture, cache_root=tmp_path)
    panel, metadata = bundle

    assert isinstance(bundle, mf.data.DataBundle)
    assert isinstance(panel, pd.DataFrame)
    assert metadata["dataset"] == "fred_md"
    info = mf.data.metadata(bundle)
    assert info["dataset"] == "fred_md"
    assert info["frequency"] == "monthly"
    assert "artifact" in info


def test_data_spec_combines_panel_metadata_target_horizons_and_sample(tmp_path):
    fixture = Path("tests/fixtures/fred_md_sample.csv")
    bundle = mf.data.load_fred_md(local_source=fixture, cache_root=tmp_path)

    data_spec = mf.data.spec(
        bundle,
        target="INDPRO",
        horizons=[1, 3],
        start="2000-01",
        end="2000-03",
        predictors=["RPI", "UNRATE"],
    )

    assert isinstance(data_spec, mf.data.DataSpec)
    assert data_spec.target == "INDPRO"
    assert data_spec.targets == ("INDPRO",)
    assert data_spec.horizons == (1, 3)
    assert list(data_spec.panel.columns) == ["RPI", "UNRATE", "INDPRO"]
    assert data_spec.panel.index[0].strftime("%Y-%m-%d") == "2000-01-01"
    assert data_spec.metadata["data_spec"]["target"] == "INDPRO"


def test_as_panel_accepts_date_column_selection_and_rename():
    raw = pd.DataFrame(
        {
            "DATE": ["2020-02-01", "2020-01-01"],
            "x": ["2.5", "1.5"],
            "y_old": [4, 3],
            "unused": ["a", "b"],
        }
    )

    panel = mf.data.as_panel(raw, date="DATE", columns=["x", "y_old"], rename={"y_old": "y"})

    assert list(panel.columns) == ["x", "y"]
    assert panel.index.name == "date"
    assert panel.index[0].strftime("%Y-%m-%d") == "2020-01-01"
    assert panel["x"].dtype.kind in {"f", "i"}


def test_data_namespace_does_not_export_raw_result_api():
    assert "load_fred_md_result" not in mf.data.__all__
    assert "RawLoadResult" not in mf.data.__all__
    assert not hasattr(mf.data, "load_fred_md_result")
    assert not hasattr(mf.data, "RawLoadResult")


def test_old_schema_namespace_is_not_public():
    assert "schema" not in mf.data.__all__
    assert "L1_LAYER_SPEC" not in mf.data.__all__
    assert "Data" not in mf.data.__all__
    assert not hasattr(mf.data, "schema")
    assert not hasattr(mf.data, "L1_LAYER_SPEC")
    assert not hasattr(mf.data, "validate_layer")


def test_internal_registry_uses_data_spec_class():
    from macroforecast.core.layers.registry import get_layer

    assert get_layer("l1").cls.__name__ == "DataSpec"
