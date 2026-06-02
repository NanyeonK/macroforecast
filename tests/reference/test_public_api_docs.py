from __future__ import annotations

import re
import importlib
from pathlib import Path

import macroforecast as mf


REFERENCE_MODULES = (
    "meta",
    "data",
    "preprocessing",
    "feature_engineering",
    "filters",
    "data_analysis",
    "feature_analysis",
    "forecast_analysis",
    "models",
    "model_ensemble",
    "model_selection",
    "forecasting",
    "metrics",
    "tests",
    "evaluation",
    "window",
    "interpretation",
    "output",
    "reporting",
)


def _documented_top_level_symbols() -> set[str]:
    text = Path("docs/reference/public_api.md").read_text()
    match = re.search(r"## Top-Level Exports\n(.*?)\n## Submodules", text, re.S)
    assert match is not None
    symbols: set[str] = set()
    for line in match.group(1).splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in {"Symbol", "---"}:
            continue
        symbols.update(re.findall(r"`([^`]+)`", cells[0]))
    return symbols


def _documented_submodules() -> set[str]:
    text = Path("docs/reference/public_api.md").read_text()
    match = re.search(r"## Submodules\n(.*)", text, re.S)
    assert match is not None
    return set(re.findall(r"`macroforecast\.([^`]+)`", match.group(1)))


def test_public_api_docs_match_top_level_all() -> None:
    documented = _documented_top_level_symbols()
    actual = set(mf.__all__)

    assert documented == actual


def test_public_api_docs_list_all_top_level_namespaces() -> None:
    namespace_exports = {
        name
        for name in mf.__all__
        if hasattr(getattr(mf, name), "__name__")
        and getattr(mf, name).__name__ == f"macroforecast.{name}"
    }

    assert _documented_submodules() == namespace_exports


def test_top_level_all_symbols_are_importable() -> None:
    for name in mf.__all__:
        getattr(mf, name)


def test_forecast_analysis_helpers_are_namespace_only() -> None:
    assert "forecast_analysis" in mf.__all__
    assert mf.forecast_analysis.diagnose_forecasts is not None
    assert not hasattr(mf, "diagnose_forecasts")
    assert not hasattr(mf, "custom_forecast_diagnostic")
    assert not hasattr(mf, "forecast_overview")


def test_feature_analysis_helpers_are_namespace_only() -> None:
    assert "feature_analysis" in mf.__all__
    assert mf.feature_analysis.diagnose_features is not None
    assert not hasattr(mf, "diagnose_features")
    assert not hasattr(mf, "custom_feature_diagnostic")
    assert not hasattr(mf, "feature_overview")


def test_data_analysis_helpers_are_namespace_only() -> None:
    assert "data_analysis" in mf.__all__
    assert mf.data_analysis.summarize_data is not None
    assert mf.data_analysis.analyze_data is not None
    assert not hasattr(mf, "summarize_data")
    assert not hasattr(mf, "analyze_data")
    assert not hasattr(mf, "panel_overview")


def test_output_helpers_are_namespace_only() -> None:
    assert "output" in mf.__all__
    assert mf.output.write_artifacts is not None
    assert mf.output.bundle_outputs is not None
    assert mf.output.forecast_table is not None
    assert not hasattr(mf, "write_artifacts")
    assert not hasattr(mf, "bundle_outputs")
    assert not hasattr(mf, "forecast_table")
    assert not hasattr(mf, "OutputBundle")


def test_reporting_helpers_are_namespace_only() -> None:
    assert "reporting" in mf.__all__
    assert mf.reporting.report_table is not None
    assert mf.reporting.metric_report_table is not None
    assert mf.reporting.test_report_table is not None
    assert not hasattr(mf, "report_table")
    assert not hasattr(mf, "metric_report_table")
    assert not hasattr(mf, "test_report_table")
    assert not hasattr(mf, "ReportTable")


def test_reference_pages_mention_module_public_symbols() -> None:
    for module_name in REFERENCE_MODULES:
        module = importlib.import_module(f"macroforecast.{module_name}")
        documented = Path(f"docs/reference/{module_name}.md").read_text()
        missing = sorted(
            symbol for symbol in module.__all__ if symbol not in documented
        )

        assert missing == [], f"{module_name} reference page is missing: {missing}"


def test_models_reference_has_one_heading_per_registered_model() -> None:
    models = importlib.import_module("macroforecast.models")
    documented = Path("docs/reference/models.md").read_text()
    headings = set(re.findall(r"^### ([^\n]+)$", documented, re.M))
    missing = sorted(name for name in models.MODEL_SPECS if name not in headings)

    assert missing == []


def test_module_reference_page_titles_use_qualified_module_names() -> None:
    for module_name in REFERENCE_MODULES:
        first_line = (
            Path(f"docs/reference/{module_name}.md").read_text().splitlines()[0]
        )

        assert first_line == f"# macroforecast.{module_name}"
