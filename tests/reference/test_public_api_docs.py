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


def test_reference_pages_mention_module_public_symbols() -> None:
    for module_name in REFERENCE_MODULES:
        module = importlib.import_module(f"macroforecast.{module_name}")
        documented = Path(f"docs/reference/{module_name}.md").read_text()
        missing = sorted(symbol for symbol in module.__all__ if symbol not in documented)

        assert missing == [], f"{module_name} reference page is missing: {missing}"


def test_models_reference_has_one_heading_per_registered_model() -> None:
    models = importlib.import_module("macroforecast.models")
    documented = Path("docs/reference/models.md").read_text()
    headings = set(re.findall(r"^### ([^\n]+)$", documented, re.M))
    missing = sorted(name for name in models.MODEL_SPECS if name not in headings)

    assert missing == []


def test_module_reference_page_titles_use_qualified_module_names() -> None:
    for module_name in REFERENCE_MODULES:
        first_line = Path(f"docs/reference/{module_name}.md").read_text().splitlines()[0]

        assert first_line == f"# macroforecast.{module_name}"
