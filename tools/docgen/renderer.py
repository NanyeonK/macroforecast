"""Generate the committed ``docs/reference`` tree from the live package API."""

from __future__ import annotations

import dataclasses
import difflib
import importlib
import inspect
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import macroforecast as mf


@dataclass(frozen=True)
class ModulePage:
    name: str
    title: str
    module: str
    purpose: str
    guide_link: str | None = None


MODULE_PAGES: tuple[ModulePage, ...] = (
    ModulePage(
        "meta",
        "Package Configuration",
        "macroforecast.meta",
        "Package-wide defaults such as random seed, worker count, metadata level, and context-managed overrides.",
    ),
    ModulePage(
        "data",
        "Data",
        "macroforecast.data",
        "Canonical date-indexed panels, metadata, FRED loaders, custom loaders, and real-time vintage sources.",
        "../guide/concepts/data.md",
    ),
    ModulePage(
        "preprocessing",
        "Preprocessing",
        "macroforecast.preprocessing",
        "Transform-code application, outlier handling, imputation, frame-edge handling, and reusable preprocessing specs.",
        "../guide/concepts/preprocessing.md",
    ),
    ModulePage(
        "data_analysis",
        "Data Analysis",
        "macroforecast.data_analysis",
        "One-panel summaries and before/after preprocessing diagnostics.",
    ),
    ModulePage(
        "filters",
        "Filters",
        "macroforecast.filters",
        "One-series filters, smoothers, decompositions, and adaptive moving averages.",
    ),
    ModulePage(
        "feature_engineering",
        "Feature Engineering",
        "macroforecast.feature_engineering",
        "Target construction, lag/MARX/factor transforms, feature-step composition, and feature selection.",
        "../guide/concepts/features.md",
    ),
    ModulePage(
        "feature_analysis",
        "Feature Analysis",
        "macroforecast.feature_analysis",
        "Feature-stage diagnostics for factors, lags, MARX transforms, selections, and distribution shift.",
    ),
    ModulePage(
        "window",
        "Window",
        "macroforecast.window",
        "Estimation, validation, test-window, split, and stage-policy definitions.",
        "../guide/concepts/windows.md",
    ),
    ModulePage(
        "models",
        "Models",
        "macroforecast.models",
        "Direct callable model fits plus the model registry used by selection and pipeline runners.",
        "../guide/model_overview.md",
    ),
    ModulePage(
        "model_ensemble",
        "Model Ensemble",
        "macroforecast.model_ensemble",
        "Fit-time ensemble estimators and ensemble model specs.",
        "../guide/concepts/models_and_arms.md",
    ),
    ModulePage(
        "model_selection",
        "Model Selection",
        "macroforecast.model_selection",
        "Hyperparameter distributions, search specs, search runners, and selection results.",
        "../guide/concepts/models_and_arms.md",
    ),
    ModulePage(
        "forecasting",
        "Forecasting",
        "macroforecast.forecasting",
        "Single-model forecasting runner, forecast result objects, checkpoint helpers, and forecast-combination specs.",
        "../guide/concepts/running.md",
    ),
    ModulePage(
        "forecast_analysis",
        "Forecast Analysis",
        "macroforecast.forecast_analysis",
        "Forecast diagnostics for fitted values, residuals, tuning traces, and forecast paths.",
    ),
    ModulePage(
        "metrics",
        "Metrics",
        "macroforecast.metrics",
        "Point, density, directional, financial, and benchmark-relative scoring callables.",
        "../guide/concepts/evaluation.md",
    ),
    ModulePage(
        "tests",
        "Tests",
        "macroforecast.tests",
        "Forecast-comparison tests, residual tests, density diagnostics, and model-confidence-set procedures.",
        "../guide/concepts/evaluation.md",
    ),
    ModulePage(
        "evaluation",
        "Evaluation",
        "macroforecast.evaluation",
        "Report-level aggregation, benchmark comparison, regime scores, and namespace access to metrics/tests.",
        "../guide/concepts/evaluation.md",
    ),
    ModulePage(
        "interpretation",
        "Interpretation",
        "macroforecast.interpretation",
        "Model, forecast, and observation attribution helpers.",
    ),
    ModulePage(
        "interpretation_dual",
        "Dual Interpretation",
        "macroforecast.interpretation.dual",
        "Observation-based dual interpretation for forecast results.",
    ),
    ModulePage(
        "output",
        "Output",
        "macroforecast.output",
        "Artifact manifests, output bundles, provenance collection, and table/record builders.",
    ),
    ModulePage(
        "reporting",
        "Reporting",
        "macroforecast.reporting",
        "Markdown, HTML, and LaTeX report-table rendering.",
    ),
    ModulePage(
        "pipeline",
        "Pipeline",
        "macroforecast.pipeline",
        "Comprehensive pseudo-out-of-sample pipeline specs, execution, evaluation, interpretation, and result stores.",
        "../guide/index.md",
    ),
)

MODULE_PAGE_BY_NAME = {page.name: page for page in MODULE_PAGES}

CUSTOM_PAGE_ENTRIES: dict[str, tuple[str, ...]] = {
    "custom_dataset": (
        "macroforecast.data.custom_dataset",
        "macroforecast.data.load_custom_csv",
        "macroforecast.data.load_custom_parquet",
        "macroforecast.data.custom_vintages",
        "macroforecast.data.with_static_extras",
    ),
    "custom_preprocess": (
        "macroforecast.preprocessing.preprocess_spec",
        "macroforecast.preprocessing.custom_preprocess",
        "macroforecast.preprocessing.custom_preprocess_step",
    ),
    "custom_features": (
        "macroforecast.feature_engineering.feature_spec",
        "macroforecast.feature_engineering.custom_features",
        "macroforecast.feature_engineering.custom_step",
    ),
    "custom_model": (
        "macroforecast.models.custom_model",
        "macroforecast.model_ensemble.custom_model_ensemble",
    ),
    "custom_window_selection_forecasting": (
        "macroforecast.window.custom_stage_policy",
        "macroforecast.model_selection.custom_search",
        "macroforecast.forecasting.custom_combination",
    ),
    "custom_evaluation_tests": (
        "macroforecast.tests.custom_test",
        "macroforecast.pipeline.EvalSpec",
    ),
    "custom_interpretation_analysis": (
        "macroforecast.interpretation.custom_interpretation",
        "macroforecast.feature_analysis.custom_feature_diagnostic",
        "macroforecast.forecast_analysis.custom_forecast_diagnostic",
    ),
    "custom_output": (
        "macroforecast.output.select_outputs",
        "macroforecast.output.write_artifacts",
        "macroforecast.reporting.report_table",
        "macroforecast.reporting.render_tables",
    ),
}

CUSTOM_PAGE_TITLES: dict[str, str] = {
    "custom_dataset": "Custom Dataset",
    "custom_preprocess": "Custom Preprocess",
    "custom_features": "Custom Features",
    "custom_model": "Custom Model",
    "custom_window_selection_forecasting": "Custom Window, Selection, And Forecasting",
    "custom_evaluation_tests": "Custom Evaluation Tests",
    "custom_interpretation_analysis": "Custom Interpretation And Analysis",
    "custom_output": "Custom Output",
}


def collect_pages() -> dict[Path, str]:
    """Return every generated reference page as ``relative_path -> text``."""

    pages: dict[Path, str] = {
        Path("index.md"): _render_reference_index(),
        Path("documentation_map.md"): _render_documentation_map(),
        Path("workflow.md"): _render_workflow(),
        Path("legacy_callable_coverage.md"): _render_legacy_callable_coverage(),
        Path("reference_verification.md"): _render_reference_verification(),
        Path("public_api.md"): _render_public_api(),
        Path("custom/index.md"): _render_custom_index(),
        Path("custom/custom.md"): _render_custom_overview(),
    }
    for page in MODULE_PAGES:
        pages[Path(f"{page.name}.md")] = _render_module_page(page)
    for key in CUSTOM_PAGE_ENTRIES:
        pages[Path(f"custom/{key}.md")] = _render_custom_page(key)
    return {path: _normalise_text(text) for path, text in sorted(pages.items())}


def write_all(out_dir: str | Path = "docs/reference") -> list[Path]:
    """Write the generated reference tree and remove stale generated Markdown."""

    out = Path(out_dir)
    pages = collect_pages()
    out.mkdir(parents=True, exist_ok=True)

    expected = set(pages)
    for existing in sorted(out.rglob("*.md")):
        rel = existing.relative_to(out)
        if rel not in expected:
            existing.unlink()

    written: list[Path] = []
    for rel, text in pages.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)

    _remove_empty_dirs(out)
    return written


def check_all(out_dir: str | Path = "docs/reference") -> tuple[bool, list[str]]:
    """Return ``(ok, messages)`` for an exact generated-tree drift check."""

    out = Path(out_dir)
    pages = collect_pages()
    expected = set(pages)
    existing = {
        path.relative_to(out)
        for path in out.rglob("*.md")
        if path.is_file()
    } if out.exists() else set()

    messages: list[str] = []
    for rel in sorted(expected - existing):
        messages.append(f"missing: {rel.as_posix()}")
    for rel in sorted(existing - expected):
        messages.append(f"stale: {rel.as_posix()}")
    for rel in sorted(expected & existing):
        current = (out / rel).read_text(encoding="utf-8")
        wanted = pages[rel]
        if current != wanted:
            messages.append(f"changed: {rel.as_posix()}")
            diff = difflib.unified_diff(
                current.splitlines(),
                wanted.splitlines(),
                fromfile=f"{rel.as_posix()} (committed)",
                tofile=f"{rel.as_posix()} (generated)",
                lineterm="",
                n=3,
            )
            messages.extend(list(diff)[:80])
    return not messages, messages


def render_to_temp() -> Path:
    """Render the tree to a temporary directory and return the path."""

    temp = Path(tempfile.mkdtemp(prefix="macroforecast-docgen-"))
    write_all(temp)
    return temp


def _remove_empty_dirs(root: Path) -> None:
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()


def _normalise_text(text: str) -> str:
    return text.rstrip() + "\n"


def _module(page: ModulePage) -> ModuleType:
    return importlib.import_module(page.module)


def _public_names(module: ModuleType) -> tuple[str, ...]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in dir(module) if not name.startswith("_")]
    return tuple(str(name) for name in names)


def _resolve_dotted(path: str) -> Any:
    parts = path.split(".")
    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            obj: Any = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for part in parts[split_at:]:
            obj = getattr(obj, part)
        return obj
    raise ModuleNotFoundError(path)


def _signature(name: str, obj: Any) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return f"{name}(...)"
    text = _clean_annotation_text(str(sig))
    text = re.sub(r"= '([^']*)'", r'= "\1"', text)
    text = re.sub(r": '([^']+)'", r": \1", text)
    text = re.sub(r"-> '([^']+)'", r"-> \1", text)
    return f"{name}{text}"


def _clean_annotation_text(text: str) -> str:
    text = re.sub(r"<object object at 0x[0-9a-fA-F]+>", "<UNSET>", text)
    text = re.sub(r"<function ([A-Za-z_][A-Za-z0-9_]*) at 0x[0-9a-fA-F]+>", r"<function \1>", text)
    return text


def _annotation(annotation: Any) -> str:
    if annotation is inspect.Signature.empty:
        return "unspecified"
    return _clean_annotation_text(_name(annotation))


def _default(default: Any) -> str:
    if default is inspect.Signature.empty:
        return "required"
    text = _stable_repr(default)
    if len(text) > 64:
        text = text[:61] + "..."
    return text.replace("|", "\\|")


def _kind(parameter: inspect.Parameter) -> str:
    if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
        return "positional only"
    if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
        return "positional or keyword"
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        return "var positional"
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
        return "keyword only"
    if parameter.kind is inspect.Parameter.VAR_KEYWORD:
        return "var keyword"
    return str(parameter.kind)


def _name(obj: Any) -> str:
    if hasattr(obj, "__name__"):
        return str(obj.__name__)
    return str(obj)


def _qualname(obj: Any) -> str:
    module = getattr(obj, "__module__", "")
    qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", ""))
    if module and qualname:
        return f"{module}.{qualname}"
    return str(obj)


def _summary(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return "No public docstring is available."


def _doc_body(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return "No public docstring is available."
    lines = []
    for line in doc.splitlines():
        if line.strip() and set(line.strip()) <= {"-", "="}:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _object_kind(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj):
        return "function"
    if callable(obj):
        return "callable"
    if isinstance(obj, ModuleType):
        return "module"
    return "data"


def _entry_anchor(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "")


def _render_reference_index() -> str:
    groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Orientation", ("documentation_map", "workflow", "legacy_callable_coverage", "reference_verification", "public_api")),
        ("Package Configuration", ("meta",)),
        ("Data Pipeline", ("data", "preprocessing", "data_analysis")),
        ("Feature Pipeline", ("filters", "feature_engineering", "feature_analysis")),
        ("Forecast Pipeline", ("window", "models", "model_ensemble", "model_selection", "forecasting", "forecast_analysis")),
        ("Evaluation And Testing", ("metrics", "tests", "evaluation")),
        ("Explanation And Delivery", ("interpretation", "interpretation_dual", "output", "reporting", "pipeline")),
    )
    lines = [
        "# Reference",
        "",
        "[Back to documentation home](../index.md)",
        "",
        "Current workflow reference for the live public Python API. These pages are generated from importable package surfaces, model registry metadata, signatures, and docstrings.",
        "",
        "Start with [Documentation Map](documentation_map.md) when deciding what to inspect, or [Public Python API](public_api.md) when checking top-level imports.",
        "",
        "## First Review Path",
        "",
        "| Order | Page | Why it comes first |",
        "| --- | --- | --- |",
        "| 1 | [Documentation Map](documentation_map.md) | Shows which page answers which question. |",
        "| 2 | [Workflow Contract](workflow.md) | Defines current module ownership and runner composition. |",
        "| 3 | [Public Python API](public_api.md) | Lists importable public symbols from the live package. |",
        "| 4 | [Reference Verification](reference_verification.md) | Records the generation source and coverage counts. |",
        "| 5 | [Custom Extensions](custom/index.md) | Shows where user-owned data, models, tests, and outputs plug in. |",
        "",
        "## Workflow Groups",
        "",
        "| Group | Pages |",
        "| --- | --- |",
    ]
    for title, names in groups:
        links = ", ".join(f"[{MODULE_PAGE_BY_NAME.get(name, ModulePage(name, name, '', '')).title if name in MODULE_PAGE_BY_NAME else _title(name)}]({name}.md)" for name in names)
        lines.append(f"| {title} | {links} |")
    lines += [
        "| Custom Hooks | [Custom Extensions](custom/index.md) |",
        "",
    ]
    for title, names in groups:
        lines.extend([
            f"## {title}",
            "",
            "```{toctree}",
            ":maxdepth: 1",
            f":caption: {title}",
            "",
        ])
        lines.extend(names)
        lines.extend(["```", ""])
    lines.extend([
        "## Custom Hooks",
        "",
        "```{toctree}",
        ":maxdepth: 1",
        ":caption: Custom Hooks",
        "",
        "custom/index",
        "```",
    ])
    return "\n".join(lines)


def _render_documentation_map() -> str:
    lines = [
        "# Documentation Map",
        "",
        "[Back to reference](index.md)",
        "",
        "Use this page to jump from a question to the generated reference page that owns it.",
        "",
        "| Question | Open first | Then open |",
        "| --- | --- | --- |",
        "| What can I import from the package? | [Public Python API](public_api.md) | The module page for the symbol. |",
        "| How do I load FRED or custom panels? | [Data](data.md) | [Custom Dataset](custom/custom_dataset.md), [FRED Datasets](../datasets/index.md). |",
        "| How do I configure preprocessing? | [Preprocessing](preprocessing.md) | [Custom Preprocess](custom/custom_preprocess.md). |",
        "| How do I build features and targets? | [Feature Engineering](feature_engineering.md) | [Feature Analysis](feature_analysis.md). |",
        "| Which models exist and what are their defaults? | [Models](models.md) | [Model Selection](model_selection.md), [Models & Features](../guide/model_overview.md). |",
        "| How do I run a full POOS study? | [Pipeline](pipeline.md) | [Forecasting](forecasting.md), [Evaluation](evaluation.md). |",
        "| Which tests and scores are available? | [Metrics](metrics.md) | [Tests](tests.md), [Evaluation](evaluation.md). |",
        "| How do I render paper tables? | [Reporting](reporting.md) | [Output](output.md). |",
        "| How do I plug in my own model or hook? | [Custom Extensions](custom/index.md) | The specific custom page for that hook. |",
    ]
    return "\n".join(lines)


def _render_workflow() -> str:
    rows = [
        ("`macroforecast.data`", "Loads or wraps a canonical pandas panel plus metadata."),
        ("`macroforecast.preprocessing`", "Builds reusable preprocessing specs and applies window-local transforms."),
        ("`macroforecast.window`", "Defines estimation, validation, and test splits."),
        ("`macroforecast.feature_engineering`", "Builds target columns and predictor matrices."),
        ("`macroforecast.models`", "Fits individual model families and owns model specs."),
        ("`macroforecast.model_selection`", "Tunes model-owned parameters."),
        ("`macroforecast.forecasting`", "Runs one model/target/horizon forecast job."),
        ("`macroforecast.pipeline`", "Runs and evaluates full multi-arm studies."),
        ("`macroforecast.metrics` and `macroforecast.tests`", "Score forecasts and run forecast-comparison tests."),
        ("`macroforecast.output` and `macroforecast.reporting`", "Collect artifacts and render tables."),
    ]
    lines = [
        "# Workflow Contract",
        "",
        "[Back to reference](index.md)",
        "",
        "The current architecture is package-surface first. Users call Python functions and dataclass specs directly; the removed layered YAML/ops registry is not part of the live workflow.",
        "",
        "## Module Ownership",
        "",
        "| Module | Owns |",
        "| --- | --- |",
    ]
    lines.extend(f"| {module} | {purpose} |" for module, purpose in rows)
    lines += [
        "",
        "## Runner Shape",
        "",
        "A full study is declared as `macroforecast.pipeline.pipeline_spec(...)` and executed by `macroforecast.pipeline.run_pipeline(...)`. A single-model run is declared directly through `macroforecast.forecasting.run(...)`.",
        "",
        "The reference pages are generated from these importable modules. No generated page depends on the removed core ops registry or layer-spec module.",
    ]
    return "\n".join(lines)


def _render_legacy_callable_coverage() -> str:
    lines = [
        "# Legacy Callable Coverage",
        "",
        "[Back to reference](index.md)",
        "",
        "The old layered-ops encyclopedia has been removed from the generated reference contract. Current coverage is tied to importable module surfaces and model specs.",
        "",
        "| Legacy item | Current status | Replacement |",
        "| --- | --- | --- |",
        "| Layer-spec registry pages | Removed | Generated module pages and model registry tables. |",
        "| Option-doc generated option tables | Removed | Function-first entries with signatures, parameters, returns, and docstrings. |",
        "| Core ops registry | Removed | Direct public callables under semantic modules. |",
        "| Former standalone callable namespace | Removed | Top-level lazy exports and semantic submodules such as `macroforecast.metrics`, `macroforecast.tests`, and `macroforecast.models`. |",
        "| Legacy layer/axis/option browse pages | Removed | [Documentation Map](documentation_map.md), [Public Python API](public_api.md), and module pages. |",
    ]
    return "\n".join(lines)


def _render_reference_verification() -> str:
    module_count = len(MODULE_PAGES)
    symbol_count = 0
    callable_count = 0
    class_count = 0
    data_count = 0
    for page in MODULE_PAGES:
        module = _module(page)
        for name in _public_names(module):
            obj = getattr(module, name)
            symbol_count += 1
            if inspect.isclass(obj):
                class_count += 1
            elif callable(obj):
                callable_count += 1
            else:
                data_count += 1
    model_count = len(mf.models.MODEL_SPECS)
    lines = [
        "# Reference Verification",
        "",
        "[Back to reference](index.md)",
        "",
        "This tree is generated by `python -m tools.docgen` from the live installed package in the repository checkout.",
        "",
        "## Current Counts",
        "",
        "| Item | Count |",
        "| --- | ---: |",
        f"| Generated module pages | {module_count} |",
        f"| Public symbols across module pages | {symbol_count} |",
        f"| Callable/function symbols | {callable_count} |",
        f"| Class symbols | {class_count} |",
        f"| Data/module symbols | {data_count} |",
        f"| Registered model specs | {model_count} |",
        "",
        "## Drift Gate",
        "",
        "Run:",
        "",
        "```bash",
        "python -m tools.docgen --check docs/reference",
        "```",
        "",
        "The command renders a temporary tree, compares every generated page with the committed tree, reports missing/stale/changed files, and exits non-zero on drift.",
    ]
    return "\n".join(lines)


def _render_public_api() -> str:
    lazy_exports: Mapping[str, str] = getattr(mf, "_LAZY_EXPORTS", {})
    lazy_modules: tuple[str, ...] = getattr(mf, "_LAZY_MODULES", ())

    lines = [
        "# Public Python API",
        "",
        "[Back to reference](index.md)",
        "",
        f"`macroforecast.__version__`: `{mf.__version__}`",
        "",
        "The top-level package uses lazy exports. Attribute access imports the owning semantic module on demand.",
        "",
        "## Top-Level Exports",
        "",
        "| Symbol | Owner | Kind |",
        "| --- | --- | --- |",
    ]
    for name in sorted(mf.__all__):
        owner = lazy_exports.get(name)
        if owner is None and name in lazy_modules:
            owner = f".{name}"
        owner_name = owner.removeprefix(".") if owner else "macroforecast"
        owner_ref = (
            f"[`macroforecast.{owner_name}`]({owner_name}.md)"
            if owner_name in MODULE_PAGE_BY_NAME
            else f"`macroforecast.{owner_name}`"
        )
        obj = getattr(mf, name)
        lines.append(f"| `{name}` | {owner_ref} | {_object_kind(obj)} |")
    lines += [
        "",
        "## Submodules",
        "",
        "| Module | Reference |",
        "| --- | --- |",
    ]
    for module_name in lazy_modules:
        obj = getattr(mf, module_name)
        if getattr(obj, "__name__", None) != f"macroforecast.{module_name}":
            continue
        ref = (
            f"[`macroforecast.{module_name}`]({module_name}.md)"
            if module_name in MODULE_PAGE_BY_NAME
            else f"`macroforecast.{module_name}`"
        )
        lines.append(f"| `macroforecast.{module_name}` | {ref} |")
    return "\n".join(lines)


def _render_module_page(page: ModulePage) -> str:
    module = _module(page)
    names = _public_names(module)
    objects = [(name, getattr(module, name)) for name in names]
    model_names = set(mf.models.MODEL_SPECS) if page.name == "models" else set()
    callables = [
        (name, obj)
        for name, obj in objects
        if (callable(obj) or inspect.isclass(obj)) and name not in model_names
    ]
    data_items = [(name, obj) for name, obj in objects if not callable(obj) and not inspect.isclass(obj)]

    lines = [
        f"# macroforecast.{page.name}",
        "",
        "[Back to reference](index.md)",
        "",
        page.purpose,
    ]
    if page.guide_link:
        lines.extend(["", f"Guide context: [{page.guide_link}]({page.guide_link})."])
    lines += [
        "",
        "## Public Symbols",
        "",
        "| Symbol | Kind | Summary |",
        "| --- | --- | --- |",
    ]
    for name, obj in objects:
        lines.append(f"| `{name}` | {_object_kind(obj)} | {_cell(_summary(obj))} |")

    if page.name == "models":
        lines.extend(["", _render_model_registry_section()])

    if data_items:
        lines += ["", "## Data And Module Values", ""]
        lines.extend(_render_data_entry(name, obj) for name, obj in data_items)

    if callables:
        lines += ["", "## Callable And Class Reference", ""]
        for name, obj in callables:
            lines.append(_render_object_entry(f"macroforecast.{page.name}.{name}", name, obj))
    return "\n".join(lines)


def _render_model_registry_section() -> str:
    rows = [
        "| Model | Family | Input kind | Default preset | Requires extra | Requires scaling | Description |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in sorted(mf.models.MODEL_SPECS):
        spec = mf.models.MODEL_SPECS[name]
        rows.append(
            "| "
            f"`{spec.name}` | `{spec.family}` | `{spec.input_kind}` | `{spec.default_preset}` | "
            f"{_extra(spec.requires_extra)} | {'yes' if spec.requires_scaling else 'no'} | {_cell(spec.description)} |"
        )
    lines = [
        "## Model Registry",
        "",
        "These rows come from `macroforecast.models.MODEL_SPECS` / `list_model_specs()`.",
        "",
        *rows,
        "",
        "## Registered Model Details",
        "",
    ]
    for name in sorted(mf.models.MODEL_SPECS):
        spec = mf.models.MODEL_SPECS[name]
        lines.extend(
            [
                f"### {name}",
                "",
                f"Family: `{spec.family}`",
                "",
                "#### Fit Signature",
                "",
                "```python",
                _signature(f"macroforecast.models.{name}", spec.fit_func),
                "```",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| `input_kind` | `{spec.input_kind}` |",
                f"| `default_preset` | `{spec.default_preset}` |",
                f"| `default_search_method` | `{spec.default_search_method}` |",
                f"| `requires_extra` | {_extra(spec.requires_extra)} |",
                f"| `requires_scaling` | {'yes' if spec.requires_scaling else 'no'} |",
                f"| `recommended_preprocessing` | `{_cell(_stable_repr(spec.recommended_preprocessing))}` |",
                "",
                spec.description or "No model description is available.",
                "",
            ]
        )
        if spec.parameters:
            lines.extend(
                [
                    "#### Model Parameters",
                    "",
                    "| Name | Default | Kind | Tunable | Description |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for parameter in spec.parameters:
                lines.append(
                    f"| `{parameter.name}` | `{_cell(_stable_repr(parameter.default))}` | `{_cell(str(parameter.kind))}` | {parameter.tunable} | {_cell(parameter.description)} |"
                )
            lines.append("")
        if spec.search_spaces:
            lines.extend(
                [
                    "#### Search Spaces",
                    "",
                    "| Preset | Parameters |",
                    "| --- | --- |",
                ]
            )
            for preset, space in sorted(spec.search_spaces.items()):
                params = ", ".join(f"`{param}`: `{_cell(_stable_repr(values))}`" for param, values in sorted(space.items()))
                lines.append(f"| `{preset}` | {params or 'none'} |")
            lines.append("")
    return "\n".join(lines)


def _extra(value: Any) -> str:
    if value is None or value == "":
        return "none"
    return f"`{value}`"


def _render_data_entry(name: str, obj: Any) -> str:
    text = _stable_repr(obj)
    if len(text) > 240:
        text = text[:237] + "..."
    return "\n".join([
        f"### `{name}`",
        "",
        f"Kind: `{_object_kind(obj)}`",
        "",
        "```python",
        f"{name} = {text}",
        "```",
    ])


def _render_object_entry(dotted_name: str, display_name: str, obj: Any) -> str:
    lines = [
        f"### {display_name}",
        "",
        f"Qualified name: `{_qualname(obj)}`",
        "",
        "#### Signature",
        "",
        "```python",
        _signature(dotted_name, obj),
        "```",
        "",
        "#### Description",
        "",
        _doc_body(obj),
    ]
    parameters = _parameters(obj)
    if parameters:
        lines += [
            "",
            "#### Parameters",
            "",
            "| Name | Kind | Type | Default |",
            "| --- | --- | --- | --- |",
        ]
        for parameter in parameters:
            lines.append(
                f"| `{parameter.name}` | {_kind(parameter)} | `{_cell(_annotation(parameter.annotation))}` | `{_default(parameter.default)}` |"
            )
    return_type = _return_annotation(obj)
    lines += [
        "",
        "#### Returns",
        "",
        return_type,
        "",
        "#### Minimal Use",
        "",
        "```python",
        "import macroforecast as mf",
        _minimal_use(dotted_name, display_name, obj),
        "```",
    ]
    if dataclasses.is_dataclass(obj):
        lines += ["", _render_dataclass_fields(obj)]
    if inspect.isclass(obj):
        method_lines = _render_public_methods(obj)
        if method_lines:
            lines += ["", method_lines]
    return "\n".join(lines)


def _parameters(obj: Any) -> list[inspect.Parameter]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    return list(sig.parameters.values())


def _return_annotation(obj: Any) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "See the description and object-specific contract."
    if sig.return_annotation is inspect.Signature.empty:
        return "See the description and object-specific contract."
    return f"`{_annotation(sig.return_annotation)}`"


def _minimal_use(dotted_name: str, display_name: str, obj: Any) -> str:
    if dotted_name.startswith("macroforecast."):
        suffix = dotted_name.removeprefix("macroforecast.")
        call = f"mf.{suffix}"
    else:
        call = display_name
    if inspect.isclass(obj):
        return f"# Construct with the signature above:\n# {call}(...)"
    if callable(obj):
        return f"# Call with the signature above:\n# {call}(...)"
    return f"# Access as {call}"


def _render_dataclass_fields(obj: Any) -> str:
    lines = [
        "#### Dataclass Fields",
        "",
        "| Field | Type | Default |",
        "| --- | --- | --- |",
    ]
    for field in dataclasses.fields(obj):
        if field.default is not dataclasses.MISSING:
            default = _stable_repr(field.default)
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
            default = "default_factory"
        else:
            default = "required"
        lines.append(f"| `{field.name}` | `{_cell(_name(field.type))}` | `{_cell(default)}` |")
    return "\n".join(lines)


def _stable_repr(value: Any) -> str:
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, Mapping):
        keys = sorted(str(key) for key in value)
        preview = ", ".join(keys[:12])
        suffix = "" if len(keys) <= 12 else ", ..."
        return f"{type(value).__name__}({len(keys)} entries: {preview}{suffix})"
    if isinstance(value, (set, frozenset)):
        items = ", ".join(repr(item) for item in sorted(value, key=str))
        return f"{type(value).__name__}({{{items}}})"
    if isinstance(value, tuple):
        return "(" + ", ".join(_stable_repr(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_stable_repr(item) for item in value) + "]"
    if callable(value):
        return _qualname(value)
    return _clean_annotation_text(repr(value))


def _render_public_methods(obj: Any) -> str:
    rows: list[str] = []
    for name, member in inspect.getmembers(obj):
        if name.startswith("_"):
            continue
        if not callable(member):
            continue
        rows.append(f"| `{name}` | `{_cell(_signature(name, member))}` | {_cell(_summary(member))} |")
    if not rows:
        return ""
    return "\n".join([
        "#### Public Methods",
        "",
        "| Method | Signature | Summary |",
        "| --- | --- | --- |",
        *rows,
    ])


def _render_custom_index() -> str:
    lines = [
        "# Custom Extensions",
        "",
        "[Back to reference](../index.md)",
        "",
        "These pages collect the user-owned hooks that enter the current Python API.",
        "",
        "| Extension point | Page |",
        "| --- | --- |",
        "| Overview | [Custom Extension Overview](custom.md) |",
    ]
    for key, title in CUSTOM_PAGE_TITLES.items():
        lines.append(f"| {title} | [{title}]({key}.md) |")
    lines += [
        "",
        "```{toctree}",
        ":maxdepth: 1",
        ":hidden:",
        "",
        "custom",
    ]
    lines.extend(CUSTOM_PAGE_TITLES)
    lines += ["```"]
    return "\n".join(lines)


def _render_custom_overview() -> str:
    lines = [
        "# Custom Extension Overview",
        "",
        "[Back to custom extensions](index.md)",
        "",
        "Custom hooks are normal Python callables wrapped by small spec builders. The spec records metadata and defaults; the runner still owns splitting, fitting, scoring, and artifact collection.",
        "",
        "| Hook | Builder | Returns |",
        "| --- | --- | --- |",
        "| Dataset | `mf.data.custom_dataset(...)`, `mf.data.load_custom_csv(...)` | `DataBundle` |",
        "| Preprocessing | `mf.preprocessing.custom_preprocess_step(...)` | preprocessing step dict |",
        "| Features | `mf.feature_engineering.custom_step(...)` | feature step dict |",
        "| Model | `mf.models.custom_model(...)` | `ModelSpec` |",
        "| Search | `mf.model_selection.custom_search(...)` | `SearchSpec` |",
        "| Forecast combination | `mf.forecasting.custom_combination(...)` | `CombinationSpec` |",
        "| Evaluation test | `mf.tests.custom_test(...)` | callable test wrapper |",
        "| Interpretation | `mf.interpretation.custom_interpretation(...)` | interpretation callable wrapper |",
        "| Output/reporting | `mf.output.write_artifacts(...)`, `mf.reporting.render_tables(...)` | artifacts or rendered tables |",
    ]
    return "\n".join(lines)


def _render_custom_page(key: str) -> str:
    title = CUSTOM_PAGE_TITLES[key]
    entries = CUSTOM_PAGE_ENTRIES[key]
    lines = [
        f"# {title}",
        "",
        "[Back to custom extensions](index.md)",
        "",
        "This page is generated from the live callable signatures.",
    ]
    if key == "custom_model":
        lines += [
            "",
            "Use `custom_model` when the estimator is not built into `macroforecast`. It returns a `ModelSpec`, so it can be passed anywhere a built-in model spec is accepted.",
            "",
            "Important current contract: there is no `metadata=` keyword. Use `description`, `parameters`, `default_params`, `search_spaces`, `input_kind`, `requires_extra`, `requires_scaling`, and `recommended_preprocessing` to describe the custom model.",
        ]
    lines += ["", "## Callable Reference"]
    for dotted in entries:
        obj = _resolve_dotted(dotted)
        lines.extend(["", _render_object_entry(dotted, dotted.split(".")[-1], obj)])
    if key == "custom_model":
        lines += [
            "",
            "## Minimal Custom Model Example",
            "",
            "```python",
            "import numpy as np",
            "import pandas as pd",
            "import macroforecast as mf",
            "",
            "class MeanFit:",
            "    def __init__(self, value: float) -> None:",
            "        self.value = value",
            "",
            "    def predict(self, X: pd.DataFrame) -> np.ndarray:",
            "        return np.full(len(X), self.value)",
            "",
            "def mean_model(X: pd.DataFrame, y: pd.Series, *, offset: float = 0.0) -> MeanFit:",
            "    return MeanFit(float(pd.Series(y).mean()) + offset)",
            "",
            "model = mf.models.custom_model(",
            "    \"mean_model\",",
            "    mean_model,",
            "    default_params={\"offset\": 0.0},",
            "    search_spaces={\"standard\": {\"offset\": (-0.1, 0.0, 0.1)}},",
            "    input_kind=\"supervised\",",
            "    requires_scaling=False,",
            "    description=\"Mean benchmark with a tunable offset.\",",
            ")",
            "```",
        ]
    return "\n".join(lines)


def _title(name: str) -> str:
    return name.replace("_", " ").title()


def copy_rendered_tree(source: Path, destination: Path) -> None:
    """Copy a rendered tree into a destination directory."""

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
