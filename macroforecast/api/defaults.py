"""Default experiment profiles for the user-facing macroforecast API."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PROFILE_NAME = "macroforecast-default-v1"

# Public defaults — single source of truth for user-facing API + recipe builder.
# Any change here propagates to:
#   - mf.forecast() / mf.Experiment() signatures (via import)
#   - DEFAULT_PROFILE (below)
#   - Docs (simple_api/quickstart.md etc.)
DEFAULT_MODEL: str = "ar_p"
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_HORIZONS: tuple[int, ...] = (1,)

# Optional convenience constants for L4 schema-level forecast settings (these
# are also the defaults at core/layers/l4.py:106-111 — kept in sync).
DEFAULT_FORECAST_STRATEGY: str = "direct"
DEFAULT_TRAINING_START_RULE: str = "expanding"
DEFAULT_REFIT_POLICY: str = "every_origin"

DEFAULT_PREPROCESSING_AXES: dict[str, str] = {
    "mixed_frequency_representation": "calendar_aligned_frame",
}

DEFAULT_PROFILE: dict[str, Any] = {
    "name": DEFAULT_PROFILE_NAME,
    "information_set_type": "final_revised_data",
    "target_structure": "single_target",
    "framework": DEFAULT_TRAINING_START_RULE,
    "benchmark_family": "zero_change",
    "feature_builder": "target_lag_features",
    "model": DEFAULT_MODEL,
    "model_family": DEFAULT_MODEL,  # deprecated alias, kept for one release
    "primary_metric": "mse",
    "importance_method": "none",
    "reproducibility_policy": "seeded_reproducible",
    "failure_policy": "fail_fast",
    "compute_policy": "serial",
    "random_seed": DEFAULT_RANDOM_SEED,
    "benchmark_config": {"minimum_train_size": 5},
    "preprocessing": dict(DEFAULT_PREPROCESSING_AXES),
}

_CUSTOM_SOURCE_SCHEMAS = {"fred_md", "fred_qd", "fred_sd"}
_CUSTOM_SOURCE_POLICIES = {"official_only", "custom_panel_only", "official_plus_custom"}
_CUSTOM_SOURCE_FORMATS = {"none", "csv", "parquet"}
_CUSTOM_SOURCE_PARQUET_SUFFIXES = {".parquet", ".pq"}


def _normalize_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(horizon) for horizon in horizons)
    if not values:
        raise ValueError("horizons must contain at least one value")
    if any(horizon <= 0 for horizon in values):
        raise ValueError("horizons must be positive integers")
    return values


def _normalize_models(models: Iterable[str] | None, model: str) -> tuple[str, ...]:
    values: tuple[str, ...]
    if models is None:
        values = (model,)
    elif isinstance(models, str):
        values = (models,)
    else:
        values = tuple(str(value) for value in models)
    if not values:
        raise ValueError("at least one model family is required")
    return values


def _slug(value: object) -> str:
    raw = str(value).strip().lower()
    chars = [ch if ch.isalnum() else "-" for ch in raw]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "value"


def _default_recipe_id(*, dataset: str, target: str, horizons: tuple[int, ...], models: tuple[str, ...]) -> str:
    horizon_token = "-".join(f"h{horizon}" for horizon in horizons)
    if len(models) == 1:
        model_token = _slug(models[0])
    else:
        model_token = f"{len(models)}-models"
    return f"experiment-{_slug(dataset)}-{_slug(target)}-{horizon_token}-{model_token}"


def _dataset_parts(dataset: str) -> set[str]:
    tokens = str(dataset).replace(",", "+").split("+")
    return {token.strip() for token in tokens if token.strip()}


def _normalize_dataset(dataset: str) -> str:
    parts = _dataset_parts(dataset)
    if not parts:
        raise ValueError("dataset is required")
    if parts & {"custom_csv", "custom_parquet"}:
        raise ValueError(
            "custom_csv/custom_parquet are no longer dataset choices; "
            "choose dataset='fred_md'/'fred_qd'/'fred_sd' and set "
            "panel_composition plus custom_source_path"
        )
    if "fred_md" in parts and "fred_qd" in parts:
        raise ValueError("fred_md and fred_qd cannot be combined in one default experiment")
    if parts == {"fred_md", "fred_sd"}:
        return "fred_md+fred_sd"
    if parts == {"fred_qd", "fred_sd"}:
        return "fred_qd+fred_sd"
    if len(parts) == 1:
        return next(iter(parts))
    raise ValueError(f"unsupported dataset combination: {dataset!r}")


def _resolve_frequency(dataset: str, frequency: str | None) -> str:
    parts = _dataset_parts(dataset)
    allowed = {"monthly", "quarterly"}
    if frequency is not None and frequency not in allowed:
        raise ValueError("frequency must be 'monthly' or 'quarterly'")

    if "fred_md" in parts:
        resolved = "monthly"
    elif "fred_qd" in parts:
        resolved = "quarterly"
    elif parts == {"fred_sd"}:
        if frequency is None:
            raise ValueError("frequency is required when dataset='fred_sd' is used alone")
        resolved = frequency
    else:
        resolved = frequency or "monthly"

    if frequency is not None and frequency != resolved:
        raise ValueError(
            f"frequency={frequency!r} conflicts with dataset={dataset!r}; "
            f"resolved frequency is {resolved!r}"
        )
    return resolved


def _custom_source_contract(
    *,
    dataset: str,
    frequency: str,
    panel_composition: str,
    custom_source_format: str,
    custom_source_schema: str | None,
    custom_source_path: str | None,
) -> tuple[str, str | None, str | None]:
    if custom_source_schema == "none":
        custom_source_schema = None
    if panel_composition not in _CUSTOM_SOURCE_POLICIES:
        raise ValueError(f"panel_composition must be one of {sorted(_CUSTOM_SOURCE_POLICIES)}")
    if custom_source_format not in _CUSTOM_SOURCE_FORMATS:
        raise ValueError(f"custom_source_format must be one of {sorted(_CUSTOM_SOURCE_FORMATS)}")

    if panel_composition == "official_only":
        if custom_source_format != "none":
            raise ValueError("custom_source_format applies only when panel_composition selects custom data")
        if custom_source_schema is not None:
            raise ValueError("custom_source_schema applies only when panel_composition selects custom data")
        if custom_source_path is not None:
            raise ValueError("custom_source_path applies only when panel_composition selects custom data")
        return "none", None, None

    if not custom_source_path:
        raise ValueError(
            "custom sources require custom_source_path; "
            "the parser is inferred from the .csv/.parquet/.pq extension"
        )
    resolved_format = _resolve_custom_source_format(
        custom_source_path=custom_source_path,
        custom_source_format=custom_source_format,
    )
    resolved_schema = custom_source_schema or _schema_from_route(
        dataset=dataset,
        frequency=frequency,
        panel_composition=panel_composition,
    )
    if resolved_schema not in _CUSTOM_SOURCE_SCHEMAS:
        raise ValueError(f"custom_source_schema must be one of {sorted(_CUSTOM_SOURCE_SCHEMAS)}")
    if panel_composition == "custom_panel_only":
        if "+" in dataset:
            raise ValueError("custom_panel_only supports a single FRED dataset, not a composite")
        if custom_source_schema is not None and resolved_schema != dataset:
            raise ValueError("legacy custom_source_schema must match dataset for custom_panel_only")
    return resolved_format, resolved_schema, str(custom_source_path)


def _custom_source_format_from_path(custom_source_path: str) -> str:
    suffix = Path(str(custom_source_path)).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in _CUSTOM_SOURCE_PARQUET_SUFFIXES:
        return "parquet"
    raise ValueError(
        "custom_source_path must end with .csv, .parquet, or .pq unless "
        "legacy custom_source_format is provided"
    )


def _resolve_custom_source_format(*, custom_source_path: str, custom_source_format: str) -> str:
    try:
        inferred = _custom_source_format_from_path(custom_source_path)
    except ValueError:
        if custom_source_format != "none":
            return custom_source_format
        raise
    if custom_source_format != "none" and custom_source_format != inferred:
        raise ValueError("custom_source_format conflicts with custom_source_path extension")
    return custom_source_format if custom_source_format != "none" else inferred


def _schema_from_route(*, dataset: str, frequency: str, panel_composition: str) -> str:
    if panel_composition == "custom_panel_only" and dataset == "fred_sd":
        return "fred_sd"
    return "fred_qd" if frequency == "quarterly" else "fred_md"


def build_default_recipe_dict(
    *,
    dataset: str,
    target: str,
    start: str | None = None,
    end: str | None = None,
    horizons: Iterable[int] = (1,),
    recipe_id: str | None = None,
    default_profile: str = DEFAULT_PROFILE_NAME,
    information_set_type: str = "final_revised_data",
    frequency: str | None = None,
    vintage: str | None = None,
    panel_composition: str = "official_only",
    custom_source_format: str = "none",
    custom_source_schema: str | None = None,
    custom_source_path: str | None = None,
    framework: str = "expanding",
    benchmark_model: str | None = None,
    benchmark_family: str | None = None,  # deprecated: use benchmark_model=
    feature_builder: str = "target_lag_features",
    model: str | None = None,
    model_family: str | None = None,  # deprecated: use model=
    models: Iterable[str] | None = None,
    model_families: Iterable[str] | None = None,  # deprecated: use models=
    primary_metric: str = "mse",
    importance_method: str = "none",
    reproducibility_policy: str = "seeded_reproducible",
    failure_policy: str = "fail_fast",
    compute_policy: str = "serial",
    random_seed: int = 42,
    benchmark_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the current minimal executable recipe for an Experiment.

    This is intentionally conservative: it uses the existing compiler and
    runtime path, chooses no-op preprocessing, and records the default profile
    in leaf config so manifests can audit how unspecified choices were filled.
    """

    from ._deprecations import resolve_model, resolve_models, resolve_benchmark_model
    resolved_model = resolve_model(model, model_family, DEFAULT_MODEL)  # deprecated alias: model_family
    resolved_models = resolve_models(models, model_families)  # deprecated alias: model_families
    resolved_benchmark = resolve_benchmark_model(benchmark_model, benchmark_family, "zero_change")  # deprecated alias: benchmark_family

    if not target:
        raise ValueError("target is required")
    if not start:
        raise ValueError("start is required")
    if not end:
        raise ValueError("end is required")
    resolved_dataset = _normalize_dataset(dataset)
    resolved_frequency = _resolve_frequency(resolved_dataset, frequency)
    resolved_custom_format, resolved_custom_schema, resolved_custom_path = _custom_source_contract(
        dataset=resolved_dataset,
        frequency=resolved_frequency,
        panel_composition=panel_composition,
        custom_source_format=custom_source_format,
        custom_source_schema=custom_source_schema,
        custom_source_path=custom_source_path,
    )
    horizon_values = _normalize_horizons(horizons)
    model_values = _normalize_models(resolved_models, resolved_model)
    resolved_recipe_id = recipe_id or _default_recipe_id(
        dataset=resolved_dataset,
        target=target,
        horizons=horizon_values,
        models=model_values,
    )
    resolved_benchmark_config = deepcopy(DEFAULT_PROFILE["benchmark_config"])
    if benchmark_config:
        resolved_benchmark_config.update(benchmark_config)

    training_fixed = {
        "framework": framework,
        "benchmark_model": resolved_benchmark,
        "feature_builder": feature_builder,
    }
    training_block: dict[str, Any] = {"fixed_axes": training_fixed}
    if len(model_values) == 1:
        training_fixed["model"] = model_values[0]
    else:
        training_block["sweep_axes"] = {"model": list(model_values)}

    data_leaf = {
        "target": target,
        "horizons": list(horizon_values),
        "sample_start_date": str(start),
        "sample_end_date": str(end),
        "training_start_date": str(start),
        "data_vintage": vintage,
    }
    if panel_composition != "official_only":
        data_leaf["custom_source_path"] = resolved_custom_path

    layer1_fixed_axes = {
        "dataset": resolved_dataset,
        "panel_composition": panel_composition,
        "frequency": resolved_frequency,
        "information_set_type": information_set_type,
        "fred_sd_frequency_policy": "report_only",
        "fred_sd_state_group": "all_states",
        "fred_sd_variable_group": "all_sd_variables",
        "target_structure": "single_target",
        "missing_availability": "zero_fill_leading_predictor_gaps",
    }
    if custom_source_format != "none":
        layer1_fixed_axes["custom_source_format"] = resolved_custom_format
    if custom_source_schema not in {None, "none"}:
        layer1_fixed_axes["custom_source_schema"] = resolved_custom_schema or "none"

    return {
        "recipe_id": resolved_recipe_id,
        "0_meta": {
            "fixed_axes": {
                "reproducibility_policy": reproducibility_policy,
                "failure_policy": failure_policy,
                "compute_policy": compute_policy,
            },
            "leaf_config": {
                "default_profile": default_profile,
                "random_seed": int(random_seed),
            },
        },
        "data": {
            "fixed_axes": layer1_fixed_axes,
            "leaf_config": data_leaf,
        },
        "preprocessing": {"fixed_axes": dict(DEFAULT_PREPROCESSING_AXES)},
        "4_forecasting_model": training_block,
        "5_evaluation": {"fixed_axes": {"primary_metric": primary_metric}},
        "6_statistical_tests": {"fixed_axes": {}},
        "7_interpretation": {"fixed_axes": {"importance_method": importance_method}},
        "8_output": {
            "leaf_config": {
                "manifest_mode": "full",
                "benchmark_config": resolved_benchmark_config,
            }
        },
    }
