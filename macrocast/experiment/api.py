"""Experiment facade for default-first macro forecasting workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ..compiler import (
    CompileValidationError,
    compile_recipe_dict,
    compile_sweep_plan,
    compiled_spec_to_dict,
)
from ..defaults import DEFAULT_PROFILE_NAME, build_default_recipe_dict
from ..execution import execute_recipe, execute_sweep
from .results import ExperimentRunResult, ExperimentSweepResult

_SWEEP_ALIASES: dict[str, tuple[str, str]] = {
    "model": ("3_training", "model_family"),
    "models": ("3_training", "model_family"),
    "model_family": ("3_training", "model_family"),
    "scaling": ("2_preprocessing", "scaling_policy"),
    "scaling_policy": ("2_preprocessing", "scaling_policy"),
    "missing": ("2_preprocessing", "x_missing_policy"),
    "x_missing": ("2_preprocessing", "x_missing_policy"),
    "x_missing_policy": ("2_preprocessing", "x_missing_policy"),
    "preprocessor": ("2_preprocessing", "custom_preprocessor"),
    "custom_preprocessor": ("2_preprocessing", "custom_preprocessor"),
    "target_transformer": ("2_preprocessing", "target_transformer"),
}


def _normalize_values(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        normalized = (values,)
    else:
        try:
            normalized = tuple(str(value) for value in values)
        except TypeError as exc:
            raise TypeError("sweep values must be a string or iterable of strings") from exc
    if not normalized:
        raise ValueError("sweep values must contain at least one value")
    return normalized


def _has_sweep_axes(recipe_dict: dict[str, Any]) -> bool:
    for layer_block in recipe_dict.get("path", {}).values():
        if isinstance(layer_block, dict) and layer_block.get("sweep_axes"):
            return True
    return False


def _mvp_runtime_blockers(recipe_dict: dict[str, Any]) -> tuple[str, ...]:
    preprocessing = recipe_dict.get("path", {}).get("2_preprocessing", {})
    if not isinstance(preprocessing, dict):
        return ()

    blockers: list[str] = []
    sweep_axes = sorted((preprocessing.get("sweep_axes") or {}).keys())
    if sweep_axes:
        blockers.append(
            "preprocessing sweeps are not executable in the Experiment MVP; "
            f"fixed preprocessing is supported, but sweep axes are blocked: {sweep_axes}"
        )

    fixed_axes = dict(preprocessing.get("fixed_axes") or {})
    extra_axes = []
    for axis in ("scaling_policy", "x_missing_policy"):
        value = fixed_axes.get(axis, "none")
        if value != "none":
            extra_axes.append(f"{axis}={value!r}")
    if extra_axes:
        blockers.append(
            "built-in extra preprocessing is not executable under the default t-code path in the Experiment MVP; "
            f"use a fixed custom_preprocessor or wait for the preprocessing layer audit: {extra_axes}"
        )
    return tuple(blockers)


def _default_output_root(recipe_id: str) -> Path:
    return Path("macrocast_outputs") / recipe_id


class Experiment:
    """User-facing forecasting experiment.

    `Experiment` is a thin facade over the existing recipe compiler and
    execution engine. It keeps the beginner path small while still lowering to
    auditable recipes and manifests.
    """

    def __init__(
        self,
        *,
        dataset: str,
        target: str,
        start: str,
        end: str,
        horizons: Iterable[int] = (1,),
        frequency: str | None = None,
        vintage: str | None = None,
        default_profile: str = DEFAULT_PROFILE_NAME,
        recipe_id: str | None = None,
        model_family: str = "ar",
        framework: str = "expanding",
        benchmark_family: str = "zero_change",
        feature_builder: str = "autoreg_lagged_target",
        primary_metric: str = "msfe",
        random_seed: int = 42,
        benchmark_config: dict[str, Any] | None = None,
        sd_tcode_policy: str = "none",
        sd_tcode_allowed_statuses: Iterable[str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.target = target
        self.start = str(start)
        self.end = str(end)
        self.horizons = tuple(int(horizon) for horizon in horizons)
        self.frequency = frequency
        self.vintage = vintage
        self.default_profile = default_profile
        self.recipe_id = recipe_id
        self.model_family = model_family
        self.framework = framework
        self.benchmark_family = benchmark_family
        self.feature_builder = feature_builder
        self.primary_metric = primary_metric
        self.random_seed = int(random_seed)
        self.benchmark_config = dict(benchmark_config or {})
        self.sd_tcode_policy = str(sd_tcode_policy)
        self.sd_tcode_allowed_statuses = (
            None if sd_tcode_allowed_statuses is None else tuple(str(status) for status in sd_tcode_allowed_statuses)
        )
        self._model_families: tuple[str, ...] = (model_family,)
        self._sweep_axes: dict[tuple[str, str], tuple[str, ...]] = {}
        self._custom_preprocessor: str | None = None
        self._target_transformer: str | None = None

    def compare_models(self, models: Iterable[str]) -> "Experiment":
        """Compare built-in or registered model names under fixed defaults."""

        self._model_families = _normalize_values(models)
        self.model_family = self._model_families[0]
        return self

    def use_preprocessor(self, name: str) -> "Experiment":
        """Use a registered custom preprocessor for this experiment."""

        self._custom_preprocessor = str(name)
        return self

    def use_target_transformer(self, name: str) -> "Experiment":
        """Use a registered target transformer for this experiment.

        Currently executable for the autoregressive lagged-target path.
        Raw-panel/exogenous target transformation remains blocked until its
        horizon-aligned supervised-target contract is fixed.
        """

        self._target_transformer = str(name)
        return self

    def use_sd_inferred_tcodes(self, statuses: Iterable[str] | None = None) -> "Experiment":
        """Opt in to reviewed, non-official FRED-SD inferred t-codes."""

        self.sd_tcode_policy = "inferred_v0_1"
        self.sd_tcode_allowed_statuses = None if statuses is None else tuple(str(status) for status in statuses)
        return self

    def sweep(self, choices: dict[str, Any]) -> "Experiment":
        """Sweep a small set of user-facing aliases.

        MVP aliases intentionally stay narrow: models, scaling, and x-missing
        policy. More aliases should be added only after the corresponding layer
        audit marks them safe for the simple API.
        """

        for alias, raw_values in choices.items():
            if alias not in _SWEEP_ALIASES:
                valid = ", ".join(sorted(_SWEEP_ALIASES))
                raise ValueError(f"unknown sweep alias {alias!r}; valid aliases: {valid}")
            layer, axis = _SWEEP_ALIASES[alias]
            values = _normalize_values(raw_values)
            if axis == "model_family":
                self.compare_models(values)
            elif axis == "custom_preprocessor" and len(values) == 1:
                self.use_preprocessor(values[0])
            elif axis == "target_transformer" and len(values) == 1:
                self.use_target_transformer(values[0])
            else:
                self._sweep_axes[(layer, axis)] = values
        return self

    def to_recipe_dict(self) -> dict[str, Any]:
        """Lower this experiment to the internal recipe dict."""

        recipe = build_default_recipe_dict(
            dataset=self.dataset,
            target=self.target,
            start=self.start,
            end=self.end,
            horizons=self.horizons,
            frequency=self.frequency,
            vintage=self.vintage,
            recipe_id=self.recipe_id,
            default_profile=self.default_profile,
            framework=self.framework,
            benchmark_family=self.benchmark_family,
            feature_builder=self.feature_builder,
            model_family=self.model_family,
            model_families=self._model_families,
            primary_metric=self.primary_metric,
            random_seed=self.random_seed,
            benchmark_config=self.benchmark_config,
        )
        if self._custom_preprocessor is not None:
            recipe["path"]["2_preprocessing"]["fixed_axes"]["custom_preprocessor"] = self._custom_preprocessor
        if self._target_transformer is not None:
            recipe["path"]["2_preprocessing"]["fixed_axes"]["target_transformer"] = self._target_transformer
        if self.sd_tcode_policy != "none":
            preprocessing_leaf = recipe["path"]["2_preprocessing"].setdefault("leaf_config", {})
            preprocessing_leaf["sd_tcode_policy"] = self.sd_tcode_policy
            preprocessing_leaf["sd_tcode_map_version"] = "sd-analog-v0.1"
            if self.sd_tcode_allowed_statuses is not None:
                preprocessing_leaf["sd_tcode_allowed_statuses"] = list(self.sd_tcode_allowed_statuses)
        for (layer, axis), values in self._sweep_axes.items():
            layer_block = recipe["path"][layer]
            fixed_axes = dict(layer_block.get("fixed_axes") or {})
            sweep_axes = dict(layer_block.get("sweep_axes") or {})
            if len(values) == 1:
                sweep_axes.pop(axis, None)
                fixed_axes[axis] = values[0]
            else:
                fixed_axes.pop(axis, None)
                sweep_axes[axis] = list(values)
            layer_block["fixed_axes"] = fixed_axes
            if sweep_axes:
                layer_block["sweep_axes"] = sweep_axes
            else:
                layer_block.pop("sweep_axes", None)
        if _has_sweep_axes(recipe):
            recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "controlled_variation"
        return recipe

    def run(
        self,
        *,
        output_root: str | Path | None = None,
        local_raw_source: str | Path | Mapping[str, str | Path] | None = None,
    ):
        """Run the experiment through the existing compiler/executor."""

        recipe = self.to_recipe_dict()
        blockers = _mvp_runtime_blockers(recipe)
        if blockers:
            raise CompileValidationError("; ".join(blockers))
        resolved_output_root = Path(output_root) if output_root is not None else _default_output_root(recipe["recipe_id"])

        if _has_sweep_axes(recipe):
            plan = compile_sweep_plan(recipe)
            sweep_result = execute_sweep(
                plan=plan,
                output_root=resolved_output_root,
                local_raw_source=local_raw_source,
                extra_provenance={"default_profile": self.default_profile},
            )
            return ExperimentSweepResult(sweep_result)

        compile_result = compile_recipe_dict(recipe)
        compiled = compile_result.compiled
        if compiled.execution_status != "executable":
            raise CompileValidationError(
                f"compiled experiment is not executable: {compiled.execution_status}; "
                f"warnings={list(compiled.warnings)}; blocked={list(compiled.blocked_reasons)}"
            )
        provenance_payload = {
            "compiler": compiled_spec_to_dict(compiled),
            "tree_context": dict(compiled.tree_context),
            "default_profile": self.default_profile,
        }
        execution = execute_recipe(
            recipe=compiled.recipe_spec,
            preprocess=compiled.preprocess_contract,
            output_root=resolved_output_root,
            local_raw_source=local_raw_source,
            provenance_payload=provenance_payload,
        )
        return ExperimentRunResult.from_execution(execution)


def forecast(
    dataset: str,
    *,
    target: str,
    start: str,
    end: str,
    horizons: Iterable[int] = (1,),
    frequency: str | None = None,
    vintage: str | None = None,
    output_root: str | Path | None = None,
    local_raw_source: str | Path | Mapping[str, str | Path] | None = None,
    **experiment_kwargs: Any,
):
    """Run one default Experiment."""

    return Experiment(
        dataset=dataset,
        target=target,
        start=start,
        end=end,
        horizons=horizons,
        frequency=frequency,
        vintage=vintage,
        **experiment_kwargs,
    ).run(output_root=output_root, local_raw_source=local_raw_source)
