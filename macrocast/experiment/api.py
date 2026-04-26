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
from ..raw.sd_inferred_tcodes import (
    MAP_VERSION as SD_ANALOG_TCODE_MAP_VERSION,
    STATE_SERIES_STATIONARITY_OVERRIDE_VERSION,
    VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION,
)
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
    "sd_state_group": ("1_data_task", "fred_sd_state_group"),
    "fred_sd_state_group": ("1_data_task", "fred_sd_state_group"),
    "sd_variable_group": ("1_data_task", "fred_sd_variable_group"),
    "fred_sd_variable_group": ("1_data_task", "fred_sd_variable_group"),
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


def _normalize_optional_selector_values(
    values: Iterable[str] | str | None,
    *,
    name: str,
    uppercase: bool = False,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        normalized = (values.strip(),)
    else:
        try:
            normalized = tuple(str(value).strip() for value in values)
        except TypeError as exc:
            raise TypeError(f"{name} must be a string or iterable of strings") from exc
    if uppercase:
        normalized = tuple(value.upper() for value in normalized)
    if not normalized:
        raise ValueError(f"{name} must be non-empty when provided")
    if any(not value for value in normalized):
        raise ValueError(f"{name} must not contain empty values")
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
        fred_sd_frequency_policy: str = "report_only",
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
        self.fred_sd_frequency_policy = str(fred_sd_frequency_policy)
        self.sd_tcode_map_version: str | None = None
        self.sd_tcode_allowed_statuses = (
            None if sd_tcode_allowed_statuses is None else tuple(str(status) for status in sd_tcode_allowed_statuses)
        )
        self.sd_tcode_code_map: dict[str, int] | None = None
        self.sd_tcode_source: str | None = None
        self.sd_tcode_audit_uri: str | None = None
        self.sd_states: tuple[str, ...] | None = None
        self.sd_variables: tuple[str, ...] | None = None
        self.fred_sd_state_group = "all_states"
        self.fred_sd_variable_group = "all_sd_variables"
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
        self.sd_tcode_map_version = SD_ANALOG_TCODE_MAP_VERSION
        self.sd_tcode_allowed_statuses = None if statuses is None else tuple(str(status) for status in statuses)
        self.sd_tcode_code_map = None
        self.sd_tcode_source = None
        self.sd_tcode_audit_uri = None
        return self

    def use_sd_empirical_tcodes(
        self,
        *,
        unit: str = "variable_global",
        code_map: Mapping[str, int] | None = None,
        source: str | None = None,
        audit_uri: str | None = None,
    ) -> "Experiment":
        """Opt in to non-official empirical FRED-SD stationarity t-codes.

        `unit="variable_global"` uses macrocast's reviewed 2026-04-26 audit
        map: one code per FRED-SD variable, shared across states. `unit`
        values `state_series` and `state_variable` require an explicit
        column-to-code map such as `{"UR_CA": 2}`.
        """

        normalized_unit = str(unit).lower().replace("-", "_")
        if normalized_unit in {"variable", "variable_global", "sd_variable"}:
            self.sd_tcode_policy = "variable_global_stationarity_v0_1"
            self.sd_tcode_map_version = VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION
            self.sd_tcode_allowed_statuses = None
            self.sd_tcode_code_map = None
        elif normalized_unit in {"state_series", "state_variable", "sd_variable_x_state", "column"}:
            if not code_map:
                raise ValueError("state-series empirical FRED-SD t-codes require a non-empty code_map")
            self.sd_tcode_policy = "state_series_stationarity_override_v0_1"
            self.sd_tcode_map_version = STATE_SERIES_STATIONARITY_OVERRIDE_VERSION
            self.sd_tcode_allowed_statuses = None
            self.sd_tcode_code_map = {str(column): int(code) for column, code in code_map.items()}
        else:
            raise ValueError("unit must be 'variable_global' or 'state_series'")
        self.sd_tcode_source = source
        self.sd_tcode_audit_uri = audit_uri
        return self

    def use_fred_sd_selection(
        self,
        *,
        states: Iterable[str] | str | None = None,
        variables: Iterable[str] | str | None = None,
    ) -> "Experiment":
        """Restrict the FRED-SD component to selected states and variables."""

        self.sd_states = _normalize_optional_selector_values(states, name="states", uppercase=True)
        self.sd_variables = _normalize_optional_selector_values(variables, name="variables")
        return self

    def use_fred_sd_groups(
        self,
        *,
        state_group: str | None = None,
        variable_group: str | None = None,
    ) -> "Experiment":
        """Restrict FRED-SD using built-in Layer 1 state or variable groups."""

        if state_group is not None:
            self.fred_sd_state_group = str(state_group)
        if variable_group is not None:
            self.fred_sd_variable_group = str(variable_group)
        return self

    def use_fred_sd_frequency_policy(self, policy: str) -> "Experiment":
        """Set the Layer 1 FRED-SD native-frequency policy."""

        self.fred_sd_frequency_policy = str(policy)
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
        if self.sd_states is not None or self.sd_variables is not None:
            data_block = recipe["path"]["1_data_task"]
            data_fixed = data_block.setdefault("fixed_axes", {})
            data_leaf = data_block.setdefault("leaf_config", {})
            if self.sd_states is not None:
                data_fixed["state_selection"] = "selected_states"
                data_leaf["sd_states"] = list(self.sd_states)
            if self.sd_variables is not None:
                data_fixed["sd_variable_selection"] = "selected_sd_variables"
                data_leaf["sd_variables"] = list(self.sd_variables)
        if self.fred_sd_state_group != "all_states":
            recipe["path"]["1_data_task"].setdefault("fixed_axes", {})[
                "fred_sd_state_group"
            ] = self.fred_sd_state_group
        if self.fred_sd_variable_group != "all_sd_variables":
            recipe["path"]["1_data_task"].setdefault("fixed_axes", {})[
                "fred_sd_variable_group"
            ] = self.fred_sd_variable_group
        if self.fred_sd_frequency_policy != "report_only":
            recipe["path"]["1_data_task"].setdefault("fixed_axes", {})[
                "fred_sd_frequency_policy"
            ] = self.fred_sd_frequency_policy
        if self.sd_tcode_policy != "none":
            preprocessing_leaf = recipe["path"]["2_preprocessing"].setdefault("leaf_config", {})
            preprocessing_leaf["sd_tcode_policy"] = self.sd_tcode_policy
            if self.sd_tcode_map_version is not None:
                preprocessing_leaf["sd_tcode_map_version"] = self.sd_tcode_map_version
            if self.sd_tcode_allowed_statuses is not None:
                preprocessing_leaf["sd_tcode_allowed_statuses"] = list(self.sd_tcode_allowed_statuses)
            if self.sd_tcode_code_map is not None:
                preprocessing_leaf["sd_tcode_code_map"] = dict(self.sd_tcode_code_map)
            if self.sd_tcode_source is not None:
                preprocessing_leaf["sd_tcode_source"] = self.sd_tcode_source
            if self.sd_tcode_audit_uri is not None:
                preprocessing_leaf["sd_tcode_audit_uri"] = self.sd_tcode_audit_uri
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
