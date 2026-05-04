"""Programmatic recipe builder.

Authoring API for users who prefer Python over hand-written YAML or the
CLI wizard. Mirrors the layer-by-layer structure of the recipe so the
builder reads top-to-bottom like an outline.

Usage::

    import macrocast as mc

    b = mc.scaffold.RecipeBuilder()
    b.l0(random_seed=42)
    b.l1.fred_md(target="CPIAUCSL", horizon_set="standard_md")
    b.l2.standard()
    b.l3.lag_only(n_lag=12)
    b.l4.fit("ridge", alpha=1.0).is_benchmark()
    b.l4.fit("random_forest", n_estimators=200)
    b.l5.standard(primary_metric="mse")

    recipe_dict = b.build()
    yaml_text = b.to_yaml()
    result = b.run(output_directory="out/")
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


class _LayerNamespace:
    """Base helper -- per-layer namespaces inherit from this and add
    convenience constructors. Common state: a mutable ``recipe`` dict
    pointing at the parent ``RecipeBuilder``."""

    layer_key: str = ""

    def __init__(self, recipe: dict[str, Any]) -> None:
        self._recipe = recipe

    @property
    def block(self) -> dict[str, Any]:
        return self._recipe.setdefault(self.layer_key, {})

    @property
    def fixed_axes(self) -> dict[str, Any]:
        return self.block.setdefault("fixed_axes", {})

    @property
    def leaf_config(self) -> dict[str, Any]:
        return self.block.setdefault("leaf_config", {})

    def set_axis(self, **kwargs: Any) -> "_LayerNamespace":
        """Set one or more fixed axes. Returns self for chaining."""

        self.fixed_axes.update(kwargs)
        return self

    def set_leaf(self, **kwargs: Any) -> "_LayerNamespace":
        """Set one or more leaf_config entries."""

        self.leaf_config.update(kwargs)
        return self


class _L0(_LayerNamespace):
    layer_key = "0_meta"

    def __call__(
        self,
        *,
        failure_policy: str = "fail_fast",
        reproducibility_mode: str = "seeded_reproducible",
        compute_mode: str = "serial",
        random_seed: int | None = 0,
        **leaf: Any,
    ) -> "_L0":
        self.set_axis(
            failure_policy=failure_policy,
            reproducibility_mode=reproducibility_mode,
            compute_mode=compute_mode,
        )
        if random_seed is not None:
            self.leaf_config["random_seed"] = int(random_seed)
        if leaf:
            self.set_leaf(**leaf)
        return self


class _L1(_LayerNamespace):
    layer_key = "1_data"

    def __call__(
        self,
        *,
        custom_source_policy: str = "official_only",
        dataset: str = "fred_md",
        frequency: str = "monthly",
        horizon_set: str = "standard_md",
        target_structure: str = "single_target",
        target: str | None = None,
        targets: list[str] | None = None,
        **leaf: Any,
    ) -> "_L1":
        self.set_axis(
            custom_source_policy=custom_source_policy,
            dataset=dataset,
            frequency=frequency,
            horizon_set=horizon_set,
            target_structure=target_structure,
        )
        if target is not None:
            self.leaf_config["target"] = target
        if targets is not None:
            self.leaf_config["targets"] = list(targets)
        if leaf:
            self.set_leaf(**leaf)
        return self

    def fred_md(self, *, target: str, **kwargs: Any) -> "_L1":
        """Preset: official FRED-MD with a single target."""

        kwargs.setdefault("custom_source_policy", "official_only")
        kwargs.setdefault("dataset", "fred_md")
        kwargs.setdefault("frequency", "monthly")
        kwargs.setdefault("horizon_set", "standard_md")
        return self(target=target, **kwargs)

    def fred_qd(self, *, target: str, **kwargs: Any) -> "_L1":
        kwargs.setdefault("custom_source_policy", "official_only")
        kwargs.setdefault("dataset", "fred_qd")
        kwargs.setdefault("frequency", "quarterly")
        kwargs.setdefault("horizon_set", "standard_qd")
        return self(target=target, **kwargs)

    def custom_panel(self, *, target: str, panel: dict[str, list[Any]], **kwargs: Any) -> "_L1":
        """Preset: inline custom panel (no FRED dependency).

        ``dataset`` is inactive when ``custom_source_policy = custom_panel_only``,
        so we deliberately omit it from the fixed-axes block (the validator
        rejects studies that set both).
        """

        # Build the L1 block manually so we don't pass the inactive ``dataset``
        # axis through the canonical ``__call__`` path.
        self.fixed_axes.update(
            {
                "custom_source_policy": kwargs.pop("custom_source_policy", "custom_panel_only"),
                "frequency": kwargs.pop("frequency", "monthly"),
                "horizon_set": kwargs.pop("horizon_set", "custom_list"),
                "target_structure": kwargs.pop("target_structure", "single_target"),
            }
        )
        self.leaf_config["target"] = target
        self.leaf_config["target_horizons"] = list(kwargs.pop("target_horizons", [1]))
        self.leaf_config["custom_panel_inline"] = panel
        if kwargs:
            self.leaf_config.update(kwargs)
        return self


class _L2(_LayerNamespace):
    layer_key = "2_preprocessing"

    def __call__(self, **axes: Any) -> "_L2":
        return self.set_axis(**axes)

    def standard(self) -> "_L2":
        """Preset: McCracken-Ng default pipeline (apply_official_tcode +
        IQR outliers + EM-factor imputation + truncate-to-balanced edges)."""

        return self(
            transform_policy="apply_official_tcode",
            outlier_policy="mccracken_ng_iqr",
            outlier_action="flag_as_nan",
            imputation_policy="em_factor",
            frame_edge_policy="truncate_to_balanced",
        )

    def no_op(self) -> "_L2":
        """Preset: pass-through (custom panels with already-clean data)."""

        return self(
            transform_policy="no_transform",
            outlier_policy="none",
            imputation_policy="none_propagate",
            frame_edge_policy="keep_unbalanced",
        )


class _L3(_LayerNamespace):
    layer_key = "3_feature_engineering"

    def lag_only(self, *, n_lag: int = 1) -> "_L3":
        """Preset: a single lag step on every predictor + target_construction."""

        block = self.block
        block.setdefault("nodes", [])
        block["nodes"] = [
            {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
            {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
            {"id": "lag_x", "type": "step", "op": "lag", "params": {"n_lag": int(n_lag)}, "inputs": ["src_X"]},
            {"id": "y_h", "type": "step", "op": "target_construction", "params": {"mode": "point_forecast", "method": "direct", "horizon": 1}, "inputs": ["src_y"]},
        ]
        block["sinks"] = {"l3_features_v1": {"X_final": "lag_x", "y_final": "y_h"}, "l3_metadata_v1": "auto"}
        return self


class _L4(_LayerNamespace):
    layer_key = "4_forecasting_model"

    def __init__(self, recipe: dict[str, Any]) -> None:
        super().__init__(recipe)
        self._fit_count = 0

    def _ensure_sources(self) -> None:
        block = self.block
        nodes = block.setdefault("nodes", [])
        if not any(n.get("id") == "src_X" for n in nodes):
            nodes.append({"id": "src_X", "type": "source", "selector": {"layer_ref": "l3", "sink_name": "l3_features_v1", "subset": {"component": "X_final"}}})
        if not any(n.get("id") == "src_y" for n in nodes):
            nodes.append({"id": "src_y", "type": "source", "selector": {"layer_ref": "l3", "sink_name": "l3_features_v1", "subset": {"component": "y_final"}}})

    def fit(
        self,
        family: str,
        *,
        forecast_strategy: str = "direct",
        training_start_rule: str = "expanding",
        refit_policy: str = "every_origin",
        search_algorithm: str = "none",
        min_train_size: int | None = None,
        is_benchmark: bool = False,
        **params: Any,
    ) -> "_FitNodeHandle":
        """Add a fit_model node with the supplied family + params. Returns
        a handle that supports ``.is_benchmark()``."""

        self._ensure_sources()
        self._fit_count += 1
        node_id = f"fit_{self._fit_count}_{family}"
        node = {
            "id": node_id,
            "type": "step",
            "op": "fit_model",
            "params": {
                "family": family,
                "forecast_strategy": forecast_strategy,
                "training_start_rule": training_start_rule,
                "refit_policy": refit_policy,
                "search_algorithm": search_algorithm,
                **params,
            },
            "inputs": ["src_X", "src_y"],
        }
        if min_train_size is not None:
            node["params"]["min_train_size"] = int(min_train_size)
        if is_benchmark:
            node["is_benchmark"] = True
        self.block["nodes"].append(node)
        # Wire predict + sinks lazily on first fit.
        if not any(n.get("id") == "predict" for n in self.block["nodes"]):
            self.block["nodes"].append({"id": "predict", "type": "step", "op": "predict", "inputs": [node_id, "src_X"]})
            self.block["sinks"] = {
                "l4_forecasts_v1": "predict",
                "l4_model_artifacts_v1": node_id,
                "l4_training_metadata_v1": "auto",
            }
        return _FitNodeHandle(self.block, node)


class _FitNodeHandle:
    """Returned by ``b.l4.fit(...)`` so the user can chain modifiers."""

    def __init__(self, l4_block: dict[str, Any], node: dict[str, Any]) -> None:
        self._block = l4_block
        self._node = node

    def is_benchmark(self) -> "_FitNodeHandle":
        self._node["is_benchmark"] = True
        # Re-wire l4_model_artifacts_v1 sink to the benchmark node so L5 can
        # find it.
        self._block.setdefault("sinks", {})["l4_model_artifacts_v1"] = self._node["id"]
        return self


class _L5(_LayerNamespace):
    layer_key = "5_evaluation"

    def __call__(self, **axes: Any) -> "_L5":
        return self.set_axis(**axes)

    def standard(self, *, primary_metric: str = "mse", **axes: Any) -> "_L5":
        return self(primary_metric=primary_metric, **axes)


class _L6(_LayerNamespace):
    layer_key = "6_statistical_tests"


class _L7(_LayerNamespace):
    layer_key = "7_interpretation"


class _L8(_LayerNamespace):
    layer_key = "8_output"


# ---------------------------------------------------------------------------
# RecipeBuilder
# ---------------------------------------------------------------------------

class RecipeBuilder:
    """Programmatic recipe authoring entry. Each ``b.l<N>`` returns a
    layer namespace; calling the namespace (or its ``.preset_name()`` /
    ``.set_axis()`` methods) populates the recipe dict in place.

    The dict is consumed by ``.build()`` (returns a deep copy) or
    ``.run(output_directory)`` (forwards to ``macrocast.run``).
    """

    def __init__(self) -> None:
        self._recipe: dict[str, Any] = {}
        self.l0 = _L0(self._recipe)
        self.l1 = _L1(self._recipe)
        self.l2 = _L2(self._recipe)
        self.l3 = _L3(self._recipe)
        self.l4 = _L4(self._recipe)
        self.l5 = _L5(self._recipe)
        self.l6 = _L6(self._recipe)
        self.l7 = _L7(self._recipe)
        self.l8 = _L8(self._recipe)

    def build(self) -> dict[str, Any]:
        """Return a deep copy of the current recipe dict."""

        return copy.deepcopy(self._recipe)

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Render the recipe as YAML. When ``path`` is supplied also write
        it to disk."""

        try:
            import yaml as _yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - PyYAML is required
            raise RuntimeError("PyYAML is required to render recipes") from exc
        text = _yaml.safe_dump(self._recipe, sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def run(self, output_directory: str | Path, **kwargs: Any) -> Any:
        """Build + run. Forwards to ``macrocast.run``; returns the
        ``ManifestExecutionResult``."""

        from .. import api

        recipe = self.build()
        return api.run(recipe, output_directory=output_directory, **kwargs)

    def validate(self) -> list[str]:
        """Run each layer's ``validate_layer`` on the partial recipe and
        return a list of human-readable error messages. Empty list = OK.

        Skips layers absent from the recipe so partial drafts can be
        validated as you build them.
        """

        from importlib import import_module

        errors: list[str] = []
        layer_validators = (
            ("0_meta", "macrocast.core.layers.l0"),
            ("1_data", "macrocast.core.layers.l1"),
            ("2_preprocessing", "macrocast.core.layers.l2"),
            ("3_feature_engineering", "macrocast.core.layers.l3"),
            ("4_forecasting_model", "macrocast.core.layers.l4"),
            ("5_evaluation", "macrocast.core.layers.l5"),
            ("6_statistical_tests", "macrocast.core.layers.l6"),
            ("7_interpretation", "macrocast.core.layers.l7"),
            ("8_output", "macrocast.core.layers.l8"),
        )
        for key, module_name in layer_validators:
            block = self._recipe.get(key)
            if not block:
                continue
            module = import_module(module_name)
            validator = getattr(module, "validate_layer", None)
            if validator is None:
                continue
            try:
                report = validator(block)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{key}: validator raised: {exc}")
                continue
            for issue in getattr(report, "hard_errors", ()):
                errors.append(f"{key}: {issue.message}")
        return errors


__all__ = ["RecipeBuilder"]
