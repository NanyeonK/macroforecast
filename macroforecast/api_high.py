"""High-level public API: ``mf.forecast`` + ``mf.Experiment`` + ``ForecastResult``.

This module is the **simple façade** for macroforecast: a thin layer over the
canonical recipe / execution engine that lets researchers write a forecasting
study in a few lines of Python without authoring YAML.

The v0.8 series ships in two PRs:

* PR 1 / v0.8.0 -- :func:`forecast`, :class:`Experiment` core
  (``compare_models`` / ``compare`` / ``sweep`` / ``run`` / ``replicate`` /
  ``to_yaml`` / ``validate``), :class:`ForecastResult` minimal shell.
* PR 2 / v0.8.5 -- :class:`Experiment` ``.use_*`` hooks (FRED-SD selection,
  state / variable groups, mixed-frequency representation, SD t-code
  policies, custom preprocessor), :class:`Experiment.variant`, rich
  :class:`ForecastResult` accessors (``forecasts`` / ``metrics`` /
  ``ranking`` / ``read_json`` / ``file_path`` / ``mean`` / ``get``), plus
  two new schema axes (``mixed_frequency_representation`` and
  ``sd_tcode_policy``).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

from .core.execution import (
    CellExecutionResult,
    ManifestExecutionResult,
    ReplicationResult,
    execute_recipe,
    replicate_recipe,
)
from .scaffold.builder import RecipeBuilder
from .defaults import (
    DEFAULT_HORIZONS,
    DEFAULT_MODEL_FAMILY,
    DEFAULT_RANDOM_SEED,
)

__all__ = ["Experiment", "ForecastResult", "forecast"]

# v0.8.5: option lists shared with the L1 / L2 schema. Kept in sync with
# ``macroforecast.core.layers.l1`` (state / variable groups) and ``l2``
# (mixed_frequency_representation). The .use_* validators consult these
# tuples so a typo on the user side raises a clear ValueError.
_FRED_SD_STATE_GROUP_OPTIONS: tuple[str, ...] = (
    "all_states",
    "census_region_northeast",
    "census_region_midwest",
    "census_region_south",
    "census_region_west",
    "census_division_new_england",
    "census_division_middle_atlantic",
    "census_division_east_north_central",
    "census_division_west_north_central",
    "census_division_south_atlantic",
    "census_division_east_south_central",
    "census_division_west_south_central",
    "census_division_mountain",
    "census_division_pacific",
    "contiguous_48_plus_dc",
    "custom_state_group",
)

_FRED_SD_VARIABLE_GROUP_OPTIONS: tuple[str, ...] = (
    "all_sd_variables",
    "labor_market_core",
    "employment_sector",
    "gsp_output",
    "housing",
    "trade",
    "income",
    "direct_analog_high_confidence",
    "provisional_analog_medium",
    "semantic_review_outputs",
    "no_reliable_analog",
    "custom_sd_variable_group",
)

_MIXED_FREQUENCY_REPRESENTATION_OPTIONS: tuple[str, ...] = (
    "calendar_aligned_frame",
    "drop_unknown_native_frequency",
    "drop_non_target_native_frequency",
    "native_frequency_block_payload",
    "mixed_frequency_model_adapter",
)

_SD_TCODE_UNIT_OPTIONS: tuple[str, ...] = ("variable_global", "state_series")


# ---------------------------------------------------------------------------
# Recipe assembly helpers
# ---------------------------------------------------------------------------

# L1 frequency that each known dataset implies; ``None`` means the user must
# supply ``frequency=`` explicitly (currently only ``fred_sd`` alone).
_DATASET_FREQUENCY: dict[str, str | None] = {
    "fred_md": "monthly",
    "fred_qd": "quarterly",
    "fred_sd": None,
    "fred_md+fred_sd": "monthly",
    "fred_qd+fred_sd": "quarterly",
}


def _resolve_frequency(dataset: str, frequency: str | None) -> str:
    """Pick the L1 frequency for ``dataset``; raise if ambiguous."""

    if dataset in _DATASET_FREQUENCY:
        implied = _DATASET_FREQUENCY[dataset]
        if implied is None:
            if frequency is None:
                raise ValueError(
                    f"dataset={dataset!r} requires an explicit `frequency=` "
                    "(it has no canonical sampling rate)"
                )
            return frequency
        if frequency is not None and frequency != implied:
            raise ValueError(
                f"dataset={dataset!r} fixes frequency={implied!r}; "
                f"got frequency={frequency!r}"
            )
        return implied
    # Unknown dataset name -- accept and require an explicit frequency.
    if frequency is None:
        raise ValueError(
            f"dataset={dataset!r} is not a known FRED panel; pass `frequency=`"
        )
    return frequency


def _build_default_recipe(
    *,
    dataset: str,
    target: str,
    horizons: Sequence[int],
    frequency: str | None,
    start: str | None,
    end: str | None,
    model_family: str,
    random_seed: int,
) -> RecipeBuilder:
    """Construct the v0.8.0 minimal default recipe via ``RecipeBuilder``.

    L0 ``fail_fast`` + ``seeded_reproducible`` + ``serial``; L1 with the
    requested dataset + target + horizons + sample window; L2 no-transform
    pass-through (the default profile already routes through the real
    McCracken-Ng pipeline when L2 is set ``standard()``, but PR 1's contract
    is "minimal viable"); L3 ``lag_only(n_lag=1)`` + ``target_construction``;
    L4 a single fit_model node with the requested family; L5 ``standard``.
    """

    horizons_list = [int(h) for h in horizons]
    if not horizons_list:
        raise ValueError("horizons must contain at least one positive integer")

    resolved_frequency = _resolve_frequency(dataset, frequency)

    b = RecipeBuilder()
    b.l0(random_seed=int(random_seed))
    # L1 -- official source path.
    b.l1(
        custom_source_policy="official_only",
        dataset=dataset,
        frequency=resolved_frequency,
        horizon_set="custom_list",
        target_structure="single_target",
        target=target,
        target_horizons=horizons_list,
    )
    if start is not None:
        b.l1.set_axis(sample_start_rule="fixed_date")
        b.l1.set_leaf(sample_start_date=start)
    if end is not None:
        b.l1.set_axis(sample_end_rule="fixed_date")
        b.l1.set_leaf(sample_end_date=end)
    # L2 -- no-transform pass-through.
    b.l2.no_op()
    # L3 -- minimal lag1 + target construction.
    b.l3.lag_only(n_lag=1)
    # L4 -- single fit_model node + predict.
    b.l4.fit(model_family).is_benchmark()
    # L5 -- mse primary metric.
    b.l5.standard(primary_metric="mse")
    return b


# ---------------------------------------------------------------------------
# forecast()
# ---------------------------------------------------------------------------


def forecast(
    dataset: str,
    target: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    *,
    frequency: str | None = None,
    start: str | None = None,
    end: str | None = None,
    model_family: str = DEFAULT_MODEL_FAMILY,
    output_directory: str | Path | None = None,
    cache_root: str | Path | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> "ForecastResult":
    """Run one default macroeconomic forecasting study.

    This is the simplest possible entry point into macroforecast: pick a
    dataset, a target series, the forecast horizons, and (for FRED-SD)
    sample window, and the function assembles the canonical default recipe,
    executes it through :func:`macroforecast.run`, and returns a
    :class:`ForecastResult`.

    Parameters
    ----------
    dataset
        FRED source panel: ``"fred_md"``, ``"fred_qd"``, ``"fred_sd"``,
        ``"fred_md+fred_sd"``, ``"fred_qd+fred_sd"``.
    target
        Target series identifier (e.g. ``"INDPRO"``, ``"CPIAUCSL"``).
    horizons
        Forecast horizons. Defaults to ``(1,)``.
    frequency
        Required when ``dataset="fred_sd"`` alone; ignored otherwise.
    start, end
        ISO sample window endpoints (e.g. ``"1985-01"``); written into L1
        ``sample_start_date`` / ``sample_end_date``.
    model_family
        L4 ``fit_model`` family. Defaults to ``"ar_p"`` (see ``macroforecast.defaults.DEFAULT_MODEL_FAMILY``).
    output_directory
        Directory to write ``manifest.json`` and per-cell artifacts.
    cache_root
        Shared raw-data cache root; forwarded to :func:`execute_recipe`.
    random_seed
        L0 ``random_seed`` (default ``42``, see ``macroforecast.defaults.DEFAULT_RANDOM_SEED``).

    Returns
    -------
    ForecastResult
        Wraps the underlying :class:`ManifestExecutionResult`.
    """

    builder = _build_default_recipe(
        dataset=dataset,
        target=target,
        horizons=horizons,
        frequency=frequency,
        start=start,
        end=end,
        model_family=model_family,
        random_seed=random_seed,
    )
    # v0.8.6 Gap 2: normalize the lone L4 fit node id for the one-shot
    # path too -- there is no chained .compare() follow-up here, but we
    # keep the id stable so ``forecast()`` and ``Experiment().run()``
    # produce identical recipe blocks for the same inputs.
    _normalize_fit_main_id(builder._recipe)
    recipe = builder.build()
    manifest = execute_recipe(
        recipe,
        output_directory=output_directory,
        cache_root=cache_root,
    )
    out_dir = Path(output_directory) if output_directory is not None else None
    return ForecastResult(manifest=manifest, output_directory=out_dir)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def _set_at(root: dict[str, Any], dotted: str, value: Any) -> None:
    """Walk ``dotted`` into ``root`` and set the leaf to ``value``.

    Behaviour:

    * The empty string is rejected with :class:`ValueError`.
    * Each step descends into ``dict`` keys; the dotted segment is matched
      verbatim against the key (no integer-index parsing -- L3/L4 ``nodes``
      lists are addressed by ``id``-keyed sub-dict at the spec level, not
      by raw list index).
    * If a segment names an existing list, we walk through entries and
      match by the ``id`` field (so ``4_forecasting_model.nodes.fit_model``
      finds the fit node regardless of position).
    * Missing intermediate dict keys are auto-created. A non-dict /
      non-list collision (e.g. trying to traverse through a leaf integer)
      raises :class:`ValueError` with the failing prefix.
    * If the final leaf already carries a ``{"sweep": [...]}`` marker we
      replace it (callers that want to extend rather than replace can
      read first via ``Experiment.to_recipe_dict()``).
    """

    if not dotted:
        raise ValueError("axis_path must be a non-empty dotted path")
    parts = dotted.split(".")
    cursor: Any = root
    walked: list[str] = []
    for part in parts[:-1]:
        walked.append(part)
        if isinstance(cursor, dict):
            nxt = cursor.get(part)
            if nxt is None:
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        elif isinstance(cursor, list):
            # Match by node id within an l3/l4-style nodes list.
            match = next(
                (entry for entry in cursor if isinstance(entry, dict) and entry.get("id") == part),
                None,
            )
            if match is None:
                raise ValueError(
                    f"axis_path {dotted!r}: no list entry with id={part!r} at "
                    f"{'.'.join(walked[:-1]) or '<root>'}"
                )
            cursor = match
        else:
            raise ValueError(
                f"axis_path {dotted!r}: cannot descend into non-collection at "
                f"{'.'.join(walked[:-1]) or '<root>'}"
            )
    final = parts[-1]
    if isinstance(cursor, dict):
        cursor[final] = value
    elif isinstance(cursor, list):
        match = next(
            (entry for entry in cursor if isinstance(entry, dict) and entry.get("id") == final),
            None,
        )
        if match is None:
            raise ValueError(
                f"axis_path {dotted!r}: no list entry with id={final!r}"
            )
        match.clear()
        if isinstance(value, dict):
            match.update(value)
        else:
            raise ValueError(
                f"axis_path {dotted!r}: cannot replace list entry id={final!r} "
                "with a non-dict value"
            )
    else:
        raise ValueError(
            f"axis_path {dotted!r}: leaf parent is not a dict/list"
        )


def _normalize_fit_main_id(recipe: dict[str, Any]) -> None:
    """Rename the lone L4 fit_model node to the stable id ``fit_main``.

    v0.8.6 Gap 2: ``RecipeBuilder.l4.fit(family)`` auto-names the node
    ``fit_<n>_<family>``. That id leaks the family name (and the fit
    counter), which makes chained ``.compare(...)`` follow-ups brittle:
    after ``compare_models([...])`` the auto-name still reflects the
    *first* family the user passed to ``Experiment(model_family=...)``,
    so users had to memorise the auto-generated id (``fit_1_ridge`` etc.).

    This helper walks the L4 nodes and renames the lone fit_model node
    to ``fit_main``. All references in the ``predict`` step inputs and
    in the L4 ``sinks`` (``l4_forecasts_v1`` / ``l4_model_artifacts_v1``)
    are updated atomically.

    Edge cases:

    * If the node id is already ``fit_main`` -- no-op.
    * If the L4 block already contains a node literally named
      ``fit_main`` *and* a different fit_model node -- skip (the user
      hand-rolled this layout via ``to_recipe_dict()`` round-trip).
    * If there are multiple fit_model nodes (ensemble / horse-race
      authored manually) -- skip; the user must use the explicit ids.
    """

    l4 = recipe.get("4_forecasting_model")
    if not isinstance(l4, dict):
        return
    nodes = l4.get("nodes")
    if not isinstance(nodes, list):
        return
    fit_nodes = [n for n in nodes if isinstance(n, dict) and n.get("op") == "fit_model"]
    if len(fit_nodes) != 1:
        return
    fit_node = fit_nodes[0]
    old_id = fit_node.get("id")
    if old_id == "fit_main":
        return
    # Guard: don't collide with a pre-existing "fit_main" node.
    if any(isinstance(n, dict) and n.get("id") == "fit_main" for n in nodes):
        return
    fit_node["id"] = "fit_main"
    # Update predict node inputs that reference the old id.
    for n in nodes:
        if not isinstance(n, dict):
            continue
        inputs = n.get("inputs")
        if isinstance(inputs, list):
            n["inputs"] = ["fit_main" if x == old_id else x for x in inputs]
    # Update sinks that reference the old id.
    sinks = l4.get("sinks")
    if isinstance(sinks, dict):
        for key, value in list(sinks.items()):
            if value == old_id:
                sinks[key] = "fit_main"


class Experiment:
    """Builder for one forecasting study.

    Workflow::

        exp = mf.Experiment(dataset="fred_md", target="INDPRO",
                            start="1980-01", end="2019-12", horizons=[1, 3])
        exp.compare_models(["ridge", "lasso"])
        result = exp.run(output_directory="out/")

    The simple sweep methods all return ``self`` so chains are legal::

        exp = (
            mf.Experiment(...)
            .compare_models(["ar_p", "ridge"])
            .compare("4_forecasting_model.nodes.fit_main.params.alpha",
                     [0.1, 1.0])
        )

    The L4 fit node is normalized to the stable id ``fit_main`` whenever
    there is exactly one fit node (default Experiment construction and
    ``compare_models([...])``); chained ``.compare(...)`` follow-ups can
    therefore address it via the predictable dotted path
    ``4_forecasting_model.nodes.fit_main....`` instead of the
    auto-generated ``fit_1_<family>``.

    PR 1 implements the basic constructor + sweep methods + run / to_yaml /
    replicate / validate. ``.use_fred_sd_inferred_tcodes()``,
    ``.use_sd_empirical_tcodes()``, ``.use_preprocessor()`` and friends are
    deferred to v0.8.1 (PR 2).
    """

    def __init__(
        self,
        dataset: str,
        target: str,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        *,
        frequency: str | None = None,
        start: str | None = None,
        end: str | None = None,
        model_family: str = DEFAULT_MODEL_FAMILY,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        self._dataset = dataset
        self._target = target
        self._horizons = [int(h) for h in horizons]
        self._frequency = frequency
        self._start = start
        self._end = end
        self._random_seed = int(random_seed)
        self._model_family = model_family

        # TODO(v0.8.1 / PR 2): track ``.use_*`` hooks here so the recipe
        # builder can apply them just before .run() / .to_yaml().

        self._builder = _build_default_recipe(
            dataset=dataset,
            target=target,
            horizons=self._horizons,
            frequency=frequency,
            start=start,
            end=end,
            model_family=model_family,
            random_seed=self._random_seed,
        )
        # v0.8.6 Gap 2: normalize the lone L4 fit node id to ``fit_main``
        # so chained ``.compare(...)`` follow-ups can use a stable dotted
        # path instead of the family-specific auto-name.
        _normalize_fit_main_id(self._builder._recipe)

    # -- sweep methods -----------------------------------------------------

    def compare_models(self, families: Sequence[str]) -> "Experiment":
        """Sweep the L4 fit_model family.

        Idiomatic shortcut around
        ``.sweep("4_forecasting_model.nodes.<fit_id>.params.family", [...])``
        -- locates the existing fit node and replaces its family with a
        ``{"sweep": [...]}`` marker.
        """

        families_list = list(families)
        if not families_list:
            raise ValueError("compare_models requires at least one family")
        l4 = self._builder._recipe.setdefault("4_forecasting_model", {})
        nodes = l4.setdefault("nodes", [])
        fit_nodes = [n for n in nodes if isinstance(n, dict) and n.get("op") == "fit_model"]
        if not fit_nodes:
            raise RuntimeError(
                "compare_models() called before any L4 fit node exists "
                "(did you instantiate Experiment without a model_family?)"
            )
        # In PR 1 we always have exactly one fit_model node. Sweep its family.
        fit_node = fit_nodes[0]
        fit_node.setdefault("params", {})["family"] = {"sweep": list(families_list)}
        # v0.8.6 Gap 2: normalize the swept fit node to ``fit_main`` so
        # ``.compare("4_forecasting_model.nodes.fit_main.params.alpha", ...)``
        # follow-ups have a stable dotted path independent of the
        # auto-generated ``fit_<n>_<initial_family>`` name.
        _normalize_fit_main_id(self._builder._recipe)
        return self

    def compare(self, axis_path: str, values: Sequence[Any]) -> "Experiment":
        """Sweep ``axis_path`` over ``values``.

        ``axis_path`` is a dotted path into the recipe dict, e.g.
        ``"4_forecasting_model.nodes.fit_1_ridge.params.alpha"``. The leaf
        is replaced with ``{"sweep": list(values)}``.
        """

        values_list = list(values)
        if not values_list:
            raise ValueError("compare() requires at least one value")
        _set_at(self._builder._recipe, axis_path, {"sweep": values_list})
        return self

    def sweep(self, axis_path: str, values: Sequence[Any]) -> "Experiment":
        """Alias for :meth:`compare`."""

        return self.compare(axis_path, values)

    def variant(self, name: str, **overrides: Any) -> "Experiment":
        """Branch a named recipe variant.

        Each :meth:`variant` call records ``overrides`` under
        ``recipe_root["variants"][name]``. At execution time the cell loop
        treats the variants block as an extra dimension (one cell per
        variant) and combines it with any explicit ``compare_models`` /
        ``compare`` / ``sweep`` axes via the configured
        ``sweep_combination.mode`` (default grid).

        ``overrides`` may use either dotted ``axis_path=value`` keys or
        the convenience aliases ``model`` / ``model_family``
        (mapped to the L4 fit_node family).

        Returns ``self`` so calls chain. Calling ``.variant`` twice with
        the same name overwrites the previous record.
        """

        if not name or "=" in name or "/" in name:
            raise ValueError(
                f"variant name must be a non-empty token without '=' or '/'; got {name!r}"
            )
        recipe = self._builder._recipe
        variants = recipe.setdefault("variants", {})
        record: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in {"model", "model_family"}:
                record["model_family"] = value
            else:
                record[key] = value
        variants[name] = record
        return self

    # -- .use_* hooks (v0.8.5) --------------------------------------------

    def use_fred_sd_selection(
        self,
        states: Sequence[str] | None = None,
        variables: Sequence[str] | None = None,
    ) -> "Experiment":
        """Restrict the FRED-SD component to specific states / variables.

        Sets the L1.D ``state_selection`` / ``sd_variable_selection`` axes
        to ``selected_states`` / ``selected_sd_variables`` and writes the
        actual lists into the L1 ``leaf_config``.

        Either or both of ``states`` and ``variables`` may be supplied;
        omitted arguments leave the corresponding axis unchanged. Returns
        ``self``.
        """

        if states is not None:
            self._builder.l1.set_axis(state_selection="selected_states")
            self._builder.l1.set_leaf(selected_states=list(states))
        if variables is not None:
            self._builder.l1.set_axis(sd_variable_selection="selected_sd_variables")
            self._builder.l1.set_leaf(selected_sd_variables=list(variables))
        return self

    def use_fred_sd_state_group(self, group: str) -> "Experiment":
        """Pick a FRED-SD state grouping (16 axis options).

        Sets the L1.D ``fred_sd_state_group`` axis. Validates ``group``
        against the published axis options and raises ``ValueError`` on a
        bad value. Returns ``self``.
        """

        if group not in _FRED_SD_STATE_GROUP_OPTIONS:
            raise ValueError(
                f"fred_sd_state_group={group!r} is not a known option; "
                f"choose from {sorted(_FRED_SD_STATE_GROUP_OPTIONS)}"
            )
        self._builder.l1.set_axis(fred_sd_state_group=group)
        return self

    def use_fred_sd_variable_group(self, group: str) -> "Experiment":
        """Pick a FRED-SD variable grouping (12 axis options).

        Sets the L1.D ``fred_sd_variable_group`` axis. Validates
        ``group`` and raises ``ValueError`` on a bad value. Returns
        ``self``.
        """

        if group not in _FRED_SD_VARIABLE_GROUP_OPTIONS:
            raise ValueError(
                f"fred_sd_variable_group={group!r} is not a known option; "
                f"choose from {sorted(_FRED_SD_VARIABLE_GROUP_OPTIONS)}"
            )
        self._builder.l1.set_axis(fred_sd_variable_group=group)
        return self

    def use_mixed_frequency_representation(self, mode: str) -> "Experiment":
        """Choose how mixed-frequency columns are rendered into L2.

        Sets the L2.A ``mixed_frequency_representation`` axis (added in
        v0.8.5). Five options:

        * ``calendar_aligned_frame`` (default)
        * ``drop_unknown_native_frequency``
        * ``drop_non_target_native_frequency``
        * ``native_frequency_block_payload``
        * ``mixed_frequency_model_adapter``

        Raises ``ValueError`` on a bad value. Returns ``self``.
        """

        if mode not in _MIXED_FREQUENCY_REPRESENTATION_OPTIONS:
            raise ValueError(
                f"mixed_frequency_representation={mode!r} is not valid; "
                f"choose from {sorted(_MIXED_FREQUENCY_REPRESENTATION_OPTIONS)}"
            )
        self._builder.l2.set_axis(mixed_frequency_representation=mode)
        return self

    def use_sd_inferred_tcodes(self) -> "Experiment":
        """Opt into the inferred (national-analog) FRED-SD t-code map.

        Sets the L2.B ``sd_tcode_policy`` axis to ``inferred``. Returns
        ``self``.
        """

        self._builder.l2.set_axis(sd_tcode_policy="inferred")
        return self

    def use_sd_empirical_tcodes(
        self,
        unit: str,
        code_map: dict[str, int] | None = None,
        audit_uri: str | None = None,
    ) -> "Experiment":
        """Opt into the empirical (stationarity-audit) FRED-SD t-code map.

        Sets the L2.B ``sd_tcode_policy`` axis to ``empirical`` and
        writes the supporting leaf_config keys:

        * ``sd_tcode_unit`` -- ``variable_global`` or ``state_series``
        * ``sd_tcode_code_map`` -- per-(variable, state) map; required
          when ``unit='state_series'``
        * ``sd_tcode_audit_uri`` -- pointer to the audit artifact

        Raises ``ValueError`` when ``unit`` is unknown, or when
        ``unit='state_series'`` but ``code_map`` is empty / missing.
        Returns ``self``.
        """

        if unit not in _SD_TCODE_UNIT_OPTIONS:
            raise ValueError(
                f"sd_tcode_unit={unit!r} is not valid; "
                f"choose from {sorted(_SD_TCODE_UNIT_OPTIONS)}"
            )
        if unit == "state_series" and not code_map:
            raise ValueError(
                "use_sd_empirical_tcodes(unit='state_series') requires a "
                "non-empty code_map={'<VAR>_<STATE>': <tcode int>}"
            )
        self._builder.l2.set_axis(sd_tcode_policy="empirical")
        leaf: dict[str, Any] = {"sd_tcode_unit": unit}
        if code_map is not None:
            leaf["sd_tcode_code_map"] = dict(code_map)
        if audit_uri is not None:
            leaf["sd_tcode_audit_uri"] = audit_uri
        self._builder.l2.set_leaf(**leaf)
        return self

    def use_preprocessor(self, name: str, applied_at: str = "l3") -> "Experiment":
        """Inject a registered custom preprocessor into the pipeline.

        ``name`` must already be registered via
        :func:`macroforecast.custom.register_preprocessor` (or the
        ``@mf.custom_preprocessor`` decorator). Two dispatch points:

        * ``applied_at='l3'`` (default) -- v0.2.5 PR #251 post-pipeline
          hook. The callable receives the L2 clean panel *after* the
          McCracken-Ng pipeline (transform / outlier / impute /
          frame_edge) has run; its output becomes the L2 clean panel
          that L3 reads. Useful for "the cleaned panel, plus my one
          extra step" workflows.
        * ``applied_at='l2'`` (v0.8.6 Gap 1) -- pre-pipeline hook. The
          callable receives the *raw* L1 panel and returns a panel that
          the canonical L2 pipeline then transforms / cleans. Useful
          for upstream cleanup (drop bad columns, deflation,
          normalisation, custom resampling) before the official t-codes
          apply.

        Both hooks dispatch to ``macroforecast.custom`` registrations
        through the same contract; only the timing within L2 differs.
        Routing the same name to both points is allowed and emits two
        distinct cleaning_log entries.

        Returns ``self``.
        """

        if applied_at == "l3":
            # v0.2.5 #251 wired this on L2.leaf_config.custom_postprocessor.
            self._builder.l2.set_leaf(custom_postprocessor=str(name))
            return self
        if applied_at == "l2":
            # v0.8.6 Gap 1: pre-pipeline hook on
            # L2.leaf_config.custom_preprocessor.
            self._builder.l2.set_leaf(custom_preprocessor=str(name))
            return self
        raise ValueError(
            f"applied_at={applied_at!r} is not valid; choose from "
            "{'l2', 'l3'} (l2 = pre-pipeline raw-panel hook, "
            "l3 = post-pipeline clean-panel hook)"
        )

    # -- inspection / serialization ---------------------------------------

    def to_recipe_dict(self) -> dict[str, Any]:
        """Return a deep copy of the in-progress recipe dict."""

        return self._builder.build()

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Render the in-progress recipe as YAML; optionally write to ``path``."""

        return self._builder.to_yaml(path)

    def validate(self) -> None:
        """Run each layer's validator over the in-progress recipe.

        Raises :class:`ValueError` if any layer reports a hard error.
        """

        errors = self._builder.validate()
        if errors:
            raise ValueError(
                "Experiment recipe failed validation:\n  - "
                + "\n  - ".join(errors)
            )

    # -- execution ---------------------------------------------------------

    def run(
        self,
        output_directory: str | Path | None = None,
        *,
        cache_root: str | Path | None = None,
    ) -> "ForecastResult":
        """Execute the recipe and return a :class:`ForecastResult`."""

        recipe = self.to_recipe_dict()
        manifest = execute_recipe(
            recipe,
            output_directory=output_directory,
            cache_root=cache_root,
        )
        out_dir = Path(output_directory) if output_directory is not None else None
        return ForecastResult(manifest=manifest, output_directory=out_dir)

    def replicate(self, manifest_path: str | Path) -> ReplicationResult:
        """Re-execute a previously-saved manifest and verify hashes."""

        return replicate_recipe(manifest_path)


# ---------------------------------------------------------------------------
# ForecastResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastResult:
    """Façade over :class:`ManifestExecutionResult`.

    Wraps the underlying execution manifest and exposes rich research-time
    accessors:

    * :attr:`forecasts` -- aggregated predictions DataFrame
    * :attr:`metrics` -- aggregated L5 metrics table
    * :attr:`ranking` -- aggregated L5 ranking table
    * :meth:`mean` -- per-(model, target, horizon) average of one metric
    * :meth:`get` -- pull one cell out by ``cell_id``
    * :meth:`read_json` / :meth:`file_path` -- per-cell artifact access

    The minimal shell from v0.8.0 (``cells`` / ``succeeded`` /
    ``manifest_path`` / ``replicate``) is preserved.
    """

    manifest: ManifestExecutionResult
    output_directory: Path | None = None

    @property
    def cells(self) -> tuple[CellExecutionResult, ...]:
        return self.manifest.cells

    @property
    def succeeded(self) -> tuple[CellExecutionResult, ...]:
        return self.manifest.succeeded

    @property
    def manifest_path(self) -> Path | None:
        """Path to the on-disk manifest, or ``None`` if not written."""

        if self.output_directory is None:
            return None
        json_path = self.output_directory / "manifest.json"
        if json_path.exists():
            return json_path
        jsonl_path = self.output_directory / "manifest.jsonl"
        if jsonl_path.exists():
            return jsonl_path
        yaml_path = self.output_directory / "manifest.yaml"
        if yaml_path.exists():
            return yaml_path
        return None

    def replicate(self) -> ReplicationResult:
        """Replicate the on-disk manifest produced by this run."""

        path = self.manifest_path
        if path is None:
            raise RuntimeError(
                "ForecastResult.replicate() requires a manifest written to "
                "disk; run with output_directory=..."
            )
        return replicate_recipe(path)

    # -- rich accessors (v0.8.5) ------------------------------------------

    @property
    def forecasts(self) -> "pd.DataFrame":
        """Concatenate per-cell ``l4_forecasts_v1`` rows.

        Columns: ``cell_id, model_id, target, horizon, origin, y_pred,
        y_pred_lo, y_pred_hi``. The two interval columns are NaN when the
        L4 forecast object is point-only.

        Returns an empty DataFrame (with the canonical columns) when no
        cells produced forecasts.
        """

        import pandas as pd

        rows: list[dict[str, Any]] = []
        for cell in self.cells:
            artifact = self._cell_artifact(cell, "l4_forecasts_v1")
            if artifact is None:
                continue
            forecasts = getattr(artifact, "forecasts", None) or {}
            intervals = getattr(artifact, "forecast_intervals", None) or {}
            # interval_lookup: (model, target, horizon, origin) -> (lo, hi)
            interval_lookup: dict[tuple[Any, Any, Any, Any], tuple[float, float]] = {}
            for key, value in intervals.items():
                if not isinstance(key, tuple) or len(key) != 5:
                    continue
                model_id, target, horizon, origin, quantile = key
                slot = interval_lookup.get((model_id, target, horizon, origin), (float("nan"), float("nan")))
                lo, hi = slot
                try:
                    qf = float(quantile)
                except (TypeError, ValueError):
                    continue
                if qf <= 0.5:
                    lo = float(value)
                else:
                    hi = float(value)
                interval_lookup[(model_id, target, horizon, origin)] = (lo, hi)
            for key, value in forecasts.items():
                if not isinstance(key, tuple) or len(key) != 4:
                    continue
                model_id, target, horizon, origin = key
                lo, hi = interval_lookup.get((model_id, target, horizon, origin), (float("nan"), float("nan")))
                rows.append({
                    "cell_id": cell.cell_id,
                    "model_id": model_id,
                    "target": target,
                    "horizon": horizon,
                    "origin": origin,
                    "y_pred": value,
                    "y_pred_lo": lo,
                    "y_pred_hi": hi,
                })
        if not rows:
            return pd.DataFrame(
                columns=[
                    "cell_id", "model_id", "target", "horizon", "origin",
                    "y_pred", "y_pred_lo", "y_pred_hi",
                ]
            )
        return pd.DataFrame(rows)

    @property
    def metrics(self) -> "pd.DataFrame":
        """Concatenate per-cell ``l5_evaluation_v1.metrics_table``.

        Adds a ``cell_id`` column to identify the source cell. Returns an
        empty DataFrame when no cells produced an L5 evaluation artifact.
        """

        import pandas as pd

        frames: list[pd.DataFrame] = []
        for cell in self.cells:
            artifact = self._cell_artifact(cell, "l5_evaluation_v1")
            if artifact is None:
                continue
            table = getattr(artifact, "metrics_table", None)
            if table is None or not isinstance(table, pd.DataFrame) or table.empty:
                continue
            with_cell = table.copy()
            with_cell.insert(0, "cell_id", cell.cell_id)
            frames.append(with_cell)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @property
    def ranking(self) -> "pd.DataFrame":
        """Concatenate per-cell ``l5_evaluation_v1.ranking_table``.

        Adds a ``cell_id`` column. Returns an empty DataFrame when no
        cell emitted a ranking table.
        """

        import pandas as pd

        frames: list[pd.DataFrame] = []
        for cell in self.cells:
            artifact = self._cell_artifact(cell, "l5_evaluation_v1")
            if artifact is None:
                continue
            table = getattr(artifact, "ranking_table", None)
            if table is None or not isinstance(table, pd.DataFrame) or table.empty:
                continue
            with_cell = table.copy()
            with_cell.insert(0, "cell_id", cell.cell_id)
            frames.append(with_cell)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def mean(self, metric: str = "mse") -> "pd.DataFrame":
        """Per-(model, target, horizon) mean of one metric across cells.

        Convenience one-liner around :attr:`metrics`; useful for
        horse-race summaries.
        """

        import pandas as pd

        table = self.metrics
        if table.empty or metric not in table.columns:
            return pd.DataFrame(
                columns=["model_id", "target", "horizon", metric]
            )
        group_cols = [
            col for col in ("model_id", "target", "horizon") if col in table.columns
        ]
        if not group_cols:
            return pd.DataFrame({metric: [table[metric].mean()]})
        return (
            table.groupby(group_cols, dropna=False)[metric]
            .mean()
            .reset_index()
        )

    def get(self, cell_id: str) -> CellExecutionResult:
        """Look one cell up by id; raise ``KeyError`` if missing."""

        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(f"no cell with cell_id={cell_id!r}")

    def read_json(self, name: str) -> dict[str, Any]:
        """Read a JSON artifact from any per-cell directory.

        Searches every ``output_directory/<cell_id>/`` directory for
        ``name`` (e.g. ``read_json('provenance.json')``) and returns the
        parsed dict from the first match. Raises ``RuntimeError`` if
        ``output_directory`` was not set, and ``FileNotFoundError`` if
        no cell carries the file.
        """

        import json

        if self.output_directory is None:
            raise RuntimeError(
                "ForecastResult.read_json() requires output_directory=..."
            )
        for cell in self.cells:
            candidate = self.output_directory / cell.cell_id / name
            if candidate.exists():
                return json.loads(candidate.read_text(encoding="utf-8"))
        # fall back to the manifest root
        candidate = self.output_directory / name
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
        raise FileNotFoundError(
            f"no artifact named {name!r} in any per-cell directory under {self.output_directory}"
        )

    def file_path(self, name: str) -> Path | None:
        """Locate a file ``name`` under any per-cell directory.

        Returns the first match (or ``None`` if no cell holds it).
        Raises ``RuntimeError`` when ``output_directory`` is None.
        """

        if self.output_directory is None:
            raise RuntimeError(
                "ForecastResult.file_path() requires output_directory=..."
            )
        for cell in self.cells:
            candidate = self.output_directory / cell.cell_id / name
            if candidate.exists():
                return candidate
        candidate = self.output_directory / name
        if candidate.exists():
            return candidate
        return None

    @staticmethod
    def _cell_artifact(cell: CellExecutionResult, sink_name: str) -> Any:
        """Pull ``sink_name`` from ``cell.runtime_result.artifacts``.

        Returns ``None`` when the cell failed or the sink is missing.
        """

        rt = cell.runtime_result
        if rt is None:
            return None
        artifacts = getattr(rt, "artifacts", None) or {}
        return artifacts.get(sink_name)
