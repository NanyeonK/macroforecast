"""High-level public API: ``mf.forecast`` + ``mf.Experiment`` + ``ForecastResult``.

This module is the **simple façade** for macroforecast: a thin layer over the
canonical recipe / execution engine that lets researchers write a forecasting
study in a few lines of Python without authoring YAML.

PR 1 of the v0.8 series ships:

* :func:`forecast` -- one-shot forecasting helper
* :class:`Experiment` -- builder class with ``compare_models`` / ``compare`` /
  ``sweep`` / ``variant`` / ``run`` / ``replicate`` / ``to_yaml`` / ``validate``
* :class:`ForecastResult` -- minimal shell wrapping :class:`ManifestExecutionResult`

PR 2 (v0.8.1) will add ``Experiment.use_*`` hooks (FRED-SD t-codes,
preprocessor injection) and richer ``ForecastResult`` accessors
(``.forecasts``, ``.metrics``, ``.ranking``, ``.read_json(...)``,
``.file_path(...)``, ``.mean()``, ``.get(...)``).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .core.execution import (
    CellExecutionResult,
    ManifestExecutionResult,
    ReplicationResult,
    execute_recipe,
    replicate_recipe,
)
from .scaffold.builder import RecipeBuilder

__all__ = ["Experiment", "ForecastResult", "forecast"]


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
    horizons: Sequence[int] = (1,),
    *,
    frequency: str | None = None,
    start: str | None = None,
    end: str | None = None,
    model_family: str = "ar_p",
    output_directory: str | Path | None = None,
    cache_root: str | Path | None = None,
    random_seed: int = 0,
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
        L4 ``fit_model`` family. Defaults to ``"ar_p"``.
    output_directory
        Directory to write ``manifest.json`` and per-cell artifacts.
    cache_root
        Shared raw-data cache root; forwarded to :func:`execute_recipe`.
    random_seed
        L0 ``random_seed`` (default ``0``).

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
            .compare("4_forecasting_model.nodes.fit_1_ridge.params.alpha",
                     [0.1, 1.0])
        )

    PR 1 implements the basic constructor + sweep methods + run / to_yaml /
    replicate / validate. ``.use_fred_sd_inferred_tcodes()``,
    ``.use_sd_empirical_tcodes()``, ``.use_preprocessor()`` and friends are
    deferred to v0.8.1 (PR 2).
    """

    def __init__(
        self,
        dataset: str,
        target: str,
        horizons: Sequence[int] = (1,),
        *,
        frequency: str | None = None,
        start: str | None = None,
        end: str | None = None,
        model_family: str = "ar_p",
        random_seed: int = 0,
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
        """Branch a named variant.

        **Not implemented in v0.8.0.** Lands in v0.8.1 (PR 2). For sweeps
        over a single axis use :meth:`compare` / :meth:`sweep`; for
        comparing model families use :meth:`compare_models`.
        """

        raise NotImplementedError(
            "Experiment.variant() lands in v0.8.1; in v0.8.0 use "
            ".compare_models([...]) / .compare(axis_path, values) / "
            ".sweep(axis_path, values)"
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
    """Thin façade over :class:`ManifestExecutionResult`.

    PR 1 ships only the **minimal shell** -- the underlying manifest is the
    source of truth. Richer accessors (``forecasts`` / ``metrics`` /
    ``ranking`` / ``read_json`` / ``file_path`` / ``mean`` / ``get``) land
    in v0.8.1 (PR 2).
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
