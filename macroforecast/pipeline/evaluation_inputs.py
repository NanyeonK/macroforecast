"""Resolve the data-backed inputs an evaluation needs, before it runs (Stage 2).

Evaluation is arithmetic over a forecast table: the same table and the same spec
should produce the same tables whatever the network, the FRED cache, and the
filesystem happen to be doing. One input broke that. A named subsample mask --
``SubsampleWindow(mask="nber_recession")`` -- is a *name*, and turning it into a
boolean state series meant a ``load_fred_series()`` call from inside
``pipeline/evaluate.py``, so scoring one fixed frame twice could disagree, or
fail offline, for reasons that had nothing to do with the forecasts.

That call lives here now. :func:`resolve_evaluation_inputs` runs once per
evaluation operation, before the evaluator sees the frame: ``run_pipeline`` and
``rescore`` both call it and pass the result to ``evaluate(..., inputs=...)``.
What comes back is data -- one already-loaded state series per named mask, plus
the provenance record the report publishes -- so the evaluator only aligns and
applies it, and never loads anything.

The two halves own different questions:

* here: which FRED series a name means at the frame's frequency, loading it (at
  most once per series, even when recession and expansion both read ``USREC``),
  the 0/1 validation, and the inversion that makes expansion the complement of
  recession.
* ``evaluate``: aligning that series to the forecast target dates, the strict
  overlap/coverage/NaN errors, and applying the result to rows.

Alignment stays with the evaluator because coverage is a question about the
forecast frame, and the frame is the evaluator's subject. The frequency and
date-set helpers are imported *from* the evaluator for the same reason: the
series a name resolves to has to be chosen by the same rule the evaluator will
align against, and two copies of that rule would be free to drift apart.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from macroforecast.data import DataBundle, load_fred_series
from macroforecast.pipeline.evaluate import (
    _SUBSAMPLE_DATE_COLUMN,
    _eval_subsamples,
    _series_summary,
    _target_mask_frequency,
)
from macroforecast.pipeline.spec import PipelineSpec

# Both indicators read the same series and differ only in polarity, which is why
# ``invert`` is a registry field rather than two near-identical entries: an
# evaluation asking for recession AND expansion loads one series, once.
_NAMED_SUBSAMPLE_MASKS: dict[str, dict[str, Any]] = {
    "nber_recession": {"monthly": "USREC", "quarterly": "USRECQ", "invert": False},
    "nber_expansion": {"monthly": "USREC", "quarterly": "USRECQ", "invert": True},
}


# ``eq=False`` on both carriers: a generated ``__eq__`` would compare a
# ``pd.Series`` with ``==`` and raise on the ambiguous truth value, and a
# generated ``__hash__`` would raise on the mapping. Identity is the only
# comparison that means anything for a loaded artifact, so that is what these get.
@dataclass(frozen=True, eq=False)
class ResolvedSubsampleMask:
    """One named subsample mask, already loaded.

    ``state`` is the indicator over its own native (FRED) dates, already
    inverted where the name asks for it and NOT yet aligned to any forecast
    frame -- aligning it is ``evaluate``'s job. ``mask_summary`` is the record
    that reaches ``report.provenance["evaluation"]["subsamples"]`` unchanged.
    """

    mask_source: str
    state: pd.Series
    mask_summary: Mapping[str, Any]


@dataclass(frozen=True, eq=False)
class ResolvedEvaluationInputs:
    """Every input one ``evaluate()`` call needs that it cannot compute itself.

    Empty -- the default -- is the common case: user-supplied masks, plain date
    windows, and evaluations without subsamples need nothing loaded, and an
    ``evaluate()`` call for those is offline with or without this object.
    """

    subsample_masks: Mapping[str, ResolvedSubsampleMask] = field(default_factory=dict)


def _resolvable_target_dates(master: pd.DataFrame) -> pd.DatetimeIndex | None:
    """The forecast target dates a mask must cover, or ``None`` if *master* cannot say.

    Read from the master frame BEFORE ``apply_combinations``: a combination
    contender is built from the rows it combines and reuses their
    ``(origin, date)`` pairs, so it contributes no date the master frame does
    not already carry, and the unique sorted date set the evaluator later sees
    is this one.

    ``None`` means the frame has no usable ``date`` column. Nothing is loaded in
    that case: ``evaluate`` raises the specific, named error for it, and
    pre-empting that with a fetch would only spend a request on a frame that
    cannot be scored and replace a precise message with a worse one.
    """
    if _SUBSAMPLE_DATE_COLUMN not in master.columns:
        return None
    dates = pd.to_datetime(master[_SUBSAMPLE_DATE_COLUMN], errors="coerce").dt.normalize()
    if bool(dates.isna().any()):
        return None
    return pd.DatetimeIndex(pd.unique(dates)).sort_values()


def _named_mask_state(
    mask_name: str,
    bundle: DataBundle,
    *,
    series_id: str,
    frequency: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Turn one loaded FRED indicator into its boolean state plus provenance."""
    raw = pd.to_numeric(bundle.panel[series_id], errors="coerce")
    observed = raw.dropna()
    invalid = observed[~observed.isin([0, 1])]
    if not invalid.empty:
        first = pd.Timestamp(invalid.index[0]).strftime("%Y-%m-%d")
        raise ValueError(
            f"EvalSpec.subsamples mask {mask_name!r} loaded FRED series "
            f"{series_id!r} with non-0/1 value at {first}"
        )

    state = pd.Series(pd.NA, index=raw.index, dtype="boolean")
    state.loc[raw == 1] = True
    state.loc[raw == 0] = False
    if bool(_NAMED_SUBSAMPLE_MASKS[mask_name]["invert"]):
        state = ~state

    artifact = dict(bundle.metadata.get("artifact", {}) or {})
    summary = {
        **_series_summary(state),
        "series_id": series_id,
        "frequency": frequency,
        "source_url": artifact.get("source_url"),
        "cache_path": artifact.get("local_path"),
        "raw_sha256": artifact.get("file_sha256"),
        "cache_hit": artifact.get("cache_hit"),
    }
    return state, summary


def resolve_evaluation_inputs(
    master: pd.DataFrame,
    spec: PipelineSpec,
) -> ResolvedEvaluationInputs:
    """Load the named subsample masks *spec* asks for, against *master*'s target dates.

    Call this once per evaluation operation and hand the result to
    ``evaluate(master, spec, inputs=...)``; ``run_pipeline`` and ``rescore``
    already do. Every distinct FRED series is fetched at most once, so an
    ``EvalSpec`` with both ``"nber_recession"`` and ``"nber_expansion"`` costs
    one load rather than two.

    Returns an empty :class:`ResolvedEvaluationInputs` when there is nothing to
    load, which is every evaluation that does not name an indicator.
    """
    subsamples = _eval_subsamples(spec)
    if subsamples is None:
        return ResolvedEvaluationInputs()
    named = {
        name: window.mask
        for name, window in subsamples.items()
        if isinstance(window.mask, str)
    }
    if not named:
        return ResolvedEvaluationInputs()

    target_dates = _resolvable_target_dates(master)
    if target_dates is None:
        return ResolvedEvaluationInputs()

    frequency = _target_mask_frequency(target_dates)
    bundles: dict[str, DataBundle] = {}
    resolved: dict[str, ResolvedSubsampleMask] = {}
    for name, mask_name in named.items():
        series_id = str(_NAMED_SUBSAMPLE_MASKS[mask_name][frequency])
        if series_id not in bundles:
            bundles[series_id] = load_fred_series(series_id, frequency=frequency)
        state, summary = _named_mask_state(
            mask_name,
            bundles[series_id],
            series_id=series_id,
            frequency=frequency,
        )
        resolved[name] = ResolvedSubsampleMask(
            mask_source=mask_name,
            state=state,
            mask_summary=summary,
        )
    return ResolvedEvaluationInputs(subsample_masks=resolved)


__all__ = [
    "ResolvedEvaluationInputs",
    "ResolvedSubsampleMask",
    "resolve_evaluation_inputs",
]
