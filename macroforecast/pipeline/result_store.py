"""Cross-run result store for pipeline forecast cells.

The store is intentionally small and file-backed.  A caller owns the directory;
macroforecast writes one parquet payload and one JSON manifest per digest under
``<store>/cells``.  There is no store-level locking: use one writer at a time.
"""
from __future__ import annotations

from macroforecast.pipeline.plan import compile_arm_plan, compile_stage_policies

import dataclasses as _dc
import base64
import datetime as _dt
import hashlib
import importlib.metadata as _metadata
import json
import os
import pickle
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from macroforecast.pipeline.spec import (
    Arm,
    PipelineSpec,
    ResolvedTarget,
    _FrozenByteArray,
    is_vintage_aware,
)

if TYPE_CHECKING:  # imported for typing only, matching the function-local runtime import
    from macroforecast.forecasting.task import ResolvedForecastTask


@dataclass(frozen=True)
class ResultCellIdentity:
    """Digest metadata for one result-store cell."""

    digest: str | None
    cell_echo: dict[str, Any] | None
    data_fingerprint: Any
    reason: str | None = None


@dataclass(frozen=True)
class ResultStoreHit:
    """Loaded forecast frame plus the manifest that justified the hit."""

    frame: pd.DataFrame
    manifest: dict[str, Any]


class _UndigestibleCell(Exception):
    """Raised internally when a cell contains an unsafe custom callable."""


def _model_definitely_differs(arm_model: Any, task_model: Any) -> bool:
    """Whether *task_model* is DEMONSTRABLY not the model this arm fits.

    Deliberately one-sided, because this guard has to stay compatible with callers
    that predate the task argument and with arbitrary user-supplied model objects:

    * a task carrying no model at all (``None``) states nothing to contradict -- the
      runner's own task check reads it the same way, so a hand-built task without a
      model keeps working;
    * the production task is built FROM ``arm.model``, so the identity check answers
      this in the common case, including across the parallel payload (spec and tasks
      travel in one pickle, which preserves sharing);
    * an ``__eq__`` that raises or returns a non-boolean (a numpy array, an estimator
      with elementwise comparison) is not evidence of anything, so it is not treated
      as a mismatch. Refusing to digest a legitimate cell over an exotic ``__eq__``
      would be a worse failure than the divergence being guarded against.
    """
    if task_model is None or arm_model is task_model:
        return False
    try:
        return not bool(arm_model == task_model)
    except Exception:
        return False


def _assert_task_describes_cell(
    task: "ResolvedForecastTask",
    arm: Arm,
    target: ResolvedTarget,
    *,
    horizon: int,
) -> None:
    """Refuse a task that answers a DIFFERENT question than the cell being digested.

    A ValueError, not an undigestible cell: an undigestible cell is a legitimate
    configuration the store declines to cache, whereas a task/cell mismatch is a
    caller bug that would silently mint a digest for one cell and file the forecast of
    another under it. Cheap to check, and it is the whole point of resolving once.

    The model is checked alongside the labels because it is the one field the digest
    is computed from that the labels cannot vouch for: two arms may share a name in
    two specs, and a digest minted for one arm's model while another's is fitted is
    precisely the cache/fit divergence the shared task exists to make impossible. The
    runner refuses the same disagreement at its own entry.
    """
    mismatches: list[tuple[str, Any, Any]] = [
        (field, supplied, resolved)
        for field, supplied, resolved in (
            ("arm", arm.name, task.arm_name),
            ("target", target.name, task.target_name),
            ("forecast_policy", target.policy, task.forecast_policy),
            ("target_transform", target.transform, task.target_transform),
            ("horizon", int(horizon), int(task.horizon)),
        )
        if supplied != resolved
    ]
    if _model_definitely_differs(arm.model, task.model):
        mismatches.append(("model", arm.model, task.model))
    if mismatches:
        detail = ", ".join(
            f"{field}={supplied!r} but the task says {resolved!r}"
            for field, supplied, resolved in mismatches
        )
        raise ValueError(
            f"result_cell_identity was given a task for a different cell: {detail}"
        )


def result_cell_identity(
    spec: PipelineSpec,
    arm: Arm,
    target: ResolvedTarget,
    *,
    horizon: int,
    data_identity: Mapping[str, Any],
    task: "ResolvedForecastTask | None" = None,
) -> ResultCellIdentity:
    """Return the digest and human-readable echo for one result-store cell.

    A ``None`` digest means the cell is deliberately not cacheable, normally
    because a user-owned callable lacks an explicit ``__mf_digest__`` opt-in.

    ``task`` is the already-resolved
    :class:`~macroforecast.forecasting.task.ResolvedForecastTask` for this cell, as
    produced once by the execution path. Passing it means the digest is computed from
    the SAME resolved features the run used, instead of re-deriving them here -- the
    two answers were previously produced by separate code that had drifted. When it is
    omitted (a direct caller with only a spec in hand) the features are resolved here
    through the same resolver, so both spellings agree by construction.
    """

    data_fingerprint = data_identity.get("fingerprint")
    try:
        _assert_digestible_data_fingerprint(data_fingerprint)
        # One resolver, shared with the execution path (A2). A retarget that fails
        # now makes the cell UNCACHEABLE rather than silently digesting the
        # un-retargeted spec -- a digest describing a task that was never run is
        # worse than no digest.
        from macroforecast.forecasting.task import FeatureRetargetError, retarget_features

        if task is not None:
            _assert_task_describes_cell(task, arm, target, horizon=horizon)
            # Already retargeted (and already raised if it could not be), so there is
            # nothing left to fail here: an unresolvable cell has no task at all.
            effective_features = task.features
        else:
            try:
                effective_features = retarget_features(
                    arm.features, target.name, arm_name=arm.name
                )
            except FeatureRetargetError as exc:
                raise _UndigestibleCell(str(exc)) from exc
        # One compiled answer per cell, shared by the window, the preprocessing spec
        # and the stage policies below, so the digest cannot describe a different
        # arrangement than the one ``run()`` is handed.
        plan = compile_arm_plan(spec, arm)
        payload: dict[str, Any] = {
            "data_fingerprint": _json_ready(data_fingerprint),
            "effective_selection_seed": _effective_selection_seed(),
            "backend_versions": _backend_versions(arm.model, params=arm.params),
            "target": {
                "name": target.name,
                "transform": target.transform,
                "forecast_policy": target.policy,
            },
            "horizon": int(horizon),
            "arm": {
                "name": arm.name,
                "model": _model_identity(arm.model, params=arm.params),
                "params": _json_ready(arm.params, path="arm.params"),
                "preset": _model_preset(arm.model),
                "features": _feature_identity(effective_features),
                "preprocessing": _preprocessing_identity(plan.preprocess.spec),
                "model_selection": _object_identity(
                    arm.model_selection, path="arm.model_selection"
                ),
                "model_selection_metric": arm.model_selection_metric,
                # The RESOLVED policies, not the raw fields they came from. A raw
                # ``None`` says only "the caller did not choose", and what it means
                # depends on package config at run time; two runs that resolve to the
                # same policy are the same fit whichever spelling asked for it.
                "stage_policies": _stage_policies_identity(
                    compile_stage_policies(spec, arm, plan=plan),
                    path="arm.stage_policies",
                ),
                "window": _object_identity(plan.window, path="arm.window"),
            },
            "evaluation_callables": _evaluation_callable_identity(spec),
        }
        if is_vintage_aware(spec):
            payload["vintage"] = {
                "actuals_vintage": getattr(spec.data, "actuals_vintage", None),
                "source_kind": type(getattr(spec.data, "source", None)).__name__,
            }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return ResultCellIdentity(
            digest=digest,
            cell_echo=payload,
            data_fingerprint=_json_ready(data_fingerprint),
        )
    except _UndigestibleCell as exc:
        return ResultCellIdentity(
            digest=None,
            cell_echo=None,
            data_fingerprint=_json_ready(data_fingerprint),
            reason=str(exc),
        )


class ResultStore:
    """Directory-backed store for pipeline cell forecast frames.

    ``root_dir`` is owned by the caller. The store writes one parquet forecast
    payload and one JSON manifest per digest under ``<root_dir>/cells`` using
    atomic file replacement. ``load(...)`` returns a hit only when the digest,
    data fingerprint, manifest, and parquet payload all agree; otherwise it
    returns ``None`` so the pipeline recomputes the cell.

    Returns
    -------
    ResultStore
        File-backed cache object with ``load`` and ``write`` methods. The
        higher-level helpers ``result_store_summary(...)`` and
        ``purge_result_store(...)`` inspect and maintain the same directory.

    Example
    -------
    >>> from macroforecast.pipeline.result_store import ResultStore
    >>> store = ResultStore("cache/results")
    >>> store.load("abc123", data_fingerprint={"sha256": "..."}) is None
    True
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root = Path(root_dir)
        self.cells = self.root / "cells"

    def load(self, digest: str, *, data_fingerprint: Any) -> ResultStoreHit | None:
        """Load *digest* when its manifest is complete and data identity matches."""

        manifest_path = self._manifest_path(digest)
        parquet_path = self._parquet_path(digest)
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            return None
        if manifest.get("digest") != digest:
            return None
        if manifest.get("data_fingerprint") != _json_ready(data_fingerprint):
            return None
        if not parquet_path.exists():
            return None
        try:
            frame = pd.read_parquet(parquet_path)
            frame = _restore_frame_encoding(frame, manifest.get("frame_encoding", {}))
        except Exception:
            return None
        return ResultStoreHit(frame=frame, manifest=manifest)

    def write(
        self,
        digest: str,
        frame: pd.DataFrame,
        *,
        data_fingerprint: Any,
        cell_echo: Mapping[str, Any],
    ) -> None:
        """Persist *frame* and its manifest with atomic replaces."""

        import macroforecast as _mf

        self.cells.mkdir(parents=True, exist_ok=True)
        parquet_path = self._parquet_path(digest)
        manifest_path = self._manifest_path(digest)

        storage_frame, frame_encoding = _encode_frame_for_parquet(frame)
        fd, tmp_parquet = tempfile.mkstemp(
            prefix=f"_{digest}_",
            suffix=".parquet",
            dir=self.cells,
        )
        os.close(fd)
        try:
            storage_frame.to_parquet(tmp_parquet, index=True)
            os.replace(tmp_parquet, parquet_path)
        except Exception:
            _unlink_quietly(tmp_parquet)
            raise

        manifest = {
            "digest": digest,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "macroforecast_version": getattr(_mf, "__version__", "unknown"),
            "data_fingerprint": _json_ready(data_fingerprint),
            "cell_echo": _json_ready(dict(cell_echo)),
            "n_rows": int(len(frame)),
            "frame_encoding": frame_encoding,
        }
        fd, tmp_manifest = tempfile.mkstemp(
            prefix=f"_{digest}_",
            suffix=".json",
            dir=self.cells,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_manifest, manifest_path)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            _unlink_quietly(tmp_manifest)
            raise

    def _parquet_path(self, digest: str) -> Path:
        return self.cells / f"{digest}.parquet"

    def _manifest_path(self, digest: str) -> Path:
        return self.cells / f"{digest}.json"


def result_store_summary(store: str | Path) -> pd.DataFrame:
    """Summarise result-store manifests, one row per readable cell manifest."""

    rows: list[dict[str, Any]] = []
    for path in sorted((Path(store) / "cells").glob("*.json")):
        try:
            manifest = json.loads(path.read_text())
        except Exception:
            continue
        echo = manifest.get("cell_echo", {})
        target = echo.get("target", {}) if isinstance(echo, Mapping) else {}
        arm = echo.get("arm", {}) if isinstance(echo, Mapping) else {}
        rows.append(
            {
                "digest": manifest.get("digest", path.stem),
                "created_at": manifest.get("created_at"),
                "version": manifest.get("macroforecast_version"),
                "target": target.get("name"),
                "horizon": echo.get("horizon") if isinstance(echo, Mapping) else None,
                "arm": arm.get("name"),
                "n_rows": manifest.get("n_rows"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["digest", "created_at", "version", "target", "horizon", "arm", "n_rows"],
    )


def _validated_purge_cutoff(value: str | datetime | None, *, label: str) -> datetime | None:
    """Parse a purge cutoff, refusing an unparseable one before anything is removed.

    :func:`_parse_datetime` is deliberately tolerant -- it returns ``None`` for a value
    it cannot read, which is right for the read and provenance paths that use it, where
    an unknown timestamp simply means "cannot compare". A DELETE cannot read it that
    way: a ``before`` that silently becomes "no cutoff" removes every entry the caller
    was trying to spare. So the tolerance stays where it belongs and the purge boundary
    validates on top of it.
    """
    if value is None:
        return None
    parsed = _parse_datetime(value)
    if parsed is None:
        raise ValueError(
            f"{label}={value!r} is not a parseable datetime, so this purge is refused "
            "before anything is enumerated or deleted -- an unreadable cutoff would "
            "otherwise act as no cutoff at all and delete every entry. Pass a datetime "
            "or an ISO-8601 string."
        )
    return parsed


def _validated_store_component(value: Any, *, label: str) -> str:
    """One filesystem NAME, refused if it can address anything but a direct child.

    Both stores address entries by a single generated component -- a digest filename, a
    sanitized arm alias -- so a filter is only ever a name. Anything that can traverse
    (``..``), re-root (an absolute path), or nest (a separator) is refused rather than
    normalized, because a purge that silently reinterprets what the caller named is the
    failure this guard exists to prevent.
    """
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
        or "\x00" in text
        or os.path.isabs(text)
        or Path(text).name != text
    ):
        raise ValueError(
            f"{label} entry {value!r} is not a plain store entry name. Purge filters "
            "address one generated entry each and must not be able to reach outside "
            "the store, so an empty name, '.', '..', a path separator, or an absolute "
            "path is refused."
        )
    return text


def _resolved_within(path: Path, root: Path) -> Path | None:
    """*path* fully resolved, or ``None`` when it does not stay inside *root*.

    Resolution follows symlinks on purpose: a name check alone is not a containment
    boundary, because a link named like an ordinary child can point anywhere. The
    caller treats ``None`` as "not ours to delete".
    """
    try:
        resolved = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        return None
    if resolved != resolved_root and resolved_root not in resolved.parents:
        return None
    return resolved


def purge_result_store(
    store: str | Path,
    *,
    before: str | datetime | None = None,
    version: str | None = None,
    digests: Sequence[str] | None = None,
) -> int:
    """Delete result-store cells matching the supplied filters and return a count.

    Every filter is validated BEFORE anything is enumerated or removed, so a call that
    is going to be refused deletes nothing at all: an unparseable ``before`` raises
    rather than acting as no cutoff, and a ``digests`` entry that is not a plain cell
    name -- ``..``, an absolute path, anything containing a separator -- raises rather
    than addressing a file outside ``<store>/cells``. One bad entry in an otherwise
    valid list therefore refuses the whole call instead of deleting the entries before
    it.

    Deletion itself stays best effort and idempotent: a cell whose files have already
    gone, or which cannot be removed, is skipped quietly rather than raising. The
    returned count is the number of CELLS for which at least one file (the parquet
    payload or its manifest) was actually removed -- not the number of files, and not
    the number of candidates considered. A digest naming a cell that does not exist
    contributes 0.
    """

    root = Path(store) / "cells"
    before_dt = _validated_purge_cutoff(before, label="before")
    digest_filter = (
        {_validated_store_component(digest, label="digests") for digest in digests}
        if digests is not None
        else None
    )
    if _resolved_within(root, Path(store)) is None:
        raise ValueError(
            f"{root!s} does not resolve inside {Path(store)!s}; refusing to purge "
            "through a link that leaves the requested result store."
        )

    deleted = 0
    candidates = sorted(root.glob("*.json"))
    if digest_filter is not None:
        seen = {path.stem for path in candidates}
        candidates.extend(root / f"{digest}.json" for digest in sorted(digest_filter - seen))
    for manifest_path in candidates:
        digest = manifest_path.stem
        if digest_filter is not None and digest not in digest_filter:
            continue
        manifest: dict[str, Any] = {}
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            if version is not None or before_dt is not None:
                continue
        if version is not None and manifest.get("macroforecast_version") != version:
            continue
        if before_dt is not None:
            created = _parse_datetime(manifest.get("created_at"))
            if created is None or created >= before_dt:
                continue
        # Both are attempted, and the cell counts when either actually went away.
        payload_removed = _unlink_quietly(root / f"{digest}.parquet")
        manifest_removed = _unlink_quietly(manifest_path)
        if payload_removed or manifest_removed:
            deleted += 1
    return deleted


def _encode_frame_for_parquet(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    storage = frame.copy()
    pickled_columns: list[str] = []
    for column in storage.columns:
        series = storage[column]
        if series.dtype != object:
            continue
        if not any(_needs_object_pickle(value) for value in series.dropna()):
            continue
        storage[column] = series.map(_pickle_to_text)
        pickled_columns.append(str(column))
    return storage, {"pickled_object_columns": pickled_columns}


def _restore_frame_encoding(frame: pd.DataFrame, encoding: Mapping[str, Any]) -> pd.DataFrame:
    restored = frame.copy()
    for column in encoding.get("pickled_object_columns", []) or []:
        if column in restored.columns:
            restored[column] = restored[column].map(_text_to_pickle)
    return restored


def _needs_object_pickle(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, int, float, bool, np.generic, pd.Timestamp)):
        return False
    return True


def _pickle_to_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return base64.b64encode(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).decode(
        "ascii"
    )


def _text_to_pickle(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return pickle.loads(base64.b64decode(str(value).encode("ascii")))  # noqa: S301


def _model_identity(model: Any, *, params: Mapping[str, Any] | None) -> dict[str, Any]:
    from macroforecast.models import ModelSpec
    from macroforecast.models.specs import MODEL_SPECS, get_model

    # Every branch routes its ``to_dict()`` through ``_json_ready``. The registry ones
    # used to embed it verbatim, so a params value the JSON encoder cannot take -- a
    # callback, an array -- reached ``json.dumps`` in ``result_cell_identity`` and raised
    # TypeError out of the whole run. Identity is allowed to say "I cannot identify this
    # cell"; it is not allowed to end the run over it.
    if isinstance(model, str):
        spec = get_model(model, params=params)
        return {"kind": "registry", "spec": _model_spec_identity(spec)}
    if isinstance(model, ModelSpec):
        spec = model.with_params(**dict(params or {})) if params else model
        registered = MODEL_SPECS.get(spec.name)
        if registered is not None and registered.fit_func is spec.fit_func:
            return {"kind": "registry", "spec": _model_spec_identity(spec)}
        return {
            "kind": "custom",
            "spec": _model_spec_identity(spec),
            "mf_digest": _callable_digest(spec, spec.fit_func, path=f"model {spec.name!r}"),
        }
    if callable(model):
        try:
            spec = get_model(model, params=params)
        except Exception:
            return {
                "kind": "custom_callable",
                "name": _callable_name(model),
                "params": _json_ready(params, path="arm.params"),
                "mf_digest": _callable_digest(model, path=f"model callable {_callable_name(model)!r}"),
            }
        return {"kind": "registry", "spec": _model_spec_identity(spec)}
    # Neither a registry name, a ModelSpec, nor a callable: there is nothing here that
    # identifies the fit. ``repr`` used to stand in, which made two such models share a
    # digest whenever their reprs matched (the default one carries an address, so it
    # also never matched itself across runs).
    raise _UndigestibleCell(
        f"model {type(model).__name__} is neither a registry name, a ModelSpec, nor a "
        "callable, so the result store cannot identify it; recomputing instead"
    )


def _stage_policies_identity(policies: Any, *, path: str) -> dict[str, Any]:
    """The three resolved stage policies, identified rather than exported.

    Same three slots and same per-policy shape as
    :meth:`~macroforecast.pipeline.plan.CompiledStagePolicies.to_dict`, which the runner
    metadata and the leakage audit publish unchanged -- only the callable entries are
    replaced, and only here. Every slot is covered even where the current runner wiring
    cannot produce a custom policy for it, so a future wiring change cannot quietly
    reintroduce the bypass.
    """
    return {
        slot: (
            None if policy is None else _stage_policy_identity(policy, path=f"{path}.{slot}")
        )
        for slot, policy in (
            ("preprocessing", policies.preprocessing),
            ("feature_engineering", policies.feature_engineering),
            ("model_selection", policies.model_selection),
        )
    }


def _stage_policy_identity(policy: Any, *, path: str) -> dict[str, Any]:
    """One resolved ``StagePolicy``, with its callables identified by marker.

    Built from the policy's own ``to_dict()`` so that everything the public export
    already canonicalises -- the normalized scope and update (including an integer or a
    ``DateOffset`` cadence), the reference bounds, ``apply_to`` -- keeps the exact
    representation it had, and an ordinary policy's digest does not move. Two entries are
    then replaced:

    * ``selector``, which the export renders as a module/qualname string. A name is not
      an identity: edit the selector's body and the name is unchanged, so a stale cell
      would be served. It must carry ``__mf_digest__``, and the marker is recorded.
    * ``metadata``, which the export walks with its own serializer and which renders a
      callable by name too. It goes through this module's serializer instead, so a
      callable anywhere inside obeys the same rule.

    A policy with no selector and ordinary metadata therefore serializes byte-identically
    to its export, which is what keeps existing stores hitting.

    What the export produces is nevertheless run through this module's serializer rather
    than embedded verbatim: ``to_dict()`` passes a value it does not recognise straight
    through, and a raw object reaching ``json.dumps`` would end the run instead of making
    the cell uncacheable. For everything an ordinary policy carries -- strings, integers,
    ``None``, the ISO-formatted reference bounds, the ``apply_to`` list -- that pass is
    the identity function, so the representation is unchanged.
    """
    exported = dict(_to_dict_or_undigestible(policy, path=path))
    # Dropped before the pass and rebuilt from the policy itself below: the exported copy
    # has already flattened any callable inside it to a name.
    exported.pop("metadata", None)
    payload = _json_ready(exported, path=path)
    payload["metadata"] = _json_ready(
        getattr(policy, "metadata", None), path=f"{path}.metadata"
    )
    update = getattr(policy, "update", None)
    if isinstance(update, DateOffset):
        # The export renders this as ``freqstr``, which does not identify it -- see
        # ``_dateoffset_identity``. A string or integer cadence is left exactly as the
        # export wrote it, so only DateOffset-backed policies change.
        payload["update"] = _json_ready(update, path=f"{path}.update")
    selector = getattr(policy, "selector", None)
    if callable(selector):
        payload["selector"] = _callable_identity(selector, path=f"{path}.selector")
    return payload


def _model_spec_identity(spec: Any) -> Any:
    """One model spec's canonical identity, params included."""
    return _json_ready(_to_dict_or_undigestible(spec, path="arm.model"), path="arm.model")


def _model_preset(model: Any) -> Any:
    return getattr(model, "preset", None) or getattr(model, "default_preset", None)


#: Fingerprint ``method`` values that do not identify the data and therefore must
#: not be cached against. ``undigestible`` is the canonical one the data layer emits
#: when it knows it cannot identify the panel; ``unavailable`` is the legacy marker a
#: fingerprint FAILURE used to produce, and it is rejected here too so a prebuilt
#: descriptor cannot route around the rule.
_UNIDENTIFIED_FINGERPRINT_METHODS: frozenset[str] = frozenset({"undigestible", "unavailable"})


def _assert_digestible_data_fingerprint(fingerprint: Any) -> None:
    if not isinstance(fingerprint, Mapping):
        return
    if str(fingerprint.get("method")) not in _UNIDENTIFIED_FINGERPRINT_METHODS:
        return
    reason = (
        fingerprint.get("reason")
        or fingerprint.get("error")
        or "data fingerprint is undigestible"
    )
    raise _UndigestibleCell(str(reason))


def _effective_selection_seed() -> int | None:
    from macroforecast.meta import get_config

    return get_config()["random_seed"]


def _backend_versions(model: Any, *, params: Mapping[str, Any] | None) -> dict[str, Any]:
    spec = _model_spec_for_backend(model, params=params)
    backend = str(getattr(spec, "backend", "")) if spec is not None else ""
    family = str(getattr(spec, "family", "")) if spec is not None else ""
    packages = _backend_packages(backend=backend, family=family)
    return {
        "model": getattr(spec, "name", _callable_name(model) if callable(model) else str(model)),
        "family": family or None,
        "backend": backend or None,
        "packages": {package: _package_version(package) for package in packages},
    }


def _model_spec_for_backend(model: Any, *, params: Mapping[str, Any] | None) -> Any | None:
    try:
        from macroforecast.models import ModelSpec
        from macroforecast.models.specs import get_model
    except ImportError:
        return None
    if isinstance(model, str):
        return get_model(model, params=params)
    if isinstance(model, ModelSpec):
        return model.with_params(**dict(params or {})) if params else model
    if callable(model):
        try:
            return get_model(model, params=params)
        except (TypeError, ValueError, KeyError):
            return None
    return None


_BACKEND_PACKAGE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("xgboost", ("xgboost",)),
    ("lightgbm", ("lightgbm",)),
    ("catboost", ("catboost",)),
    ("torch", ("torch",)),
    ("arch.", ("arch",)),
    ("statsmodels", ("statsmodels",)),
    ("sklearn", ("scikit-learn",)),
    ("scipy", ("scipy",)),
)

_FAMILY_BACKEND_PACKAGES: Mapping[str, tuple[str, ...]] = {
    "tree": ("scikit-learn",),
    "support_vector": ("scikit-learn",),
    "nonparametric": ("scikit-learn",),
    "linear": ("scikit-learn",),
    "factor": ("numpy",),
    "timeseries": ("statsmodels",),
    "volatility": ("arch",),
    "neural": ("torch",),
}


def _backend_packages(*, backend: str, family: str) -> tuple[str, ...]:
    found: list[str] = []
    lowered = backend.lower()
    for marker, packages in _BACKEND_PACKAGE_MARKERS:
        if marker in lowered:
            found.extend(packages)
    if not found:
        found.extend(_FAMILY_BACKEND_PACKAGES.get(family, ()))
    # Stable de-duplication without changing declared package order.
    return tuple(dict.fromkeys(found))


def _package_version(package: str) -> str | None:
    try:
        return _metadata.version(package)
    except _metadata.PackageNotFoundError:
        return None


def _feature_identity(features: Any) -> Any:
    if features is None:
        return None
    return {
        "spec": _object_identity(features, path="arm.features"),
        "custom_callable_digests": _custom_feature_digests(features),
    }


def _preprocessing_identity(preprocessing: Any) -> Any:
    if preprocessing is None:
        return None
    return {
        "spec": _object_identity(preprocessing, path="arm.preprocessing"),
        "custom_callable_digests": _custom_preprocessing_digests(preprocessing),
    }


def _custom_feature_digests(features: Any) -> list[dict[str, str]]:
    digests: list[dict[str, str]] = []
    for idx, step in enumerate(getattr(features, "feature_steps", ()) or ()):
        if not isinstance(step, Mapping) or step.get("method") != "custom":
            continue
        for key in ("func", "callable", "fit_func", "transform_func"):
            func = step.get(key)
            if callable(func):
                digests.append(
                    {
                        "step": str(step.get("name", idx)),
                        "slot": key,
                        "digest": _callable_digest(func, path=f"feature step {step.get('name', idx)!r}.{key}"),
                    }
                )
    return digests


def _custom_preprocessing_digests(preprocessing: Any) -> list[dict[str, str]]:
    options = getattr(preprocessing, "options", None)
    if not isinstance(options, Mapping):
        return []
    raw_steps = options.get("custom_steps")
    if raw_steps is None:
        return []
    if callable(raw_steps):
        return [
            {
                "step": _callable_name(raw_steps),
                "slot": "func",
                "digest": _callable_digest(raw_steps, path="preprocessing custom_steps"),
            }
        ]
    steps = [raw_steps] if isinstance(raw_steps, Mapping) else list(raw_steps)
    digests: list[dict[str, str]] = []
    for idx, step in enumerate(steps):
        if callable(step):
            digests.append(
                {
                    "step": _callable_name(step),
                    "slot": "func",
                    "digest": _callable_digest(step, path=f"preprocessing custom step {idx}"),
                }
            )
            continue
        if not isinstance(step, Mapping):
            continue
        func = step.get("func", step.get("callable"))
        if callable(func):
            digests.append(
                {
                    "step": str(step.get("name", idx)),
                    "slot": "func",
                    # Two specs differing ONLY in where the step runs are different
                    # pipelines; without this they would share a cache key (#453a).
                    "position": str(step.get("position", "last")),
                    "digest": _callable_digest(func, path=f"preprocessing custom step {step.get('name', idx)!r}"),
                }
            )
    return digests


def _evaluation_callable_identity(spec: PipelineSpec) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for metric in getattr(spec.evaluation, "metrics", ()):
        if callable(metric) and not _is_registered_metric_callable(metric):
            out.append(
                {
                    "slot": "metric",
                    "name": _callable_name(metric),
                    "digest": _callable_digest(metric, path=f"evaluation metric {_callable_name(metric)!r}"),
                }
            )
    loss = getattr(spec.evaluation, "loss", None)
    if callable(loss):
        out.append(
            {
                "slot": "loss",
                "name": _callable_name(loss),
                "digest": _callable_digest(loss, path=f"evaluation loss {_callable_name(loss)!r}"),
            }
        )
    return out


def _is_registered_metric_callable(func: Callable[..., Any]) -> bool:
    try:
        from macroforecast.metrics import _METRICS
    except Exception:
        return False
    return any(func is registered for registered in _METRICS.values())


def _object_identity(value: Any, *, path: str = "value") -> Any:
    """Canonical identity for one spec object.

    A ``to_dict()`` is preferred where the object publishes one, because that is the
    form its own module calls canonical. A :class:`~macroforecast.model_selection.SearchSpec`
    is the exception and is walked field by field instead: its ``to_dict()`` renders a
    custom search function by NAME, which is a public-export choice this module must not
    change and must not trust for identity -- two different functions sharing a qualname
    would otherwise share a cell digest.
    """
    if value is None:
        return None
    if not _is_search_spec(value) and hasattr(value, "to_dict"):
        payload = _to_dict_or_undigestible(value, path=path)
        return _json_ready(payload, path=path)
    return _json_ready(value, path=path)


def _to_dict_or_undigestible(value: Any, *, path: str) -> Any:
    """``value.to_dict()``, or an undigestible cell naming where it failed.

    The previous fallback recorded ``repr(value)`` plus the error text, which put a
    generic repr into the digest: two objects whose ``to_dict()`` failed the same way
    became the same cell.
    """
    try:
        return value.to_dict()
    except Exception as exc:
        raise _UndigestibleCell(
            f"{path}: {type(value).__name__}.to_dict() raised "
            f"{type(exc).__name__}: {exc}, so this value cannot be identified; "
            "recomputing instead"
        ) from exc


def _search_spec_identity(value: Any, *, path: str, seen: frozenset[int]) -> dict[str, Any]:
    """A search spec's identity, taken from its FIELDS rather than its export.

    ``SearchSpec.to_dict()`` is the public JSON export and renders ``custom_func`` as a
    module/qualname string. That is right for an export a human reads and wrong for a
    cache key, and the two must not be conflated: repairing identity by changing the
    export would alter a documented public format. So the pipeline boundary reads the
    dataclass itself, which sends ``custom_func``, ``custom_params``, ``metadata``, the
    grids and the splitter through the ordinary rules -- meaning every callable among
    them has to carry ``__mf_digest__``, and that marker lands in the digest.
    """
    return {
        str(field.name): _json_ready(
            getattr(value, field.name), path=f"{path}.{field.name}", seen=seen
        )
        for field in sorted(_dc.fields(value), key=lambda item: item.name)
    }


def _is_search_spec(value: Any) -> bool:
    try:
        from macroforecast.model_selection import SearchSpec
    except ImportError:
        return False
    return isinstance(value, SearchSpec)


def _is_model_spec(value: Any) -> bool:
    try:
        from macroforecast.models import ModelSpec
    except ImportError:
        return False
    return isinstance(value, ModelSpec)


def _callable_digest(*candidates: Any, path: str) -> str:
    for candidate in candidates:
        marker = getattr(candidate, "__mf_digest__", None)
        if marker is not None:
            return str(marker)
    raise _UndigestibleCell(
        f"{path} is a custom callable without __mf_digest__; recomputing instead"
    )


def _callable_identity(value: Any, *, path: str) -> dict[str, Any]:
    """Name AND stable marker for a callable reached anywhere in an identity payload.

    The name alone is not an identity -- editing a function body leaves it unchanged --
    so the marker is required here rather than merely checked elsewhere, and it is
    recorded so that changing it changes the digest.
    """
    return {"callable": _callable_name(value), "mf_digest": _callable_digest(value, path=path)}


def _type_name(value: Any) -> str:
    """A value's concrete type, named by its stable PUBLIC path where it has one.

    Two values can agree on every field this module records and still be different
    things -- a ``datetime.timedelta`` and a ``pandas.Timedelta`` of one second, say --
    so where the fields alone do not separate them, the type does. That is why the name
    is here, and it is also why the name has to mean the same thing everywhere: it goes
    into a cache digest, so a library that MOVES a class without changing it moves every
    digest carrying one. ``pandas.Timedelta`` did exactly that, reporting
    ``pandas._libs.tslibs.timedeltas`` as its module on pandas 2 and ``pandas`` on
    pandas 3, which silently split one cell in two across an upgrade.

    The repair is narrow on purpose. A class is renamed to ``<root package>.<qualname>``
    only when that root package exports THE SAME CLASS OBJECT under that qualname, so
    the shortened name provably still refers to this class and nothing else. Anything
    else -- a class its package does not re-export, a nested qualname, a root that was
    never imported -- keeps its full module path.

    Two properties follow, and both matter more than the shortening does. Nothing
    private leaks into a digest for a class its package publishes. And nothing
    collapses: an identity test cannot be satisfied by two different classes, so two
    same-named classes in different private submodules stay distinguishable rather than
    both becoming the root name. A bare class name would give up exactly that, which is
    why the package prefix is kept rather than dropped.
    """
    cls = type(value)
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", None) or cls.__name__
    full = f"{module}.{qualname}" if module else str(qualname)
    root = module.split(".", 1)[0]
    if not root or root == module:
        # Already the root path (``datetime.timedelta``, and ``pandas.Timedelta`` as
        # pandas 3 reports it): there is nothing to shorten, and the answer is the
        # same one the check below would reach.
        return full
    root_module = sys.modules.get(root)
    if root_module is None:
        # Not imported, so there is nothing to ask. The class exists, so its own
        # submodule is; the root package simply may not be a package at all.
        return full
    try:
        exported = getattr(root_module, qualname, None)
    except Exception:
        # A module-level ``__getattr__`` is free to raise. A type name is never worth
        # ending a run over when the full path is right here.
        return full
    return f"{root}.{qualname}" if exported is cls else full


def _callable_name(func: Any) -> str:
    """A textual name for a callable that is the same in every process.

    A function carries ``__qualname__``; a callable OBJECT usually does not, and falling
    back to ``repr`` put its memory address into the name -- so two instances of one
    functor read as different callables and the same instance read differently on the
    next run. The instance's TYPE is the stable answer for those.

    The name is not the identity on its own -- ``__mf_digest__`` is, and
    :func:`_callable_identity` records both -- but it has to be stable, or a digest
    containing it cannot be either.
    """
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    if qualname is None:
        cls = type(func)
        module = getattr(cls, "__module__", "")
        qualname = getattr(cls, "__qualname__", None) or cls.__name__
    else:
        module = getattr(func, "__module__", "")
    return f"{module}.{qualname}" if module else str(qualname)


def _ndarray_identity(value: "np.ndarray", *, path: str, seen: frozenset[int]) -> dict[str, Any]:
    """Full content, plus the dtype and shape that give the content its meaning.

    A repr will not do: NumPy truncates it above a thousand elements, so two arrays
    differing only in the elided middle printed identically and shared a cell digest.
    The hash reads every byte, and dtype/shape are carried alongside because the same
    bytes mean different things at a different dtype or shape.

    A buffer containing pointers says nothing about what they point at, so any dtype that
    holds objects ANYWHERE recurses over semantic elements instead of being hashed. The
    test is ``dtype.hasobject`` rather than ``dtype == object``: a structured dtype with
    one object field is not equal to ``object`` but still stores a pointer, and hashing
    that buffer compared addresses -- two arrays with equal elements differed because
    their Python objects were allocated separately. The full dtype string is carried
    either way, so two structured layouts with the same elements stay distinguishable,
    and an element that cannot be identified makes the cell uncacheable rather than
    silently hashing an address.
    """
    if value.dtype.hasobject:
        return {
            "__ndarray__": {
                "dtype": str(value.dtype),
                "shape": [int(size) for size in value.shape],
                "items": _json_ready(value.tolist(), path=f"{path}.tolist()", seen=seen),
            }
        }
    contiguous = np.ascontiguousarray(value)
    return {
        "__ndarray__": {
            "dtype": str(value.dtype),
            "shape": [int(size) for size in value.shape],
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    }


def _busdaycalendar_identity(value: "np.busdaycalendar") -> dict[str, Any]:
    """A business-day calendar by what it MEANS: its week mask and its holidays.

    ``numpy.busdaycalendar`` has no readable form of its own -- its repr is an address --
    and it is the only place a custom business offset's real calendar lives. It is not
    redundant with the offset's ``kwds``: constructing ``CustomBusinessDay(calendar=...)``
    leaves ``kwds["weekmask"]`` reporting the DEFAULT ``'Mon Tue Wed Thu Fri'`` while the
    calendar holds the mask that actually applies, so identifying the offset by kwds alone
    would make two different calendars look alike.

    Holidays are rendered as ISO day strings rather than hashed, because they are few,
    they are the readable part of what makes two calendars differ, and a hash would hide
    them from the provenance echo.
    """
    return {
        "__busdaycalendar__": {
            "weekmask": [bool(flag) for flag in value.weekmask],
            "holidays": [str(day) for day in value.holidays],
        }
    }


def _dateoffset_identity(value: "DateOffset", *, path: str, seen: frozenset[int]) -> dict[str, Any]:
    """A pandas offset by its semantics, not by the label it prints.

    ``freqstr`` is what the public export records and it is not an identity: every
    ``CustomBusinessDay`` reports ``"C"`` whatever holidays or week mask it carries, so
    two policies whose refit cadence genuinely differs shared one cell digest. The
    concrete type, the multiplier, ``normalize`` and the full constructor state
    (``kwds``) are what actually decide when the offset lands.

    ``kwds`` goes through the ordinary rules, so a calendar inside it is serialized
    semantically and anything this module cannot identify -- a nested value of an unknown
    type -- makes the cell uncacheable with its field path rather than being guessed at.
    """
    try:
        multiplier = int(value.n)
        normalize = bool(value.normalize)
        keywords = dict(value.kwds)
    except Exception as exc:
        raise _UndigestibleCell(
            f"{path}: {type(value).__name__} did not expose its offset state "
            f"({type(exc).__name__}: {exc}), so its cadence cannot be identified; "
            "recomputing instead"
        ) from exc
    return {
        "__dateoffset__": {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "n": multiplier,
            "normalize": normalize,
            "kwds": _json_ready(keywords, path=f"{path}.kwds", seen=seen),
        }
    }


def _set_identity(value: "set[Any] | frozenset[Any]", *, path: str, seen: frozenset[int]) -> dict[str, Any]:
    """Canonical order, and the container type kept.

    ``repr`` of a set follows hash iteration order, which for strings varies with
    ``PYTHONHASHSEED`` -- so an arm carrying a set-valued parameter could never match its
    own cell from a previous process. Elements are serialized first and then ordered by
    their canonical JSON text, because the elements themselves need not be orderable.
    """
    items = [
        _json_ready(item, path=f"{path}{{{index}}}", seen=seen)
        for index, item in enumerate(value)
    ]
    return {
        "__set__": {
            "type": type(value).__name__,
            "items": sorted(items, key=lambda item: json.dumps(item, sort_keys=True)),
        }
    }


def _json_ready(value: Any, *, path: str = "value", seen: frozenset[int] = frozenset()) -> Any:
    """Canonical, deterministic identity for one value, or refusal.

    Everything this returns is JSON-serializable and depends only on the value, never on
    a repr, an address, or hash iteration order. What cannot be identified that way
    raises :class:`_UndigestibleCell` naming ``path``, so
    :func:`result_cell_identity` returns ``digest=None`` with an actionable reason and
    the cell is recomputed -- rather than minting a digest that two different values
    could share.

    ``path`` accumulates as the traversal descends so the reason points at the field.
    ``seen`` carries the containers currently being serialized, so a self-referential
    structure is refused instead of recursing until the interpreter gives out.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        # Base64 rather than a repr, and the type kept: b"a" and bytearray(b"a") are
        # different objects to anything that consumes them.
        return {
            "__bytes__": {
                "type": (
                    "bytearray"
                    if isinstance(value, _FrozenByteArray)
                    else type(value).__name__
                ),
                "base64": base64.b64encode(bytes(value)).decode("ascii"),
            }
        }
    if isinstance(value, np.generic):
        # ``.item()`` may yield something still unsupported (a complex, say), so it goes
        # back through the rules rather than being returned unchecked.
        return _json_ready(value.item(), path=path, seen=seen)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, _dt.timedelta):
        # Nanoseconds, exactly: ``pd.Timedelta`` carries them and subclasses
        # ``timedelta``, whose own fields stop at microseconds, so reading days/seconds/
        # microseconds off it would silently drop the finest resolution it can express.
        # The concrete type is recorded alongside, because a duration of one second is
        # not the same VALUE as a pandas duration of one second -- they differ in what
        # they do downstream, and the count alone cannot tell them apart.
        nanoseconds = getattr(value, "value", None)
        if nanoseconds is None:
            nanoseconds = (
                (value.days * 86_400 + value.seconds) * 1_000_000_000
                + value.microseconds * 1_000
            )
        return {
            "__timedelta__": {"type": _type_name(value), "ns": int(nanoseconds)}
        }
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        # After the Timestamp branch on purpose: a Timestamp IS a datetime, and it keeps
        # its long-standing bare-ISO rendering so existing digests do not move. These are
        # tagged because a bare ISO string would be indistinguishable from a str that
        # happens to spell a date.
        moment: dict[str, Any] = {"type": _type_name(value), "iso": value.isoformat()}
        fold = getattr(value, "fold", None)
        if fold is not None:
            # ``isoformat`` does not carry fold, so the two sides of a repeated
            # wall-clock hour would otherwise be one value. ``date`` has no fold, which
            # is why this is conditional rather than assumed.
            moment["fold"] = int(fold)
        return {"__datetime__": moment}

    marker = id(value)
    if marker in seen:
        raise _UndigestibleCell(
            f"{path} closes a reference cycle (a self-referential structure), which "
            "has no finite canonical form; recomputing instead"
        )
    descended = seen | {marker}

    if type(value) is np.ndarray:
        return _ndarray_identity(value, path=path, seen=descended)
    if isinstance(value, np.ndarray):
        # A subclass carries meaning the base array's buffer does not. A masked array is
        # the clear case: its mask decides which elements exist at all, and hashing the
        # data view alone gave two arrays that behave differently everywhere the same
        # cell. Rather than guess which attributes of an unknown subclass matter, only
        # the exact type is identified and the rest are recomputed.
        raise _UndigestibleCell(
            f"{path} is a {type(value).__name__}, a numpy.ndarray subclass whose "
            "identity is not carried by the base array's dtype, shape and buffer alone "
            "(a masked array's mask, for instance, is invisible to them). Only a plain "
            "numpy.ndarray is identified; recomputing instead."
        )
    if isinstance(value, (set, frozenset)):
        return _set_identity(value, path=path, seen=descended)
    if isinstance(value, np.busdaycalendar):
        return _busdaycalendar_identity(value)
    if isinstance(value, DateOffset):
        return _dateoffset_identity(value, path=path, seen=descended)

    # A ModelSpec is callable, but it is not an anonymous piece of code: the registry
    # already identifies its fit function by name plus the backend versions recorded
    # elsewhere in the payload. _model_identity owns that registry-versus-custom rule,
    # so a nested spec asks it rather than being forced to carry a marker a registered
    # model has no reason to have.
    if _is_model_spec(value):
        return _model_identity(value, params=None)
    # Everything else callable is identified AS CODE, before any structural view of it.
    # A callable object that also exposes to_dict() or dataclass fields would otherwise
    # be serialized by that structure and never asked for __mf_digest__ -- the same
    # bypass this module closes for a plain function, wearing a different hat. Structure
    # cannot stand in for a body: two functors can carry identical fields and compute
    # entirely different things.
    if callable(value):
        return _callable_identity(value, path=path)

    if isinstance(value, Mapping):
        # Keys are checked BEFORE anything is rendered, so a mapping with one bad key is
        # refused whole rather than half-serialized.
        for key in value:
            if not isinstance(key, str):
                raise _UndigestibleCell(
                    f"{path} has a key of type {type(key).__name__}, and identity "
                    "records a mapping by its keys' text. Only a str has text that is "
                    "both unambiguous and the same in every process: an arbitrary "
                    "object falls back to a repr carrying its address, and an int key "
                    "is indistinguishable from the string of the same digits. Use "
                    "string keys; recomputing instead."
                )
        rendered: dict[str, Any] = {}
        for key in sorted(value):
            if key in rendered:
                # Unreachable for a dict, whose keys are unique by construction; kept
                # for a Mapping implementation that yields a key twice, because silently
                # keeping the last would make two different configurations look alike.
                raise _UndigestibleCell(
                    f"{path} yields the key {key!r} more than once, so it has no "
                    "unambiguous canonical form; recomputing instead"
                )
            rendered[key] = _json_ready(value[key], path=f"{path}.{key}", seen=descended)
        return rendered
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_ready(item, path=f"{path}[{index}]", seen=descended)
            for index, item in enumerate(value)
        ]
    if _is_search_spec(value):
        return _search_spec_identity(value, path=path, seen=descended)
    if _dc.is_dataclass(value) and not isinstance(value, type):
        return {
            str(field.name): _json_ready(
                getattr(value, field.name), path=f"{path}.{field.name}", seen=descended
            )
            for field in sorted(_dc.fields(value), key=lambda item: item.name)
        }
    if hasattr(value, "to_dict"):
        payload = _to_dict_or_undigestible(value, path=path)
        return _json_ready(payload, path=path, seen=descended)
    raise _UndigestibleCell(
        f"{path} is a {type(value).__name__}, which the result store cannot identify "
        "deterministically; recomputing instead. Supported values are JSON scalars, "
        "paths, timestamps, dates, times, timedeltas, bytes, NumPy scalars, plain "
        "numpy.ndarray (not its subclasses), pandas date offsets and business-day "
        "calendars, sets, string-keyed mappings, sequences, dataclasses, objects "
        "exposing to_dict(), and callables carrying __mf_digest__."
    )


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _unlink_quietly(path: str | Path) -> bool:
    """Remove *path* if it is there; return whether it actually went away.

    The write paths call this to clean up a temp file and ignore the answer -- a
    cleanup that fails is not worth raising over. The purge helpers read it, so their
    counts describe deletions that happened rather than deletions attempted.
    """
    try:
        Path(path).unlink()
    except OSError:
        return False
    return True


__all__ = ["ResultStore", "purge_result_store", "result_store_summary"]
