"""Cross-run result store for pipeline forecast cells.

The store is intentionally small and file-backed.  A caller owns the directory;
macroforecast writes one parquet payload and one JSON manifest per digest under
``<store>/cells``.  There is no store-level locking: use one writer at a time.
"""
from __future__ import annotations

import dataclasses as _dc
import base64
import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from macroforecast.pipeline.spec import Arm, PipelineSpec, ResolvedTarget, is_vintage_aware


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


def result_cell_identity(
    spec: PipelineSpec,
    arm: Arm,
    target: ResolvedTarget,
    *,
    horizon: int,
    data_identity: Mapping[str, Any],
) -> ResultCellIdentity:
    """Return the digest and human-readable echo for one result-store cell.

    A ``None`` digest means the cell is deliberately not cacheable, normally
    because a user-owned callable lacks an explicit ``__mf_digest__`` opt-in.
    """

    data_fingerprint = data_identity.get("fingerprint")
    try:
        effective_features = _retargeted_features(arm.features, target.name)
        payload: dict[str, Any] = {
            "data_fingerprint": _json_ready(data_fingerprint),
            "target": {
                "name": target.name,
                "transform": target.transform,
                "forecast_policy": target.policy,
            },
            "horizon": int(horizon),
            "arm": {
                "name": arm.name,
                "model": _model_identity(arm.model, params=arm.params),
                "params": _json_ready(arm.params),
                "preset": _model_preset(arm.model),
                "features": _feature_identity(effective_features),
                "feature_policy": _object_identity(arm.feature_policy),
                "preprocessing": _preprocessing_identity(
                    arm.preprocessing if arm.preprocessing is not None else spec.preprocessing
                ),
                "preprocessing_policy": _object_identity(
                    arm.preprocessing_policy
                    if arm.preprocessing is not None
                    else spec.preprocessing_policy
                ),
                "model_selection": _object_identity(arm.model_selection),
                "model_selection_metric": arm.model_selection_metric,
                "window": _object_identity(arm.window if arm.window is not None else spec.window),
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


def purge_result_store(
    store: str | Path,
    *,
    before: str | datetime | None = None,
    version: str | None = None,
    digests: Sequence[str] | None = None,
) -> int:
    """Delete result-store cells matching the supplied filters and return a count."""

    root = Path(store) / "cells"
    digest_filter = {str(d) for d in digests} if digests is not None else None
    before_dt = _parse_datetime(before) if before is not None else None
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
        _unlink_quietly(root / f"{digest}.parquet")
        _unlink_quietly(manifest_path)
        deleted += 1
    return deleted


def _retargeted_features(features: Any, target_name: str) -> Any:
    if features is None:
        return None
    needs_retarget = (
        getattr(features, "target", None) != target_name
        or bool(getattr(features, "targets", ()))
    )
    if not needs_retarget:
        return features
    try:
        kwargs: dict[str, Any] = {"target": target_name}
        if getattr(features, "targets", None):
            kwargs["targets"] = ()
        return _dc.replace(features, **kwargs)
    except Exception:
        return features


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

    if isinstance(model, str):
        spec = get_model(model, params=params)
        return {"kind": "registry", "spec": spec.to_dict()}
    if isinstance(model, ModelSpec):
        spec = model.with_params(**dict(params or {})) if params else model
        registered = MODEL_SPECS.get(spec.name)
        if registered is not None and registered.fit_func is spec.fit_func:
            return {"kind": "registry", "spec": spec.to_dict()}
        return {
            "kind": "custom",
            "spec": _json_ready(spec.to_dict()),
            "mf_digest": _callable_digest(spec, spec.fit_func, path=f"model {spec.name!r}"),
        }
    if callable(model):
        try:
            spec = get_model(model, params=params)
        except Exception:
            return {
                "kind": "custom_callable",
                "name": _callable_name(model),
                "params": _json_ready(params),
                "mf_digest": _callable_digest(model, path=f"model callable {_callable_name(model)!r}"),
            }
        return {"kind": "registry", "spec": spec.to_dict()}
    return {"kind": type(model).__name__, "repr": repr(model), "params": _json_ready(params)}


def _model_preset(model: Any) -> Any:
    return getattr(model, "preset", None) or getattr(model, "default_preset", None)


def _feature_identity(features: Any) -> Any:
    if features is None:
        return None
    return {
        "spec": _object_identity(features),
        "custom_callable_digests": _custom_feature_digests(features),
    }


def _preprocessing_identity(preprocessing: Any) -> Any:
    if preprocessing is None:
        return None
    return {
        "spec": _object_identity(preprocessing),
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


def _object_identity(value: Any) -> Any:
    _assert_safe_generic_callables(value)
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        try:
            return _json_ready(value.to_dict())
        except Exception as exc:
            return {"repr": repr(value), "to_dict_error": f"{type(exc).__name__}: {exc}"}
    return _json_ready(value)


def _assert_safe_generic_callables(value: Any, *, path: str = "value") -> None:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    if callable(value):
        _callable_digest(value, path=path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_safe_generic_callables(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, item in enumerate(value):
            _assert_safe_generic_callables(item, path=f"{path}[{idx}]")


def _callable_digest(*candidates: Any, path: str) -> str:
    for candidate in candidates:
        marker = getattr(candidate, "__mf_digest__", None)
        if marker is not None:
            return str(marker)
    raise _UndigestibleCell(
        f"{path} is a custom callable without __mf_digest__; recomputing instead"
    )


def _callable_name(func: Any) -> str:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    if _dc.is_dataclass(value):
        return _json_ready(_dc.asdict(cast(Any, value)))
    if hasattr(value, "to_dict"):
        try:
            return _json_ready(value.to_dict())
        except Exception:
            return repr(value)
    if callable(value):
        return {"callable": _callable_name(value), "mf_digest": getattr(value, "__mf_digest__", None)}
    return value if _json_scalar(value) else repr(value)


def _json_scalar(value: Any) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


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


def _unlink_quietly(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


__all__ = ["ResultStore", "purge_result_store", "result_store_summary"]
