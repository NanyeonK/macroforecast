from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SavedModel:
    """Paths and status returned by model fit persistence helpers."""

    model_path: str | None
    metadata_path: str
    save_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "metadata_path": self.metadata_path,
            "save_error": self.save_error,
        }


def save_fit(
    fit: Any,
    model_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    allow_pickle_error: bool = True,
) -> SavedModel:
    """Persist a fitted model object and a JSON metadata sidecar.

    This helper owns only the storage format. Forecasting runners decide which
    fit to save, where to save it, and which experiment metadata to attach.
    """

    pickle_path = Path(model_path)
    sidecar_path = (
        Path(metadata_path) if metadata_path is not None else pickle_path.with_suffix(".json")
    )
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    saved_model_path: Path | None = pickle_path
    save_error = None
    try:
        with pickle_path.open("wb") as handle:
            pickle.dump(fit, handle)
    except Exception as exc:  # noqa: BLE001 - custom/local fits may not pickle.
        if not allow_pickle_error:
            raise
        saved_model_path = None
        save_error = f"{type(exc).__name__}: {exc}"

    sidecar = {
        **dict(metadata or {}),
        "fit": fit.to_metadata() if hasattr(fit, "to_metadata") else None,
        "model_path": None if saved_model_path is None else str(saved_model_path),
        "metadata_path": str(sidecar_path),
        "save_error": save_error,
    }
    sidecar_path.write_text(
        json.dumps(_json_ready(sidecar), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return SavedModel(
        model_path=None if saved_model_path is None else str(saved_model_path),
        metadata_path=str(sidecar_path),
        save_error=save_error,
    )


def load_fit(model_path: str | Path) -> Any:
    """Load a fitted model object saved by `save_fit()`."""

    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, pd.Series):
        return {
            "name": value.name,
            "index": [_json_ready(item) for item in value.index],
            "data": [_json_ready(item) for item in value.to_list()],
        }
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "index": [_json_ready(item) for item in value.index],
            "data": _json_ready(value.to_dict(orient="list")),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["SavedModel", "load_fit", "save_fit"]
