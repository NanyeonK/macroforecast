from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..design import DesignFrame


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    stage0: DesignFrame
    target: str
    horizons: tuple[int, ...]
    raw_dataset: str
    benchmark_config: dict[str, Any] = field(default_factory=dict)
    data_task_spec: dict[str, Any] = field(default_factory=dict)
    training_spec: dict[str, Any] = field(default_factory=dict)
    data_vintage: str | None = None
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    recipe_id: str
    route_owner: str
    artifact_subdir: str
