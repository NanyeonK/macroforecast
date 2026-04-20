from __future__ import annotations

from typing import Any

from .types import RecipeSpec
from ..design import DesignFrame


def build_recipe_spec(
    *,
    recipe_id: str,
    stage0: DesignFrame,
    target: str,
    horizons: tuple[int, ...],
    raw_dataset: str,
    benchmark_config: dict[str, Any] | None = None,
    data_task_spec: dict[str, Any] | None = None,
    training_spec: dict[str, Any] | None = None,
    data_vintage: str | None = None,
    targets: tuple[str, ...] | None = None,
) -> RecipeSpec:
    return RecipeSpec(
        recipe_id=recipe_id,
        stage0=stage0,
        target=target,
        horizons=tuple(horizons),
        raw_dataset=raw_dataset,
        benchmark_config=dict(benchmark_config or {}),
        data_task_spec=dict(data_task_spec or {}),
        training_spec=dict(training_spec or {}),
        data_vintage=data_vintage,
        targets=tuple(targets or ()),
    )
