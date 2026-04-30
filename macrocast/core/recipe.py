from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cache import recipe_hash
from .dag import DAG, LayerId
from .sweep import Cell, SweepCombination, expand_sweeps
from .yaml import RecipeMetadata, LayerYamlSpec, normalize_to_dag_form, parse_recipe_yaml, recipe_layers_from_yaml


@dataclass(frozen=True)
class Recipe:
    metadata: RecipeMetadata = field(default_factory=RecipeMetadata)
    layers: dict[LayerId, LayerYamlSpec] = field(default_factory=dict)
    sweep_combination: SweepCombination = field(default_factory=SweepCombination)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Recipe":
        meta_raw = raw.get("metadata", {}) or {}
        metadata = RecipeMetadata(
            name=meta_raw.get("name", ""),
            description=meta_raw.get("description", ""),
            author=meta_raw.get("author", ""),
            created_at=meta_raw.get("created_at", ""),
            extra={key: value for key, value in meta_raw.items() if key not in {"name", "description", "author", "created_at"}},
        )
        combo_raw = raw.get("sweep_combination", {}) or {}
        return cls(
            metadata=metadata,
            layers=recipe_layers_from_yaml(raw),
            sweep_combination=SweepCombination(
                mode=combo_raw.get("mode", "grid"),
                groups=tuple(combo_raw.get("groups", ())),
            ),
        )

    @classmethod
    def from_yaml(cls, text: str) -> "Recipe":
        return cls.from_dict(parse_recipe_yaml(text))

    def to_dag_form(self) -> dict[LayerId, DAG]:
        return {
            layer_id: normalize_to_dag_form(spec.raw_yaml, layer_id)
            for layer_id, spec in self.layers.items()
            if spec.enabled
        }

    @property
    def cells(self) -> list[Cell]:
        return expand_sweeps(self.to_dag_form(), self.sweep_combination)

    @property
    def hash(self) -> str:
        return recipe_hash(self.to_dag_form())
