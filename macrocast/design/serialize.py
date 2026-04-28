from __future__ import annotations

from dataclasses import asdict

from .normalize import (
    normalize_comparison_contract,
    normalize_fixed_design,
    normalize_replication_input,
    normalize_varying_design,
)
from .types import DesignFrame
from .build import build_design_frame


def design_to_dict(stage0: DesignFrame) -> dict:
    return asdict(stage0)


def design_from_dict(payload: dict) -> DesignFrame:
    return build_design_frame(
        fixed_design=normalize_fixed_design(payload["fixed_design"]),
        comparison_contract=normalize_comparison_contract(payload["comparison_contract"]),
        varying_design=normalize_varying_design(payload.get("varying_design")),
        replication_input=normalize_replication_input(payload.get("replication_input")),
        experiment_unit=payload.get("experiment_unit"),
    )
