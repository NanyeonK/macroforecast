"""Offline-friendly helpers for the v0.8 high-level API tests.

The simple-API surface (``mf.forecast`` / ``mf.Experiment``) deliberately
defaults to FRED-MD/QD/SD official sources because that is the canonical
research workflow. Inside the test suite we want to exercise the full
``RecipeBuilder -> execute_recipe -> ForecastResult`` flow without
hitting the network, so we swap the recipe's L1 block for an inline
custom panel via the ``Experiment._builder`` private attribute. The
swap stays inside the test layer; the public surface remains
network-by-default.
"""
from __future__ import annotations

from typing import Any

import macroforecast as mf


def _custom_panel(n: int = 24) -> dict[str, list[Any]]:
    """A tiny synthetic monthly panel suitable for ar_p / ridge fits."""

    dates = [f"2018-{m:02d}-01" for m in range(1, 13)] + [f"2019-{m:02d}-01" for m in range(1, 13)]
    return {
        "date": dates[:n],
        "y": [float(v) for v in range(1, n + 1)],
        "x1": [float(v) / 2 for v in range(1, n + 1)],
        "x2": [float((-1) ** v) for v in range(1, n + 1)],
    }


def install_custom_panel(experiment: "mf.Experiment", *, n: int = 24, target: str = "y") -> None:
    """Replace the experiment's L1 block with a custom_panel_inline source.

    Mutates ``experiment._builder._recipe`` in place so subsequent
    ``.run()`` / ``.to_recipe_dict()`` calls see the offline panel.
    """

    panel = _custom_panel(n=n)
    recipe = experiment._builder._recipe
    recipe["1_data"] = {
        "fixed_axes": {
            "custom_source_policy": "custom_panel_only",
            "frequency": "monthly",
            "horizon_set": "custom_list",
            "target_structure": "single_target",
        },
        "leaf_config": {
            "target": target,
            "target_horizons": [1],
            "custom_panel_inline": panel,
        },
    }
    # Ensure L4 fit nodes carry a min_train_size so the small panel can
    # produce at least a few origins.
    l4 = recipe.get("4_forecasting_model", {})
    for node in l4.get("nodes", []):
        if isinstance(node, dict) and node.get("op") == "fit_model":
            node.setdefault("params", {}).setdefault("min_train_size", 4)
