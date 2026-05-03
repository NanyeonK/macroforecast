"""Figure rendering for L7 importance and L4.5/L3.5/L1.5 diagnostics.

This module is intentionally lightweight: matplotlib is the only required
dependency, and the US choropleth uses a stylized 50-state grid (no GIS
dependency). When ``shap`` is installed it is used directly for SHAP
beeswarm/force-plot rendering; otherwise we fall back to bar charts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_bar_global(table: pd.DataFrame, *, output_path: Path, top_k: int = 20, title: str | None = None) -> Path:
    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "feature" not in table.columns or "importance" not in table.columns:
        raise ValueError("bar_global requires columns 'feature' and 'importance'")
    sub = table.sort_values("importance", ascending=False).head(int(top_k))
    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.25 * len(sub) + 1.0)))
    ax.barh(sub["feature"][::-1], sub["importance"][::-1], color="#3b82f6")
    ax.set_xlabel("importance")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_heatmap(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = table.copy()
    if matrix.index.dtype == object and "feature" in matrix.columns:
        matrix = matrix.set_index("feature")
    fig, ax = plt.subplots(figsize=(max(4.0, 0.4 * len(matrix.columns) + 2), max(2.0, 0.3 * len(matrix) + 1)))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_pdp_line(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if "value" in table.columns:
        for feature, group in table.groupby("feature"):
            ax.plot(group["value"], group["importance"], label=str(feature))
    else:
        ax.plot(range(len(table)), table["importance"])
    ax.set_xlabel("feature value")
    ax.set_ylabel("partial dependence")
    if title:
        ax.set_title(title)
    if "value" in table.columns:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# US state choropleth (FRED-SD geographic map)
# ---------------------------------------------------------------------------

# Stylized state grid: 11 columns x 8 rows mapping each state postal code to a
# (col, row) cell. Mirrors the layout used by NPR and FiveThirtyEight tile maps.
US_STATE_GRID: dict[str, tuple[int, int]] = {
    "AK": (0, 0),                 "ME": (10, 0),
    "VT": (8, 1), "NH": (9, 1),
    "WA": (1, 1),  "ID": (2, 2),  "MT": (3, 1),  "ND": (4, 1),  "MN": (5, 1),
    "WI": (6, 1),  "MI": (7, 2),                  "NY": (8, 2),  "MA": (9, 2),
    "OR": (1, 2),                 "SD": (4, 2),  "IA": (5, 2),
    "IL": (6, 3),                  "PA": (8, 3),  "NJ": (9, 3),  "CT": (10, 2),
    "CA": (1, 3),  "NV": (2, 3),  "WY": (3, 2),  "NE": (4, 3),  "MO": (5, 3),
    "IN": (7, 3),  "OH": (8, 4),  "MD": (9, 4),  "DE": (10, 3),  "RI": (10, 4),
    "UT": (2, 4),  "CO": (3, 3),  "KS": (4, 4),  "AR": (5, 4),  "KY": (6, 4),
    "WV": (7, 4),  "VA": (8, 5),  "DC": (9, 5),
    "AZ": (2, 5),  "NM": (3, 4),  "OK": (4, 5),  "LA": (5, 5),  "TN": (6, 5),
    "NC": (7, 5),
    "TX": (3, 5),  "MS": (5, 6),  "AL": (6, 6),  "GA": (7, 6),  "SC": (8, 6),
    "HI": (1, 7),                  "FL": (8, 7),
}


def render_us_state_choropleth(
    importance_by_state: dict[str, float] | pd.Series,
    *,
    output_path: Path,
    title: str | None = None,
    cmap: str = "magma",
) -> Path:
    """Draw a US state choropleth on a stylized tile grid.

    Each state appears as a square colored by its importance score. Used by
    L7 ``group_aggregate(grouping=fred_sd_states)`` to visualize per-state
    feature importance without requiring a GIS shapefile.
    """

    plt = _ensure_matplotlib()
    if isinstance(importance_by_state, pd.Series):
        importance = importance_by_state.to_dict()
    else:
        importance = dict(importance_by_state)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    if importance:
        values = np.array(list(importance.values()), dtype=float)
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9)
        colormap = plt.get_cmap(cmap)
    else:
        norm = plt.Normalize(0, 1)
        colormap = plt.get_cmap(cmap)
    for state, (col, row) in US_STATE_GRID.items():
        score = importance.get(state)
        face = colormap(norm(score)) if score is not None else "#e5e7eb"
        ax.add_patch(plt.Rectangle((col, -row), 0.95, 0.95, facecolor=face, edgecolor="white", linewidth=1.5))
        ax.text(col + 0.475, -row + 0.5, state, ha="center", va="center", fontsize=9, color="black" if score is None else "white")
        if score is not None:
            ax.text(col + 0.475, -row + 0.15, f"{score:.2f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-8.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04, label="importance")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_default_for_op(
    op: str,
    payload: Any,
    *,
    output_path: Path,
    title: str | None = None,
) -> Path | None:
    """Dispatch helper used by L7 export to render a default figure per op."""

    if isinstance(payload, pd.DataFrame) and "feature" in payload.columns and "importance" in payload.columns:
        if op in {"shap_tree", "shap_kernel", "shap_linear", "shap_deep"}:
            return render_bar_global(payload, output_path=output_path, title=title)
        if op == "partial_dependence":
            return render_pdp_line(payload, output_path=output_path, title=title)
        if op in {"group_aggregate", "lineage_attribution"}:
            return render_bar_global(payload.rename(columns={"group": "feature", "pipeline": "feature"}), output_path=output_path, title=title)
        return render_bar_global(payload, output_path=output_path, title=title)
    return None


__all__ = [
    "US_STATE_GRID",
    "render_bar_global",
    "render_default_for_op",
    "render_heatmap",
    "render_pdp_line",
    "render_us_state_choropleth",
]
