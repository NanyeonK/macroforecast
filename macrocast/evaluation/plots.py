"""Visualization functions replicating CLSS 2021 Figures 1, 2, 3, and 6.

Three public functions:

    marginal_effect_plot     — Fig 1 (MARX), Fig 2 (path-avg), Fig 4 (MAF),
                               Fig 5 (F): horizontal dot-and-whisker grid
                               showing marginal contribution by model/horizon.

    variable_importance_plot — Fig 3: stacked-bar panels showing importance
                               share by semantic group, one panel per target.

    cumulative_squared_error_plot — Fig 6: cumulative squared error lines
                               over OOS dates, with optional recession shading.

All functions return a matplotlib Figure and do NOT call plt.show().
Callers are responsible for display or saving.

Reference: Coulombe, Leroux, Stevanovic, Surprenant (2021),
    "Macroeconomic Data Transformations Matter", IJF.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# Default 10-series target palette (CLSS 2021 style)
_TARGET_PALETTE: dict[str, str] = {
    "INDPRO":   "#1f7a1f",   # dark green
    "PAYEMS":   "#ff7f0e",   # orange
    "UNRATE":   "#d62728",   # red
    "RPI":      "#1f77b4",   # blue
    "PCE":      "#9467bd",   # purple
    "RSAFS":    "#8c564b",   # brown
    "HOUST":    "#e377c2",   # pink
    "M2SL":     "#7f7f7f",   # gray
    "CPIAUCSL": "#bcbd22",   # yellow-green
    "PPIACO":   "#17becf",   # teal
}

# Group palette for stacked VI bars (CLSS 2021 style)
_VI_GROUP_PALETTE: dict[str, str] = {
    "AR":       "#2ca02c",   # teal/green
    "AR-MARX":  "#ff7f0e",   # orange
    "Factors":  "#1a1a1a",   # black
    "MARX":     "#d62728",   # red
    "X":        "#1f77b4",   # blue
    "Level":    "#9467bd",   # purple
    "other":    "#aaaaaa",   # gray
}

# Preferred draw order for VI groups (bottom to top in stacked bars)
_VI_GROUP_ORDER: list[str] = [
    "AR", "AR-MARX", "Factors", "MARX", "X", "Level", "other"
]


# ---------------------------------------------------------------------------
# Function 1 — marginal_effect_plot (Fig 1, 2, 4, 5)
# ---------------------------------------------------------------------------


def marginal_effect_plot(
    mc_df: pd.DataFrame,
    feature: str,
    models: list[str] | None = None,
    horizons: list[int] | None = None,
    target_palette: dict[str, str] | None = None,
    zero_line: bool = True,
    xlim: tuple[float, float] = (-0.4, 0.4),
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Horizontal dot-and-whisker grid replicating CLSS 2021 Fig 1/2/4/5.

    Each grid cell covers one (horizon, model) combination.  Within each cell
    a single dot per target variable is drawn at the estimated marginal
    contribution ``alpha`` with horizontal whiskers spanning ``[ci_low,
    ci_high]``.  An optional vertical dashed line marks x=0.

    Parameters
    ----------
    mc_df : pd.DataFrame
        Output of ``marginal_contribution()`` (or ``marginal_contribution_all``).
        Required columns: ``feature``, ``model``, ``horizon``, ``alpha``,
        ``ci_low``, ``ci_high``.  Optional column: ``target``.
        When ``target`` is absent every row is treated as a single target pool.
    feature : str
        Which feature to display.  Rows where ``mc_df["feature"] != feature``
        are dropped before plotting.
    models : list of str, optional
        Ordered list of model ids to show as columns.  Auto-detected from data
        if None (sorted alphabetically).
    horizons : list of int, optional
        Ordered list of horizons to show as rows.  Auto-detected from data if
        None (sorted numerically).
    target_palette : dict of str -> str, optional
        Mapping from target name to hex color.  Falls back to
        ``_TARGET_PALETTE`` for known targets; unknown targets receive a
        matplotlib tab10 color.
    zero_line : bool, default True
        Draw a vertical dashed line at x=0 in each cell.
    xlim : (float, float), default (-0.4, 0.4)
        Shared x-axis limits for all cells.
    figsize : (float, float), optional
        Override figure size.  Default scales with grid dimensions.
    title : str, optional
        Suptitle text.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Filter to the requested feature
    df = mc_df.loc[mc_df["feature"] == feature].copy()
    if df.empty:
        raise ValueError(
            f"No rows for feature='{feature}' in mc_df. "
            f"Available features: {mc_df['feature'].unique().tolist()}"
        )

    # Determine whether a target column is present
    has_target = "target" in df.columns
    if not has_target:
        # Treat as single target with a placeholder name
        df = df.copy()
        df["target"] = "_all_"

    # Resolve models and horizons
    if models is None:
        models = sorted(df["model"].unique().tolist())
    if horizons is None:
        horizons = sorted(df["horizon"].unique().tolist())

    # Build effective palette: start with defaults, then extend with tab10
    palette = dict(_TARGET_PALETTE)
    if target_palette is not None:
        palette.update(target_palette)

    all_targets = df["target"].unique().tolist()
    tab10 = plt.get_cmap("tab10")
    extra_idx = 0
    for tgt in all_targets:
        if tgt not in palette:
            palette[tgt] = tab10(extra_idx % 10)
            extra_idx += 1

    n_rows = len(horizons)
    n_cols = len(models)

    # Default figsize: 2.8 inches per column, 1.8 per row plus header/footer
    if figsize is None:
        figsize = (max(4.0, 2.8 * n_cols), max(3.0, 1.8 * n_rows + 1.0))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharey=False,
    )

    for row_idx, h in enumerate(horizons):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx][col_idx]

            # Subset to this (model, horizon) cell
            cell = df.loc[(df["model"] == model) & (df["horizon"] == h)]

            if cell.empty:
                ax.set_visible(False)
                continue

            # One dot per target — sort for consistent vertical ordering
            targets_in_cell = sorted(cell["target"].unique().tolist())

            for y_pos, tgt in enumerate(targets_in_cell):
                row = cell.loc[cell["target"] == tgt]
                if row.empty:
                    continue

                # Take first row (should be unique per target/model/horizon)
                alpha_val = float(row["alpha"].iloc[0])
                ci_lo = float(row["ci_low"].iloc[0])
                ci_hi = float(row["ci_high"].iloc[0])
                color = palette.get(tgt, "#333333")

                # Horizontal whiskers
                ax.plot(
                    [ci_lo, ci_hi],
                    [y_pos, y_pos],
                    color=color,
                    linewidth=1.2,
                    solid_capstyle="butt",
                )
                # Central dot
                ax.plot(
                    alpha_val,
                    y_pos,
                    marker="o",
                    markersize=5,
                    color=color,
                    zorder=3,
                    label=tgt if tgt != "_all_" else None,
                )

            # Vertical zero reference line
            if zero_line:
                ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", zorder=1)

            ax.set_xlim(xlim)
            ax.set_ylim(-0.5, len(targets_in_cell) - 0.5)

            # Remove y-tick labels (targets identified by color in legend)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=7)

            # Light grid on x only
            ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
            ax.set_axisbelow(True)

            # Row label on left edge
            if col_idx == 0:
                ax.set_ylabel(f"H={h}", fontsize=8, labelpad=4)

            # Column label on top edge
            if row_idx == 0:
                ax.set_title(model, fontsize=8, pad=4)

    # Build a legend for target colors using the first visible axes
    # Only shown when there are multiple named targets
    legend_targets = [t for t in all_targets if t != "_all_"]
    if legend_targets:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=palette.get(t, "#333333"),
                markersize=6,
                label=t,
            )
            for t in legend_targets
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(legend_targets), 5),
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
        # Extra bottom margin so legend does not overlap subplots
        fig.subplots_adjust(bottom=0.12)

    if title is not None:
        fig.suptitle(title, fontsize=10, y=1.01)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Function 2 — variable_importance_plot (Fig 3)
# ---------------------------------------------------------------------------


def variable_importance_plot(
    vi_avg_df: pd.DataFrame,
    targets: list[str],
    direct_vi_avg_df: pd.DataFrame | None = None,
    group_palette: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Stacked bar panels replicating CLSS 2021 Fig 3.

    One panel per target variable.  For each panel the x-axis carries one bar
    per horizon (showing average importance share for the path-average model)
    plus a "SGRAGR" bar (grand average across all horizons in vi_avg_df) and
    optionally an "AGR" bar (average across all horizons in direct_vi_avg_df).
    Bars are stacked by semantic group using the CLSS 2021 color scheme.

    Parameters
    ----------
    vi_avg_df : pd.DataFrame
        Output of ``average_vi_by_horizon()``.  Required columns:
        ``model_id``, ``feature_set``, ``horizon``, ``group``,
        ``importance_share``.  Optional column: ``target``.
        When ``target`` is absent all rows are treated as one target pool.
    targets : list of str
        Target labels to display (used both to filter vi_avg_df and to title
        each panel).  If vi_avg_df has no ``target`` column the list is used
        only for panel titles and must have length 1.
    direct_vi_avg_df : pd.DataFrame or None
        When provided, adds an "AGR" bar at the right of each panel using this
        alternative model's importances (e.g., the direct forecasting scheme).
        Same schema as vi_avg_df.
    group_palette : dict of str -> str, optional
        Override color mapping.  Falls back to ``_VI_GROUP_PALETTE`` for
        known groups; unknown groups receive a tab10 color.
    figsize : (float, float), optional
        Override figure size.  Default scales with number of targets.
    title : str, optional
        Suptitle text.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Resolve palette
    palette = dict(_VI_GROUP_PALETTE)
    if group_palette is not None:
        palette.update(group_palette)

    # If vi_avg_df lacks a target column, synthesise one using the first entry
    # in targets (caller guarantees len(targets)==1 in that case).
    df = vi_avg_df.copy()
    if "target" not in df.columns:
        if len(targets) != 1:
            raise ValueError(
                "vi_avg_df has no 'target' column but targets has more than "
                "one entry.  Add a 'target' column or pass a single-element "
                "targets list."
            )
        df["target"] = targets[0]

    # Filter to requested targets
    df = df.loc[df["target"].isin(targets)]

    # Similarly handle direct_vi_avg_df
    direct_df: pd.DataFrame | None = None
    if direct_vi_avg_df is not None:
        direct_df = direct_vi_avg_df.copy()
        if "target" not in direct_df.columns:
            direct_df["target"] = targets[0]
        direct_df = direct_df.loc[direct_df["target"].isin(targets)]

    n_panels = len(targets)
    if figsize is None:
        figsize = (max(4.0, 3.5 * n_panels), 4.0)

    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize, squeeze=False, sharey=True
    )

    # Determine all groups present and their draw order
    all_groups_in_data = df["group"].unique().tolist()
    if direct_df is not None:
        all_groups_in_data += direct_df["group"].unique().tolist()
    draw_order = [g for g in _VI_GROUP_ORDER if g in all_groups_in_data]
    # Append any unknown groups not in the canonical order
    for g in all_groups_in_data:
        if g not in draw_order:
            draw_order.append(g)

    # Assign fallback colors for unknown groups
    tab10 = plt.get_cmap("tab10")
    extra_idx = 0
    for g in draw_order:
        if g not in palette:
            palette[g] = tab10(extra_idx % 10)
            extra_idx += 1

    for panel_idx, target in enumerate(targets):
        ax = axes[0][panel_idx]
        target_df = df.loc[df["target"] == target]

        horizons_in_data = sorted(target_df["horizon"].unique().tolist())

        # Build x-tick labels: horizon values + "SGRAGR" + optional "AGR"
        x_labels: list[str] = [str(h) for h in horizons_in_data] + ["SGRAGR"]
        if direct_df is not None:
            x_labels.append("AGR")

        x_positions = list(range(len(x_labels)))
        bar_width = 0.7

        # Helper: pivot one DataFrame subset into a {group: share} dict
        def _pivot_shares(sub: pd.DataFrame) -> dict[str, float]:
            """Sum importance shares by group (should already be normalised)."""
            if sub.empty:
                return {}
            return (
                sub.groupby("group")["importance_share"].mean().to_dict()
            )

        # Build per-bar share dicts
        bar_shares: list[dict[str, float]] = []
        for h in horizons_in_data:
            sub = target_df.loc[target_df["horizon"] == h]
            bar_shares.append(_pivot_shares(sub))

        # SGRAGR: grand average across all horizons in vi_avg_df
        bar_shares.append(_pivot_shares(target_df))

        # AGR: grand average in direct_vi_avg_df
        if direct_df is not None:
            direct_target_df = direct_df.loc[direct_df["target"] == target]
            bar_shares.append(_pivot_shares(direct_target_df))

        # Draw stacked bars
        for x_pos, shares in zip(x_positions, bar_shares):
            bottom = 0.0
            for group in draw_order:
                share = shares.get(group, 0.0)
                if share <= 0.0:
                    continue
                ax.bar(
                    x_pos,
                    share,
                    bottom=bottom,
                    width=bar_width,
                    color=palette[group],
                    linewidth=0.0,
                    label=group,  # duplicate labels filtered in legend
                )
                bottom += share

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(target, fontsize=9, pad=4)

        if panel_idx == 0:
            ax.set_ylabel("Importance share", fontsize=8)

        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

    # Deduplicated legend using the last axes
    handles_seen: dict[str, plt.Artist] = {}
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in handles_seen:
                handles_seen[label] = handle

    # Reorder legend entries by draw_order
    ordered_labels = [g for g in draw_order if g in handles_seen]
    ordered_handles = [handles_seen[g] for g in ordered_labels]

    if ordered_handles:
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="lower center",
            ncol=min(len(ordered_labels), 4),
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
        fig.subplots_adjust(bottom=0.18)

    if title is not None:
        fig.suptitle(title, fontsize=10, y=1.01)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Function 3 — cumulative_squared_error_plot (Fig 6)
# ---------------------------------------------------------------------------


def cumulative_squared_error_plot(
    result_df: pd.DataFrame,
    model_feature_combos: list[tuple[str, str]],
    target: str,
    horizon: int,
    combo_labels: list[str] | None = None,
    combo_colors: list[str] | None = None,
    recession_shading: list[tuple[str, str]] | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Cumulative squared error plot replicating CLSS 2021 Fig 6.

    For each (model_id, feature_set) combination the function computes the
    cumulative sum of squared forecast errors over the OOS period and draws
    one line per combination.  Lower values indicate better performance.

    Parameters
    ----------
    result_df : pd.DataFrame
        Forecast result table.  Required columns: ``model_id``,
        ``feature_set``, ``horizon``, ``date`` (or ``forecast_date``),
        ``y_hat``, ``y_true``.  Optionally a ``target`` column when result_df
        pools multiple target variables.
    model_feature_combos : list of (str, str)
        Each element is a ``(model_id, feature_set)`` pair to include.
    target : str
        Target variable to plot.  Used to filter by ``result_df["target"]``
        when that column is present.
    horizon : int
        Forecast horizon to plot.
    combo_labels : list of str, optional
        Display labels for each combo.  Defaults to "model_id / feature_set".
    combo_colors : list of str, optional
        Line colors.  Defaults to matplotlib tab10 cycle.
    recession_shading : list of (str, str), optional
        Recession periods as ``(start_date, end_date)`` pairs in any format
        accepted by ``pd.Timestamp``.  Shaded with a light gray fill.
    figsize : (float, float), default (8, 5)
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Detect date column
    date_col: str
    if "date" in result_df.columns:
        date_col = "date"
    elif "forecast_date" in result_df.columns:
        date_col = "forecast_date"
    else:
        raise ValueError(
            "result_df must have a 'date' or 'forecast_date' column."
        )

    # Filter to the requested horizon
    df = result_df.loc[result_df["horizon"] == horizon].copy()

    # Filter to the requested target when the column is present
    if "target" in df.columns:
        df = df.loc[df["target"] == target]

    if df.empty:
        raise ValueError(
            f"No rows for horizon={horizon}"
            + (f", target='{target}'" if "target" in result_df.columns else "")
            + "."
        )

    # Resolve labels and colors
    n_combos = len(model_feature_combos)
    if combo_labels is None:
        combo_labels = [f"{mid} / {fs}" for mid, fs in model_feature_combos]
    if combo_colors is None:
        tab10 = plt.get_cmap("tab10")
        combo_colors = [tab10(i % 10) for i in range(n_combos)]

    fig, ax = plt.subplots(figsize=figsize)

    # Recession shading first (drawn behind lines)
    if recession_shading is not None:
        for rec_start, rec_end in recession_shading:
            ax.axvspan(
                pd.Timestamp(rec_start),
                pd.Timestamp(rec_end),
                alpha=0.15,
                color="gray",
                linewidth=0,
                zorder=0,
            )

    for (model_id, feature_set), label, color in zip(
        model_feature_combos, combo_labels, combo_colors
    ):
        sub = df.loc[
            (df["model_id"] == model_id) & (df["feature_set"] == feature_set)
        ].copy()

        if sub.empty:
            continue

        # Sort by date to ensure cumulative sum is chronological
        sub = sub.sort_values(date_col)
        sq_err = (sub["y_true"] - sub["y_hat"]) ** 2
        cum_se = sq_err.cumsum().values
        dates = pd.to_datetime(sub[date_col].values)

        ax.plot(dates, cum_se, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Cumulative squared error", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    ax.legend(fontsize=8, frameon=False)

    if title := f"Cumulative squared error — {target}, H={horizon}":
        ax.set_title(title, fontsize=9, pad=4)

    fig.tight_layout()
    return fig
