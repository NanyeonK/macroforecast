"""L7 interpretation -- L7.B output / export axes documentation.

L7.A's 30 importance ops are documented in ``l7_a.py``. This module
documents the L7.B output-shape & export sub-layer:

* ``output_table_format`` (long / wide) -- tidy vs matrix-shaped tables,
* ``figure_type`` (auto / bar / boxplot / heatmap / lineplot / scatter)
  -- which matplotlib renderer to use,
* ``figure_format`` (pdf / png / svg) -- file format,
* ``latex_table_export`` (true / false) -- co-emit LaTeX snippets,
* ``markdown_table_export`` (true / false) -- co-emit Markdown tables.
"""
from __future__ import annotations

from . import register
from .types import OptionDoc, Reference

_REVIEWED = "2026-05-05"
_REVIEWER = "macrocast author"

_REF_DESIGN_L7 = Reference(
    citation="macrocast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'",
)
_REF_WICKHAM_TIDY = Reference(
    citation="Wickham (2014) 'Tidy Data', Journal of Statistical Software 59(10): 1-23.",
    doi="10.18637/jss.v059.i10",
)


def _e(axis: str, option: str, summary: str, description: str, when_to_use: str,
       *, when_not_to_use: str = "",
       references: tuple[Reference, ...] = (_REF_DESIGN_L7,),
       related: tuple[str, ...] = ()) -> OptionDoc:
    return OptionDoc(
        layer="l7", sublayer="L7_B_output_shape_export", axis=axis, option=option,
        summary=summary, description=description, when_to_use=when_to_use,
        when_not_to_use=when_not_to_use, references=references,
        related_options=related,
        last_reviewed=_REVIEWED, reviewer=_REVIEWER,
    )


# ---------------------------------------------------------------------------
# output_table_format -- long / wide
# ---------------------------------------------------------------------------

register(
    _e("output_table_format", "long",
       "Long-form (tidy) tables: one row per (model, feature, metric).",
       (
           "Returns importance tables in the tidy data format -- each "
           "row is a single observation of (model_id, feature, "
           "metric_value). Default for downstream pandas / R analysis "
           "since aggregation, filtering, and ggplot-style faceting "
           "all expect this shape.\n\n"
           "Wickham's tidy-data principles (one variable per column, "
           "one observation per row, one type per table) underpin the "
           "long format."
       ),
       "Default for downstream pandas / R analysis; required for ``seaborn`` faceting.",
       when_not_to_use="Paper-quality matrix-shaped reporting (use ``wide`` instead).",
       references=(_REF_DESIGN_L7, _REF_WICKHAM_TIDY),
       related=("wide",)),
    _e("output_table_format", "wide",
       "Wide-form tables: one row per feature, columns per (model, metric).",
       (
           "Returns importance tables in the matrix-shaped format -- "
           "each row is one feature, columns vary across "
           "(model_id × metric) combinations. Compact for paper-quality "
           "reporting and the natural shape for LaTeX ``tabular`` "
           "export."
       ),
       "Compact paper-quality reporting; LaTeX table generation.",
       when_not_to_use="Downstream pandas analysis -- use ``long`` instead.",
       references=(_REF_DESIGN_L7,),
       related=("long",)),
)


# ---------------------------------------------------------------------------
# figure_type -- auto / bar / boxplot / heatmap / lineplot / scatter
# ---------------------------------------------------------------------------

_FIG_TYPE_DOCS: dict[str, tuple[str, str, str]] = {
    "auto": (
        "Pick the figure type matching the importance op's default mapping.",
        (
            "Each L7.A op declares its canonical figure type "
            "(``shap_*`` → bar/beeswarm; ``partial_dependence`` → "
            "lineplot; ``shap_interaction`` → heatmap; "
            "``rolling_recompute`` → heatmap; etc.). Setting "
            "``figure_type = auto`` honours that default."
        ),
        "Default; lets each L7.A op choose the canonical figure for its output.",
    ),
    "bar": (
        "Horizontal bar chart -- one bar per feature, length = importance score.",
        (
            "The standard global-importance visualisation. Renders "
            "features sorted by mean-``|importance|`` so the most "
            "important variables surface at the top of the chart. "
            "Pair with ``output_table_format = wide`` for direct "
            "table-figure cross-reference."
        ),
        "Global importance rankings (linear coefficients, permutation, mean SHAP).",
    ),
    "boxplot": (
        "Boxplot of per-fold / per-bootstrap importance distributions.",
        (
            "Renders each feature as a box capturing the distribution "
            "of its importance score across folds (cross-validation, "
            "bootstrap, rolling windows). Reveals stability information "
            "that a single bar cannot convey."
        ),
        "Stability-of-importance audits; ``bootstrap_jackknife`` / ``rolling_recompute`` outputs.",
    ),
    "heatmap": (
        "Two-axis heatmap (feature × time / model / state).",
        (
            "Visualises importance across an additional dimension. "
            "Used for time-varying importance (``rolling_recompute``), "
            "per-state aggregation (``group_aggregate`` over FRED-SD "
            "blocks), and pairwise interaction strength "
            "(``shap_interaction``). The ``us_state_choropleth`` figure "
            "is a specialised heatmap on the US state grid."
        ),
        "Time-varying importance, FRED-SD state choropleth, group-aggregate matrices.",
    ),
    "lineplot": (
        "Line plot of importance over time / origin.",
        (
            "Tracks importance evolution across walk-forward origins. "
            "Pair with ``rolling_recompute`` to surface trends in which "
            "features matter as new data arrives."
        ),
        "Tracking importance evolution across walk-forward origins.",
    ),
    "scatter": (
        "Scatter plot (e.g. SHAP value vs feature value).",
        (
            "PDP / ALE / SHAP dependence-plot family. Each point is a "
            "single observation; the x-axis is the feature value, the "
            "y-axis is the importance contribution. Reveals "
            "non-linearity in the model's response."
        ),
        "PDP / ALE / SHAP dependence-plot family.",
    ),
}

for option, (summary, desc, when) in _FIG_TYPE_DOCS.items():
    register(_e("figure_type", option, summary, desc, when,
        related=tuple(k for k in _FIG_TYPE_DOCS if k != option)))


# ---------------------------------------------------------------------------
# figure_format -- pdf / png / svg
# ---------------------------------------------------------------------------

register(
    _e("figure_format", "pdf",
       "Vector PDF figures (matplotlib backend).",
       (
           "Vector graphics that scale without pixelation. "
           "Recommended for paper figures where journals require "
           "sub-pixel-precise typography. File sizes larger than PNG "
           "but renderable at any zoom level."
       ),
       "Publication-grade plots; LaTeX-rendered figures.",
       when_not_to_use="Web embedding -- prefer PNG or SVG.",
       related=("png", "svg")),
    _e("figure_format", "png",
       "Raster PNG figures (matplotlib AGG backend).",
       (
           "300dpi-by-default raster images. Smaller than PDF for "
           "plot-heavy reports; the natural choice for slides, HTML "
           "embeddings, and Markdown documents that render through "
           "GitHub / Slack / web viewers."
       ),
       "Slide / web embedding where vector formats are unnecessary.",
       related=("pdf", "svg")),
    _e("figure_format", "svg",
       "Vector SVG figures (matplotlib SVG backend).",
       (
           "XML-based vector format renderable in browsers. Selectable "
           "text and zoom-without-pixelation; useful when the "
           "consumer wants to interactively inspect / edit the figure "
           "(e.g. via Inkscape) before final publication."
       ),
       "Web embedding with selectable text; pre-publication editable figures.",
       related=("pdf", "png")),
)


# ---------------------------------------------------------------------------
# latex_table_export / markdown_table_export -- boolean co-emission flags
# ---------------------------------------------------------------------------

register(
    _e("latex_table_export", "true",
       "Co-emit LaTeX ``tabular`` snippets alongside the JSON / CSV outputs.",
       (
           "Saves an extra round-trip through ``pandas.to_latex`` when "
           "the recipe is feeding a paper draft. Booktabs-friendly "
           "alignment and automatic column-name escaping; the "
           "resulting ``.tex`` file is ``\\input``-able directly into "
           "a manuscript without further processing."
       ),
       "When the recipe is feeding a paper draft.",
       related=("false",)),
    _e("latex_table_export", "false",
       "Skip the LaTeX-table co-emission step.",
       (
           "Default. Avoids the small-but-non-trivial overhead of "
           "pandas-to-LaTeX rendering on every importance table when "
           "no paper-quality output is needed."
       ),
       "Default; LaTeX adds tooling overhead that is wasted for non-paper runs.",
       related=("true",)),
    _e("markdown_table_export", "true",
       "Co-emit Markdown tables alongside the JSON / CSV outputs.",
       (
           "Useful for README / wiki / GitHub-flavoured Markdown "
           "documents. Pipe-aligned columns; pairs naturally with "
           "the ``markdown_report`` L8 export format for end-to-end "
           "Markdown reporting."
       ),
       "Generating GitHub README / wiki pages from runs.",
       related=("false",)),
    _e("markdown_table_export", "false",
       "Skip the Markdown-table co-emission step.",
       (
           "Default. JSON / CSV outputs cover most consumers; "
           "Markdown is opt-in for documentation pipelines."
       ),
       "Default; Markdown is opt-in.",
       related=("true",)),
)
