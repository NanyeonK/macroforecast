"""Sphinx configuration for macroforecast documentation."""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))

project = "macroforecast"
author = "NanyeonK"
copyright = "2026, NanyeonK"

try:
    from macroforecast import __version__ as _pkg_version  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _pkg_version = "0.1.0"

version = _pkg_version
release = _pkg_version

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"
language = "en"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- MyST ---------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_inline",
    "substitution",
    "smartquotes",
]

myst_heading_anchors = 3

# -- HTML output --------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = "macroforecast"
html_static_path: list[str] = []
html_extra_path = ["_html_extra"]

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "titles_only": True,
    "prev_next_buttons_location": "bottom",
}

html_context = {
    "display_github": True,
    "github_user": "NanyeonK",
    "github_repo": "macroforecast",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Autodoc ------------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

autodoc_mock_imports = [
    "xgboost",
    "lightgbm",
    "catboost",
    "shap",
    "lime",
    "optuna",
    "openpyxl",
    "torch",
    "pytorch_lightning",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# -- Intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
}

# -- Warnings -----------------------------------------------------------------

nitpicky = False
# During the docs IA migration, some audit and compatibility pages are
# intentionally link-only so they do not dominate the main navigation.
suppress_warnings = ["myst.xref_missing", "toc.not_included"]


# -- Auto-emit OptionDoc reference pages -------------------------------------
# Build-time hook: regenerate ``docs/reference/{l0..l8}.rst`` from the
# scaffold OptionDoc registry on every sphinx build, so ReadTheDocs picks
# up the latest 578-entry registry without manual file shuffling.

def _emit_optiondoc_reference() -> None:
    """Render per-layer reference pages into ``docs/reference/`` from
    the live OptionDoc registry. Idempotent: overwrites existing files.

    Failure is non-fatal -- if the scaffold subpackage cannot be imported
    (e.g. during a partial-environment build) we log a warning rather
    than break the rest of the docs build.
    """

    docs_root = Path(__file__).resolve().parent
    target = docs_root / "reference"
    try:
        from macroforecast.scaffold import render_rst as _render_rst
    except Exception as exc:  # pragma: no cover - environment-dependent
        import warnings

        warnings.warn(
            "macroforecast.scaffold.render_rst unavailable; skipping "
            f"OptionDoc reference auto-emit ({exc!r})."
        )
        return
    written = _render_rst.write_all(target)
    print(f"[macroforecast docs] emitted {len(written)} reference pages -> {target}")


_emit_optiondoc_reference()

