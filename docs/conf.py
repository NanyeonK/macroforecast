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
    "sphinx_design",
    "sphinxcontrib.mermaid",
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
    "dollarmath",
]

myst_heading_anchors = 4

# -- HTML output --------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "macroforecast"
html_static_path: list[str] = ["_static"]
html_css_files: list[str] = ["custom.css"]
html_extra_path: list[str] = []

html_theme_options = {
    "logo": {"text": "macroforecast"},
    "github_url": "https://github.com/NanyeonK/macroforecast",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_nav_level": 1,
    "header_links_before_dropdown": 7,
    "navigation_with_keys": True,
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/macroforecast/", "icon": "fa-brands fa-python"},
    ],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

# Leaf top-level pages have no sub-toctree, so the primary (left) sidebar would
# render empty. Suppress it on those pages; pydata-sphinx-theme then lays them
# out full-width. In-page section navigation still appears in the right-hand
# "On this page" panel.
html_sidebars = {
    "guide/glossary": [],
}

html_context = {
    "github_user": "NanyeonK",
    "github_repo": "macroforecast",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
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
    "openpyxl",
    "pyarrow",
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
suppress_warnings = ["myst.xref_missing", "toc.not_included", "myst.header"]
