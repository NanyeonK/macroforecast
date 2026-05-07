"""Vendored MacroRandomForest with numpy 2.x / pandas 2.x compatibility
patches.

Upstream: https://github.com/RyanLucas3/MacroRandomForest (Ryan Lucas,
2022, MIT) -- reference implementation of Goulet Coulombe (2024) "The
Macroeconomy as a Random Forest" (arXiv:2006.12724). The upstream
release 1.0.6 (2022-07-28) has not been updated and fails to import on
modern numpy / pandas; this vendored copy applies four surgical
compatibility patches documented in ``PATCHES.md``. No algorithmic
changes.

Public surface mirrors upstream: ``MacroRandomForest`` class with
``_ensemble_loop()`` returning a dict with keys ``{'pred', 'betas',
'pred_ensemble', ...}``. See ``LICENSE`` for the upstream MIT licence
preserved alongside the patch attribution.
"""
from .MRF import MacroRandomForest

__all__ = ["MacroRandomForest"]
__version__ = "1.0.6.post1+macroforecast"
__upstream_url__ = "https://github.com/RyanLucas3/MacroRandomForest"
