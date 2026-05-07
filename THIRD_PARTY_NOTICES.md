# Third-party notices

macroforecast bundles vendored third-party code under
``macroforecast/_vendor/``. Each subpackage carries the upstream licence
file alongside a ``PATCHES.md`` describing the delta from the original
release. This file is the consolidated attribution table.

## macroforecast/_vendor/macro_random_forest/

* **Upstream**: <https://github.com/RyanLucas3/MacroRandomForest>
* **Upstream version**: 1.0.6 (2022-07-28; sole release)
* **Upstream author**: Ryan Lucas
* **Method reference**: Goulet Coulombe, P. (2024) "The Macroeconomy as
  a Random Forest", *Journal of Applied Econometrics*. arXiv:2006.12724.
* **Licence**: MIT (preserved at
  ``macroforecast/_vendor/macro_random_forest/LICENSE`` with dual
  copyright for the upstream and patch authors).
* **Patches**: four surgical numpy 2.x / pandas 2.x compatibility fixes;
  full list at ``macroforecast/_vendor/macro_random_forest/PATCHES.md``.
  No algorithmic changes.
* **Citation requirement**: research using the
  ``macroeconomic_random_forest`` L4 family must cite Goulet Coulombe
  (2024) and acknowledge Ryan Lucas's upstream implementation. The
  citation is surfaced in the encyclopedia entry for the family
  (``docs/encyclopedia/l4/axes/family.md``) and in the ``OPTION_DOCS``
  prose.
