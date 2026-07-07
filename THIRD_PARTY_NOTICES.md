# Third-party notices

macroforecast bundles one vendored third-party reference backend. The upstream
licence file is preserved beside the adapted source file, and this file is the
consolidated attribution table.

## macroforecast/models/_mrf_reference.py

* **Upstream**: <https://github.com/RyanLucas3/MacroRandomForest>
* **Upstream version**: 1.0.6 (2022-07-28; sole release)
* **Upstream author**: Ryan Lucas
* **Method reference**: Goulet Coulombe, P. (2024) "The Macroeconomy as
  a Random Forest", *Journal of Applied Econometrics*. arXiv:2006.12724.
* **Licence**: MIT (preserved at
  ``macroforecast/models/_mrf_reference.LICENSE``).
* **Local source**: ``macroforecast/models/_mrf_reference.py``.
* **Patches**: package integration and numpy/pandas compatibility fixes called
  out in the source-file header. No algorithmic changes are intended.
* **Citation requirement**: research using the
  ``macro_random_forest`` model must cite Goulet Coulombe
  (2024) and acknowledge Ryan Lucas's upstream implementation. The
  backend is documented in the model reference page
  (``docs/reference/models.md``).
