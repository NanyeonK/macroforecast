# Sources

Datasets that ship as first-class `dataset` values in macrocast. This section documents each one at the **source level** — what the maintainers publish, how macrocast downloads and caches it, which variable groups exist, how transformation codes work, and what "vintage" means per dataset.

Selection of *which* dataset a recipe uses is an axis, documented in [Source & Frame (1.1)](../user_guide/data/source.md). The pages here cover the data itself.

## Datasets

| Dataset | Frequency | Maintainer | Paper | Page |
|---|---|---|---|---|
| **FRED-MD** | monthly | St. Louis Fed | McCracken & Ng (2016) | [fred_md.md](fred_md.md) |
| **FRED-QD** | quarterly | St. Louis Fed | McCracken & Ng (2021) | [fred_qd.md](fred_qd.md) |
| **FRED-SD** | monthly / quarterly, state-level | St. Louis Fed | Bokun, Jackson, Kliesen, Owyang (2022) | [fred_sd.md](fred_sd.md) |

All three are loaded via canonical macrocast entry points (`macrocast.load_fred_md()` etc.). Each page follows the same structure:

1. **At a glance** — authors, span, size, loader.
2. **Citation & authoritative source** — paper, landing page, appendix.
3. **What macrocast downloads** — exact URL pattern used by the loader + cache layout.
4. **Variable groups** — group labels and representative members.
5. **Transformation codes (T-codes)** — the per-series transform the upstream maintainers publish.
6. **Vintage & revisions** — which `information_set_type` regime each vintage satisfies.
7. **Real-time release lag** — default behaviour for this dataset.
8. **Schema contract** — what a `custom_csv` / `custom_parquet` file must look like to pass as this `dataset`.
9. **Known quirks / breaking changes** — methodology shifts, COVID outliers, series retirements.
10. **See also** — cross-refs to related axes.

## Bring-your-own-data

`dataset_source = custom_csv` or `custom_parquet` loads a user-supplied file under the schema of whichever `dataset` value is declared. The schema contract lives in each dataset page's section 8. See [Source & Frame (1.1)](../user_guide/data/source.md#custom-csv--parquet--implementation-contract) for the axis-level mechanics.

```{toctree}
:hidden:
:maxdepth: 1

fred_md
fred_qd
fred_sd
```
