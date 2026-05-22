# FRED Datasets in Recipes

Recipe dataset choices are `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, and `fred_qd+fred_sd`.

FRED-MD and FRED-QD use the St. Louis Fed current CSV endpoints by default. FRED-SD resolves the latest official Data by Series workbook from the FRED-SD landing page unless a vintage is requested.

Generated column-count pages are snapshots. They must say:

- generation date,
- data-through date,
- source URL,
- column count,
- whether counts were generated from cache or live download.

Do not describe a generated count as permanently current. Regenerate the reference pages when the raw adapter cache or official landing page changes.
