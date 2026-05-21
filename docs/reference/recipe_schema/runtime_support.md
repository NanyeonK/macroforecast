# Runtime Support Matrix

This matrix describes current runtime behavior. When option docs and runtime differ, this page must say so.

| Layer | Runtime state |
| --- | --- |
| L0 | `fail_fast` / `continue_on_failure`, seeded/exploratory, serial/parallel schema. |
| L1 | FRED-MD, FRED-QD, FRED-SD, custom inline/path panels, current vintage. `real_time_alfred` is future. |
| L2 | Mixed-frequency alignment, transform, outlier, imputation, frame-edge policies. |
| L3 | DAG feature pipelines and `target_construction`; `lag` is a universal op now enumerated in `option_docs/l3.py`. |
| L4 | DAG model nodes; 42 documented model families; `ar_p` is the default family constant. |
| L5 | Metrics include `mse`, `rmse`, `mae`, `medae`, `theil_u1`, `theil_u2`, direction, relative, and density metrics. |
| L6 | Runtime has A-G sublayers. Option docs currently expose only part of that surface. |
| L7 | Interpretation DAG with output controls and model-family gates. |
| L8 | JSON/CSV/Parquet/LaTeX/Markdown/HTML-report formats, gzip/zip compression, manifests, recipe snapshot, per-cell artifacts. `saved_objects: all` excludes the HTML report (Cycle 11b1 doc-fix). |
