# Vintage and release lag (Phase 3)

Two orthogonal axes govern the information set realism of every forecast in macrocast:

* `vintage_policy` (Phase 0) selects *which historical revision* of the series is used.
* `release_lag_rule` (Phase 3) selects *how the publication delay* is applied to predictor availability.

Both must be set jointly to claim "pseudo real-time" evaluation.

## Notation

Let $X_{i,t}^{(v)}$ denote the value of series $i$ at calendar date $t$ as published in vintage $v$. Let $\\ell_i$ denote the publication lag of series $i$ (months between reference period $t$ and publication date $t + \\ell_i$). Let $\\tau$ be the forecast origin (the date at which the forecaster commits to a prediction).

The information set available at origin $\\tau$ under release lag rule $r$ is

$$
\\mathcal{I}_\\tau^{(r)} = \\{ X_{i,t}^{(v)} : t \\le \\tau - \\delta_i^{(r)} \\}.
$$

The lag offset $\\delta_i^{(r)}$ depends on the rule:

| Rule | $\\delta_i^{(r)}$ |
|---|---|
| `ignore_release_lag` | $0$ for all $i$ |
| `fixed_lag_all_series` | $\\bar{\\ell}$ — a single constant lag, default 1 month |
| `series_specific_lag` | $\\ell_i$ — a series-specific table |
| `calendar_exact_lag` | $\\ell_i^{\\text{cal}}$ — exact day-of-month from a release calendar |
| `lag_conservative` | $\\max(\\ell_i, c)$ — never less than the conservative floor $c$ (default 2) |
| `lag_aggressive` | $\\max(\\ell_i - 1, 0)$ — assume early publication |

In the current operational slice (v0.5), `series_specific_lag` and `calendar_exact_lag` use a default lag of 1 month with placeholder series-/calendar-specific tables. Real release-calendar integration is scheduled for v1.1 (Phase 10).

## Pseudo real-time vs revised

A forecast is *pseudo real-time* when both:

$$
\\mathcal{I}_\\tau \\subseteq \\{ X_{i,t}^{(v)} : v \\le \\tau, \; t \\le \\tau - \\delta_i \\}.
$$

The vintage constraint $v \\le \\tau$ ($\\Leftrightarrow$ no revisions published after origin are visible) is enforced by `vintage_policy ∈ {single_vintage, rolling_vintage}`. The release-lag constraint $t \\le \\tau - \\delta_i$ is enforced by any non-`ignore_release_lag` value of `release_lag_rule`.

Setting `release_lag_rule=ignore_release_lag` and `vintage_policy=latest_only` corresponds to a *fully revised, no-lag* evaluation — fast and reproducible, but optimistic. This is the default for unit tests; production studies should override both.

## Series-specific vs calendar-exact

`series_specific_lag` applies $\\delta_i = \\ell_i$ to *every* observation of series $i$. This is correct on average but wrong on dates where the lag deviates from the table value (e.g. a delayed release).

`calendar_exact_lag` applies $\\delta_{i,t}^{\\text{cal}}$ — the exact lag for *that observation* taken from a release calendar table. This is the gold standard and matches what a real-time forecaster would have observed.

The runtime hook `_apply_release_lag` in `macrocast/execution/build.py` is currently a `.shift(δ)` of the predictor matrix. The series-specific table and calendar table are placeholders to be filled in by Phase 10 (`plans/phases/phase_10_data_extension.md`).
