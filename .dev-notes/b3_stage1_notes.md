# B3 Stage-1 Notes: ZWW 2023 Corrected Data Build

Date: 2026-07-13

Scope: stage-1 data build only. No `macroforecast/**` patch, no Table 3 run, no push.

## 1. RV Oracle Parity

- Status: PASS.
- Months checked: 451.
- Date range: 1986-06-30 to 2023-12-31.
- Max absolute difference: 0.
- Missing daily price rows dropped before log returns: 359.

Output: `runs/zww_b3_stage1/wti_rv_monthly.csv`.

## 2. FRED-MD 2019:06 Predictors

- Loader: `mf.data.load_fred_md(vintage="2019-06")`.
- Raw loader variables: 127; raw `ACOGNO` present: True.
- `ACOGNO` excluded: True.
- Output predictor variables: 126.
- McCracken-Ng t-code transforms applied: True (126 variables).
- One-month publication lag applied: True (126 variables).
- Predictor date range: 1985-01-31 to 2018-12-31 (408 rows).
- FRED source hash: `3382f00edc0cf146c739c1fe4f74a1f9409fa7d1e9c08f0937fdc3e032826329`.

Count discrepancy recorded: The package-returned official 2019-06 vintage has 127 raw fields including ACOGNO, so excluding ACOGNO leaves 126 predictors. No replacement column was fabricated.

Missing predictor cells after tcode+lag/sample: 0 across 0 columns.

Output: `runs/zww_b3_stage1/fred_md_2019_06.csv`.

## 3. Multi-Horizon Target Verification

h=1 PASS max_abs_diff=0 n=450; h=3 PASS max_abs_diff=0 n=448; h=6 PASS max_abs_diff=0 n=445; h=12 PASS max_abs_diff=0 n=439

Output: `runs/zww_b3_stage1/targets_log_average_value.csv`.

## 4. Spot vs Futures

This build uses WTI spot `DCOILWTICO`, matching ZWW's Table 4 spot robustness target. ZWW's main Table 3 requires daily WTI futures RV from EIA futures data; that source is not in the staged archive and remains an open data dependency.

## 5. Remaining Package Addition

The confirmed Table 3 fix-lane package addition remains a general AICc lambda-selection unit for `lasso`/`elastic_net` in the A2 information-criterion lane. This Stage-1 task did not patch `macroforecast/**`.
