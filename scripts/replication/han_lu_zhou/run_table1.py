"""HLZ Table 1 -- summary statistics of the 14 predictors under Xt, MA3, MA6, MA12.

Cheap, but it is the one exhibit that checks the PANEL rather than the estimator:
if our predictor construction diverged from the authors', every later table
inherits the error silently. Sample 1926:12-2022:12, per the Table 1 note.

Units. The note says both statistics are "expressed as percentages". Welch-Goyal
deliver the yields and spreads already in percent (TBL, LTY, LTR, TMS, DFY, DFR)
and INFL as a percent inflation rate; the remaining seven are ratios or log ratios
(DP, DY, EP, DE, BM, NTIS, SVAR) and must be multiplied by 100 to be percentages.
The rule is applied by SOURCE UNIT, uniformly -- not fitted per cell to the printed
values. The rate block is the check on it: it is compared with no rescaling at all.
"""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd

panel = pd.read_parquet("/tmp/hlz_panel.parquet")
NAMES = ["DP", "DY", "EP", "DE", "BM", "NTIS", "TBL", "LTY", "LTR", "TMS",
         "DFY", "DFR", "INFL", "SVAR"]
AS_RATIO = {"DP", "DY", "EP", "DE", "BM", "NTIS", "SVAR"}   # x100 -> percent
econ = panel[NAMES]
print(f"panel: {econ.index[0].date()} .. {econ.index[-1].date()}  n={len(econ)}")

PRINTED = {  # (mean, SD) under Xt, MA3, MA6, MA12
    "DP":   [(-340.93, 47.67), (-340.91, 47.51), (-340.88, 47.29), (-340.79, 46.85)],
    "DY":   [(-340.43, 47.47), (-340.41, 47.30), (-340.38, 47.09), (-340.30, 46.64)],
    "EP":   [(-276.15, 42.05), (-276.15, 41.79), (-276.16, 41.35), (-276.15, 40.31)],
    "DE":   [( -64.77, 32.77), ( -64.75, 32.61), ( -64.72, 32.15), ( -64.64, 30.75)],
    "BM":   [(  55.19, 26.84), (  55.23, 26.67), (  55.28, 26.47), (  55.40, 26.13)],
    "NTIS": [(   1.60,  2.57), (   1.60,  2.55), (   1.60,  2.51), (   1.59,  2.43)],
    "TBL":  [(   3.29,  3.06), (   3.29,  3.06), (   3.29,  3.04), (   3.30,  3.02)],
    "LTY":  [(   4.96,  2.81), (   4.97,  2.80), (   4.97,  2.80), (   4.98,  2.80)],
    "LTR":  [(   0.45,  2.49), (   0.45,  1.48), (   0.45,  1.04), (   0.46,  0.74)],
    "TMS":  [(   1.67,  1.29), (   1.67,  1.27), (   1.68,  1.24), (   1.68,  1.19)],
    "DFY":  [(   1.12,  0.68), (   1.12,  0.67), (   1.12,  0.66), (   1.12,  0.64)],
    "DFR":  [(   0.05,  1.41), (   0.04,  0.74), (   0.04,  0.50), (   0.04,  0.33)],
    "INFL": [(   0.25,  0.53), (   0.25,  0.42), (   0.25,  0.37), (   0.25,  0.32)],
    "SVAR": [(   0.29,  0.60), (   0.29,  0.50), (   0.29,  0.44), (   0.29,  0.40)],
}
WINDOWS = [1, 3, 6, 12]

rows, deltas, worst = [], [], (0.0, "")
for v in NAMES:
    k = 100.0 if v in AS_RATIO else 1.0
    cells = []
    for L in WINDOWS:
        s = econ[v] if L == 1 else econ[v].rolling(L, min_periods=L).mean()
        cells.append((float(s.mean()) * k, float(s.std(ddof=1)) * k))
    rows.append((v, cells))
    for j, ((m, sd), (pm, psd)) in enumerate(zip(cells, PRINTED[v])):
        for got, want, what in ((m, pm, "mean"), (sd, psd, "SD")):
            d = abs(got - want)
            deltas.append(d)
            if d > worst[0]:
                worst = (d, f"{v} {what} @{['Xt','MA3','MA6','MA12'][j]}: {got:.4f} vs {want}")

print("\n########## TABLE 1 -- mine vs printed  (bold = |Δ| > 0.01) ##########")
print("| predictor | Xt mean | Xt SD | MA3 mean | MA3 SD | MA6 mean | MA6 SD | MA12 mean | MA12 SD |")
print("|---|---|---|---|---|---|---|---|---|")
for v, cells in rows:
    out = []
    for (m, sd), (pm, psd) in zip(cells, PRINTED[v]):
        for got, want in ((m, pm), (sd, psd)):
            cell = f"{got:.2f}"
            out.append(f"**{cell}**" if abs(got - want) > 0.01 else cell)
    print(f"| {v} | " + " | ".join(out) + " |")

d = np.array(deltas)
print(f"\nALL {len(d)} cells (14 predictors x 4 windows x 2 stats):")
print(f"  max|Δ| = {d.max():.4f}   mean|Δ| = {d.mean():.4f}")
print(f"  within 0.005 = {(d <= 0.005).sum()}/{len(d)}    within 0.01 = {(d <= 0.01).sum()}/{len(d)}"
      f"    within 0.05 = {(d <= 0.05).sum()}/{len(d)}")
print(f"  worst cell: {worst[1]}")
rate = np.array([abs(m - pm) for v, c in rows if v not in AS_RATIO
                 for (m, sd), (pm, psd) in zip(c, PRINTED[v]) for m, pm in ((m, pm), (sd, psd))])
print(f"  UNSCALED rate/INFL block alone ({len(rate)} cells): max|Δ| = {rate.max():.4f}")
print("TABLE1_DONE")
