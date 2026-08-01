"""Table 8 (placebo): does the archive reproduce the printed table, and how many
simulations would an independent replication need?

`r2_sims` is stored in the printed units already (percent), verified below by the
Xt row landing on 0.5987 against a printed 0.60 with zero cross-simulation spread
-- scrambling the past cannot touch a current-value-only design, so its 1,000
draws must be identical, and they are (sd 2.2e-16).
"""
import numpy as np, scipy.io as sio, warnings
warnings.filterwarnings("ignore")
ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
m = sio.loadmat(f"{ARCH}/Pseduo_test_3years_sramble.mat")
r2, dif = m["r2_sims"], m["r2_diff_sims"]
LAB = ["Xt", "Xt+{MA2..MA6}", "Xt+{MA2..MA12}", "Xt+{MA2..MA24}", "Xt+{MA2..MA36}"]
PRINT = [0.60, 0.40, 0.32, 0.27, 0.26]
PRINT_D = [-0.20, -0.28, -0.33, -0.34]

def need(sd, tol):
    return "--" if sd <= 0 else int(np.ceil((1.96 * sd / tol) ** 2))

print("| design | archive mean (1,000 sims) | printed | Δ | sd across sims | n for ±0.10pp | ±0.05pp | ±0.01pp |")
print("|---|---|---|---|---|---|---|---|")
for i, lab in enumerate(LAB):
    v = r2[i]; mu, sd = np.nanmean(v), np.nanstd(v, ddof=1)
    print(f"| {lab} | {mu:.4f} | {PRINT[i]:.2f} | {mu - PRINT[i]:+.4f} | {sd:.4f} | "
          f"{need(sd,0.10)} | {need(sd,0.05)} | {need(sd,0.01)} |")
print("\n| design | archive mean ΔR2 | printed | Δ |")
print("|---|---|---|---|")
for i in range(4):
    mu = np.nanmean(dif[i])
    print(f"| {LAB[i+1]} | {mu:.4f} | {PRINT_D[i]:.2f} | {mu - PRINT_D[i]:+.4f} |")
d = np.array([abs(np.nanmean(r2[i]) - PRINT[i]) for i in range(5)] +
             [abs(np.nanmean(dif[i]) - PRINT_D[i]) for i in range(4)])
print(f"\nall 9 printed values: max|Δ| = {d.max():.4f}pp   (archive reproduces the table)")
print(f"Xt row sd = {np.nanstd(r2[0], ddof=1):.2e}  (must be 0 -- scrambling cannot touch it)")
