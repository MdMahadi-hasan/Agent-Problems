"""
Replication script for
“Two Heuristic Agents Walk into a Bar: Monte-Carlo Simulation
 and Optimal Inaction in LLM Adoption”

Implements three agents
  • Curious  – adopts every period
  • Lazy     – never adopts
  • Optimal  – threshold rule from Proposition 1

Author: 2025-06-22
"""

# ───────────────────────────────── 1. Imports ───────────────────────────────── #
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)       # deterministic RNG

# ───────────────────────────────── 2. Calibration ───────────────────────────── #
T        = 200       # horizon
N        = 10_000    # Monte-Carlo paths

rho      = 0.05
alpha    = 1.3
bar_delta = 1.2
mu_C,  sigma_C  = 0.1, 0.5
p_h      = 0.20
mu_H,  sigma_H  = 0.5, 0.7
lam      = 5e-6
beta     = 2
phi      = 0.5

# -------- OPTION: uncomment / tweak to force B < 0 (referee-friendly regime) --
# alpha      = 1.6
# bar_delta  = 0.8
# mu_C       = 0.5
# p_h        = 0.30
# mu_H       = 0.8

# ─────────── 2·1  Derived moment: expected one-period net benefit B ─────────── #
E_Delta = (alpha * bar_delta) / (alpha - 1)          # Pareto mean
E_C     = np.exp(mu_C + 0.5 * sigma_C**2)
E_H     = np.exp(mu_H + 0.5 * sigma_H**2)
B       = E_Delta - E_C - p_h * E_H
print(f"Expected net benefit  B = {B:0.3f}")

# ───────────────────────────────── 3. State arrays ──────────────────────────── #
P_cur  = np.ones(N)
P_lazy = np.ones(N)
P_opt  = np.ones(N)

A_cur  = np.zeros(N, dtype=int)
A_opt  = np.zeros(N, dtype=int)

crash_cur = np.zeros(N, dtype=int)
crash_opt = np.zeros(N, dtype=int)

# ───────────────────────────────── 4. Main loop ─────────────────────────────── #
for t in range(T):
    # -------- stochastic primitives (draw once per period) ------------------- #
    Delta_t = bar_delta * (rng.pareto(alpha, N) + 1.0)       # support [bar_delta,∞)
    C_t     = rng.lognormal(mu_C, sigma_C, N)
    H_raw   = rng.lognormal(mu_H, sigma_H, N)
    H_t     = np.where(rng.random(N) < p_h, H_raw, 0.0)

    # ===================== Curious: always adopt ============================ #
    d_cur   = 1
    A_cur  += d_cur
    crash_p_cur   = lam * (A_cur ** beta)
    crash_mask_cur = rng.random(N) < crash_p_cur
    crash_cur    += crash_mask_cur

    # crash branch first (reset)
    P_cur[crash_mask_cur] *= phi

    # learning update for paths that did NOT crash
    idx_nc_cur = ~crash_mask_cur
    P_cur[idx_nc_cur] = (1 - rho) * P_cur[idx_nc_cur] + \
                        rho * (P_cur[idx_nc_cur] +
                               Delta_t[idx_nc_cur] -
                               C_t[idx_nc_cur]     -
                               H_t[idx_nc_cur])

    # ===================== Lazy: never adopt =============================== #
    # formula collapses to identity ⇒ stays at 1 forever (already initialised)

    # ===================== Optimal: threshold rule ========================= #
    # threshold P(A_t)  = [ρB] / [λ(1-φ)((A+1)^β − A^β)]
    delta_A      = (A_opt + 1) ** beta - A_opt ** beta
    denom        = lam * (1 - phi) * delta_A
    P_thresh     = (rho * B) / (denom + 1e-12)   # ε guards against /0

    d_opt_vec    = (P_opt <= P_thresh).astype(int)
    A_opt       += d_opt_vec
    crash_p_opt  = lam * (A_opt ** beta)
    crash_mask_opt = rng.random(N) < crash_p_opt
    crash_opt   += crash_mask_opt

    # crash branch
    P_opt[crash_mask_opt] *= phi

    # learning update ONLY for adopters who did not crash
    idx_nc_opt   = ~crash_mask_opt
    adopt_mask   = (d_opt_vec == 1) & idx_nc_opt

    gain = np.zeros(N)
    gain[adopt_mask] = Delta_t[adopt_mask] - C_t[adopt_mask] - H_t[adopt_mask]

    P_opt[idx_nc_opt] = (1 - rho) * P_opt[idx_nc_opt] + \
                        rho * (P_opt[idx_nc_opt] + gain[idx_nc_opt])

# ───────────────────────────────── 5. Diagnostics ───────────────────────────── #
def describe(P, name):
    pct = np.percentile(P, [1, 5, 50, 95, 99])
    return pd.Series({
        f"{name} Mean":       P.mean(),
        f"{name} Std":        P.std(ddof=1),
        f"{name} P1":         pct[0],
        f"{name} P5":         pct[1],
        f"{name} Median":     pct[2],
        f"{name} P95":        pct[3],
        f"{name} P99":        pct[4],
        f"{name} ES 5%":      P[P <= pct[1]].mean(),
        f"{name} Sharpe":     P.mean() / P.std(ddof=1),
        f"{name} Skew":       skew(P),
        f"{name} Kurtosis":   kurtosis(P, fisher=False),
    })

summary = pd.concat([
    describe(P_cur,  "Curious"),
    describe(P_lazy, "Lazy"),
    describe(P_opt,  "Optimal"),
    pd.Series({
        "Pr(Lazy > Curious)":   np.mean(P_lazy > P_cur),
        "Pr(Opt  > Curious)":   np.mean(P_opt  > P_cur),
        "Pr(Opt  > Lazy)":      np.mean(P_opt  > P_lazy),
        "Crash Freq Curious":   np.mean(crash_cur > 0),
        "Crash Freq Optimal":   np.mean(crash_opt > 0),
        "Avg Crashes Curious":  crash_cur.mean(),
        "Avg Crashes Optimal":  crash_opt.mean()
    })
])

print("\n=== Monte-Carlo Summary (N = {:,}, T = {}) ==="
      .format(N, T))
print(summary.to_string(float_format="{:0.4f}".format))

# ───────────────────────────────── 6. Optional plot ─────────────────────────── #
sorted_cur  = np.sort(P_cur)
sorted_lazy = np.sort(P_lazy)
sorted_opt  = np.sort(P_opt)
cdf         = np.linspace(0, 1, N)

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.plot(sorted_cur,  cdf, label="Curious", lw=1.2)
ax.plot(sorted_lazy, cdf, label="Lazy",    lw=1.2, ls='--')
ax.plot(sorted_opt,  cdf, label="Optimal", lw=1.2, ls='-.')
ax.set_xlabel(r"Terminal performance $P_T$")
ax.set_ylabel("Cumulative probability")
ax.legend(frameon=False)
fig.savefig("cdf_simulation.png", dpi=300)
plt.close(fig)
