import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ──────────────────────────────────────────────────────────────────────
#  A.  CORE SIMULATION FUNCTION
# ──────────────────────────────────────────────────────────────────────
def simulate(params, rng_seed=42):
    """
    Monte-Carlo simulation of three agents
        • Curious  (always adopts)
        • Lazy     (never adopts)
        • Optimal  (threshold rule from Proposition 1)

    Returns a dict of summary statistics (1 run, N paths).
    """
    # ── unpack
    (T, N, rho, alpha, bar_delta,
     mu_C, sigma_C, p_h, mu_H, sigma_H,
     lam, beta, phi) = (
        params['T'],     params['N'],     params['rho'],
        params['alpha'], params['bar_delta'],
        params['mu_C'],  params['sigma_C'],
        params['p_h'],   params['mu_H'],  params['sigma_H'],
        params['lam'],   params['beta'],  params['phi'])

    # ── expected values → B
    #   Δ = bar_delta*(Y+1) with Y~Pareto(α) (support≥0)
    E_delta = bar_delta * (alpha / (alpha - 1))
    E_C     = np.exp(mu_C + 0.5 * sigma_C**2)
    E_H     = np.exp(mu_H + 0.5 * sigma_H**2)
    B       = E_delta - E_C - p_h * E_H          # one-period net benefit

    # ── initialise states
    P_cur = np.ones(N)
    P_lazy = np.ones(N)
    P_opt = np.ones(N)

    A_cur = np.zeros(N, dtype=int)
    A_opt = np.zeros(N, dtype=int)

    crashes_cur = np.zeros(N, dtype=int)
    crashes_opt = np.zeros(N, dtype=int)

    rng = np.random.default_rng(rng_seed)

    # ── main loop
    for t in range(T):
        #  ▸ primitives (same draws for all agents)
        Delta_t = bar_delta * (rng.pareto(alpha, N) + 1.0)      # ≥ bar_delta
        C_t     = rng.lognormal(mu_C, sigma_C, N)
        H_raw   = rng.lognormal(mu_H, sigma_H, N)
        H_t     = np.where(rng.random(N) < p_h, H_raw, 0.0)

        #  ■ Curious: always adopt
        A_cur += 1
        crash_p_cur  = lam * (A_cur ** beta)
        crash_mask_c = rng.random(N) < crash_p_cur
        crashes_cur += crash_mask_c

        # crash branch first
        P_cur[crash_mask_c] *= phi

        # learning branch
        idx_nc_c = ~crash_mask_c
        P_cur[idx_nc_c] = (1 - rho) * P_cur[idx_nc_c] + \
                          rho * (P_cur[idx_nc_c] +
                                 Delta_t[idx_nc_c] - C_t[idx_nc_c] - H_t[idx_nc_c])

        #  ■ Lazy: never adopts  → stays at 1  (no update)

        #  ■ Optimal: threshold rule
        delta_A   = (A_opt + 1)**beta - A_opt**beta
        denom     = lam * (1 - phi) * delta_A
        P_thresh  = (rho * B) / (denom + 1e-12)          # ε avoids /0

        adopt     = P_opt <= P_thresh                    # boolean vector
        A_opt    += adopt

        crash_p_opt  = lam * (A_opt ** beta)
        crash_mask_o = rng.random(N) < crash_p_opt
        crashes_opt += crash_mask_o

        # crash first
        P_opt[crash_mask_o] *= phi

        # learning update only for adopters who did NOT crash
        learn_mask = adopt & (~crash_mask_o)
        gain       = Delta_t[learn_mask] - C_t[learn_mask] - H_t[learn_mask]
        P_opt[learn_mask] = (1 - rho) * P_opt[learn_mask] + \
                            rho * (P_opt[learn_mask] + gain)

        # non-adopters or crashed paths keep the (1-ρ) carry-over only
        keep_mask = (~adopt) & (~crash_mask_o)
        P_opt[keep_mask] = (1 - rho) * P_opt[keep_mask] + rho * P_opt[keep_mask]

    # ── diagnostics
    percentiles = lambda x: np.percentile(x, [1, 5, 50, 95, 99])
    pct_c = percentiles(P_cur)
    pct_o = percentiles(P_opt)

    out = dict(
        MeanCur  = P_cur.mean(),   MedianCur=pct_c[2],
        ES5Cur   = P_cur[P_cur<=pct_c[1]].mean(),
        SharpeCur= P_cur.mean()/P_cur.std(ddof=1),
        MeanOpt  = P_opt.mean(),   MedianOpt=pct_o[2],
        PrLazy_Cur = np.mean(P_lazy > P_cur),
        PrOpt_Cur  = np.mean(P_opt  > P_cur),
        PrOpt_Lazy = np.mean(P_opt  > P_lazy),
        AvgCrashCur= crashes_cur.mean(),
        AvgCrashOpt= crashes_opt.mean()
    )
    return out

# ──────────────────────────────────────────────────────────────────────
#  B.  BASELINE & ONE-AT-A-TIME SWEEP
# ──────────────────────────────────────────────────────────────────────
baseline = dict(
    T=200, N=10_000,
    rho=0.05,
    alpha=1.5,
    bar_delta=1.2,
    mu_C=0.1, sigma_C=0.5,
    p_h=0.20,
    mu_H=0.5, sigma_H=0.7,
    lam=5e-6,
    beta=2,
    phi=0.6,
)

sweep = {
    'alpha': [1.1, 1.5, 1.5, 1.7],
    'lam'  : [2e-6, 5e-6, 1e-5, 2e-5],
    'rho'  : [0.01, 0.05, 0.10],
    'phi'  : [0.3, 0.6, 0.7, 0.9],
    'p_h'  : [0.0, 0.10, 0.20, 0.50],
    'beta' : [1, 2, 3, 4]
}

# ──────────────────────────────────────────────────────────────────────
#  C.  OAT DRIVER
# ──────────────────────────────────────────────────────────────────────
records = []
for key, grid in sweep.items():
    for value in grid:
        pars = baseline.copy()
        pars[key] = value
        rec  = simulate(pars, rng_seed=42)
        rec.update(Param=key, Value=value)
        records.append(rec)

results = pd.DataFrame(records)

# ──────────────────────────────────────────────────────────────────────
#  D.  SAVE / PRINT
# ──────────────────────────────────────────────────────────────────────
results.to_csv("OAT_summary_with_optimal.csv", index=False)
print("\n===== OAT SENSITIVITY SUMMARY (Curious · Lazy · Optimal) =====")
print(results.to_string(index=False, float_format="{:0.4f}".format))
