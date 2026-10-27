"""
The Mexican nominal yield curve (CETES + BONOS M), primary auctions, monthly averages.
Oct-2006 to Sep-2026. Source: Banco de México (primary auctions).

Answers three questions:
 Q1  Is there structure in the co-movement of the nine maturities?   -> PCA on changes + Nelson-Siegel
 Q2  Can the curve be forecast 1Q (h=3) and 4Q (h=12) ahead?         -> expanding window, direct forecasts
 Q3  Does the Q1 structure improve the Q2 forecasts?                 -> RMSE relative to RW, DM-HLN, Clark-West

Run from the repo root:  python yield_curve.py
  Writes CSV tables and PNG figures to ./outputs/ (or ./outputs_original/ with DATA=original).
  DATA=corrected (default) applies the Banxico corrections in banxico_corrections.py; DATA=original uses the CSV as is.
  Importing this module (e.g. `import yield_curve as C` from a notebook in the repo root) runs the full pipeline.
Dependencies: pandas, numpy, scipy, matplotlib (statsmodels is not needed)
"""
import warnings, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
# DATA = "original" (CSV as is) or "corrected" (with the corrections against Banxico, see banxico_corrections.py)
DATA = os.environ.get("DATA", "corrected")
OUT = os.path.join(ROOT, "outputs" if DATA == "corrected" else "outputs_original"); os.makedirs(OUT, exist_ok=True)
np.set_printoptions(suppress=True, precision=3)

# --------------------------------------------------------------------------------------
# 0. DATA
# --------------------------------------------------------------------------------------
# Maturities in months (30.4 days/month). CETES are discount/zero-coupon; BONOS M pay a semiannual coupon,
# so the "curve" mixes conventions: we treat it as an approximate par curve.
MAT = {"CETEs, 28 days": 28/30.4, "CETEs, 91 days": 91/30.4, "CETEs, 182 days": 182/30.4,
       "CETEs, 364 days": 364/30.4, "MBONOs, 3 years": 36, "MBONOs, 5 years": 60,
       "MBONOs, 10 years": 120, "MBONOs, 20 years": 240, "MBONOs, 30 years": 360}
LABEL = {"CETEs, 28 days": "1m", "CETEs, 91 days": "3m", "CETEs, 182 days": "6m", "CETEs, 364 days": "12m",
         "MBONOs, 3 years": "3y", "MBONOs, 5 years": "5y", "MBONOs, 10 years": "10y",
         "MBONOs, 20 years": "20y", "MBONOs, 30 years": "30y"}
ORDER = list(MAT.keys())
TAU = np.array([MAT[k] for k in ORDER])          # maturities in months
COLS = [LABEL[k] for k in ORDER]

df = pd.read_csv(os.path.join(ROOT, "data", "banxico_auction_rates_monthly.csv"))
df["DATE"] = pd.to_datetime(df["DATE"])
raw = df.pivot_table(index="DATE", columns="INSTRUMENT", values="VALUE_PCT")[ORDER]
raw.columns = COLS
raw = raw.asfreq("MS")                                   # 240 months, no gaps in the index
if DATA == "corrected":
    import banxico_corrections
    raw = banxico_corrections.correct(raw)
import external_data
EXT = external_data.load(raw.index)                      # policy rate, UST 10y/3m, inflation t-1
OBS = raw.notna()                                        # mask: observed at auction that month

# Missing values: almost all are 1-month gaps in 10y/20y/30y (6-week auction cycle).
miss = raw.isna().groupby(raw.index.year).sum()
miss.to_csv(f"{OUT}/t0_missing_by_year.csv")
print("Total months:", len(raw), "| months with all 9 rates:", raw.dropna().shape[0])
print("Missing by maturity:\n", raw.isna().sum().to_string())

# Decision: balanced panel by LINEAR interpolation over time (for estimation only).
# Forecast TARGETS are evaluated only on observed values (OBS==True).
Y = raw.interpolate(method="linear", limit_direction="both")
DY = Y.diff().dropna()

# Figure 0: the nine rates
fig, ax = plt.subplots(figsize=(11, 5))
for c in COLS:
    ax.plot(Y.index, Y[c], lw=1, label=c)
for a, b, lab in [("2008-10", "2009-08", "Crisis 08-09"), ("2015-12", "2018-12", "Hikes 16-18"),
                  ("2020-02", "2021-05", "Pandemic"), ("2021-06", "2023-03", "Hikes 21-23"),
                  ("2024-03", "2026-09", "Cuts 24-26")]:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=.08, color="k")
    ax.text(pd.Timestamp(a), 12.4, lab, fontsize=7)
ax.set_ylabel("% per year (monthly auction average)"); ax.legend(ncol=5, fontsize=7); ax.set_ylim(2, 13)
ax.set_title("Primary auction rates, CETES and BONOS M, Oct-2006 to Sep-2026")
fig.tight_layout(); fig.savefig(f"{OUT}/f0_rates.png", dpi=150); plt.close()

# --------------------------------------------------------------------------------------
# Q1. STRUCTURE
# --------------------------------------------------------------------------------------
def pca(X):
    """PCA on the covariance matrix. Returns (explained variance, loadings [k x p], scores)."""
    Xc = X - X.mean(0)
    cov = np.cov(Xc, rowvar=False)
    val, vec = np.linalg.eigh(cov)
    idx = np.argsort(val)[::-1]; val, vec = val[idx], vec[:, idx]
    # sign: positive level; slope increasing with maturity; positive curvature in the middle
    for j in range(vec.shape[1]):
        if j == 0 and vec[:, 0].sum() < 0: vec[:, 0] *= -1
        if j == 1 and vec[-1, 1] < vec[0, 1]: vec[:, 1] *= -1
        if j == 2 and vec[4, 2] < 0: vec[:, 2] *= -1
    return val / val.sum(), vec.T, Xc @ vec

# 1a. PCA on monthly CHANGES (full sample, interpolated panel)
ve, load, sc = pca(DY.values)
t1 = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(9)], "var_explained": ve, "cumulative": ve.cumsum()})
t1.to_csv(f"{OUT}/t1_pca_changes_variance.csv", index=False)
print("\nQ1. PCA on monthly changes (interpolated panel). Explained variance:\n", t1.head(4).to_string(index=False))
pd.DataFrame(load[:3].T, index=COLS, columns=["PC1 level", "PC2 slope", "PC3 curvature"]
             ).to_csv(f"{OUT}/t1_pca_changes_loadings.csv")

# 1b. Robustness: only changes between consecutive OBSERVED months (pairwise covariance)
dy_obs = raw.diff(); dy_obs = dy_obs.where(OBS & OBS.shift(1))
cov_pw = dy_obs.cov(min_periods=24).values            # pairwise-complete
val_pw, vec_pw = np.linalg.eigh(cov_pw); idx = np.argsort(val_pw)[::-1]
ve_pw = val_pw[idx] / val_pw.sum()
print("Robustness (pairwise covariance, observed only): expl. var. PC1-3 =", np.round(ve_pw[:3], 3),
      "| cum 3 =", round(ve_pw[:3].sum(), 3))

# 1c. PCA on LEVELS (for contrast; inflates PC1 due to non-stationarity)
ve_lv, load_lv, _ = pca(Y.values)
print("Contrast, PCA on levels: expl. var. PC1-3 =", np.round(ve_lv[:3], 3))

# 1d. Loading stability across subsamples
sub = {"2006-2015": DY.loc[:"2015-12"], "2016-2026": DY.loc["2016-01":]}
stab = {}
for k, d in sub.items():
    v, l, _ = pca(d.values); stab[k] = (v, l)
    print(f"Subsample {k}: expl. var. PC1-3 = {np.round(v[:3], 3)}")
rows = []
for hh in [1, 3, 12]:                                   # stability by change horizon
    d = Y.diff(hh).dropna(); a = pca(d.loc[:"2015-12"].values)[1]; b = pca(d.loc["2016-01":].values)[1]
    rows.append({"horizon_months": hh, **{f"corr_PC{j+1}": abs(np.corrcoef(a[j], b[j])[0, 1]) for j in range(3)}})
STAB = pd.DataFrame(rows); STAB.round(3).to_csv(f"{OUT}/t1_loading_stability.csv", index=False)
print("|Loading| correlation 2006-15 vs 2016-26, by change horizon:\n", STAB.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(9)
for j, nm in enumerate(["PC1 level", "PC2 slope", "PC3 curvature"]):
    axs[0].plot(x, load[j], marker="o", label=f"{nm} ({ve[j]*100:.1f}%)")
    axs[1].plot(x, stab["2006-2015"][1][j], marker="o", ls="-", color=f"C{j}", label=f"PC{j+1} 06-15")
    axs[1].plot(x, stab["2016-2026"][1][j], marker="s", ls="--", color=f"C{j}", label=f"PC{j+1} 16-26")
for a in axs:
    a.set_xticks(x); a.set_xticklabels(COLS); a.axhline(0, color="k", lw=.5); a.legend(fontsize=7)
axs[0].set_title("PCA loadings on monthly changes (full sample)")
axs[1].set_title("Loading stability across subsamples")
fig.tight_layout(); fig.savefig(f"{OUT}/f1_pca_loadings.png", dpi=150); plt.close()

# 1d-bis. Structure by horizon: PCA on 1-, 3- and 12-month changes.
# At monthly frequency the auction-calendar noise (bonds auctioned in different weeks, monthly
# averages, interpolation) dilutes the co-movement; at 3 and 12 months the signal dominates.
rows = []
for hh in [1, 3, 12]:
    d = Y.diff(hh).dropna(); v, l, _ = pca(d.values)
    rows.append({"horizon_months": hh, "PC1": v[0], "PC2": v[1], "PC3": v[2], "cum3": v[:3].sum()})
HPCA = pd.DataFrame(rows); HPCA.round(3).to_csv(f"{OUT}/t1_pca_by_horizon.csv", index=False)
print("\nPCA on changes at different horizons (explained variance):\n", HPCA.round(3).to_string(index=False))
DY.corr().round(2).to_csv(f"{OUT}/t1_corr_changes_monthly.csv")
Y.diff(12).dropna().corr().round(2).to_csv(f"{OUT}/t1_corr_changes_12m.csv")

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for i, hh in enumerate([1, 3, 12]):
    d = Y.diff(hh).dropna(); v, l, _ = pca(d.values)
    for j, nm in enumerate(["level", "slope", "curvature"]):
        axs[i].plot(range(9), l[j], marker="o", label=f"PC{j+1} {nm} ({v[j]*100:.0f}%)")
    axs[i].set_xticks(range(9)); axs[i].set_xticklabels(COLS, fontsize=8); axs[i].axhline(0, color="k", lw=.5)
    axs[i].set_title(f"{hh}-month changes: 3 PCs = {v[:3].sum()*100:.0f}%"); axs[i].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/f1_pca_horizons.png", dpi=150); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for a, (d, tt) in zip(axs, [(DY, "1-month changes"), (Y.diff(12).dropna(), "12-month changes")]):
    im = a.imshow(d.corr().values, vmin=0, vmax=1, cmap="Blues"); a.set_xticks(range(9)); a.set_yticks(range(9))
    a.set_xticklabels(COLS, fontsize=8); a.set_yticklabels(COLS, fontsize=8); a.set_title(f"Correlation, {tt}")
    for i in range(9):
        for j in range(9): a.text(j, i, f"{d.corr().values[i,j]:.2f}", ha="center", va="center", fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/f1_corr.png", dpi=150); plt.close()

# 1e. Nelson-Siegel month by month (tolerates missing maturities: uses only those observed each month)
def ns_loadings(tau, lam):
    a = (1 - np.exp(-lam * tau)) / (lam * tau)
    return np.column_stack([np.ones_like(tau), a, a - np.exp(-lam * tau)])

def ns_fit(yrow, obsrow, lam):
    """OLS of observed yields on the NS loadings. Requires >=5 nodes, 1 CETE and 1 bond >=10y."""
    m = obsrow.values
    if m.sum() < 5 or not m[:4].any() or not m[6:].any():
        return np.full(3, np.nan)
    X = ns_loadings(TAU[m], lam)
    b, *_ = np.linalg.lstsq(X, yrow.values[m], rcond=None)
    return b

LAM_DL = 0.0609                                        # Diebold-Li: maximum curvature at ~30 months
def ns_factors(Yp, Op, lam):
    F = np.array([ns_fit(Yp.iloc[i], Op.iloc[i], lam) for i in range(len(Yp))])
    return pd.DataFrame(F, index=Yp.index, columns=["L", "S", "C"])

# grid search for lambda (full sample; decision documented as "with full information")
grid = np.linspace(0.01, 0.25, 49); rm = []
for lam in grid:
    F = ns_factors(raw, OBS, lam).dropna()
    fit = F.values @ ns_loadings(TAU, lam).T
    e = (raw.loc[F.index].values - fit)[OBS.loc[F.index].values]
    rm.append(np.sqrt(np.nanmean(e**2)))
LAM_OPT = grid[int(np.argmin(rm))]
print(f"\nNelson-Siegel: optimal lambda on grid = {LAM_OPT:.4f} (fit RMSE {min(rm):.3f} pp) "
      f"vs Diebold-Li lambda 0.0609 (RMSE {rm[np.argmin(abs(grid-LAM_DL))]:.3f} pp)")

F_ns = ns_factors(raw, OBS, LAM_DL)
fit = F_ns.values @ ns_loadings(TAU, LAM_DL).T
res = pd.DataFrame(raw.values - fit, index=raw.index, columns=COLS).where(OBS)
t1ns = pd.DataFrame({"RMSE_fit_pp": np.sqrt((res**2).mean()), "MAE_pp": res.abs().mean()})
t1ns.to_csv(f"{OUT}/t1_ns_fit_by_maturity.csv")
print("NS cross-sectional fit RMSE (lambda=0.0609) by maturity, pp:\n", t1ns["RMSE_fit_pp"].round(3).to_string())

# NS factors vs empirical proxies
prox = pd.DataFrame({"level_30y": raw["30y"], "slope_30y-1m": raw["30y"] - raw["1m"],
                     "curv_2*3y-1m-30y": 2*raw["3y"] - raw["1m"] - raw["30y"]})
cors = [F_ns["L"].corr(prox.iloc[:, 0]), F_ns["S"].corr(-prox.iloc[:, 1]), F_ns["C"].corr(prox.iloc[:, 2])]
print("Corr(L, y30y) = %.3f | Corr(S, -(y30y-y1m)) = %.3f | Corr(C, empirical curvature) = %.3f" % tuple(cors))
pd.DataFrame({"factor": ["L", "S", "C"], "corr_with_proxy": cors}).to_csv(f"{OUT}/t1_ns_vs_proxies.csv", index=False)

fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
axs[0].plot(F_ns["L"], label="L (NS level)"); axs[0].plot(prox.iloc[:, 0], ls="--", label="y 30y")
axs[1].plot(-F_ns["S"], label="-S (NS slope)"); axs[1].plot(prox.iloc[:, 1], ls="--", label="y30y - y1m")
axs[2].plot(F_ns["C"], label="C (NS curvature)"); axs[2].plot(prox.iloc[:, 2], ls="--", label="2·y3y - y1m - y30y")
for a in axs: a.legend(fontsize=8); a.axhline(0, color="k", lw=.4)
axs[0].set_title("Nelson-Siegel factors (λ=0.0609) vs empirical proxies")
fig.tight_layout(); fig.savefig(f"{OUT}/f1_ns_factors.png", dpi=150); plt.close()

# --------------------------------------------------------------------------------------
# Q2. OUT-OF-SAMPLE FORECASTING
# --------------------------------------------------------------------------------------
# Target: y_{t+h} (monthly auction average for month t+h), h=3 (1Q) and h=12 (4Q).
# Information at t: all auctions up to the end of month t. EXPANDING window from 2006-10.
# Everything (means, PCA loadings, coefficients) is re-estimated at each origin using only data <= t.
# DIRECT forecast: regression of z_{t+h} on (1, z_t) estimated with pairs (s, s+h), s+h <= t.
FIRST_ORIGIN = pd.Timestamp("2012-12-01")
H = [3, 12]
ROLL = 72                                              # rolling window (robustness), months

def direct_ols(Z, h, X=None):
    """z_{t+h} = a + B x_t. Z: target matrix (T x k); X: regressors (default X=Z). Returns (a, B)."""
    X = Z if X is None else X
    Yt, Xt = Z[h:], X[:-h]
    Xd = np.column_stack([np.ones(len(Xt)), Xt])
    coef, *_ = np.linalg.lstsq(Xd, Yt, rcond=None)
    return coef[0], coef[1:]

def fc_rw(Yw, h):          return Yw.values[-1]
def fc_mean(Yw, h):        return Yw.values.mean(0)
def fc_ar1(Yw, h):         # direct AR(1) per maturity (univariate)
    out = np.empty(9)
    for j in range(9):
        z = Yw.values[:, [j]]; a, B = direct_ols(z, h); out[j] = a[0] + B[0, 0]*z[-1, 0]
    return out
def fc_var1(Yw, h):        # unrestricted direct VAR(1) on the 9 yields
    a, B = direct_ols(Yw.values, h); return a + Yw.values[-1] @ B
def fc_pca_ar(Yw, h, k=3):  # k components (loadings from changes, up to t), direct AR(1) per factor
    mu = Yw.values.mean(0); dy = np.diff(Yw.values, axis=0)
    _, L, _ = pca(dy); L = L[:k]                        # k x 9
    Fs = (Yw.values - mu) @ L.T                         # scores of centered levels
    fF = np.empty(k)
    for j in range(k):
        a, B = direct_ols(Fs[:, [j]], h); fF[j] = a[0] + B[0, 0]*Fs[-1, j]
    return mu + fF @ L
def fc_pca_var(Yw, h, k=3):
    mu = Yw.values.mean(0); dy = np.diff(Yw.values, axis=0)
    _, L, _ = pca(dy); L = L[:k]; Fs = (Yw.values - mu) @ L.T
    a, B = direct_ols(Fs, h); return mu + (a + Fs[-1] @ B) @ L
def fc_dns_ar(Yw, h, lam=LAM_DL):   # Diebold-Li: NS with fixed lambda + direct AR(1) per factor
    X = ns_loadings(TAU, lam)
    F = np.linalg.lstsq(X, Yw.values.T, rcond=None)[0].T   # T x 3 (interpolated panel => 9 nodes)
    fF = np.empty(3)
    for j in range(3):
        a, B = direct_ols(F[:, [j]], h); fF[j] = a[0] + B[0, 0]*F[-1, j]
    return X @ fF
def fc_dns_var(Yw, h, lam=LAM_DL):
    X = ns_loadings(TAU, lam); F = np.linalg.lstsq(X, Yw.values.T, rcond=None)[0].T
    a, B = direct_ols(F, h); return X @ (a + F[-1] @ B)

def fc_slope(Yw, h):       # RW + slope: y_{j,t+h} - y_{j,t} = a_j + b_j (y12m - y1m)_t. A single regressor.
    yv = Yw.values; sl = (Yw["12m"] - Yw["1m"]).values[:, None]
    out = np.empty(9)
    for j in range(9):
        dz = (yv[h:, j] - yv[:-h, j])[:, None]
        Xd = np.column_stack([np.ones(len(dz)), sl[:-h]])
        coef, *_ = np.linalg.lstsq(Xd, dz, rcond=None)
        out[j] = yv[-1, j] + coef[0, 0] + coef[1, 0]*sl[-1, 0]
    return out

def _direct_dy(Yw, h, Xfun):
    """For each maturity j: y_{j,t+h} - y_{j,t} = a_j + b_j' x_{j,t}; x built only with information at t."""
    yv = Yw.values; out = np.empty(9)
    for j in range(9):
        X = Xfun(Yw, j)                                        # T x k
        dz = yv[h:, j] - yv[:-h, j]
        Xd = np.column_stack([np.ones(len(dz)), X[:-h]])
        coef, *_ = np.linalg.lstsq(Xd, dz, rcond=None)
        out[j] = yv[-1, j] + coef[0] + X[-1] @ coef[1:]
    return out
_E = lambda Yw: EXT.loc[Yw.index]
_slope = lambda Yw: (Yw["12m"] - Yw["1m"]).values
# (a) Spread over the policy rate: how much each maturity prices in relative to today's Banxico rate
def fc_pol(Yw, h):  return _direct_dy(Yw, h, lambda Yw, j: (Yw.iloc[:, j] - _E(Yw).policy_rate).values[:, None])
# (b) Slope + real policy rate (rate minus known inflation): monetary stance
def fc_slope_real(Yw, h):
    return _direct_dy(Yw, h, lambda Yw, j: np.column_stack([_slope(Yw), (_E(Yw).policy_rate - _E(Yw).inflation_lag1).values]))
# (c) Slope + spread over the US Treasury (10y): a Mexico premium that mean-reverts
def fc_slope_ust(Yw, h):
    return _direct_dy(Yw, h, lambda Yw, j: np.column_stack([_slope(Yw), (Yw.iloc[:, j] - _E(Yw).ust10).values]))
# (d) Slope + recent (3-month) change in the 10-year Treasury: US pass-through
def fc_slope_dust(Yw, h):
    return _direct_dy(Yw, h, lambda Yw, j: np.column_stack([_slope(Yw), _E(Yw).ust10.diff(3).fillna(0).values]))

MODELS = {"RW": fc_rw, "Mean": fc_mean, "AR1": fc_ar1, "Slope": fc_slope, "VAR1": fc_var1,
          "PCA3-AR1": fc_pca_ar, "PCA3-VAR1": fc_pca_var, "DNS-AR1": fc_dns_ar, "DNS-VAR1": fc_dns_var,
          "Ext:Policy": fc_pol, "Ext:Slope+RealRate": fc_slope_real, "Ext:Slope+SpreadUST": fc_slope_ust,
          "Ext:Slope+ΔUST": fc_slope_dust}

def run_oos(window=None):
    recs = []
    origins = Y.index[(Y.index >= FIRST_ORIGIN)]
    for t in origins:
        Yw = Y.loc[:t] if window is None else Y.loc[:t].iloc[-window:]
        for h in H:
            tgt = t + pd.DateOffset(months=h)
            if tgt > Y.index[-1]: continue
            for nm, f in MODELS.items():
                p = f(Yw, h)
                for j, c in enumerate(COLS):
                    if OBS.loc[tgt, c]:                   # only targets observed at auction
                        recs.append((t, tgt, h, nm, c, p[j], raw.loc[tgt, c]))
    r = pd.DataFrame(recs, columns=["origin", "target", "h", "model", "maturity", "forecast", "obs"])
    r["err"] = r["obs"] - r["forecast"]
    return r

R = run_oos(); R.to_csv(f"{OUT}/t2_forecasts_long.csv", index=False)
R_roll = run_oos(ROLL)
print(f"\nQ2. Origins: {R.origin.nunique()} (from {R.origin.min():%Y-%m} to {R.origin.max():%Y-%m}); "
      f"evaluated pairs h=3: {len(R[(R.h==3)&(R.model=='RW')])}, h=12: {len(R[(R.h==12)&(R.model=='RW')])}")

# --------------------------------------------------------------------------------------
# Q3. EVALUATION
# --------------------------------------------------------------------------------------
def rmse_table(R):
    t = R.groupby(["h", "model", "maturity"])["err"].apply(lambda e: np.sqrt((e**2).mean())).unstack("maturity")[COLS]
    t["Curve"] = R.groupby(["h", "model"])["err"].apply(lambda e: np.sqrt((e**2).mean()))
    return t

RM = rmse_table(R)
ratio = RM.copy()
for h in H:
    ratio.loc[h] = (RM.loc[h] / RM.loc[(h, "RW")]).values
RM.round(3).to_csv(f"{OUT}/t3_rmse.csv"); ratio.round(3).to_csv(f"{OUT}/t3_rmse_ratio_vs_RW.csv")
order_m = list(MODELS.keys())
for h in H:
    print(f"\nQ3. RMSE relative to the random walk, h={h} (<1 = the model wins):")
    print(ratio.loc[h].loc[order_m].round(3).to_string())

def dm_hln(e1, e2, h):
    """Diebold-Mariano, squared loss, HAC variance (Bartlett, h-1 lags), HLN correction.
    H0: equal accuracy. Statistic <0 => model 1 (e1) more accurate than model 2 (e2)."""
    d = e1**2 - e2**2; T = len(d); dbar = d.mean(); dc = d - dbar
    g = [np.sum(dc[k:]*dc[:T-k])/T for k in range(h)]
    lrv = g[0] + 2*sum((1 - k/h)*g[k] for k in range(1, h))
    if lrv <= 0: lrv = g[0]
    dm = dbar / np.sqrt(lrv/T)
    hln = dm*np.sqrt((T + 1 - 2*h + h*(h-1)/T)/T)
    return hln, 2*stats.t.sf(abs(hln), df=T-1)

def clark_west(e_small, e_big, f_small, f_big, h):
    """Clark-West (2007) adjusted MSPE, for nested models. H0: the small model is correct.
    Positive and significant statistic => the big model (which nests the small one) improves."""
    f = e_small**2 - (e_big**2 - (f_small - f_big)**2); T = len(f); fbar = f.mean(); fc = f - fbar
    g = [np.sum(fc[k:]*fc[:T-k])/T for k in range(h)]
    lrv = g[0] + 2*sum((1 - k/h)*g[k] for k in range(1, h)); lrv = lrv if lrv > 0 else g[0]
    z = fbar/np.sqrt(lrv/T); return z, stats.norm.sf(z)

def paired(R, m1, m2, h, maturity=None):
    a = R[(R.model == m1) & (R.h == h)]; b = R[(R.model == m2) & (R.h == h)]
    if maturity: a, b = a[a.maturity == maturity], b[b.maturity == maturity]
    m = a.merge(b, on=["origin", "target", "maturity"], suffixes=("_1", "_2"))
    return m

# DM-HLN of each model against RW, by maturity and for the curve (stacked errors)
rows = []
for h in H:
    for m in order_m:
        if m == "RW": continue
        for pl in COLS + ["Curve"]:
            mm = paired(R, m, "RW", h, None if pl == "Curve" else pl)
            s, p = dm_hln(mm.err_1.values, mm.err_2.values, h)
            rows.append({"h": h, "model": m, "maturity": pl, "DM_HLN": s, "p": p})
DM = pd.DataFrame(rows); DM.round(3).to_csv(f"{OUT}/t3_dm_hln_vs_RW.csv", index=False)
pv = DM.pivot_table(index=["h", "model"], columns="maturity", values="DM_HLN")[COLS + ["Curve"]]
for h in H:
    print(f"\nDM-HLN vs RW, h={h} (negative = model better than RW; |t|>1.65 ~10%, >1.96 ~5%):")
    print(pv.loc[h].loc[[m for m in order_m if m != 'RW']].round(2).to_string())

# Clark-West on nested pairs: RW ⊂ AR1 (univariate); AR1 ⊂ VAR1; DNS-AR1 ⊂ DNS-VAR1; PCA3-AR1 ⊂ PCA3-VAR1
rows = []
for h in H:
    for small, big in [("RW", "AR1"), ("RW", "Slope"), ("RW", "Ext:Policy"), ("Slope", "Ext:Slope+RealRate"), ("Slope", "Ext:Slope+SpreadUST"), ("Slope", "Ext:Slope+ΔUST"), ("AR1", "VAR1"), ("DNS-AR1", "DNS-VAR1"), ("PCA3-AR1", "PCA3-VAR1")]:
        for pl in COLS + ["Curve"]:
            mm = paired(R, small, big, h, None if pl == "Curve" else pl)
            z, p = clark_west(mm.err_1.values, mm.err_2.values, mm.forecast_1.values, mm.forecast_2.values, h)
            rows.append({"h": h, "small": small, "big": big, "maturity": pl, "CW": z, "p": p})
CW = pd.DataFrame(rows); CW.round(3).to_csv(f"{OUT}/t3_clark_west.csv", index=False)
print("\nClark-West (curve), >1.65 => the big model improves on the one it nests:")
print(CW[CW.maturity == "Curve"].pivot_table(index=["small", "big"], columns="h", values="CW").round(2).to_string())

# Comparisons that answer Q3 directly: does structure help against models without structure?
rows = []
for h in H:
    for struct, free in [("Slope", "AR1"), ("DNS-VAR1", "AR1"), ("VAR1", "AR1"), ("DNS-AR1", "AR1"), ("PCA3-AR1", "AR1"),
                         ("DNS-VAR1", "VAR1"), ("PCA3-VAR1", "VAR1"), ("Slope", "VAR1")]:
        mm = paired(R, struct, free, h)
        s, p = dm_hln(mm.err_1.values, mm.err_2.values, h)
        r1 = np.sqrt((mm.err_1**2).mean()); r2 = np.sqrt((mm.err_2**2).mean())
        rows.append({"h": h, "with_pattern": struct, "without_pattern": free, "RMSE_ratio": r1/r2, "DM_HLN": s, "p": p})
Q3 = pd.DataFrame(rows); Q3.round(3).to_csv(f"{OUT}/t3_pattern_vs_no_pattern.csv", index=False)
print("\nQ3 directly. Models with structure vs without structure (full curve):")
print(Q3.round(3).to_string(index=False))

# External data: do they add anything over the best model without external data (Slope)?
rows = []
for h in H:
    for m in ["Ext:Policy", "Ext:Slope+RealRate", "Ext:Slope+SpreadUST", "Ext:Slope+ΔUST"]:
        for segment, pls in [("CETES", COLS[:4]), ("BONOS", COLS[4:]), ("Curve", COLS)]:
            a = paired(R, m, "Slope", h); a = a[a.maturity.isin(pls)]
            b = paired(R, m, "RW", h); b = b[b.maturity.isin(pls)]
            s1, p1 = dm_hln(a.err_1.values, a.err_2.values, h)
            rows.append({"h": h, "model": m, "segment": segment,
                         "RMSE_vs_RW": np.sqrt((b.err_1**2).mean()/(b.err_2**2).mean()),
                         "RMSE_vs_Slope": np.sqrt((a.err_1**2).mean()/(a.err_2**2).mean()), "DM_vs_Slope": s1, "p": p1})
EXTT = pd.DataFrame(rows); EXTT.round(3).to_csv(f"{OUT}/t4_external_data.csv", index=False)
print("\nExternal data (relative RMSE; <1 wins):"); print(EXTT.round(3).to_string(index=False))

# Subperiods and rolling window
R["sub"] = np.where(R.target < "2020-01-01", "2013-2019", "2020-2026")
SUB = R.groupby(["h", "sub", "model"])["err"].apply(lambda e: np.sqrt((e**2).mean())).unstack("model")[order_m]
SUBr = SUB.div(SUB["RW"], axis=0); SUBr.round(3).to_csv(f"{OUT}/t3_rmse_ratio_subperiods.csv")
print("\nCurve RMSE relative to RW by subperiod (by target date):"); print(SUBr.round(3).to_string())
RMr = rmse_table(R_roll); ratio_roll = RMr.copy()
for h in H: ratio_roll.loc[h] = (RMr.loc[h] / RMr.loc[(h, "RW")]).values
ratio_roll.round(3).to_csv(f"{OUT}/t3_rmse_ratio_rolling72.csv")
print("\nRobustness, 72-month rolling window, curve RMSE relative to RW:")
print(ratio_roll["Curve"].unstack("h").loc[order_m].round(3).to_string())

# Q3 figures
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for i, h in enumerate(H):
    for m in ["AR1", "Slope", "VAR1", "PCA3-AR1", "DNS-AR1", "DNS-VAR1"]:
        axs[i].plot(range(9), ratio.loc[(h, m)][COLS].values, marker="o", label=m)
    axs[i].axhline(1, color="k", lw=1, label="RW"); axs[i].set_xticks(range(9)); axs[i].set_xticklabels(COLS)
    axs[i].set_title(f"RMSE relative to RW, h={h} months"); axs[i].legend(fontsize=7)
axs[0].set_ylabel("Model RMSE / RW RMSE")
fig.tight_layout(); fig.savefig(f"{OUT}/f3_rmse_ratio.png", dpi=150); plt.close()

fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for i, h in enumerate(H):
    for m, ls in [("RW", "--"), ("Slope", "-"), ("DNS-VAR1", ":")]:
        s = R[(R.h == h) & (R.model == m) & (R.maturity == "3m")].set_index("target")
        axs[i].plot(s.index, s.forecast, ls=ls, label=f"{m}")
    axs[i].plot(s.index, s.obs, color="k", lw=1.5, label="Observed"); axs[i].legend(fontsize=8)
    axs[i].set_title(f"91-day CETE: observed vs forecast h={h}")
fig.tight_layout(); fig.savefig(f"{OUT}/f3_forecast_3m.png", dpi=150); plt.close()

fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for i, h in enumerate(H):
    for m, ls in [("RW", "--"), ("Slope", "-"), ("DNS-VAR1", ":")]:
        s = R[(R.h == h) & (R.model == m) & (R.maturity == "10y")].set_index("target")
        axs[i].plot(s.index, s.forecast, ls=ls, label=f"{m}")
    axs[i].plot(s.index, s.obs, color="k", lw=1.5, label="Observed"); axs[i].legend(fontsize=8)
    axs[i].set_title(f"10-year BONO: observed vs forecast h={h}")
fig.tight_layout(); fig.savefig(f"{OUT}/f3_forecast_10y.png", dpi=150); plt.close()

# Cumulative errors (RW vs Slope): where each one wins over time
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for i, h in enumerate(H):
    mm = paired(R, "Slope", "RW", h); d = (mm.err_2**2 - mm.err_1**2).groupby(mm.target).mean().cumsum()
    axs[i].plot(d.index, d.values); axs[i].axhline(0, color="k", lw=.5)
    axs[i].set_title(f"Cumulative sum of (e²RW - e²Slope), h={h}\n(up = Slope wins; down = RW wins)")
fig.tight_layout(); fig.savefig(f"{OUT}/f3_cumdiff.png", dpi=150); plt.close()
print(f"\nDone. Outputs in {OUT}/")
