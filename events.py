"""
Extension: treatment of policy episodes and events (Banxico cycle, crises, pandemic, breaks).
Runs after yield_curve.py (it imports it, which reruns the main pipeline). Everything respects the information
available at each origin: no model uses, at origin t, an event that had not yet happened.

  A. Real-time policy regime: direction of Banxico's last move within the previous 6 months
     (+1 hikes, -1 cuts, 0 pause). Inspired by threshold/regime models (Ibarra, Banxico DI 2021-07).
  B. Estimation robust to atypical episodes: when estimating, pairs (s, s+h) that touch the 2008-10..2009-06
     crisis, the 2013-05..2013-09 taper tantrum or the 2020-03..2020-08 pandemic are excluded (only once they
     have happened). Equivalent to a dummy for each month of the episode in the estimation.
  C. Window combination (Pesaran and Timmermann 2007): simple average of the forecasts estimated with an
     expanding window, a 10-year window and a 6-year window.
  D. Evaluation without the pandemic: does the ranking change if Mar-Dec 2020 targets are excluded?

Run from the repo root:  python events.py   (writes t5_events.csv, t5_events_forecasts.csv and
f5_regime_episodes.png to ./outputs/, or ./outputs_original/ with DATA=original)
"""
import io, contextlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import yield_curve as C
Y, EXT, COLS, OUT = C.Y, C.EXT, C.COLS, C.OUT

# A. policy regime known at t
pol = EXT.policy_rate
d = pol.diff().fillna(0).where(lambda s: s.abs() > 0.15, 0)        # ignores noise in the 2006-07 funding rate
last = np.sign(d).replace(0, np.nan).ffill().fillna(0)
quiet = d.abs().rolling(6, min_periods=1).sum() == 0
CYCLE = last.where(~quiet, 0)                                         # +1 hikes, -1 cuts, 0 pause
slope = lambda Yw: (Yw["12m"] - Yw["1m"]).values
def fc_slope_cycle(Yw, h):
    c = CYCLE.loc[Yw.index].values
    return C._direct_dy(Yw, h, lambda Yw, j: np.column_stack([slope(Yw), c, slope(Yw) * c]))

# B. estimation excluding atypical episodes that have already happened
EPIS = [("2008-10-01", "2009-06-01"), ("2013-05-01", "2013-09-01"), ("2020-03-01", "2020-08-01")]
def _in_epis(idx):
    m = np.zeros(len(idx), bool)
    for a, b in EPIS: m |= (idx >= a) & (idx <= b)
    return m
def fc_slope_excl_epis(Yw, h):
    yv = Yw.values; s = slope(Yw); bad = _in_epis(Yw.index); out = np.empty(9)
    keep = ~(bad[:-h] | bad[h:])                                       # drop pairs that touch an episode
    for j in range(9):
        dz = yv[h:, j] - yv[:-h, j]
        Xd = np.column_stack([np.ones(len(dz)), s[:-h]])[keep]
        coef, *_ = np.linalg.lstsq(Xd, dz[keep], rcond=None)
        out[j] = yv[-1, j] + coef[0] + coef[1] * s[-1]
    return out

# C. window combination
def fc_slope_pool(Yw, h):
    return np.mean([C.fc_slope(Yw, h), C.fc_slope(Yw.iloc[-120:], h), C.fc_slope(Yw.iloc[-72:], h)], axis=0)

C.MODELS = {"RW": C.fc_rw, "Slope": C.fc_slope, "Slope+Cycle": fc_slope_cycle,
            "Slope excl. episodes": fc_slope_excl_epis, "Slope pooled windows": fc_slope_pool,
            "Ext:Slope+RealRate": C.fc_slope_real}
R = C.run_oos()
R.to_csv(f"{OUT}/t5_events_forecasts.csv", index=False)

rows = []
for h in C.H:
    for m in [k for k in C.MODELS if k != "RW"]:
        for segment, pls in [("CETES", COLS[:4]), ("BONOS", COLS[4:]), ("Curve", COLS)]:
            for excl in [False, True]:
                rr = R if not excl else R[~((R.target >= "2020-03-01") & (R.target <= "2020-12-01"))]
                a = C.paired(rr, m, "RW", h); a = a[a.maturity.isin(pls)]
                b = C.paired(rr, m, "Slope", h); b = b[b.maturity.isin(pls)]
                s1, p1 = C.dm_hln(a.err_1.values, a.err_2.values, h)
                s2, p2 = (np.nan, np.nan) if m == "Slope" else C.dm_hln(b.err_1.values, b.err_2.values, h)
                rows.append({"h": h, "model": m, "segment": segment, "excl_pandemic": excl,
                             "RMSE_vs_RW": np.sqrt((a.err_1**2).mean() / (a.err_2**2).mean()), "p_vs_RW": p1,
                             "RMSE_vs_Slope": np.nan if m == "Slope" else np.sqrt((b.err_1**2).mean() / (b.err_2**2).mean()),
                             "DM_vs_Slope": s2, "p_vs_Slope": p2})
T = pd.DataFrame(rows); T.round(3).to_csv(f"{OUT}/t5_events.csv", index=False)
pd.set_option("display.width", 200)
print(T[T.segment != "BONOS"].round(3).to_string(index=False))

# subperiods for Slope+Cycle
R["sub"] = np.where(R.target < "2020-01-01", "2013-2019", "2020-2026")
S = R.groupby(["h", "sub", "model"])["err"].apply(lambda e: np.sqrt((e**2).mean())).unstack("model")
print((S.div(S["RW"], axis=0)).round(3).to_string())
print("Regime distribution across origins:", CYCLE.loc["2012-12":].value_counts().to_dict())

# Figure: regime + episodes over the 28-day CETE and the 10-year bond
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(Y.index, Y["1m"], color="#2a6f97", lw=1.3, label="CETE 28d"); ax.plot(Y.index, Y["10y"], color="#c97c1f", lw=1.1, label="Bono 10y")
ax.step(EXT.index, EXT.policy_rate, where="post", color="k", lw=.9, ls="--", label="Target / funding rate")
for t, c in CYCLE.items():
    if c != 0: ax.axvspan(t, t + pd.offsets.MonthBegin(1), color="#d62728" if c > 0 else "#2ca02c", alpha=.10, lw=0)
for (a, b), nm in zip(EPIS, ["Crisis 08-09", "Taper tantrum", "Pandemic"]):
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#555", alpha=.22, lw=0); ax.text(pd.Timestamp(a), 12.2, nm, fontsize=7)
ax.set_ylim(2, 12.8); ax.legend(fontsize=7, ncol=3, loc="lower left")
ax.set_title("Policy regime known each month (red: hikes, green: cuts) and atypical episodes (grey)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/f5_regime_episodes.png", dpi=150)
