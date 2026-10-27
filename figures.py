"""Brief figures with event annotations, plus two wide slide figures. Run after yield_curve.py and events.py
(it reads their CSV outputs; importing yield_curve reruns the main pipeline).
Reference categorical palette (fixed order): blue, orange, aqua, yellow; text in neutral ink.

Run from the repo root:
  python figures.py           -> fig1_curve_events.png, fig2_movements.png, fig3_error_grid.png,
                                 fig4_cete91_forecast.png, fig5_external_events.png (needs surveys.py and extra_predictors.py first),
                                 slide_policy_rate_vs_average.png, slide_cete91_forecast.png
  TRANSP=1 python figures.py  -> transparent background, ink readable in light and dark mode; adds suffix _t
  WIDE=1 python figures.py    -> wide version of figure 1 for slides (fig1_curve_events_slide.png)
Outputs go to ./outputs/ (or ./outputs_original/ with DATA=original).
"""
import io, contextlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import yield_curve as C
Y, raw, EXT, COLS = C.Y, C.raw, C.EXT, C.COLS
B, O, A, Yl, V = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
import os
TRANSP = os.environ.get("TRANSP") == "1"          # transparent background, neutral ink readable in light and dark mode
INK, INK2, GRID = ("#000000", "#000000", (0.5, 0.5, 0.5, 0.25)) if TRANSP else ("#000000", "#000000", "#e4e3df")
ANN = "#1b1b1a"                                    # annotation text (always on a white box)
SUF = "_t" if TRANSP else ""
SHADE_G = (0.18, 0.55, 0.32, 0.16) if TRANSP else "#e9f3ec"
SHADE_R = (0.85, 0.30, 0.30, 0.22) if TRANSP else "#f8e3e3"
SHADE_N = (0.5, 0.5, 0.5, 0.18) if TRANSP else "#efeeea"
ZONE = "#3f9a5f" if TRANSP else "#2f6b44"
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 9.5, "axes.edgecolor": "#000000", "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK2, "text.color": INK2, "legend.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "legend.frameon": False})
def note(ax, x, y, xt, yt, txt, ha="left"):
    ax.annotate(txt, xy=(pd.Timestamp(x), y), xytext=(pd.Timestamp(xt), yt), fontsize=8, color=ANN, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-|>", color="#6d6b66", lw=.8, shrinkA=2, shrinkB=2),
                bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#c9c7c1", lw=.6))

# Actual size: the figures are inserted at ~5.2 inches wide, so text appears close to its nominal size.
MX, US = O, A          # Mexico (Bono M) in orange; US in aqua, distinct from the CETE blue

# F1. The curve over time, with events (Mexico on top, Mexico vs US below)
WIDE = os.environ.get("WIDE") == "1"
fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.6, 4.7) if WIDE else (5.6, 4.9), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
ax.plot(raw.index, raw["1m"], color=B, lw=1.5, label="CETE 28 days")
ax.plot(Y.index, Y["10y"], color=MX, lw=1.2, label="Bono M 10 years")
ax.step(EXT.index, EXT.policy_rate, where="post", color=INK, lw=.9, ls=(0, (3, 2)), label="Banxico policy rate")
ax.set_ylabel("Annual rate (%)"); ax.set_ylim(0, 14.8); ax.set_xlim(pd.Timestamp("2006-09"), pd.Timestamp("2026-12"))
note(ax, "2008-11", 9.8, "2006-11", 13.4, "2008 crisis: long\nbonds spike")
note(ax, "2017-06", 6.8, "2011-01" if not WIDE else "2012-09", 1.2, "Banxico hikes\n2015-2018")
note(ax, "2020-05", 5.6, "2017-11", 1.2, "Pandemic:\ncuts")
note(ax, "2023-03", 11.3, "2014-06" if not WIDE else "2015-06", 13.4, "Banxico at 11.25%: short\nrates above long rates")
note(ax, "2024-06", 10.9, "2022-01", 1.6, "Cuts\nsince 2024")
if WIDE:
    ax.legend(loc="lower left", fontsize=8.5, ncol=3, bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.6, columnspacing=1)
    ax.text(0, 1.13, "Mexico: rates and Banxico policy", transform=ax.transAxes, fontsize=9.5, color=INK, fontweight="bold")
else:
    ax.legend(loc="lower left", fontsize=8.5, ncol=2, bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.6, columnspacing=1)
    ax.text(0, 1.30, "Mexico: rates and Banxico policy", transform=ax.transAxes, fontsize=9.5, color=INK, fontweight="bold")
bx.axvspan(pd.Timestamp("2008-11"), pd.Timestamp("2014-10"), color=SHADE_N, lw=0)
bx.text(pd.Timestamp("2011-10"), 12.6 if WIDE else 15.2, "Fed bond purchases\n(QE), Nov-2008\nto Oct-2014" if WIDE else "Fed bond purchases\n(QE), 2008-2014", fontsize=8, color=INK2, ha="center", va="top")
bx.plot(Y.index, Y["10y"], color=MX, lw=1.2, label="Mexico: Bono M 10 years")
bx.plot(EXT.index, EXT.ust10, color=US, lw=1.4, label="US: 10-year Treasury")
bx.set_ylabel("Annual rate (%)"); bx.set_ylim(0, 12.8 if WIDE else 15.6)
note(bx, "2013-06", 2.3, "2015-02" if WIDE else "2015-03", 11.4 if WIDE else 13.5, "2013 taper tantrum:\nboth rates rise")
note(bx, "2020-04", 7.4, "2020-11" if WIDE else "2020-09", 11.6 if WIDE else 13.5, "Mar-Apr 2020: US\nfalls, Mexico rises")
bx.legend(loc="lower left", fontsize=8.5, ncol=2 if WIDE else 1, bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.6, columnspacing=1)
bx.text(0, 1.21 if WIDE else 1.42, "Mexico and US: 10-year bonds (correlation 0.76)", transform=bx.transAxes, fontsize=9.5, color=INK, fontweight="bold")
fig.tight_layout(h_pad=2.2 if WIDE else 2.4); fig.savefig(f"{C.OUT}/fig1_curve_events{SUF}{'_slide' if WIDE else ''}.png", dpi=220, transparent=TRANSP); plt.close()

# F2. The three movements (12-month changes): how much each maturity takes part
fig, a = plt.subplots(figsize=(4.0, 2.55))
v, l, _ = C.pca(Y.diff(12).dropna().values)
for j_, (nm, c) in enumerate([("Level", B), ("Slope", O), ("Curvature", A)]):
    a.plot(range(9), l[j_], marker="o", ms=4, lw=1.8, color=c, label=f"{nm} ({v[j_]*100:.0f}%)")
a.axhline(0, color="#9a9893", lw=.7); a.set_xticks(range(9)); a.set_xticklabels(["28", "91", "182", "364", "3", "5", "10", "20", "30"], fontsize=8.5)
a.axvline(3.5, color="black", lw=1.6, ls=(0, (2, 2)))
a.text(3.5, .74, "◄ CETES (days)   |   BONOS M (years) ►", ha="center", va="bottom", fontsize=8.5, color="black", fontweight="bold")
a.set_ylim(-.62, .72); a.set_ylabel("Weight of each maturity")
a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.42, -.12), ncol=3, columnspacing=.8, handlelength=1.2, handletextpad=.4)
fig.tight_layout(); fig.savefig(f"{C.OUT}/fig2_movements{SUF}.png", dpi=220, transparent=TRANSP); plt.close()

# F3. Results grid: slope model vs "the rate does not change", by maturity and horizon
ratio = pd.read_csv(f"{C.OUT}/t3_rmse_ratio_vs_RW.csv").set_index(["h", "model"])
dm = pd.read_csv(f"{C.OUT}/t3_dm_hln_vs_RW.csv").set_index(["h", "model", "maturity"])
labs = ["28 d", "91 d", "182 d", "364 d", "3 y", "5 y", "10 y", "20 y", "30 y"]
fig, a = plt.subplots(figsize=(5.3, 1.75))
a.set_xlim(-2.1, 9.45); a.set_ylim(-.15, 2.75); a.axis("off")
GREEN, PINK, GREY = "#1f7a45", "#f3d9d3", "#e4e7ec"
for r_, h in enumerate([3, 12]):
    y = 1 - r_
    a.text(-2.05, y + .42, "{h}-month\nforecast".format(h=h), fontsize=8.5, color=INK, va="center", fontweight="bold")
    for j, c in enumerate(COLS):
        x = j + (.35 if j > 3 else 0)
        val = ratio.loc[(h, "Slope")][c]; p = dm.loc[(h, "Slope", c)].p
        fc, tc = (GREEN, "white") if (p < .05 and val < 1) else ((PINK, "#b23b3b") if (p < .05 and val > 1) else (GREY, "#3d4452"))
        a.add_patch(plt.Rectangle((x + .04, y + .06), .92, .78, fc=fc, ec="none"))
        a.text(x + .5, y + .45, f"{(val-1)*100:+.0f}%".replace("-", "−"), ha="center", va="center", fontsize=9, color=tc, fontweight="bold")
for j, l_ in enumerate(labs):
    x = j + (.35 if j > 3 else 0); a.text(x + .5, 2.05, l_, ha="center", fontsize=8.5, color=INK2)
a.text(2.0, 2.5, "CETES", ha="center", fontsize=9.5, color=B, fontweight="bold"); a.plot([0.05, 3.95], [2.36, 2.36], color=B, lw=1.5)
a.text(6.85, 2.5, "BONOS M", ha="center", fontsize=9.5, color=INK2, fontweight="bold"); a.plot([4.4, 9.3], [2.36, 2.36], color="#9a9893", lw=1.5)
fig.tight_layout(pad=.2); fig.savefig(f"{C.OUT}/fig3_error_grid{SUF}.png", dpi=220, transparent=TRANSP); plt.close()

# F4. Forecast vs observed, 91-day CETE at 3 months
R = pd.read_csv(f"{C.OUT}/t2_forecasts_long.csv", parse_dates=["origin", "target"])
s = R[(R.h == 3) & (R.maturity == "3m")]
rw = s[s.model == "RW"].set_index("target"); pe = s[s.model == "Slope"].set_index("target")
import fig_shading as FS
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(7.6, 2.5))
ax.plot(rw.index, rw.obs, color=INK, lw=1.8, label="Observed rate")
ax.plot(rw.index, rw.forecast, color="#9a9893", lw=1.3, ls=(0, (3, 2)), label="Assume no change")
ax.plot(pe.index, pe.forecast, color=B, lw=1.5, label="Slope model")
ax.set_xlim(pd.Timestamp("2018-06"), pd.Timestamp("2026-10")); ax.set_ylim(1.8, 15.2); ax.set_ylabel("Annual rate (%)")
FS.shade(ax, rw, pe, fs=7.5)   # green: slope model errs less than "no change" (3-month rolling); red: errs more
hs, ls_ = ax.get_legend_handles_labels()
hs += [Patch(color=(0.12, 0.48, 0.27, 0.35)), Patch(color=(0.70, 0.23, 0.23, 0.3))]; ls_ += ["Model errs less", "Model errs more"]
ax.legend(hs, ls_, fontsize=8, loc="upper left", ncol=1, bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); fig.savefig(f"{C.OUT}/fig4_cete91_forecast{SUF}.png", dpi=220, transparent=TRANSP); plt.close()

# F5. External data and event treatment: error relative to the slope model (horizontal bars)
T4 = pd.read_csv(f"{C.OUT}/t4_external_data.csv"); T5 = pd.read_csv(f"{C.OUT}/t5_events.csv")
T6 = pd.read_csv(f"{C.OUT}/t6_surveys.csv")    # written by surveys.py (run it before this script)
T7 = pd.read_csv(f"{C.OUT}/t7_market.csv"); T7 = T7[T7.period == "all"]   # written by extra_predictors.py
items = [("Ext:Policy", T4, "Gap to Banxico rate"), ("Ext:Slope+RealRate", T4, "Banxico real rate"),
         ("Ext:Slope+SpreadUST", T4, "Gap to US bond"), ("Ext:Slope+ΔUST", T4, "Change in US bond"),
         ("Slope+Cycle", T5, "Banxico cycle"), ("Slope excl. episodes", T5, "Exclude crises and pandemic"),
         ("Slope+PolicySurvey", T6, "Survey: Banxico rate (MX)"), ("Slope+Bono10Survey", T6, "Survey: 10y Bono M (MX)"),
         ("Slope+SPF", T6, "Survey: 10y Treasury (US)"),
         ("Slope+ΔFX3", T7, "Peso per dollar, 3-month change"), ("Slope+ΔEMBIG", T7, "EMBIG spread, 3-month change"),
         ("Slope+ΔForeign", T7, "Foreign holdings of BONOS M, growth")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 5.6), sharey=True)
yy = np.arange(len(items))[::-1].astype(float); yy[4:] -= .5; yy[6:] -= .5; yy[9:] -= .5
for a, h in zip(axs, [3, 12]):
    hh = .44
    for k, (tr, c, lab) in enumerate([("CETES", B, "CETES (Mexico)"), ("BONOS", O, "BONOS M (Mexico)")]):
        vals, ps = [], []
        for m, T, _ in items:
            r = T[(T.h == h) & (T.model == m) & (T.segment == tr)]
            if "excl_pandemic" in r: r = r[r.excl_pandemic == False]
            col_p = "p" if "p" in r else "p_vs_Slope"
            vals.append(r["RMSE_vs_Slope"].iloc[0]); ps.append(r[col_p].iloc[0])
        yk = yy + (.5 - k) * hh
        a.barh(yk, np.array(vals) - 1, hh * .9, left=1, color=c, label=lab)
        for y_, v_, p_ in zip(yk, vals, ps):
            if p_ < .05: a.text(v_ + (.008 if v_ > 1 else -.008), y_, "•", ha="left" if v_ > 1 else "right", va="center", fontsize=12, color=INK)
    a.axvline(1, color=INK, lw=1.1); a.set_xlim(.82, 1.26); a.grid(axis="y", visible=False)
    a.set_title("{h}-month forecast".format(h=h), fontsize=11, color=INK)
    for k_ in (3, 5, 8): a.axhline(yy[k_] - .75, color="#9a9893", lw=.7, ls=":")
    a.tick_params(axis="x", labelsize=9.5)
axs[0].set_yticks(yy); axs[0].set_yticklabels([t for *_, t in items], fontsize=10)
for a in axs: a.set_xlabel("← better   |   worse →", fontsize=10)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=10, loc="lower center", ncol=2, bbox_to_anchor=(.62, 0))
fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(f"{C.OUT}/fig5_external_events{SUF}.png", dpi=220, transparent=TRANSP); plt.close()

# --------------------------------------------------------------------------------------
# Slide figures (wide, own style): Banxico policy rate vs its full-sample average; 91-day CETE forecast.
# --------------------------------------------------------------------------------------
plt.rcdefaults()
INK, INK2, GRID = "#000000", "#000000", "#e4e3df"
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": "#000000", "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK2, "text.color": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "legend.frameon": False})
r = EXT.policy_rate.dropna(); r = r[r.index >= "2006-10"]
m = r.mean()
fig, ax = plt.subplots(figsize=(8.6, 3.9))
ax.step(r.index, r, where="post", color=B, lw=2.2, label="Banxico policy rate")
ax.axhline(m, color=O, lw=2, ls=(0, (4, 3)), label=f"Full-sample average ({m:.1f}%)")
ax.fill_between(r.index, r, m, step="post", color=B, alpha=.08)
for x, y, t in [("2011-01", 1.0, "Years below\nthe average"), ("2023-06", 12.4, "Years above\nthe average")]:
    ax.text(pd.Timestamp(x), y, t, ha="center", fontsize=11, color=INK)
ax.set_ylim(0, 13.5); ax.set_ylabel("Annual rate (%)"); ax.legend(loc="upper left", ncol=2, fontsize=11)
fig.tight_layout(); fig.savefig(f"{C.OUT}/slide_policy_rate_vs_average.png", dpi=200); plt.close()
print("Average policy rate:", round(m, 2), "| sample:", r.index.min(), "to", r.index.max())
s = R[(R.h == 3) & (R.maturity == "3m")]
rw = s[s.model == "RW"].set_index("target"); pe = s[s.model == "Slope"].set_index("target")
fig, ax = plt.subplots(figsize=(8.6, 3.9))
ax.plot(rw.index, rw.obs, color=INK, lw=2.4, label="Observed rate")
ax.plot(rw.index, rw.forecast, color="#9a9893", lw=1.8, ls=(0, (3, 2)), label="Assume no change")
ax.plot(pe.index, pe.forecast, color=B, lw=2.0, label="Slope model")
ax.set_xlim(pd.Timestamp("2018-06"), pd.Timestamp("2026-10")); ax.set_ylim(1.8, 13.2); ax.set_ylabel("Annual rate (%)")
kw = dict(fontsize=11, color=INK, va="center", arrowprops=dict(arrowstyle="-|>", color="#6d6b66", lw=1), bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#c9c7c1"))
ax.annotate("2020 cuts: the model\nanticipates the fall", xy=(pd.Timestamp("2020-05"), 5.4), xytext=(pd.Timestamp("2018-08"), 2.9), **kw)
ax.annotate("2021-22 hikes: the same\nin reverse", xy=(pd.Timestamp("2022-02"), 6.4), xytext=(pd.Timestamp("2022-09"), 3.4), **kw)
ax.legend(loc="upper left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{C.OUT}/slide_cete91_forecast.png", dpi=200); plt.close()
print("ok")
