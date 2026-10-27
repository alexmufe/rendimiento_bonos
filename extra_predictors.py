"""Do market variables used in the emerging-market literature improve the forecasts? Variants of the slope model.

Run after yield_curve.py (it imports it):  python extra_predictors.py
Same design as surveys.py: direct regressions per maturity, expanding window, only pairs (s, s+h) with s+h <= t,
at least 24 pairs, compared with the slope model and with "the rate does not change" on the same forecast dates.

Variants (x_t is known at the end of month t)
  Slope+EMBIG          slope + EMBIG Mexico spread (level, basis points / 100)                   sovereign risk
  Slope+ΔEMBIG         slope + 3-month change in the EMBIG spread                               change in risk
  Slope+ΔFX1           slope + 1-month log change of the peso per dollar (monthly average)      currency pressure
  Slope+ΔFX3           slope + 3-month log change of the peso per dollar
  Slope+ΔForeign       slope + 3-month log change of foreign holdings of BONOS M                 foreign flows
  Slope+LongSlope      slope + long slope (10-year Bono M minus 12-month CETE)                  term premium
  CP-type              one factor in the spirit of Cochrane and Piazzesi (2005): the average change of the nine
                       rates over h regressed on five rates (1m, 12m, 3y, 10y, 30y), estimated inside each window;
                       each maturity is then regressed on that single factor
"""
import io, contextlib, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    import yield_curve as C
import market_data as MD

Y, raw, COLS, OUT, OBS = C.Y, C.raw, C.COLS, C.OUT, C.OBS
H, MIN_PAIRS = C.H, 24

MK = MD.load(pd.date_range("2006-01-01", Y.index[-1], freq="MS"))
VAR = pd.DataFrame({
    "embig": MK.embig / 100,
    "d_embig": MK.embig.diff(3) / 100,
    "d_fx1": 100 * np.log(MK.fx).diff(1),
    "d_fx3": 100 * np.log(MK.fx).diff(3),
    "d_foreign": 100 * np.log(MK.foreign_bonos).diff(3),
}).reindex(Y.index)
VAR["long_slope"] = Y["10y"] - Y["12m"]


def _fit_predict(Yw, h, xcols):
    X = pd.concat([(Yw["12m"] - Yw["1m"]).rename("slope")] + [VAR.loc[Yw.index, c] for c in xcols], axis=1)
    yv = Yw.values; T = len(Yw)
    ok = X.iloc[:T - h].notna().all(axis=1).values
    if ok.sum() < MIN_PAIRS or X.iloc[-1].isna().any(): return None
    out = np.empty(9)
    for j in range(9):
        dz = (yv[h:, j] - yv[:-h, j])[ok]
        Xd = np.column_stack([np.ones(ok.sum()), X.values[:T - h][ok]])
        coef, *_ = np.linalg.lstsq(Xd, dz, rcond=None)
        out[j] = yv[-1, j] + coef[0] + X.values[-1] @ coef[1:]
    return out


CP_COLS = ["1m", "12m", "3y", "10y", "30y"]
def _cp(Yw, h):
    yv = Yw.values; L = Yw[CP_COLS].values
    avg = (yv[h:] - yv[:-h]).mean(1)
    Ld = np.column_stack([np.ones(len(avg)), L[:-h]])
    g, *_ = np.linalg.lstsq(Ld, avg, rcond=None)
    f = np.column_stack([np.ones(len(L)), L]) @ g                     # CP-type factor at every date of the window
    out = np.empty(9)
    for j in range(9):
        dz = yv[h:, j] - yv[:-h, j]
        Xd = np.column_stack([np.ones(len(dz)), f[:-h]])
        coef, *_ = np.linalg.lstsq(Xd, dz, rcond=None)
        out[j] = yv[-1, j] + coef[0] + coef[1] * f[-1]
    return out


MODELS = {"Slope+EMBIG": ["embig"], "Slope+ΔEMBIG": ["d_embig"], "Slope+ΔFX1": ["d_fx1"],
          "Slope+ΔFX3": ["d_fx3"], "Slope+ΔForeign": ["d_foreign"], "Slope+LongSlope": ["long_slope"]}
FUN = {k: (lambda v: (lambda Yw, h: _fit_predict(Yw, h, v)))(v) for k, v in MODELS.items()}
FUN["CP-type"] = _cp

recs = []
for t in Y.index[Y.index >= C.FIRST_ORIGIN]:
    Yw = Y.loc[:t]
    for h in H:
        tgt = t + pd.DateOffset(months=h)
        if tgt > Y.index[-1]: continue
        for nm, f in FUN.items():
            p = f(Yw, h)
            if p is None: continue
            for j, c in enumerate(COLS):
                if OBS.loc[tgt, c]: recs.append((t, tgt, h, nm, c, p[j], raw.loc[tgt, c]))
RX = pd.DataFrame(recs, columns=["origin", "target", "h", "model", "maturity", "forecast", "obs"])
RX["err"] = RX.obs - RX.forecast
RX.to_csv(f"{OUT}/t7_market_forecasts.csv", index=False)

base = C.R[C.R.model.isin(["Slope", "RW"])]
SEG = [("CETES", COLS[:4]), ("BONOS", COLS[4:]), ("10y", ["10y"]), ("Curve", COLS)]
PER = [("all", "2000", "2100"), ("2013-19", "2000", "2019-12-31"), ("2020-26", "2020-01-01", "2100")]
rows = []
for h in H:
    for m in FUN:
        a = RX[(RX.model == m) & (RX.h == h)]
        for seg, pls in SEG:
            for per, d0, d1 in PER:
                aa = a[a.maturity.isin(pls) & (a.origin >= d0) & (a.origin <= d1)]
                row = {"h": h, "model": m, "segment": seg, "period": per, "n_origins": aa.origin.nunique()}
                for b in ["Slope", "RW"]:
                    bb = base[(base.model == b) & (base.h == h)]
                    mm = aa.merge(bb, on=["origin", "target", "maturity"], suffixes=("_1", "_2"))
                    s, p = C.dm_hln(mm.err_1.values, mm.err_2.values, h)
                    row[f"RMSE_vs_{b}"] = np.sqrt((mm.err_1**2).mean() / (mm.err_2**2).mean()); row[f"p_vs_{b}"] = p
                    if b == "Slope" and m != "CP-type":   # nested in the slope model: Clark-West
                        z, pcw = C.clark_west(mm.err_2.values, mm.err_1.values, mm.forecast_2.values, mm.forecast_1.values, h)
                        row["p_CW_vs_Slope"] = pcw
                rows.append(row)
T7 = pd.DataFrame(rows); T7.round(3).to_csv(f"{OUT}/t7_market.csv", index=False)
if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(T7[T7.period == "all"].round(3).to_string(index=False))
