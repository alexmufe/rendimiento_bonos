"""Do survey expectations improve the forecasts? Variants of the slope model that add survey data.

Run after yield_curve.py (it imports it):  python surveys.py
Every survey value is used only once it was public (see survey_data.py). Survey models are estimated with the
same direct regressions as the rest, using only pairs (s, s+h) with s+h <= t, and they start once at least
24 such pairs exist; they are compared with the slope model and with "the rate does not change" on exactly
the same forecast dates.

Variants
  Slope+PolicySurvey   slope + expected change in Banxico's rate (Banxico survey)            all maturities
  PolicySurvey         no estimation: every rate moves by the expected change in Banxico's rate
  Slope+Bono10Survey   slope + expected change in the 10-year Bono M (Banxico survey)         all maturities
  Bono10Survey         no estimation: every rate moves by the expected change in the 10-year Bono M
  Slope+SPF            slope + expected change in the 10-year US Treasury (Philadelphia Fed SPF)
"""
import io, contextlib, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    import yield_curve as C
import survey_data as SD

Y, raw, EXT, COLS, OUT, OBS = C.Y, C.raw, C.EXT, C.COLS, C.OUT, C.OBS
H, MIN_PAIRS = C.H, 24

# Expected changes known at the end of each month t, for each horizon h
EXP = {}
for h in H:
    pol_e = SD.banxico_policy_expected(Y.index, EXT.policy_rate, h)
    b10_e = SD.banxico_bono10_expected(Y.index, Y["10y"], h)
    spf_e = SD.spf_tbond_expected(Y.index, EXT.ust10, h)
    EXP[h] = pd.DataFrame({"policy": pol_e - EXT.policy_rate, "bono10": b10_e - Y["10y"], "spf": spf_e - EXT.ust10})


def _fit_predict(Yw, h, xcols):
    """Direct regression per maturity: y_{j,s+h} - y_{j,s} = a_j + b_j' x_s, using only rows with survey data."""
    X = pd.concat([(Yw["12m"] - Yw["1m"]).rename("slope")] + [EXP[h].loc[Yw.index, c] for c in xcols], axis=1)
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


def _shift(Yw, h, col):
    e = EXP[h].loc[Yw.index[-1], col]
    return None if np.isnan(e) else Yw.values[-1] + e


MODELS = {"Slope+PolicySurvey": lambda Yw, h: _fit_predict(Yw, h, ["policy"]),
          "PolicySurvey": lambda Yw, h: _shift(Yw, h, "policy"),
          "Slope+Bono10Survey": lambda Yw, h: _fit_predict(Yw, h, ["bono10"]),
          "Bono10Survey": lambda Yw, h: _shift(Yw, h, "bono10"),
          "Slope+SPF": lambda Yw, h: _fit_predict(Yw, h, ["spf"])}

recs = []
for t in Y.index[Y.index >= C.FIRST_ORIGIN]:
    Yw = Y.loc[:t]
    for h in H:
        tgt = t + pd.DateOffset(months=h)
        if tgt > Y.index[-1]: continue
        for nm, f in MODELS.items():
            p = f(Yw, h)
            if p is None: continue
            for j, c in enumerate(COLS):
                if OBS.loc[tgt, c]: recs.append((t, tgt, h, nm, c, p[j], raw.loc[tgt, c]))
RS = pd.DataFrame(recs, columns=["origin", "target", "h", "model", "maturity", "forecast", "obs"])
RS["err"] = RS.obs - RS.forecast
RS.to_csv(f"{OUT}/t6_survey_forecasts.csv", index=False)

# Compare with the slope model and with no change on the same forecast dates
base = C.R[C.R.model.isin(["Slope", "RW"])]
rows = []
for h in H:
    for m in MODELS:
        a = RS[(RS.model == m) & (RS.h == h)]
        for seg, pls in [("CETES", COLS[:4]), ("BONOS", COLS[4:]), ("10y", ["10y"]), ("Curve", COLS)]:
            aa = a[a.maturity.isin(pls)]
            row = {"h": h, "model": m, "segment": seg, "n_origins": aa.origin.nunique(),
                   "first_origin": aa.origin.min()}
            for b in ["Slope", "RW"]:
                bb = base[(base.model == b) & (base.h == h)]
                mm = aa.merge(bb, on=["origin", "target", "maturity"], suffixes=("_1", "_2"))
                s, p = C.dm_hln(mm.err_1.values, mm.err_2.values, h)
                row[f"RMSE_vs_{b}"] = np.sqrt((mm.err_1**2).mean() / (mm.err_2**2).mean()); row[f"p_vs_{b}"] = p
            rows.append(row)
T6 = pd.DataFrame(rows); T6.round(3).to_csv(f"{OUT}/t6_surveys.csv", index=False)
if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(T6.round(3).to_string(index=False))
