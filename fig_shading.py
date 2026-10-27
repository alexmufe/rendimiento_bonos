"""Shaded windows where the slope model beats (green) or loses to (red) the no-change forecast, 91-day CETE at 3 months."""
import numpy as np, pandas as pd

LABELS = {  # (start, end, kind, English, Spanish)
    "2019-02": ("Rate held\nat 8.25%", "Tasa fija\nen 8.25%"),
    "2019-08": ("Cuts\n2019-20", "Recortes\n2019-20"),
    "2021-07": ("Hikes\n2021-23", "Alzas\n2021-23"),
    "2023-05": ("Rate held at 11.25%,\nthen slow cuts", "Tasa fija en 11.25%,\nluego recortes lentos"),
    "2024-10": ("Cuts\n2024-25", "Recortes\n2024-25"),
    "2026-03": ("Slow\ncuts", "Recortes\nlentos"),
}

def windows(rw, pe, start="2018-06", min_len=3):
    d = (rw.err.abs() - pe.err.abs()).rolling(3, center=True, min_periods=1).mean()
    d = d[d.index >= start]; sg = np.sign(d); runs = (sg != sg.shift()).cumsum(); out = []
    for _, g in d.groupby(runs):
        if len(g) >= min_len: out.append((g.index[0], g.index[-1], "g" if g.mean() > 0 else "r"))
    return out

def shade(ax, rw, pe, es=False, green=(0.12, 0.48, 0.27, 0.16), red=(0.70, 0.23, 0.23, 0.14), fs=8, ytop=None):
    ytop = ytop if ytop is not None else ax.get_ylim()[1]
    for a, b, k in windows(rw, pe):
        ax.axvspan(a - pd.DateOffset(days=15), b + pd.DateOffset(days=15), color=green if k == "g" else red, lw=0, zorder=0)
        key = a.strftime("%Y-%m")
        if key in LABELS:
            ax.text(a + (b - a) / 2, ytop * .985, LABELS[key][1 if es else 0], ha="center", va="top", fontsize=fs,
                    color="#1F7A45" if k == "g" else "#B23B3B", fontweight="bold", linespacing=1.1)
