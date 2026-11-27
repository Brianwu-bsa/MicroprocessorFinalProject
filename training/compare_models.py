# GenAI declaration: written with Claude (Anthropic) for the SEH500 project.
"""Compare network sizes and int8 / int16 fixed-point quantization on the ECCC data.

Usage: python compare_models.py <folder with h_<station>_<year>_<month>.csv files>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

DATA = sys.argv[1]
COLS = {"Date/Time (LST)": "time", "Temp (°C)": "temp", "Rel Hum (%)": "rh", "Wind Spd (km/h)": "wind"}


def load(station_id):
    files = sorted(glob.glob(os.path.join(DATA, f"h_{station_id}_*.csv")))
    df = pd.concat(pd.read_csv(f, encoding="utf-8-sig", usecols=list(COLS)) for f in files)
    df = df.rename(columns=COLS)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


df = load(51459).join(load(31688), lsuffix="_p", rsuffix="_d", how="inner")
df["dT"] = df["temp_d"] - df["temp_p"]
df = df.dropna(subset=["temp_p", "temp_d", "rh_p", "wind_p"])
hour, doy, yr = df.index.hour.values, df.index.dayofyear.values, df.index.year.values
X = np.column_stack([df["temp_p"], df["rh_p"], df["wind_p"],
                     np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                     np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)])
y = df["dT"].values
fit, te = yr <= 2023, yr >= 2024
mu, sd = X[yr <= 2022].mean(0), X[yr <= 2022].std(0)
Xs = np.clip((X - mu) / sd, -4, 4)
ymu, ysd = y[yr <= 2022].mean(), y[yr <= 2022].std()


def forward_float(m, x):
    for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
        x = x @ W + b
        if i < len(m.coefs_) - 1:
            x = np.maximum(x, 0)
    return x.ravel()


def forward_fixed(m, x, bits, frac):
    """Integer-only inference: Qm.frac values, 32-bit accumulator, shift, saturate."""
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q = lambda v: np.clip(np.round(v * 2 ** frac), lo, hi).astype(np.int64)
    a = q(x)
    for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
        acc = a @ q(W) + (q(b) << frac)          # products are Q(2*frac)
        a = np.clip(acc >> frac, lo, hi)          # back to Q(frac), saturate
        if i < len(m.coefs_) - 1:
            a = np.maximum(a, 0)                  # ReLU
    return a.ravel() / 2 ** frac


def mae(z):
    return np.abs(z * ysd + ymu - y[te]).mean()


print(f"{'layers':<14}{'params':>7}{'float':>8}{'int16 Q4.11':>13}{'int8 Q3.4':>11}{'int8 Q2.5':>11}")
for layers in [(8,), (16,), (8, 8), (16, 8), (32, 16)]:
    m = MLPRegressor(hidden_layer_sizes=layers, activation="relu", early_stopping=True,
                     max_iter=300, random_state=0).fit(Xs[fit], (y[fit] - ymu) / ysd)
    params = sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)
    print(f"7-{'-'.join(map(str, layers))}-1".ljust(14) + f"{params:>7}"
          f"{mae(forward_float(m, Xs[te])):>8.3f}"
          f"{mae(forward_fixed(m, Xs[te], 16, 11)):>13.3f}"
          f"{mae(forward_fixed(m, Xs[te], 8, 4)):>11.3f}"
          f"{mae(forward_fixed(m, Xs[te], 8, 5)):>11.3f}")
