# GenAI declaration: written with Claude (Anthropic) for the SEH500 project.
"""Train the 7-16-8-1 heat-island network, quantize it to Q4.11 int16 and export it for the K64F.

Usage: python train_export.py <folder with ECCC h_<station>_<year>_<month>.csv files> <output folder>

Outputs (in <output folder>):
  weights.bin       int16 little-endian: header [3, 7, 16, 8, 1], then W1 b1 W2 b2 W3 b3.
                    Each W has shape (out, in), row-major. 5 + 273 values = 556 bytes.
  model.h           Same numbers as C arrays, plus the scaling constants and time-feature tables.
  scaling.json      Integer constants for turning raw serial values into network inputs and back.
  test_vectors.csv  Sample hours with every intermediate value. The board must match these exactly.

The board's whole pipeline is integer-only and is defined by board_pipeline() below.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

DATA, OUT = sys.argv[1], sys.argv[2]
COLS = {"Date/Time (LST)": "time", "Temp (°C)": "temp", "Rel Hum (%)": "rh", "Wind Spd (km/h)": "wind"}
FRAC = 11                      # Q4.11: real value = int / 2048
ONE = 1 << FRAC
S = 16                         # shift used by the input/output scaling multipliers
ZMAX = 4 * ONE                 # inputs clipped to +-4 standard deviations
I16_LO, I16_HI = -32768, 32767


# ---------- 1. Build the table ----------
def load(station_id):
    files = sorted(glob.glob(os.path.join(DATA, f"h_{station_id}_*.csv")))
    df = pd.concat(pd.read_csv(f, encoding="utf-8-sig", usecols=list(COLS)) for f in files)
    df = df.rename(columns=COLS)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


df = load(51459).join(load(31688), lsuffix="_p", rsuffix="_d", how="inner")
df = df.dropna(subset=["temp_p", "temp_d", "rh_p", "wind_p"])
# Raw values exactly as they will travel over serial: integer tenths, hour 0-23, day of year 1-366
temp10 = np.round(df["temp_p"].values * 10).astype(np.int64)
rh10 = np.round(df["rh_p"].values * 10).astype(np.int64)
wind10 = np.round(df["wind_p"].values * 10).astype(np.int64)
hour = df.index.hour.values
doy = df.index.dayofyear.values
yr = df.index.year.values
gap10_true = np.round((df["temp_d"].values - df["temp_p"].values) * 10).astype(np.int64)


def time_features(h, d):
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365)])


X = np.column_stack([temp10 / 10, rh10 / 10, wind10 / 10, time_features(hour, doy)])
y = gap10_true / 10

# ---------- 2. Split by year ----------
tr, va, te = yr <= 2022, yr == 2023, yr >= 2024

# ---------- 3. Standardize with training statistics only ----------
mu, sd = X[tr].mean(0), X[tr].std(0)
ymu, ysd = y[tr].mean(), y[tr].std()
Xs = np.clip((X - mu) / sd, -4, 4)
ys = (y - ymu) / ysd

# ---------- 4. Train ----------
m = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", early_stopping=True,
                 max_iter=300, random_state=0).fit(Xs[tr], ys[tr])
print(f"trained for {m.n_iter_} epochs")

# ---------- 5. Quantize ----------
def q16(v):
    return np.clip(np.round(np.asarray(v) * ONE), I16_LO, I16_HI).astype(np.int64)


Wq = [q16(W.T) for W in m.coefs_]          # (out, in) like the board reads them
bq = [q16(b) for b in m.intercepts_]
sat_w = sum(int((np.abs(np.asarray(W.T) * ONE) > I16_HI).sum()) for W in m.coefs_)
print(f"weights saturated by int16: {sat_w}; largest |weight| = {max(np.abs(W).max() for W in m.coefs_):.3f}")

# Integer scaling constants
M10 = np.round(mu[:3] * 10).astype(np.int64)                 # means in tenths
K = np.round(ONE * (1 << S) / (sd[:3] * 10)).astype(np.int64)  # z_q11 = ((x10 - M10) * K) >> S
hour_tab = np.clip(np.round((time_features(np.arange(24), np.zeros(24))[:, :2] - mu[3:5]) / sd[3:5] * ONE),
                   -ZMAX, ZMAX).astype(np.int64)             # [hour] -> (sin, cos) inputs
d = np.arange(1, 367)
doy_tab = np.clip(np.round((time_features(np.zeros(366), d)[:, 2:] - mu[5:7]) / sd[5:7] * ONE),
                  -ZMAX, ZMAX).astype(np.int64)              # [doy - 1] -> (sin, cos) inputs
KO = int(round(ysd * 10 * (1 << S) / ONE))                   # gap10 = ((y_q11 * KO) >> S) + MO10
MO10 = int(round(ymu * 10))


def board_pipeline(t10, r10, w10, h, dy):
    """Exactly what the board computes, with int64 here standing in for 32-bit registers."""
    raw = np.column_stack([t10, r10, w10])
    x = np.clip(((raw - M10) * K) >> S, -ZMAX, ZMAX)
    x = np.column_stack([x, hour_tab[h], doy_tab[dy - 1]])
    acts, accs = [x], []
    a = x
    for i, (W, b) in enumerate(zip(Wq, bq)):
        acc = a @ W.T + (b << FRAC)            # 32-bit multiply-accumulate, products are Q8.22
        accs.append(acc)
        a = np.clip(acc >> FRAC, I16_LO, I16_HI)  # asr #11, then saturate to int16
        if i < len(Wq) - 1:
            a = np.maximum(a, 0)               # ReLU
        acts.append(a)
    y_q = a[:, 0]
    gap10 = ((y_q * KO) >> S) + MO10
    return acts, accs, gap10


# ---------- 6. Check ----------
def mae10(pred10, mask):
    return np.abs(pred10 - gap10_true[mask]).mean() / 10


acts, accs, gap10_int = board_pipeline(temp10, rh10, wind10, hour, doy)
pred_float = m.predict(Xs) * ysd + ymu
hourly = pd.Series(y[tr]).groupby(hour[tr]).mean()

print("\nMAE in °C           val 2023   test 2024-25")
print(f"no heat island      {np.abs(y[va]).mean():8.3f}   {np.abs(y[te]).mean():8.3f}")
print(f"hourly average      {np.abs(hourly.reindex(hour[va]).values - y[va]).mean():8.3f}   "
      f"{np.abs(hourly.reindex(hour[te]).values - y[te]).mean():8.3f}")
print(f"network, float      {np.abs(pred_float[va] - y[va]).mean():8.3f}   {np.abs(pred_float[te] - y[te]).mean():8.3f}")
print(f"network, int16      {mae10(gap10_int[va], va):8.3f}   {mae10(gap10_int[te], te):8.3f}")
diff = np.abs(gap10_int / 10 - pred_float)
print(f"\nint16 vs float on all hours: mean diff {diff.mean():.4f} °C, max diff {diff.max():.4f} °C")

worst = [int((np.abs(W).sum(1) * ZMAX + np.abs(b) * ONE).max()) for W, b in zip(Wq, bq)]
seen = [int(np.abs(acc).max()) for acc in accs]
print(f"accumulator |max| seen per layer: {seen}  (int32 limit {2**31 - 1})")
print(f"input-scaling product max: {int(np.abs((np.column_stack([temp10, rh10, wind10]) - M10) * K).max())}")
assert max(seen) < 2**31, "32-bit accumulator would overflow"
clipped = [int(((a == I16_HI) | (a == I16_LO)).sum()) for a in acts[1:]]
print(f"activations hitting int16 limits per layer: {clipped}")

# ---------- 7. Export ----------
os.makedirs(OUT, exist_ok=True)
header = np.array([3, 7, 16, 8, 1], dtype="<i2")
blob = np.concatenate([v.ravel() for pair in zip(Wq, bq) for v in pair]).astype("<i2")
with open(os.path.join(OUT, "weights.bin"), "wb") as f:
    f.write(header.tobytes() + blob.tobytes())

scaling = {
    "format": "Q4.11 int16 (real = int / 2048)",
    "raw_inputs": "temp, humidity, wind sent as integer tenths; hour 0-23 (LST); day of year 1-366",
    "input_rule": "x_q11 = clamp(((raw10 - M10) * K) >> 16, -8192, 8192)",
    "M10": {"temp": int(M10[0]), "rh": int(M10[1]), "wind": int(M10[2])},
    "K": {"temp": int(K[0]), "rh": int(K[1]), "wind": int(K[2])},
    "S": S,
    "time_inputs": "x3,x4 = hour_tab[hour]; x5,x6 = doy_tab[doy - 1] (see model.h)",
    "output_rule": "gap10 = ((y_q11 * KO) >> 16) + MO10   (expected downtown - Pearson, tenths of °C)",
    "KO": KO, "MO10": MO10,
    "float_stats": {"mu": mu.tolist(), "sd": sd.tolist(), "y_mu": ymu, "y_sd": ysd},
}
json.dump(scaling, open(os.path.join(OUT, "scaling.json"), "w", encoding="utf-8"), indent=1)


def c_array(name, arr):
    vals = ", ".join(str(int(v)) for v in np.asarray(arr).ravel())
    return f"static const int16_t {name}[{np.asarray(arr).size}] = {{{vals}}};\n"


with open(os.path.join(OUT, "model.h"), "w", encoding="utf-8", newline="\n") as f:
    f.write("/* Generated by train_export.py. Q4.11 int16. Weight matrices are (out, in) row-major. */\n")
    f.write("#pragma once\n#include <stdint.h>\n\n")
    for i, (W, b) in enumerate(zip(Wq, bq), 1):
        f.write(f"/* W{i}: {W.shape[0]} x {W.shape[1]} */\n" + c_array(f"W{i}", W) + c_array(f"B{i}", b))
    f.write("\n/* x_q11 = clamp(((raw10 - M10) * K) >> S, -8192, 8192); order: temp, rh, wind */\n")
    f.write(c_array("IN_M10", M10).replace("int16_t", "int32_t") + c_array("IN_K", K).replace("int16_t", "int32_t"))
    f.write(f"#define IN_S {S}\n#define OUT_KO {KO}\n#define OUT_MO10 {MO10}\n")
    f.write("/* hour_tab[hour*2 + 0/1] = sin/cos input; doy_tab[(doy-1)*2 + 0/1] likewise */\n")
    f.write(c_array("HOUR_TAB", hour_tab) + c_array("DOY_TAB", doy_tab))

# Test vectors: extremes plus random test-set hours
rng = np.random.default_rng(0)
idx_te = np.flatnonzero(te)
pick = list(rng.choice(idx_te, 14, replace=False))
for arr in (temp10, rh10, wind10, gap10_int):
    pick += [idx_te[arr[te].argmin()], idx_te[arr[te].argmax()]]
pick = sorted(set(int(i) for i in pick))
rows = {"time": df.index[pick].strftime("%Y-%m-%d %H:%M"), "temp10": temp10[pick], "rh10": rh10[pick],
        "wind10": wind10[pick], "hour": hour[pick], "doy": doy[pick]}
for name, a in zip(["x", "h1", "h2", "y"], acts):
    for j in range(a.shape[1]):
        rows[f"{name}_{j}" if a.shape[1] > 1 else name] = a[pick, j]
rows["gap10"] = gap10_int[pick]
rows["true_gap10"] = gap10_true[pick]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "test_vectors.csv"), index=False)

print(f"\nwrote weights.bin ({len(header) * 2 + blob.nbytes} bytes), model.h, scaling.json, "
      f"test_vectors.csv ({len(pick)} rows) to {OUT}")
