# GenAI declaration: written with Claude (Anthropic) for the SEH500 project.
"""Explore ECCC hourly data: Pearson (51459) vs Toronto City downtown (31688)."""
import glob
import os
import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1]
COLS = {"Date/Time (LST)": "time", "Temp (°C)": "temp", "Rel Hum (%)": "rh", "Wind Spd (km/h)": "wind"}


def load(station_id):
    files = sorted(glob.glob(os.path.join(DATA, f"h_{station_id}_*.csv")))
    df = pd.concat(pd.read_csv(f, encoding="utf-8-sig", usecols=list(COLS)) for f in files)
    df = df.rename(columns=COLS)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


pearson = load(51459)
downtown = load(31688)

print("=== Rows and missing values (% of hours) ===")
for name, df in [("Pearson", pearson), ("Downtown", downtown)]:
    miss = (df.isna().mean() * 100).round(1).to_dict()
    print(f"{name}: {len(df)} rows, {df.index.min()} -> {df.index.max()}, missing % {miss}")
print("Downtown missing % by year (temp):")
print((downtown["temp"].isna().groupby(downtown.index.year).mean() * 100).round(1).to_dict())

df = pearson.join(downtown, lsuffix="_p", rsuffix="_d", how="inner")
df["dT"] = df["temp_d"] - df["temp_p"]
df = df.dropna(subset=["temp_p", "temp_d", "rh_p", "wind_p"])
print(f"\nUsable joined hours (both temps + Pearson RH + wind): {len(df)}")

print("\n=== Gap dT = downtown - Pearson (°C) ===")
print(df["dT"].describe().round(2).to_dict())
print("Mean dT by hour of day:")
print(df.groupby(df.index.hour)["dT"].mean().round(2).to_dict())
season = df.index.month.map(lambda m: {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                                         6: "summer", 7: "summer", 8: "summer"}.get(m, "fall"))
print("Mean dT by season:", df.groupby(season)["dT"].mean().round(2).to_dict())
night = (df.index.hour >= 22) | (df.index.hour <= 4)
calm = df["wind_p"] <= 5
windy = df["wind_p"] >= 25
print("Night, calm (wind<=5):", round(df.loc[night & calm, "dT"].mean(), 2), "n=", int((night & calm).sum()))
print("Night, windy (wind>=25):", round(df.loc[night & windy, "dT"].mean(), 2), "n=", int((night & windy).sum()))
print("Summer nights, calm:", round(df.loc[night & calm & (season == "summer"), "dT"].mean(), 2))

# Features as in the proposal doc
hour = df.index.hour.values
doy = df.index.dayofyear.values
X = np.column_stack([df["temp_p"], df["rh_p"], df["wind_p"],
                     np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                     np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)])
y = df["dT"].values
yr = df.index.year.values
tr, va, te = yr <= 2022, yr == 2023, yr >= 2024
print(f"\nSplit sizes: train {tr.sum()}, val {va.sum()}, test {te.sum()}")

mu, sd = X[tr].mean(0), X[tr].std(0)
Xs = (X - mu) / sd
ymu, ysd = y[tr].mean(), y[tr].std()


def mae(pred):
    return round(float(np.abs(pred - y[te]).mean()), 3)


print("\n=== Test MAE on 2024-2025 (°C) ===")
print("1. No heat island (dT=0):", mae(np.zeros(te.sum())))
hourly = pd.Series(y[tr]).groupby(hour[tr]).mean()
print("2. Hourly average dT:", mae(hourly.reindex(hour[te]).values))
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

lin = LinearRegression().fit(Xs[tr], y[tr])
print("3. Linear regression:", mae(lin.predict(Xs[te])))
mlp = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", early_stopping=True,
                   max_iter=300, random_state=0)
mlp.fit(Xs[tr | va], (y[tr | va] - ymu) / ysd)
pred = mlp.predict(Xs[te]) * ysd + ymu
print("4. Neural net 7-16-8-1:", mae(pred))
print("   params:", sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))
