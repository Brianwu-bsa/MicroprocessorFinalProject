# GenAI declaration: written with Claude (Anthropic) for the SEH500 project.
"""Charts of the downtown - Pearson temperature gap, and the network diagram.

Usage: python visualize.py <folder with ECCC h_<station>_<year>_<month>.csv files> <output folder>
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

DATA, OUT = sys.argv[1], sys.argv[2]
COLS = {"Date/Time (LST)": "time", "Temp (°C)": "temp", "Rel Hum (%)": "rh", "Wind Spd (km/h)": "wind"}

# Reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, BLUE_LIGHT = "#2a78d6", "#eb6834", "#9ec5f4"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def load(station_id):
    files = sorted(glob.glob(os.path.join(DATA, f"h_{station_id}_*.csv")))
    df = pd.concat(pd.read_csv(f, encoding="utf-8-sig", usecols=list(COLS)) for f in files)
    df = df.rename(columns=COLS)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


df = load(51459).join(load(31688), lsuffix="_p", rsuffix="_d", how="inner")
df["dT"] = df["temp_d"] - df["temp_p"]
df = df.dropna(subset=["temp_p", "temp_d", "rh_p", "wind_p"])
hour, month = df.index.hour, df.index.month

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
fig.suptitle("Downtown is warmer than Pearson at night, by about 1 °C (ECCC hourly data, 2014–2025, "
             f"{len(df):,} hours)", x=0.01, ha="left", fontsize=13, fontweight="bold")

# 1. Gap by hour of day: median line + interquartile band
ax = axes[0, 0]
q = df.groupby(hour)["dT"].quantile([0.25, 0.5, 0.75]).unstack()
ax.fill_between(q.index, q[0.25], q[0.75], color=BLUE_LIGHT, alpha=0.6, linewidth=0, label="Middle 50% of hours")
ax.plot(q.index, q[0.5], color=BLUE, linewidth=2, label="Median")
ax.axhline(0, color=AXIS, linewidth=1)
ax.set(title="Gap by hour of day", xlabel="Hour (local standard time)", ylabel="Downtown − Pearson (°C)",
       xticks=range(0, 24, 3), xlim=(0, 23))
ax.legend(frameon=False, loc="upper center")

# 2. Month x hour heatmap of the mean gap (diverging, gray at 0)
ax = axes[0, 1]
grid = df.groupby([month, hour])["dT"].mean().unstack()
cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", ORANGE])
lim = float(np.abs(grid.values).max())
im = ax.imshow(grid.values, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), origin="upper")
ax.grid(False)
ax.set(title="Mean gap by month and hour", xlabel="Hour", ylabel="Month",
       xticks=range(0, 24, 3), yticks=range(12),
       yticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("°C (orange = downtown warmer)", color=INK2)
cb.outline.set_edgecolor(AXIS)

# 3. Distribution of the gap (the target)
ax = axes[1, 0]
ax.hist(df["dT"].clip(-4, 5), bins=np.arange(-4, 5.01, 0.2), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
ax.axvline(0, color=INK2, linewidth=1)
ax.set(title="What the network predicts: distribution of the gap", xlabel="Downtown − Pearson (°C), clipped to −4…5",
       ylabel="Hours")
ax.text(0.98, 0.95, f"mean {df['dT'].mean():+.2f} °C\nstd {df['dT'].std():.2f} °C", transform=ax.transAxes,
        ha="right", va="top", color=INK2)

# 4. Gap vs Pearson temperature, night vs day
ax = axes[1, 1]
bins = np.arange(-25, 36, 5)
df["tbin"] = pd.cut(df["temp_p"], bins)
night = (hour >= 22) | (hour <= 4)
day = (hour >= 11) & (hour <= 16)
for mask, label, color in [(night, "Night (10 PM–4 AM)", BLUE), (day, "Afternoon (11 AM–4 PM)", ORANGE)]:
    m = df[mask].groupby("tbin", observed=True)["dT"].agg(["mean", "size"])
    m = m[m["size"] >= 200]
    x = [b.mid for b in m.index]
    ax.plot(x, m["mean"], color=color, linewidth=2, marker="o", markersize=6, label=label)
ax.axhline(0, color=AXIS, linewidth=1)
ax.set(title="Gap vs Pearson temperature", xlabel="Pearson temperature (°C, 5° bins)", ylabel="Mean gap (°C)")
ax.legend(frameon=False)

fig.savefig(os.path.join(OUT, "gap_overview.png"), dpi=150)

# Network diagram
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.set_axis_off()
ax.set_xlim(0, 12)
ax.set_ylim(-0.6, 10)
layers = [
    ("Inputs (7)", ["Pearson temp", "Pearson humidity", "Pearson wind", "sin(hour)", "cos(hour)",
                     "sin(day of year)", "cos(day of year)"], 1.6),
    ("Hidden 1 (16)\nReLU", [""] * 16, 4.6),
    ("Hidden 2 (8)\nReLU", [""] * 8, 7.2),
    ("Output (1)\nlinear", ["expected gap\n(°C, scaled)"], 9.8),
]
pos = []
for title, labels, x in layers:
    n = len(labels)
    ys = np.linspace(8.6, 0.4, n) if n > 1 else np.array([4.5])
    pos.append((x, ys))
    ax.text(x, 9.5, title, ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    for y, lab in zip(ys, labels):
        ax.scatter([x], [y], s=110, color=BLUE if title.startswith(("Hidden", "Output")) else ORANGE, zorder=3)
        if lab:
            side = -0.3 if x < 3 else 0.3
            ax.text(x + side, y, lab, ha="right" if side < 0 else "left", va="center", fontsize=9, color=INK2)
for (x1, y1s), (x2, y2s) in zip(pos, pos[1:]):
    for a in y1s:
        for b in y2s:
            ax.plot([x1, x2], [a, b], color=GRID, linewidth=0.5, zorder=1)
counts = ["7×16 weights + 16 biases = 128", "16×8 + 8 = 136", "8×1 + 1 = 9"]
for (x1, _), (x2, _), c in zip(pos, pos[1:], counts):
    ax.text((x1 + x2) / 2, -0.45, c, ha="center", fontsize=9, color=MUTED)
fig.suptitle("7 → 16 → 8 → 1 fully connected network: 273 numbers, 546 bytes as int16",
             x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.savefig(os.path.join(OUT, "network.png"), dpi=150, bbox_inches="tight")

# Numbers quoted in the explanation
print("rows", len(df))
print("night mean by temp bin:", df[night].groupby("tbin", observed=True)["dT"].mean().round(2).to_dict())
print("day mean by temp bin:", df[day].groupby("tbin", observed=True)["dT"].mean().round(2).to_dict())
print("feature ranges:", df[["temp_p", "rh_p", "wind_p"]].describe().loc[["mean", "std", "min", "max"]].round(1).to_dict())
