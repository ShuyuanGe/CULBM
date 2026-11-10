import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

CELL_BYTES = 108.0
PANELS = (
    {
        "title": "L1 optimistic upper bound ($\\lambda=0$)",
        "capacities": np.linspace(1.0e3, 128.0e3, 1024),
        "capacity_scale": 1.0e3,
        "capacity_label": "Resident capacity $C$ (KB)",
        "depths": np.linspace(1.0, 5.0, 1024),
        "depth_ticks": np.arange(1, 6),
        "lambda": 0.0,
        "marker_capacity": 108.0e3,
        "marker_label": "108 KB",
    },
    {
        "title": "L2 representative model ($\\lambda=0.2$)",
        "capacities": np.linspace(2.0e6, 128.0e6, 1024),
        "capacity_scale": 1.0e6,
        "capacity_label": "Resident capacity $C$ (MB)",
        "depths": np.linspace(1.0, 30.0, 1024),
        "depth_ticks": np.arange(5, 31, 5),
        "lambda": 0.2,
        "marker_capacity": 64.0e6,
        "marker_label": "64 MB",
    },
)


def S0(I, C, lam):
    b = np.cbrt(C / CELL_BYTES)
    interior = 1.0 - 2.0 * (I - 1.0) / b
    h = interior**3
    speedup = h / ((1.0 + h) / (2.0 * I) + lam * (1.0 - 1.0 / I))
    return np.ma.masked_where(interior <= 0.0, speedup)


assert len(sys.argv) == 2
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

plot_data = []
for panel in PANELS:
    Is, Cs = np.meshgrid(panel["depths"], panel["capacities"])
    Ss = S0(Is, Cs, panel["lambda"])
    plot_data.append((panel, Is, Cs, Ss))

minS = min(float(Ss.min()) for _, _, _, Ss in plot_data)
maxS = max(float(Ss.max()) for _, _, _, Ss in plot_data)
assert minS < 1.0 < maxS
levels = np.unique(np.concatenate((np.linspace(minS, maxS, 41), [1.0])))
norm = TwoSlopeNorm(vmin=minS, vcenter=1.0, vmax=maxS)

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
filled = None
break_even_handle = Line2D([0], [0], color="#4A4A4A", linestyle="--", linewidth=1.0, label=r"Speedup--slowdown boundary ($S_0=1$)")

for ax, (panel, Is, Cs, Ss) in zip(axes, plot_data, strict=True):
    displayed_capacity = Cs / panel["capacity_scale"]

    filled = ax.contourf(Is, displayed_capacity, Ss, levels=levels, cmap="coolwarm", norm=norm)

    ax.contour(Is, displayed_capacity, Ss, levels=[1.0], colors="#4A4A4A", linestyles="--", linewidths=1.0, alpha=0.95)

    integer_depths = np.arange(1, int(panel["depths"][-1]) + 1)
    marker_capacities = np.full(integer_depths.shape, panel["marker_capacity"], dtype=float)
    marker_values = S0(integer_depths, marker_capacities, panel["lambda"])
    marker_index = int(np.argmax(marker_values))
    marker_depth = int(integer_depths[marker_index])
    marker_value = float(marker_values[marker_index])
    displayed_marker = panel["marker_capacity"] / panel["capacity_scale"]
    ax.axhline(displayed_marker, color="#1A1A1A", linestyle="--", linewidth=0.4, alpha=0.8)
    ax.plot(marker_depth, displayed_marker, color="#111111", marker="x", markersize=5.5, markeredgewidth=1.2)
    ax.annotate(f"{panel['marker_label']}: $S_0={marker_value:.2f}$", xy=(marker_depth, displayed_marker), xytext=(5, 5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5)
    ax.set_title(panel["title"], fontsize=8)
    ax.set_xlabel("Local depth $I$", fontsize=7.5)
    ax.set_ylabel(panel["capacity_label"], fontsize=7.5)
    ax.set_xticks(panel["depth_ticks"])
    ax.tick_params(labelsize=6.5, length=2.5)

fig.legend(handles=[break_even_handle], loc="upper center", bbox_to_anchor=(0.43, 1.035), frameon=False, fontsize=6.5, handlelength=2.4)

assert filled is not None
fig.subplots_adjust(left=0.085, right=0.89, bottom=0.20, top=0.88, wspace=0.27)
cbar = fig.colorbar(filled, ax=axes, pad=0.025, ticks=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
cbar.set_label("Ideal speedup $S_0$", fontsize=7.5)
cbar.ax.tick_params(labelsize=6.5)
fig.savefig(sys.argv[1], bbox_inches="tight")
