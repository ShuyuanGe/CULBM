#!/usr/bin/env python3
"""Regenerate analytical Figure4 and archived-reference Figures5--7."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox

# Measured from the ACM acmart template used by this manuscript.
ACM_COLUMN_WIDTH = 3.3368
ACM_TEXT_WIDTH = 7.0056
FIGURE_2_HEIGHT = 1.6
FIGURE_3_HEIGHT = 1.4
FIGURE_4_HEIGHT = 1.7
FIGURE_5_HEIGHT = 2.7
FIGURE_2_HORIZONTAL_PADDING = 0.0
FIGURE_3_HORIZONTAL_PADDING = 0.0
FIGURE_4_HORIZONTAL_PADDING = 0.7
FIGURE_5_HORIZONTAL_PADDING = 0.2
PAPER_DIR = Path(__file__).resolve().parents[2]
FIGURE_DIR = PAPER_DIR / "results/reference/figures"
DATA_DIR = PAPER_DIR / "data/reference"
FIXED_DOMAIN_CSV = DATA_DIR / "fig5/fixed_domain_speedup.csv"
CROSS_VELOCITY_JSON = DATA_DIR / "fig6/cross_velocity_speedup.json"
SELECTED_OUTCOMES_JSON = DATA_DIR / "fig7/selected_outcomes.json"
DEFAULT_MEASUREMENT_DIR = DATA_DIR / "fig7"
DEFAULT_CALIBRATION_DIR = DATA_DIR / "calibration/middle_efficiency"
FONT_PATH = Path(font_manager.findfont("DejaVu Serif"))

GREEN = "#159947"
MAGENTA = "#B66AA8"
GRAY = "#555B61"
LIGHT_GRAY = "#E7E5E1"
WARM_WHITE = "#F7F5F1"
GAIN_MAP = LinearSegmentedColormap.from_list(
    "culbm_gain",
    (WARM_WHITE, "#CFE7C9", "#69B978", "#2F9256"),
)
GPU_ORDER = (
    ("rtx4090d", "RTX 4090 D"),
    ("local", "RTX 5090 D v2"),
    ("rtx6000ada", "RTX 6000 Ada"),
    ("rtxpro6000", "RTX PRO 6000"),
    ("l20", "L20"),
    ("l40", "L40"),
)
VELOCITY_SETS = ("D3Q15", "D3Q19", "D3Q27")
VALIDATION_GPU_ORDER = (
    "rtx4090d",
    "rtx6000ada",
    "l20",
    "rtx5090dv2",
    "rtxpro6000",
    "l40",
)
STATIC_BASELINE_MLUPS = {
    "rtx5090dv2": 5013.25955,
    "rtx4090d": 4069.55445,
    "rtxpro6000": 6409.61670,
    "rtx6000ada": 3659.63110,
    "l20": 2768.71485,
    "l40": 2717.67215,
}


def configure_style(font_path: Path) -> None:
    assert font_path.is_file()
    font_manager.fontManager.addfont(str(font_path))
    font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 8.2,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.6,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.7,
            "axes.edgecolor": GRAY,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
        }
    )


def save_figure(
    figure,
    output_path: Path,
    target_width: float,
    horizontal_padding: float,
    dpi: int | None,
) -> None:
    assert horizontal_padding >= 0
    target_content_width = target_width - 2 * horizontal_padding
    assert target_content_width > 0
    for _ in range(12):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        tight_width = figure.get_tightbbox(renderer).width
        error = target_content_width - tight_width
        if abs(error) < 1.0e-7:
            break
        current_width, current_height = figure.get_size_inches()
        figure.set_size_inches(
            current_width * target_content_width / tight_width,
            current_height,
            forward=True,
        )
    figure.canvas.draw()
    tight_bbox = figure.get_tightbbox(figure.canvas.get_renderer())
    tight_width = tight_bbox.width
    assert abs(tight_width - target_content_width) < 1.0e-6
    output_bbox = Bbox.from_bounds(
        tight_bbox.x0 - horizontal_padding,
        tight_bbox.y0,
        target_width,
        tight_bbox.height,
    )
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches=output_bbox,
        pad_inches=0,
    )


def ideal_speedup(local_depth, capacity, l2_ratio):
    tile_edge = np.cbrt(capacity / 108.0)
    interior_edge = 1.0 - 2.0 * (local_depth - 1.0) / tile_edge
    valid_fraction = interior_edge**3
    speedup = valid_fraction / (
        (1.0 + valid_fraction) / (2.0 * local_depth) + l2_ratio * (1.0 - 1.0 / local_depth)
    )
    return np.ma.masked_where(interior_edge <= 0.0, speedup)


def simplify_axes(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRAY)
    axis.spines["bottom"].set_color(GRAY)


def forward_fill(values: np.ndarray) -> np.ndarray:
    filled = np.empty_like(values)
    last = np.nan
    for index, value in enumerate(values):
        if np.isfinite(value):
            last = value
        filled[index] = last
    return filled


def draw_figure_2() -> None:
    panels = (
        {
            "title": r"L1 ($\lambda=0$)",
            "capacities": np.linspace(1.0e3, 140.0e3, 1024),
            "capacity_scale": 1.0e3,
            "capacity_label": r"Capacity $C$ (KB)",
            "depths": np.linspace(1.0, 5.0, 1024),
            "depth_ticks": np.arange(1, 6),
            "lambda": 0.0,
            "markers": ((108.0e3, "108 KB", (1.5, 5)),),
            "annotation_mode": "above",
        },
        {
            "title": r"L2 ($\lambda=0.2$)",
            "capacities": np.linspace(2.0e6, 140.0e6, 1024),
            "capacity_scale": 1.0e6,
            "capacity_label": r"Capacity $C$ (MB)",
            "depths": np.linspace(1.0, 30.0, 1024),
            "depth_ticks": np.arange(5, 31, 5),
            "lambda": 0.2,
            "markers": (
                (32.0e6, "32 MB", (14, 44)),
                (80.0e6, "80 MB", (17, 82)),
                (128.0e6, "128 MB", (20, 122)),
            ),
            "annotation_mode": "leader",
        },
    )
    plot_data = []
    for panel in panels:
        depths, capacities = np.meshgrid(panel["depths"], panel["capacities"])
        speedups = ideal_speedup(depths, capacities, panel["lambda"])
        plot_data.append((panel, depths, capacities, speedups))

    minimum = min(float(values.min()) for _, _, _, values in plot_data)
    data_maximum = max(float(values.max()) for _, _, _, values in plot_data)
    assert minimum < 1.0 < data_maximum
    maximum = np.ceil(data_maximum * 2.0) / 2.0
    negative_power = 1.4
    positive_power = 1.35

    linear_positions = np.linspace(0.0, 1.0, 257)
    linear_values = minimum + linear_positions * (maximum - minimum)
    distance = linear_values - 1.0
    negative = 0.5 - 0.5 * (np.maximum(-distance, 0.0) / (1.0 - minimum)) ** negative_power
    positive = 0.5 + 0.5 * (np.maximum(distance, 0.0) / (maximum - 1.0)) ** positive_power
    remapped = np.where(distance < 0.0, negative, positive)
    base_map = LinearSegmentedColormap.from_list(
        "culbm_speedup_base",
        (
            (0.00, "#AD6A9F"),
            (0.26, "#D0A1C5"),
            (0.43, "#EAD7E6"),
            (0.50, "#F4F2EE"),
            (0.54, "#E0EEE3"),
            (0.60, "#B7DCC0"),
            (0.72, "#7FC18F"),
            (0.86, "#3F9A5D"),
            (1.00, "#176F39"),
        ),
    )
    color_map = LinearSegmentedColormap.from_list(
        "culbm_speedup",
        list(
            zip(
                linear_positions,
                base_map(remapped),
                strict=True,
            )
        ),
    )
    norm = Normalize(vmin=minimum, vmax=maximum)
    levels = np.linspace(minimum, maximum, 51)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(ACM_COLUMN_WIDTH, FIGURE_2_HEIGHT),
        gridspec_kw={"width_ratios": (0.72, 1.50)},
    )
    filled = None
    for axis, (panel, depths, capacities, speedups) in zip(axes, plot_data, strict=True):
        shown_capacity = capacities / panel["capacity_scale"]
        filled = axis.contourf(
            depths,
            shown_capacity,
            speedups,
            levels=levels,
            cmap=color_map,
            norm=norm,
        )
        axis.contour(
            depths,
            shown_capacity,
            speedups,
            levels=[1.0],
            colors="#4A4A4A",
            linestyles="--",
            linewidths=1.0,
            alpha=0.95,
        )
        integer_depths = np.arange(1, int(panel["depths"][-1]) + 1)
        for capacity, label, text_position in panel["markers"]:
            values = ideal_speedup(
                integer_depths,
                np.full(integer_depths.shape, capacity),
                panel["lambda"],
            )
            index = int(np.argmax(values))
            depth = int(integer_depths[index])
            value = float(values[index])
            shown_marker = capacity / panel["capacity_scale"]
            axis.plot(
                depth,
                shown_marker,
                marker="x",
                color="#292D2F",
                markersize=4.8,
                markeredgewidth=1.05,
                clip_on=False,
            )
            if panel["annotation_mode"] == "above":
                axis.annotate(
                    f"{label}\n${value:.2f}\\times$",
                    xy=(depth, shown_marker),
                    xytext=text_position,
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color="#292D2FBA",
                    fontsize=5.5,
                    linespacing=0.88,
                    annotation_clip=False,
                )
            else:
                axis.annotate(
                    f"{label}\n${value:.2f}\\times$",
                    xy=(depth, shown_marker),
                    xytext=text_position,
                    textcoords="data",
                    ha="right",
                    va="center",
                    color="#292D2FBA",
                    fontsize=6,
                    linespacing=0.88,
                    annotation_clip=False,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#292D2FBA",
                        "linewidth": 0.65,
                        "shrinkA": 1.0,
                        "shrinkB": 1.0,
                    },
                )
        axis.set_title(panel["title"], fontsize=8.2, pad=2.0)
        axis.set_ylabel(panel["capacity_label"], fontsize=8.0, labelpad=2.0)
        axis.set_xticks(panel["depth_ticks"])
        axis.set_ylim(0, 140)
        axis.set_yticks((25, 50, 75, 100, 125))
        axis.tick_params(labelsize=7.2, length=2.5, width=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.75)
        axis.spines["bottom"].set_linewidth(0.75)
    assert filled is not None
    figure.supxlabel(r"Local depth $I$", fontsize=8.0, y=0.005)
    figure.subplots_adjust(
        left=0.125,
        right=0.875,
        bottom=0.21,
        top=0.90,
        wspace=0.58,
    )
    colorbar = figure.colorbar(
        filled,
        ax=axes,
        pad=0.020,
        fraction=0.042,
        ticks=np.arange(0.0, 2.51, 0.5),
    )
    colorbar.outline.set_linewidth(0.75)
    colorbar.set_label(r"Ideal speedup $S_0$", fontsize=8.0, labelpad=3.0)
    colorbar.ax.tick_params(labelsize=7.2, length=2.5, width=0.75)
    colorbar.ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    colorbar.ax.tick_params(which="minor", length=1.5, width=0.6)
    save_figure(
        figure,
        FIGURE_DIR / "ideal_speedup.pdf",
        ACM_COLUMN_WIDTH,
        FIGURE_2_HORIZONTAL_PADDING,
        None,
    )
    plt.close(figure)


def load_fixed_domain_rows() -> list[dict[str, str]]:
    assert FIXED_DOMAIN_CSV.is_file()
    with FIXED_DOMAIN_CSV.open(newline="") as stream:
        return list(csv.DictReader(stream))


def draw_figure_3() -> None:
    rows = load_fixed_domain_rows()
    by_device = {}
    for device_id, _ in GPU_ORDER:
        by_device[device_id] = [row for row in rows if row["target_id"] == device_id]

    figure, axes = plt.subplots(
        1,
        6,
        figsize=(ACM_TEXT_WIDTH, FIGURE_3_HEIGHT),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    for axis, (device_id, device_name) in zip(axes, GPU_ORDER, strict=True):
        device_rows = by_device[device_id]
        n_values = np.array([int(row["n"]) for row in device_rows])
        temporal = np.array(
            [
                float(row["culbm_median_mlups"]) / 1000
                if row["culbm_status"] == "measured"
                else np.nan
                for row in device_rows
            ]
        )
        fluidx3d = np.array(
            [
                float(row["fluidx3d_median_mlups"]) / 1000
                if row["fluidx3d_status"] == "measurement"
                else np.nan
                for row in device_rows
            ]
        )
        axis.plot(n_values, temporal, "-o", color=GREEN, lw=1.45, ms=3.1)
        axis.plot(n_values, fluidx3d, "-s", color=GRAY, lw=1.15, ms=2.7)
        culbm_follow = forward_fill(temporal)
        fluidx3d_follow = forward_fill(fluidx3d)
        axis.plot(
            n_values[~np.isfinite(temporal)],
            culbm_follow[~np.isfinite(temporal)],
            linestyle="none",
            marker="x",
            color=GREEN,
            ms=3.2,
            mew=0.8,
        )
        axis.plot(
            n_values[~np.isfinite(fluidx3d)],
            fluidx3d_follow[~np.isfinite(fluidx3d)],
            linestyle="none",
            marker="x",
            color=GRAY,
            ms=3.2,
            mew=0.8,
        )
        axis.set_title(device_name, pad=2.0)
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55)
        axis.set_xlim(128 - 64, 896 + 64)
        axis.set_xticks((128, 384, 640, 896))
        axis.set_xticks((256, 512, 768), minor=True)
        axis.tick_params(length=2.4)
        axis.tick_params(which="minor", length=1.5, width=0.6)
        simplify_axes(axis)

    axes[0].set_ylabel("Throughput\n(GLUP/s)")
    axes[0].set_ylim(0, 17)
    axes[0].set_yticks((0, 5, 10, 15))
    axes[0].set_yticks((2.5, 7.5, 12.5), minor=True)
    axes[0].tick_params(which="minor", length=1.5, width=0.6)
    figure.supxlabel(r"Domain edge $N$", y=0.015, fontsize=8.2)
    legend = (
        Line2D([], [], color=GREEN, marker="o", lw=1.45, ms=3.1, label="L2 LBM"),
        Line2D([], [], color=GRAY, marker="s", lw=1.15, ms=2.7, label="FluidX3D"),
        (
            Line2D([], [], color=GREEN, marker="x", linestyle="none", ms=3.2, mew=0.8),
            Line2D([], [], color=GRAY, marker="x", linestyle="none", ms=3.2, mew=0.8),
        ),
    )
    figure.legend(
        handles=legend,
        labels=("L2 LBM", "FluidX3D", "Unavailable"),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.27, top=0.67, wspace=0.24)
    save_figure(
        figure,
        FIGURE_DIR / "r4_fixed_domain_performance.pdf",
        ACM_TEXT_WIDTH,
        FIGURE_3_HORIZONTAL_PADDING,
        None,
    )
    plt.close(figure)


def draw_figure_4() -> None:
    experiments = json.loads(CROSS_VELOCITY_JSON.read_text())
    source_gpu_order = {
        "rtx4090d": "4090D",
        "local": "5090Dv2",
        "rtx6000ada": "RTX 6000 Ada",
        "rtxpro6000": "RTX PRO 6000",
        "l20": "L20",
        "l40": "L40",
    }
    n_values = (128, 256, 384, 512, 640)
    values = np.full((len(GPU_ORDER), len(VELOCITY_SETS) * len(n_values)), np.nan)
    for gpu_index, (device_id, _) in enumerate(GPU_ORDER):
        source_device = source_gpu_order[device_id]
        for velocity_index, velocity_set in enumerate(VELOCITY_SETS):
            measurements = experiments[source_device][velocity_set]
            for n_index, domain_edge in enumerate(n_values):
                domain_key = str(domain_edge)
                if domain_key in measurements:
                    values[gpu_index, velocity_index * len(n_values) + n_index] = measurements[
                        domain_key
                    ]

    figure, axis = plt.subplots(figsize=(ACM_TEXT_WIDTH, FIGURE_4_HEIGHT))
    masked = np.ma.masked_invalid(values)
    color_map = GAIN_MAP.copy()
    color_map.set_bad(color="white")
    image = axis.imshow(
        masked,
        cmap=color_map,
        vmin=1.7,
        vmax=3.1,
        aspect="auto",
        interpolation="bilinear",
        interpolation_stage="rgba",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.1,
                    color="black",
                )
            else:
                axis.text(column, row, "—", ha="center", va="center", color="0.55")

    axis.set_xticks(np.arange(values.shape[1]))
    axis.set_xticklabels(n_values * len(VELOCITY_SETS))
    axis.set_yticks(np.arange(len(GPU_ORDER)))
    axis.set_yticklabels([name for _, name in GPU_ORDER])
    axis.tick_params(length=0, pad=2)
    axis.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.7)
    axis.tick_params(which="minor", bottom=False, left=False)
    for boundary in (4.5, 9.5):
        axis.axvline(boundary, color="white", linewidth=3.0)
    for index, velocity_set in enumerate(VELOCITY_SETS):
        axis.text(
            (index + 0.5) / len(VELOCITY_SETS),
            1.08,
            velocity_set,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
        )
    axis.set_xlabel(r"Cubic domain edge $N$")
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, pad=0.018, aspect=13)
    colorbar.set_label("Speedup (×)")
    colorbar.ax.tick_params(length=2.4)
    colorbar.outline.set_visible(False)
    figure.subplots_adjust(left=0.125, right=0.925, bottom=0.20, top=0.84)
    save_figure(
        figure,
        FIGURE_DIR / "CrossVelocitySetApplicability.pdf",
        ACM_TEXT_WIDTH,
        FIGURE_4_HORIZONTAL_PADDING,
        None,
    )
    save_figure(
        figure,
        FIGURE_DIR / "CrossVelocitySetApplicability.png",
        ACM_TEXT_WIDTH,
        FIGURE_4_HORIZONTAL_PADDING,
        300,
    )
    plt.close(figure)


def final_store_factors(block_x: int, iteration: int) -> tuple[float, float]:
    radius = iteration - 1
    length = block_x - 2 * radius
    warp = np.arange(0, 1024, 32)
    x_lo = np.maximum(radius, warp)
    x_hi = np.minimum(block_x - radius, warp + 32)
    phase = np.arange(8)
    sector_lo = (phase[:, None] + x_lo[None, :]) // 8
    sector_hi = (phase[:, None] + x_hi[None, :] - 1) // 8 + 1
    sectors = np.where(x_hi[None, :] > x_lo[None, :], sector_hi - sector_lo, 0)
    gcd = np.gcd(length, 8)
    period = 8 // gcd
    orbit = phase % gcd == 0
    sector_count = (sectors.sum(axis=1) * orbit).sum()
    left_edge = (phase + radius) % 8 != 0
    right_edge = (phase + block_x - radius) % 8 != 0
    edge_count = (left_edge & orbit).sum() + (right_edge & orbit).sum()
    return 8 * sector_count / (period * length), 4 * edge_count / (period * length)


def corrected_speedup(
    valid_fraction: np.ndarray | float,
    final_store_factor: np.ndarray | float,
    iteration: np.ndarray | float,
    l2_ratio: float,
) -> np.ndarray | float:
    return valid_fraction / (
        (1 + final_store_factor * valid_fraction) / (2 * iteration) + l2_ratio * (1 - 1 / iteration)
    )


def validation_model_values(
    target: dict,
    block: list[int],
    iteration: int,
) -> tuple[float, float]:
    block_x, block_y, block_z = block
    radius = iteration - 1
    valid_fraction = (
        (1 - 2 * radius / block_x) * (1 - 2 * radius / block_y) * (1 - 2 * radius / block_z)
    )
    beta_sector, beta_edge = final_store_factors(block_x, iteration)
    final_store_factor = (
        1
        + target["lambda_"] * (beta_sector - 1)
        + (2 * target["gamma"] - (1 + target["lambda_"])) * beta_edge
    )
    ideal = valid_fraction / (
        (1 + valid_fraction) / (2 * iteration) + target["lambda_"] * (1 - 1 / iteration)
    )
    corrected = corrected_speedup(
        valid_fraction,
        final_store_factor,
        iteration,
        target["lambda_"],
    )
    return float(ideal), float(corrected)


def feasible_yz(block_x: int, iteration: int, capacity_cells: int) -> tuple:
    minimum = 2 * (iteration - 1) + 1
    assert block_x >= minimum
    ys = np.arange(minimum, capacity_cells // (block_x * minimum) + 1)
    ends = np.minimum(ys, capacity_cells // (block_x * ys))
    counts = np.maximum(0, ends - minimum + 1)
    by = np.repeat(ys, counts)
    starts = np.cumsum(counts) - counts
    bz = np.arange(int(counts.sum())) - np.repeat(starts, counts) + minimum
    return by, bz


def full_model_envelopes(target: dict) -> tuple[list[dict], list[dict]]:
    corrected_maxima = []
    ideal_maxima = []
    capacity_cells = target["persisting_l2_bytes"] // 108
    for iteration in range(2, 21):
        radius = iteration - 1
        minimum = 2 * radius + 1
        best_corrected = -np.inf
        best_ideal = -np.inf
        for bx in range(32 * ((minimum + 31) // 32), 1025, 32):
            by, bz = feasible_yz(bx, iteration, capacity_cells)
            h = (1 - 2 * radius / bx) * (1 - 2 * radius / by) * (1 - 2 * radius / bz)
            sector, edge = final_store_factors(bx, iteration)
            factor = (
                1
                + target["lambda_"] * (sector - 1)
                + (2 * target["gamma"] - (1 + target["lambda_"])) * edge
            )
            ideal = h / ((1 + h) / (2 * iteration) + target["lambda_"] * (1 - 1 / iteration))
            corrected = corrected_speedup(h, factor, iteration, target["lambda_"])
            best_corrected = max(best_corrected, np.max(corrected, initial=-np.inf))
            best_ideal = max(best_ideal, np.max(ideal, initial=-np.inf))
        corrected_maxima.append({"I": iteration, "S": float(best_corrected)})
        ideal_maxima.append({"I": iteration, "S0": float(best_ideal)})
    return corrected_maxima, ideal_maxima


def validation_panel_data(
    measurement_path: Path,
    calibration_path: Path,
) -> tuple:
    measurement = json.loads(measurement_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    target = measurement["target"]
    assert calibration["target"]["key"] == target["key"]
    rows = []
    for raw in measurement["records"]:
        if raw["returncode"] != 0 or len(raw["speeds"]) < 5 or raw["inner_loop"] < 2:
            continue
        ideal, corrected = validation_model_values(
            target,
            raw["block_size"],
            raw["inner_loop"],
        )
        rows.append(
            {
                "I": raw["inner_loop"],
                "B": raw["block_size"],
                "speedup": raw["median_mlups"] / STATIC_BASELINE_MLUPS[target["key"]],
                "S0": ideal,
                "S": corrected,
            }
        )
    measured_maxima = []
    conditional_measurements = []
    for iteration in sorted({row["I"] for row in rows}):
        group = [row for row in rows if row["I"] == iteration]
        measured_maxima.append(max(group, key=lambda row: row["speedup"]))
        conditional_measurements.append(max(group, key=lambda row: row["S"]))
    corrected_maxima, ideal_maxima = full_model_envelopes(target)
    # The cross is the exact model-selected outcome reported in Table 2,
    # not the sampled candidate with the highest model score.
    selected = next(
        r for r in json.loads(SELECTED_OUTCOMES_JSON.read_text()) if r["key"] == target["key"]
    )
    selected_measurement = {
        "I": selected["I"],
        "B": selected["B"],
        "speedup": selected["S"],
        "S": selected["S"],
    }
    return (
        target,
        rows,
        measured_maxima,
        conditional_measurements,
        corrected_maxima,
        ideal_maxima,
        selected_measurement,
    )


def draw_validation_panel(axis, data: tuple) -> tuple:
    (
        target,
        rows,
        measured_maxima,
        conditional_measurements,
        corrected_maxima,
        ideal_maxima,
        selected_measurement,
    ) = data
    axis.scatter(
        [row["I"] for row in rows],
        [row["speedup"] for row in rows],
        s=1.4,
        color=GRAY,
        alpha=0.035,
        linewidths=0,
    )
    measured_line = axis.plot(
        [row["I"] for row in measured_maxima],
        [row["speedup"] for row in measured_maxima],
        "o-",
        color=GREEN,
        lw=1.05,
        ms=2.7,
        alpha=0.90,
        label=r"$\widehat S$ envelope",
    )[0]
    conditional_line = axis.plot(
        [row["I"] for row in conditional_measurements],
        [row["speedup"] for row in conditional_measurements],
        "^-",
        color=MAGENTA,
        lw=1.00,
        ms=2.9,
        alpha=0.88,
        label=r"$\widehat S$ at sampled $\max S$",
    )[0]
    corrected_line = axis.plot(
        [row["I"] for row in corrected_maxima],
        [row["S"] for row in corrected_maxima],
        "--",
        color=GREEN,
        lw=0.95,
        alpha=0.78,
        label=r"$S$ envelope",
    )[0]
    ideal_line = axis.plot(
        [row["I"] for row in ideal_maxima],
        [row["S0"] for row in ideal_maxima],
        "--",
        color=GRAY,
        lw=0.90,
        alpha=0.72,
        label=r"$S_0$ envelope",
    )[0]
    selected_marker = axis.scatter(
        selected_measurement["I"],
        selected_measurement["speedup"],
        marker="x",
        s=42,
        color=MAGENTA,
        linewidths=1.35,
        alpha=0.92,
        zorder=8,
        label=r"$\widehat S(\theta^{\star})$",
    )
    axis.axhline(1.0, color=GRAY, ls=":", lw=0.8)
    axis.set_xlim(1, 20)
    axis.set_xticks(np.arange(1, 20, 2))
    axis.set_xticks(np.arange(2, 21, 2), minor=True)
    axis.tick_params(which="minor", length=1.6, width=0.6)
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5)
    axis.set_title(target["label"], pad=2.5)
    simplify_axes(axis)
    axis.set_ylim(0, 3)
    axis.set_yticks(np.arange(0, 3.01, 0.5))
    return (
        measured_line,
        conditional_line,
        corrected_line,
        ideal_line,
        selected_marker,
    )


def draw_figure_5(measurement_dir: Path, calibration_dir: Path) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(ACM_TEXT_WIDTH, FIGURE_5_HEIGHT),
        sharex=True,
        sharey=True,
    )
    handles = None
    for axis, key in zip(axes.flat, VALIDATION_GPU_ORDER, strict=True):
        data = validation_panel_data(
            measurement_dir / f"{key}.json",
            calibration_dir / f"{key}.json",
        )
        handles = draw_validation_panel(axis, data)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Speedup ($\times$)")
    for axis in axes[1, :]:
        axis.set_xlabel(r"Inner-loop depth $I$")
    assert handles is not None
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=5,
        frameon=False,
        fontsize=7.4,
        columnspacing=1.1,
        handlelength=1.8,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.12,
        top=0.86,
        wspace=0.18,
        hspace=0.34,
    )
    save_figure(
        figure,
        FIGURE_DIR / "d3q27_joint_tile_depth_validation.pdf",
        ACM_TEXT_WIDTH,
        FIGURE_5_HORIZONTAL_PADDING,
        None,
    )
    save_figure(
        figure,
        FIGURE_DIR / "d3q27_joint_tile_depth_validation.png",
        ACM_TEXT_WIDTH,
        FIGURE_5_HORIZONTAL_PADDING,
        300,
    )
    plt.close(figure)


def render(output: Path) -> None:
    global FIGURE_DIR
    FIGURE_DIR = output
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_style(FONT_PATH)
    draw_figure_2()
    draw_figure_3()
    draw_figure_4()
    draw_figure_5(DEFAULT_MEASUREMENT_DIR, DEFAULT_CALIBRATION_DIR)
