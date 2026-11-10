#!/usr/bin/env python3
"""Single average-GPU FP32 Roofline for dense D3Q27 LBM kernels."""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from matplotlib.transforms import Bbox

# ============================ User configuration =============================

BYTES_PER_POPULATION = 4
EXTRA_BYTES_PER_LUP = 0

MEMORY_EFFICIENCY = 1.0
COMPUTE_EFFICIENCY = 1.0

SAVE_DPI = 300

# Full data range; the average plot is displayed with tighter limits below.
FULL_X_MIN = 1.0e-2
FULL_X_MAX = 8.0e2
FULL_Y_MIN = 5.0e-3
FULL_Y_MAX = 2.0e2

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results/reference/figures"
FONT_PATH = Path(font_manager.findfont("DejaVu Serif"))
KERNEL_OPS_JSON = Path(__file__).resolve().parents[2] / "data/reference/roofline/kernel_ops.json"

LATTICES = ("D2Q9", "D3Q19", "D3Q27")
Q_BY_LATTICE = {
    "D2Q9": 9,
    "D3Q19": 19,
    "D3Q27": 27,
}

# ACM single-column width measured from the acmart template.
ACM_COLUMN_WIDTH = 3.3368
ROOFLINE_AVERAGE_GPU_HEIGHT = 2.2
ROOFLINE_AVERAGE_GPU_HORIZONTAL_PADDING = 0.35
AVERAGE_X_MIN = 0.5
AVERAGE_Y_MIN = 0.5
DRAM_ROOF_COLOR = "#333333"
DRAM_RANGE_COLOR = "#C9C9C9"
L2_ROOF_COLOR = "#007A55"
L2_RANGE_COLOR = "#A8D8C2"
KERNEL_COLOR = "#1F2933"


@dataclass(frozen=True)
class GPU:
    fp32_tflops: float
    dram_gbps: float
    l2_gbps: float

    @property
    def fp32_roof(self) -> float:
        return self.fp32_tflops * COMPUTE_EFFICIENCY

    @property
    def dram_roof(self) -> float:
        return self.dram_gbps * MEMORY_EFFICIENCY

    @property
    def l2_roof(self) -> float | None:
        if self.l2_gbps is None:
            return None
        return self.l2_gbps * MEMORY_EFFICIENCY

    @property
    def dram_ridge_ai(self) -> float:
        # TFLOP/s * 1000 / GB/s = FLOP/byte.
        return self.fp32_roof * 1000.0 / self.dram_roof

    @property
    def l2_ridge_ai(self) -> float | None:
        l2_roof = self.l2_roof
        if l2_roof is None:
            return None
        return self.fp32_roof * 1000.0 / l2_roof


GPU_SPECS: dict[str, GPU] = {
    # Measured R_DRAM / R_L2 from ppopp27/main.tex, tab:device_model_summary.
    "RTX 4090 D": GPU(
        fp32_tflops=74.0,
        dram_gbps=798.0,
        l2_gbps=3840.0,
    ),
    # Derived ordinary FP32 peak:
    # 21760 CUDA cores * 2 FLOP/cycle * 2.41 GHz.
    "RTX 5090 D V2": GPU(
        fp32_tflops=104.8832,
        dram_gbps=991.0,
        l2_gbps=5310.0,
    ),
    "RTX 6000 Ada": GPU(
        fp32_tflops=91.1,
        dram_gbps=697.0,
        l2_gbps=4430.0,
    ),
    "RTX PRO 6000 Blackwell WS": GPU(
        fp32_tflops=125.0,
        dram_gbps=1309.0,
        l2_gbps=6870.0,
    ),
    "L20": GPU(
        fp32_tflops=59.8,
        dram_gbps=459.0,
        l2_gbps=4030.0,
    ),
    "L40": GPU(
        fp32_tflops=90.5,
        dram_gbps=490.0,
        l2_gbps=4380.0,
    ),
}


def load_kernel_ops() -> dict[str, dict[str, int]]:
    payload = json.loads(KERNEL_OPS_JSON.read_text())
    ops = {lattice: {} for lattice in payload["columns"]["datasets"]}
    for row in payload["data"]:
        method = row["method"]
        for lattice in payload["columns"]["datasets"]:
            ops[lattice][method] = row[lattice]["S+CSE"]
    return ops


KERNEL_OPS = load_kernel_ops()

METHOD_LABELS = {
    "SRT": "BGK / SRT",
    "TRT": "TRT",
    "O-MRT": "O-MRT",
    "R-O-MRT": "R-O-MRT",
    "WO-MRT": "WO-MRT",
    "R-WO-MRT": "R-WO-MRT",
    "CM": "Central moment",
    "R-CM": "R-CM",
    "K": "Cumulant",
    "R-K": "R-K",
}

KERNEL_ORDER = [
    "SRT",
    "TRT",
    "O-MRT",
    "WO-MRT",
    "CM",
    "K",
]

KERNEL_MARKERS = ["o", "s", "v", "^", "D", "P"]


# ================================ Model ======================================


def bytes_per_lup(lattice: str) -> int:
    return 2 * Q_BY_LATTICE[lattice] * BYTES_PER_POPULATION + EXTRA_BYTES_PER_LUP


def selected_kernels() -> list[str]:
    return KERNEL_ORDER


def arithmetic_intensity(lattice: str, kernel_name: str) -> float:
    return KERNEL_OPS[lattice][kernel_name] / bytes_per_lup(lattice)


def _bandwidth_ceiling(
    ai: np.ndarray | float,
    gbps: float,
) -> np.ndarray:
    return np.asarray(ai, dtype=float) * gbps / 1000.0


def dram_ceiling(ai: np.ndarray | float, gpu: GPU) -> np.ndarray:
    return _bandwidth_ceiling(ai, gpu.dram_roof)


def l2_ceiling(ai: np.ndarray | float, gpu: GPU) -> np.ndarray:
    assert gpu.l2_roof is not None
    return _bandwidth_ceiling(ai, gpu.l2_roof)


def compute_ceiling(ai: np.ndarray | float, gpu: GPU) -> np.ndarray:
    ai_arr = np.asarray(ai, dtype=float)
    return np.full_like(ai_arr, gpu.fp32_roof, dtype=float)


def attainable_roof(ai: np.ndarray | float, gpu: GPU) -> np.ndarray:
    return np.minimum(dram_ceiling(ai, gpu), compute_ceiling(ai, gpu))


def kernel_performance(lattice: str, kernel_name: str, gpu: GPU) -> float:
    return float(attainable_roof(arithmetic_intensity(lattice, kernel_name), gpu))


def save_figure(
    fig: plt.Figure,
    stem: str,
    target_width: float,
    horizontal_padding: float,
) -> None:
    assert horizontal_padding >= 0
    target_content_width = target_width - 2 * horizontal_padding
    assert target_content_width > 0
    for _ in range(40):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight_width = fig.get_tightbbox(renderer).width
        error = target_content_width - tight_width
        if abs(error) < 1.0e-7:
            break
        current_width, current_height = fig.get_size_inches()
        scale = target_content_width / tight_width
        fig.set_size_inches(
            current_width * scale,
            current_height * scale,
            forward=True,
        )
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    tight_width = tight_bbox.width
    assert abs(tight_width - target_content_width) < 1.0e-6
    output_bbox = Bbox.from_bounds(
        tight_bbox.x0 - horizontal_padding,
        tight_bbox.y0,
        target_width,
        tight_bbox.height,
    )
    for extension in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{extension}",
            dpi=SAVE_DPI if extension == "png" else None,
            bbox_inches=output_bbox,
            pad_inches=0,
        )
    plt.close(fig)


def configure_style() -> None:
    assert FONT_PATH.is_file()
    font_manager.fontManager.addfont(str(FONT_PATH))
    font_family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_family,
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )


def configure_log_axes(
    ax: plt.Axes,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax.set_ylabel("Throughput [TFLOP/s]")
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.tick_params(pad=2)
    ax.grid(True, which="both", linewidth=0.35, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_kernel_points(
    ax: plt.Axes,
    lattice: str,
    gpu: GPU,
    kernels: list[str],
    *,
    color: str,
    label: bool = False,
    markersize: int = 4,
) -> None:
    for marker, kernel_name in zip(KERNEL_MARKERS, kernels):
        ai = arithmetic_intensity(lattice, kernel_name)
        plot_kwargs = {
            "marker": marker,
            "color": color,
            "linestyle": "None",
            "markersize": markersize,
            "zorder": 10,
        }
        if label:
            plot_kwargs["label"] = METHOD_LABELS[kernel_name]
        ax.plot(
            ai,
            kernel_performance(lattice, kernel_name, gpu),
            **plot_kwargs,
        )


# ========================== Average GPU roofline =============================


def plot_average_gpu_roofline(lattice: str) -> None:
    l2_values = [gpu.l2_gbps for gpu in GPU_SPECS.values()]
    assert all(value is not None for value in l2_values)
    average_gpu = GPU(
        fp32_tflops=float(np.mean([gpu.fp32_tflops for gpu in GPU_SPECS.values()])),
        dram_gbps=float(np.mean([gpu.dram_gbps for gpu in GPU_SPECS.values()])),
        l2_gbps=float(np.mean(l2_values)),
    )

    kernels = selected_kernels()
    ai_grid = np.logspace(
        np.log10(FULL_X_MIN),
        np.log10(FULL_X_MAX),
        1800,
    )
    roofs = np.stack([attainable_roof(ai_grid, gpu) for gpu in GPU_SPECS.values()])
    l2_roofs = np.stack(
        [
            np.minimum(
                l2_ceiling(ai_grid, gpu),
                compute_ceiling(ai_grid, gpu),
            )
            for gpu in GPU_SPECS.values()
        ]
    )
    fig, ax = plt.subplots(figsize=(ACM_COLUMN_WIDTH, ROOFLINE_AVERAGE_GPU_HEIGHT))
    ax.fill_between(
        ai_grid,
        roofs.min(axis=0),
        roofs.max(axis=0),
        color=DRAM_RANGE_COLOR,
        alpha=0.40,
        linewidth=0,
    )
    ax.fill_between(
        ai_grid,
        l2_roofs.min(axis=0),
        l2_roofs.max(axis=0),
        color=L2_RANGE_COLOR,
        alpha=0.32,
        linewidth=0,
    )
    ax.plot(
        ai_grid,
        attainable_roof(ai_grid, average_gpu),
        color=DRAM_ROOF_COLOR,
        linewidth=1.0,
        alpha=0.65,
        label="DRAM roof",
        zorder=5,
    )
    ax.plot(
        ai_grid,
        np.minimum(
            l2_ceiling(ai_grid, average_gpu),
            compute_ceiling(ai_grid, average_gpu),
        ),
        color=L2_ROOF_COLOR,
        linestyle="-",
        linewidth=0.9,
        alpha=0.65,
        label="L2 roof",
        zorder=4,
    )
    ridge = average_gpu.dram_ridge_ai
    l2_ridge = average_gpu.l2_ridge_ai
    assert l2_ridge is not None
    ax.axvline(
        l2_ridge,
        color=L2_ROOF_COLOR,
        linestyle="--",
        linewidth=0.55,
        alpha=0.8,
        zorder=1,
    )
    ax.axvline(
        ridge,
        color="#555B61",
        linestyle="--",
        linewidth=0.55,
        alpha=0.9,
        zorder=1,
    )
    region_label_y = 8.0
    for x, label, color in (
        (1.2 * np.sqrt(AVERAGE_X_MIN * l2_ridge), "L2\nbound", L2_ROOF_COLOR),
        (np.sqrt(l2_ridge * ridge), "memory\nbound", DRAM_ROOF_COLOR),
        (np.sqrt(ridge * FULL_X_MAX), "compute\nbound", DRAM_ROOF_COLOR),
    ):
        ax.text(
            x,
            region_label_y,
            label,
            ha="center",
            va="center",
            multialignment="center",
            rotation=45,
            fontsize=6,
            color=color,
        )
    plot_kernel_points(
        ax,
        lattice,
        average_gpu,
        kernels,
        color=KERNEL_COLOR,
        label=False,
        markersize=2.2,
    )

    configure_log_axes(
        ax,
        (AVERAGE_X_MIN, FULL_X_MAX),
        (AVERAGE_Y_MIN, FULL_Y_MAX),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")
    kernel_handles = [
        Line2D(
            [],
            [],
            marker=marker,
            color=KERNEL_COLOR,
            linestyle="None",
            markersize=3,
            label=METHOD_LABELS[kernel_name],
        )
        for marker, kernel_name in zip(KERNEL_MARKERS, kernels)
    ]
    ax.legend(
        handles=kernel_handles,
        loc="center left",
        bbox_to_anchor=(1.04, 0.55),
        ncol=1,
        fontsize=6,
        handlelength=1.0,
        columnspacing=0.8,
        frameon=False,
    )
    fig.subplots_adjust(left=0.12, right=0.27, bottom=0.14, top=0.98)
    fig.canvas.draw()
    diagonal = ax.transData.transform(((1.0, 1.0), (10.0, 10.0)))
    dx, dy = diagonal[1] - diagonal[0]
    assert np.isclose(abs(dy / dx), 1.0)

    save_figure(
        fig,
        f"roofline_average_gpu_{lattice.lower()}_complete",
        ACM_COLUMN_WIDTH,
        ROOFLINE_AVERAGE_GPU_HORIZONTAL_PADDING,
    )


# ================================ Report =====================================


def render(output: Path) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = output
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_average_gpu_roofline("D3Q27")
