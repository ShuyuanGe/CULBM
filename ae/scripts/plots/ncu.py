#!/usr/bin/env python3
import csv
import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO / "data/reference/ncu"
RAW_DIR = RESULT_DIR / "raw_ncu"
TIMING_DIR = RESULT_DIR / "raw_timing"
USEFUL_LUP = 512**3 * 18
PERSISTING_L2_BYTES = 83_886_080
TOTAL_L2_BYTES = 134_217_728
PLAIN_NCU_FILES = [f"plain_rep{repeat}.csv" for repeat in range(1, 4)]

CASES = {
    "enabled_in_capacity": {
        "persisting_l2": "enabled",
        "blocking_dim": [96, 94, 86],
        "block_dim": [96, 2, 2],
        "grid_dim": [1, 47, 43],
        "raw_files": [
            "enabled_in_capacity_rep1.csv",
            "enabled_in_capacity_rep2.csv",
            "enabled_in_capacity_rep3.csv",
        ],
    },
    "disabled_in_capacity": {
        "persisting_l2": "disabled",
        "blocking_dim": [96, 94, 86],
        "block_dim": [96, 2, 2],
        "grid_dim": [1, 47, 43],
        "raw_files": [
            "disabled_in_capacity_rep1.csv",
            "disabled_in_capacity_rep2.csv",
            "disabled_in_capacity_rep3.csv",
        ],
    },
    "disabled_over_total_l2": {
        "persisting_l2": "disabled",
        "blocking_dim": [96, 188, 86],
        "block_dim": [96, 2, 2],
        "grid_dim": [1, 94, 43],
        "raw_files": [
            "disabled_over_total_l2_rep1.csv",
            "disabled_over_total_l2_rep2.csv",
            "disabled_over_total_l2_rep3.csv",
        ],
    },
}

TIMING_CASES = {
    "plain": {
        "block_dim": [128, 1, 1],
        "grid_dim": [4, 512, 512],
        "raw_files": [f"plain_rep{repeat}.log" for repeat in range(1, 6)],
    },
    "enabled_in_capacity": {
        "block_dim": [96, 2, 2],
        "grid_dim": [1, 47, 43],
        "raw_files": [f"enabled_in_capacity_rep{repeat}.log" for repeat in range(1, 6)],
    },
    "disabled_in_capacity": {
        "block_dim": [96, 2, 2],
        "grid_dim": [1, 47, 43],
        "raw_files": [f"disabled_in_capacity_rep{repeat}.log" for repeat in range(1, 6)],
    },
    "disabled_over_total_l2": {
        "block_dim": [96, 2, 2],
        "grid_dim": [1, 94, 43],
        "raw_files": [f"disabled_over_total_l2_rep{repeat}.log" for repeat in range(1, 6)],
    },
}

BASE_STAGES = ["First", "Middle (false)", "Middle (true)", "Last"]
STAGE_ORDER = ["First", "Middle (false)", "Middle (true)", "Middle total", "Last", "Pipeline"]
METRIC_NAMES = [
    "dram__sectors.sum",
    "lts__t_requests_aperture_device_lookup_hit.sum",
    "lts__t_requests_aperture_device_lookup_miss.sum",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_name(kernel_name: str) -> str | None:
    if "FirstKernel" in kernel_name:
        return "First"
    if "MiddleKernel<0>" in kernel_name:
        return "Middle (false)"
    if "MiddleKernel<1>" in kernel_name:
        return "Middle (true)"
    if "LastKernel" in kernel_name:
        return "Last"
    return None


def parse_raw(path: Path) -> dict[str, dict[str, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith('"ID"'))
    values = {stage: {metric: 0.0 for metric in METRIC_NAMES} for stage in BASE_STAGES}
    launch_ids = {stage: set() for stage in BASE_STAGES}
    for row in csv.DictReader(lines[header_index:]):
        stage = stage_name(row["Kernel Name"])
        if stage is None:
            continue
        metric_name = row["Metric Name"]
        if metric_name not in METRIC_NAMES:
            continue
        values[stage][metric_name] += float(row["Metric Value"].replace(",", ""))
        launch_ids[stage].add(row["ID"])

    output: dict[str, dict[str, float]] = {}
    for stage in BASE_STAGES:
        metrics = values[stage]
        hits = metrics["lts__t_requests_aperture_device_lookup_hit.sum"]
        misses = metrics["lts__t_requests_aperture_device_lookup_miss.sum"]
        output[stage] = {
            "calls": float(len(launch_ids[stage])),
            "dram_sectors_per_lup": metrics["dram__sectors.sum"] / USEFUL_LUP,
            "l2_requests_per_lup": (hits + misses) / USEFUL_LUP,
            "l2_request_hit_pct": 100.0 * hits / (hits + misses),
        }

    middle_metrics = {
        metric_name: values["Middle (false)"][metric_name] + values["Middle (true)"][metric_name]
        for metric_name in METRIC_NAMES
    }
    hits = middle_metrics["lts__t_requests_aperture_device_lookup_hit.sum"]
    misses = middle_metrics["lts__t_requests_aperture_device_lookup_miss.sum"]
    output["Middle total"] = {
        "calls": output["Middle (false)"]["calls"] + output["Middle (true)"]["calls"],
        "dram_sectors_per_lup": middle_metrics["dram__sectors.sum"] / USEFUL_LUP,
        "l2_requests_per_lup": (hits + misses) / USEFUL_LUP,
        "l2_request_hit_pct": 100.0 * hits / (hits + misses),
    }

    pipeline_metrics = {
        metric_name: sum(values[stage][metric_name] for stage in BASE_STAGES)
        for metric_name in METRIC_NAMES
    }
    hits = pipeline_metrics["lts__t_requests_aperture_device_lookup_hit.sum"]
    misses = pipeline_metrics["lts__t_requests_aperture_device_lookup_miss.sum"]
    output["Pipeline"] = {
        "calls": sum(output[stage]["calls"] for stage in BASE_STAGES),
        "dram_sectors_per_lup": pipeline_metrics["dram__sectors.sum"] / USEFUL_LUP,
        "l2_requests_per_lup": (hits + misses) / USEFUL_LUP,
        "l2_request_hit_pct": 100.0 * hits / (hits + misses),
    }
    return output


def parse_plain_raw(path: Path) -> dict[str, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith('"ID"'))
    metrics = {metric: 0.0 for metric in METRIC_NAMES}
    launch_ids = set()
    for row in csv.DictReader(lines[header_index:]):
        if "StaticD3Q27PullKernel" not in row["Kernel Name"]:
            continue
        metric_name = row["Metric Name"]
        if metric_name not in METRIC_NAMES:
            continue
        metrics[metric_name] += float(row["Metric Value"].replace(",", ""))
        launch_ids.add(row["ID"])
    hits = metrics["lts__t_requests_aperture_device_lookup_hit.sum"]
    misses = metrics["lts__t_requests_aperture_device_lookup_miss.sum"]
    return {
        "calls": float(len(launch_ids)),
        "dram_sectors_per_lup": metrics["dram__sectors.sum"] / USEFUL_LUP,
        "l2_requests_per_lup": (hits + misses) / USEFUL_LUP,
        "l2_request_hit_pct": 100.0 * hits / (hits + misses),
    }


def resident_bytes(blocking_dim: list[int]) -> int:
    return 108 * blocking_dim[0] * blocking_dim[1] * blocking_dim[2]


def parse_timing(path: Path) -> list[float]:
    matches = re.findall(
        r"speed = ([0-9]+(?:\.[0-9]+)?) \(MLUPS\)", path.read_text(encoding="utf-8")
    )
    assert len(matches) == 6, path
    return [float(value) for value in matches]


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
