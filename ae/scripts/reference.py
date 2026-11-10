"""Verify archived measurements and regenerate paper reference statistics."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from checks import MODEL_ID
from plots import evaluation, ncu, roofline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/reference"
GPU_KEYS = ("rtx4090d", "rtx5090dv2", "rtx6000ada", "rtxpro6000", "l20", "l40")
GPU_LABELS = ("4090D", "5090Dv2", "RTX 6000 Ada", "RTX PRO 6000", "L20", "L40")
SEED = 20260803
DRAWS = 5000


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list) -> None:
    assert rows
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geomean(values: list) -> float:
    assert values and all(value > 0 for value in values)
    return math.exp(statistics.mean(math.log(value) for value in values))


def verify_inventory() -> dict:
    inventory = read_json(ROOT / "data/provenance/reference_inventory.json")
    for row in inventory["files"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"], row["path"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"], row["path"]
    return {
        "files": len(inventory["files"]),
        "bytes": sum(row["bytes"] for row in inventory["files"]),
    }


def fixed_domain(output: Path) -> dict:
    rows = read_csv(DATA / "fig5/fixed_domain_speedup.csv")
    paired = [row for row in rows if row["culbm_status"] == "measured" and row["fluidx3d_status"] == "measurement"]
    for row in paired:
        ratio = float(row["culbm_median_mlups"]) / float(row["fluidx3d_median_mlups"])
        assert math.isclose(ratio, float(row["speedup"]), rel_tol=1e-12)
    write_csv(output / "figure5_paired.csv", paired)
    values = [float(row["speedup"]) for row in paired]
    return {
        "paired_domains": len(values),
        "geomean_speedup": geomean(values),
        "min_speedup": min(values),
        "max_speedup": max(values),
        "historical_protocol_recovered": False,
    }


def cross_velocity(output: Path) -> dict:
    extracted = read_json(DATA / "fig6/raw_samples.json")
    snapshot = read_json(DATA / "fig6/cross_velocity_speedup.json")
    rows = []
    for label in GPU_LABELS:
        for velocity in ("D3Q15", "D3Q19", "D3Q27"):
            for n, samples in extracted["EXPERIMENTS"][label][velocity].items():
                ratio = statistics.median(samples["temporal_mlups"]) / statistics.median(samples["plain_mlups"])
                assert math.isclose(ratio, snapshot[label][velocity][n], rel_tol=1e-12)
                rows.append(
                    {
                        "gpu": label,
                        "velocity_set": velocity,
                        "n": int(n),
                        "plain_median_mlups": statistics.median(samples["plain_mlups"]),
                        "temporal_median_mlups": statistics.median(samples["temporal_mlups"]),
                        "speedup": ratio,
                    }
                )
    write_csv(output / "figure6_speedups.csv", rows)
    return {
        velocity: {
            "paired_domains": sum(row["velocity_set"] == velocity for row in rows),
            "geomean_speedup": geomean([row["speedup"] for row in rows if row["velocity_set"] == velocity]),
        }
        for velocity in ("D3Q15", "D3Q19", "D3Q27")
    }


def density_redraw(iterations: np.ndarray, speeds: np.ndarray) -> list:
    full = np.zeros(19)
    np.maximum.at(full, iterations - 2, speeds)
    assert np.all(full > 0)
    rng = np.random.default_rng(SEED)
    rows = []
    for count in range(1000, 8001, 1000):
        errors = np.empty(DRAWS)
        for draw in range(DRAWS):
            indices = rng.choice(len(speeds), size=count, replace=False)
            envelope = np.zeros(19)
            np.maximum.at(envelope, iterations[indices] - 2, speeds[indices])
            errors[draw] = np.sqrt(np.mean((envelope - full) ** 2)) / full.max()
        rows.append(
            {
                "n": count,
                "draws": DRAWS,
                "seed": SEED,
                "median_percent": float(np.median(errors) * 100),
                "p95_percent": float(np.percentile(errors, 95) * 100),
            }
        )
    return rows


def feasible_count(target: dict) -> int:
    count = 0
    cells = target["persisting_l2_bytes"] // 108
    for iteration in range(2, 21):
        minimum = 2 * (iteration - 1) + 1
        for bx in range(32 * ((minimum + 31) // 32), 1025, 32):
            by, _ = evaluation.feasible_yz(bx, iteration, cells)
            count += len(by)
    return count


def model_predictions(target: dict, records: list, selected: dict) -> dict:
    points = []
    for row in records:
        _, score = evaluation.validation_model_values(target, row["block_size"], row["inner_loop"])
        points.append(
            {
                "case_id": row["case_id"],
                "B": row["block_size"],
                "I": row["inner_loop"],
                "model_score": score,
                "measured_mlups": row["median_mlups"],
            }
        )
    winners = [
        max((row for row in points if row["I"] == depth), key=lambda row: row["model_score"])
        for depth in range(2, 21)
    ]
    envelope, ideal_envelope = evaluation.full_model_envelopes(target)
    _, selected_score = evaluation.validation_model_values(target, selected["B"], selected["I"])
    return {
        "inputs": {
            "q": 27,
            "lambda": target["lambda_"],
            "gamma": target["gamma"],
            "capacity_bytes": target["persisting_l2_bytes"],
        },
        "theoretical_envelope": envelope,
        "ideal_envelope": ideal_envelope,
        "sampled_model_winners": winners,
        "selected": {"B": selected["B"], "I": selected["I"], "model_score": selected_score},
    }


def selection_quality(output: Path) -> dict:
    selected = {row["key"]: row for row in read_json(DATA / "fig7/selected_outcomes.json")}
    coverage = []
    density = []
    quality = []
    predictions = {}
    for key in GPU_KEYS:
        payload = read_json(DATA / "fig7" / f"{key}.json")
        records = payload["records"]
        valid = [row for row in records if row["returncode"] == 0 and len(row["speeds"]) >= 5 and row["inner_loop"] >= 2]
        full_count = feasible_count(payload["target"])
        coverage.append(
            {
                "gpu": key,
                "sampled_attempts": len(records),
                "sampled_I1_attempts": sum(row["inner_loop"] == 1 for row in records),
                "valid_I_ge2": len(valid),
                "full_Theta_eval_count": full_count,
                "historical_theta_and_i1_count": payload["protocol"]["theta_and_i1_count"],
                "valid_I_ge2_fraction_of_full_set": len(valid) / full_count,
            }
        )
        row = selected[key]
        assert math.isclose(row["P_l2lbm"] / row["P_plain"], row["S"], rel_tol=1e-12)
        sample_maximum = max(record["median_mlups"] / row["P_plain"] for record in valid)
        quality.append(
            {
                "gpu": key,
                "selected_speedup": row["S"],
                "sampled_best_speedup": sample_maximum,
                "selected_over_reference": row["S"] / max(row["S"], sample_maximum),
            }
        )
        iterations = np.array([record["inner_loop"] for record in valid])
        speeds = np.array([record["median_mlups"] for record in valid])
        for result in density_redraw(iterations, speeds):
            density.append({"gpu": key, **result})
        predictions[key] = model_predictions(payload["target"], valid, row)
        print(f"Reference {key}: {len(valid)} valid I>=2, density redraw complete", flush=True)
    aggregated = []
    for count in range(1000, 8001, 1000):
        group = [row for row in density if row["n"] == count]
        aggregated.append(
            {
                "n": count,
                "largest_per_gpu_median_percent": max(row["median_percent"] for row in group),
                "largest_per_gpu_p95_percent": max(row["p95_percent"] for row in group),
                "geomean_per_gpu_p95_percent": geomean([row["p95_percent"] for row in group]),
            }
        )
    write_csv(output / "figure7_coverage.csv", coverage)
    write_csv(output / "figure7_selection_quality.csv", quality)
    write_csv(output / "sampling_density_per_gpu.csv", density)
    write_csv(output / "sampling_density_table_D1.csv", aggregated)
    write_json(output / "figure7_model_predictions.json", {"model_id": MODEL_ID, "devices": predictions})
    return {
        "geomean_selected_fraction": geomean([row["selected_over_reference"] for row in quality]),
        "geomean_selected_percent": 100 * geomean([row["selected_over_reference"] for row in quality]),
        "coverage": coverage,
        "sampling_density": aggregated,
        "density_protocol": (
            "Per-GPU default_rng(20260803), n ascending, 5000 independent uniform redraws "
            "without replacement per n; full successful I>=2 pool is the reference, "
            "and n=8000 is a subsample."
        ),
    }


def calibration(output: Path) -> list:
    gamma = read_csv(DATA / "calibration/legacy_gamma.csv")
    bandwidth = {row["gpu"]: row for row in read_csv(DATA / "calibration/legacy_lambda.csv")}
    rows = []
    for row in gamma:
        bw = bandwidth[row["gpu"]]
        rows.append(
            {
                "gpu": row["gpu"],
                "gamma_from_printed_times": 4 * float(row["t_3f1p_ms"]) / float(row["t_4f_ms"]) - 3,
                "gamma_reported": float(row["gamma"]),
                "lambda_from_printed_bandwidths": float(bw["dram_gbps"]) / float(bw["l2_gbps"]),
                "lambda_reported": float(bw["lambda"]),
                "note": (
                    "Printed inputs have finite precision; standalone bandwidths do not "
                    "establish Table2 absolute-bandwidth provenance."
                ),
            }
        )
    write_csv(output / "table2_calibration_check.csv", rows)
    return rows


def cache_metrics(output: Path) -> dict:
    rows = []
    for repeat, name in enumerate(ncu.PLAIN_NCU_FILES, 1):
        result = ncu.parse_plain_raw(ncu.RAW_DIR / name)
        rows.append({"case": "plain", "repeat": repeat, "stage": "Fused pull-stream", **result})
    for case, config in ncu.CASES.items():
        for repeat, name in enumerate(config["raw_files"], 1):
            for stage, result in ncu.parse_raw(ncu.RAW_DIR / name).items():
                rows.append({"case": case, "repeat": repeat, "stage": stage, **result})
    write_json(output / "table3_raw_recomputed.json", rows)
    aggregated = []
    for case, stage in dict.fromkeys((row["case"], row["stage"]) for row in rows):
        group = [row for row in rows if row["case"] == case and row["stage"] == stage]
        assert len(group) == 3
        aggregated.append(
            {
                "case": case,
                "stage": stage,
                **{
                    metric: statistics.median(row[metric] for row in group)
                    for metric in ("calls", "dram_sectors_per_lup", "l2_requests_per_lup", "l2_request_hit_pct")
                },
            }
        )
    write_csv(output / "table3_counters.csv", aggregated)
    frozen = read_csv(DATA / "ncu/summary.csv")
    for row in aggregated:
        expected = next(item for item in frozen if item["case"] == row["case"] and item["stage"] == row["stage"])
        assert row["calls"] == int(expected["calls"])
        for metric in ("dram_sectors_per_lup", "l2_requests_per_lup", "l2_request_hit_pct"):
            assert abs(row[metric] - float(expected[metric + "_median"])) < 1e-8
    timing = []
    for case, config in ncu.TIMING_CASES.items():
        samples = [ncu.parse_timing(ncu.TIMING_DIR / name) for name in config["raw_files"]]
        assert all(len(sample) == 6 for sample in samples)
        estimates = [statistics.median(sample[1:]) for sample in samples]
        timing.append(
            {
                "case": case,
                "processes": len(estimates),
                "median_mlups": statistics.median(estimates),
                "process_cv_percent": 100 * statistics.stdev(estimates) / statistics.mean(estimates),
            }
        )
    write_csv(output / "table3_non_profiler_timing.csv", timing)
    plain = next(row for row in aggregated if row["case"] == "plain")
    temporal = next(row for row in aggregated if row["case"] == "enabled_in_capacity" and row["stage"] == "Pipeline")
    return {
        "raw_profiles": 12,
        "plain_hit_percent": plain["l2_request_hit_pct"],
        "temporal_hit_percent": temporal["l2_request_hit_pct"],
        "dram_traffic_reduction": plain["dram_sectors_per_lup"] / temporal["dram_sectors_per_lup"],
        "note": (
            "Counters recomputed from raw Nsight Compute CSVs; timing separately recomputed "
            "from non-profiler logs."
        ),
    }


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 2,
        "model_id": MODEL_ID,
        "mode": "paper_reference_reanalysis",
        "new_hardware_measurements": False,
        "inventory": verify_inventory(),
    }
    summary["figure5"] = fixed_domain(output)
    summary["figure6"] = cross_velocity(output)
    summary["figure7"] = selection_quality(output)
    summary["table2"] = calibration(output)
    summary["table3"] = cache_metrics(output)
    evaluation.render(output / "figures")
    roofline.render(output / "figures")
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = run(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "figure5_geomean": summary["figure5"]["geomean_speedup"],
                "figure7_selected_percent": summary["figure7"]["geomean_selected_percent"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
