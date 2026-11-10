"""Run packaged reference analysis and fresh, explicitly labelled GPU experiments."""

import argparse
import csv
import importlib
import io
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from checks import (
    MODEL_ID,
    compare_ddf,
    compare_macroscopic,
    compare_macroscopic_bits,
    full_halo_domain,
    make_input,
    model_score,
    parse_windows,
    sample_candidates,
    sha256,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
NCU_METRICS = (
    "gpu__time_duration.sum,dram__sectors.sum,"
    "lts__t_requests_aperture_device_lookup_hit.sum,"
    "lts__t_requests_aperture_device_lookup_miss.sum"
)


def vector(values):
    return "[" + ",".join(str(value) for value in values) + "]"


def execute(command, output: Path, metadata, environment):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    (output / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (output / "stderr.log").write_text(process.stderr, encoding="utf-8")
    configuration = dict(metadata)
    if "--block" in command and "--grid" in command:
        configuration["actual_launch"] = {
            "block": json.loads(command[command.index("--block") + 1]),
            "grid": json.loads(command[command.index("--grid") + 1]),
        }
    record = {
        "schema_version": 2,
        "model_id": MODEL_ID,
        "command": [str(part).replace(str(ROOT) + "/", "") for part in command],
        "returncode": process.returncode,
        "wall_seconds": time.monotonic() - started,
        "status": "passed" if process.returncode == 0 else "failed",
        "configuration": configuration,
    }
    write_json(output / "run.json", record)
    return record, process.stdout, process.stderr


def successful(command, output: Path, metadata, environment):
    record, stdout, stderr = execute(command, output, metadata, environment)
    assert record["returncode"] == 0, {"run": str(output), "stderr": stderr[-4000:]}
    return record, stdout


def host_available_bytes():
    memory = Path("/proc/meminfo").read_text(encoding="utf-8")
    available = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", memory, re.MULTILINE)
    assert available is not None
    return int(available[1]) * 1024


def gpu_context(args):
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",") if "CUDA_VISIBLE_DEVICES" in os.environ else []
    physical = visible[args.device] if visible else str(args.device)
    raw = subprocess.check_output(
        [
            "nvidia-smi", "--id", physical,
            "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, uuid, total, free, driver = [part.strip() for part in raw.split(",")]
    files = sorted((ROOT / "code/src").rglob("*")) + sorted((ROOT / "code/tools").rglob("*"))
    context = {
        "schema_version": 2,
        "model_id": MODEL_ID,
        "runner_sha256": sha256(Path(__file__)),
        "implementation": "refactor-validation",
        "cuda_visible_device": args.device,
        "cuda_visible_devices": visible,
        "gpu": {"name": name, "uuid": uuid, "memory_mib": int(total), "driver": driver},
        "available_memory_bytes_at_start": int(free) * 1024**2,
        "available_host_memory_bytes_at_start": host_available_bytes(),
        "host_memory_measurement": "/proc/meminfo MemAvailable (kB converted to bytes)",
        "source_files": {str(path.relative_to(ROOT)): sha256(path) for path in files if path.is_file()},
        "binaries": {name: sha256(args.binary_dir / name) for name in ("culbm-single", "culbm-tune")},
        "tile_allocation": "physical",
        "physical_alignment_bytes": 6 * 1024**2,
        "original_source": json.loads((ROOT / "data/provenance/freeze.json").read_text()),
    }
    cache = args.binary_dir.parent / "CMakeCache.txt"
    if cache.is_file():
        context["cmake_cache"] = cache.read_text(encoding="utf-8")
    write_json(args.output / "provenance.json", context)
    return context


def solver_command(args, q, domain, mode, tile, depth, batch, steps, initial, output, dump, extra):
    assert batch > 0 and steps % batch == 0 and depth > 0 and batch % depth == 0
    command = [
        str(args.binary_dir / "culbm-single"),
        "--device", str(args.device), "--domain", vector(domain),
        "--q", str(q), "--collision", "bgk", "--inv-tau", str(args.inv_tau),
        "--batch-steps", str(batch), "--steps", str(steps),
    ]
    if mode in ("plain", "twist"):
        block = [math.gcd(length, edge) for length, edge in zip(domain, args.plain_block, strict=True)]
        grid = [length // edge for length, edge in zip(domain, block, strict=True)]
        command += [
            "--block", vector(block), "--grid", vector(grid),
            "--opt-policy", "plain", "--stream-policy", {"plain": "pull", "twist": "esotwist"}[mode],
            "--ddf-buffer", {"plain": "double", "twist": "single"}[mode],
        ]
    else:
        bx, by, bz = tile
        assert min(tile) > 2 * (depth - 1)
        command += [
            "--block", vector((bx, 1, 1)), "--grid", vector((1, by, bz)), "--inner-steps", str(depth),
            "--opt-policy", "blocking", "--stream-policy", "esotwist", "--ddf-buffer", mode,
        ]
    if initial is not None:
        command += ["--init", str(initial)]
    if dump:
        command += [
            "--output", str(output),
            "--dump-ddf", "--dump-rho", "--dump-vx", "--dump-vy", "--dump-vz",
        ]
    return command + extra


def estimated_bytes(q, domain, mode, tile, dump):
    nx, ny, nz = domain
    cells = math.prod(domain)
    ddf_cells = {
        "plain": 2 * cells,
        "twist": cells,
        "double": 2 * cells,
        "ring": nx * ny * (nz + tile[2] + 1),
    }[mode]
    # Upstream write_ddf holds a complete Q*N device snapshot alongside state.
    return 4 * q * (ddf_cells + math.prod(tile) + cells * int(dump)) + 20 * cells


def estimated_host_bytes(q, domain, dump):
    # write retains one macro field while holding the full DDF host snapshot.
    # Add a five-field input/work allowance; this is not an exact process peak.
    return (4 * (q + 1) * int(dump) + 20) * math.prod(domain)


def tune(args, output: Path, q):
    report = output / "calibration.json"
    successful(
        [
            str(args.binary_dir / "culbm-tune"),
            "--device", str(args.device), "--q", str(q),
            "--report-json", str(report),
        ],
        output,
        {"q": q},
        dict(os.environ),
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["schema_version"] == 2 and result["model_id"] == MODEL_ID
    return result


def timed_process(args, folder, q, domain, mode, tile, depth, initial, extra, cells):
    batch = math.lcm(90, depth)
    if args.command == "performance":
        batch = math.lcm(batch, args.performance_depth)
    command = solver_command(args, q, domain, mode, tile, depth, batch, 6 * batch, initial, folder, False, extra)
    config = {
        "q": q,
        "domain": list(domain),
        "mode": mode,
        "B": list(tile),
        "I": depth,
        "inv_tau": args.inv_tau,
        "input_manifest_sha256": sha256(initial / "input.json") if initial is not None else None,
        "input_kind": "uniform-default-all-fluid" if initial is None else json.loads((initial / "input.json").read_text())["kind"],
        "batch_steps": batch,
        "actual_steps": 6 * batch,
        "cells_per_step": cells,
        "denominator": "full-domain" if cells == math.prod(domain) else "full-halo-interior",
    }
    record, stdout, stderr = execute(command, folder, config, dict(os.environ))
    if record["returncode"] != 0:
        assert args.command == "selection", {"run": str(folder), "stderr": stderr[-4000:]}
        return record
    windows = parse_windows(stdout, 6)
    record.update(
        {
            "windows_mlups": windows,
            "discarded_windows": 1,
            "process_median_mlups": statistics.median(windows[1:]),
        }
    )
    write_json(folder / "run.json", record)
    return record


def run_reference(args):
    return importlib.import_module("reference").run(args.output)


def run_build(args):
    configure = [
        "cmake", "-S", str(ROOT / "code"), "-B", str(args.build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}",
    ]
    if args.toolchain_prefix is not None:
        prefix = args.toolchain_prefix
        configure += [
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DCMAKE_CXX_COMPILER={prefix / 'bin/g++'}",
            f"-DCMAKE_CUDA_COMPILER={prefix / 'bin/nvcc'}",
            f"-DCMAKE_CUDA_HOST_COMPILER={prefix / 'bin/g++'}",
        ]
    successful(configure, args.output / "configure", {}, dict(os.environ))
    successful(
        [
            "cmake", "--build", str(args.build_dir), "--parallel", str(args.jobs),
            "--target", "culbm-single", "culbm-tune", "device_prop",
        ],
        args.output / "compile",
        {},
        dict(os.environ),
    )
    return {"status": "passed", "binary_dir": str(args.build_dir / "bin")}


def correctness_case(args, context, folder, q, domain, depth, steps, kind, tile):
    initial = folder / "input"
    assert steps % depth == 0
    modes = ("plain", "twist", "double", "ring")
    estimates = {mode: estimated_bytes(q, domain, mode, tile, True) for mode in modes}
    estimate = max(estimates.values())
    host_estimate = estimated_host_bytes(q, domain, True)
    memory = {
        "estimated_device_peak_bytes": estimate,
        "estimated_device_bytes_by_mode": estimates,
        "device_ddf_snapshot_bytes": 4 * q * math.prod(domain),
        "available_device_bytes_at_start": context["available_memory_bytes_at_start"],
        "estimated_host_peak_bytes": host_estimate,
        "host_ddf_snapshot_bytes": 4 * q * math.prod(domain),
        "host_macro_snapshot_bytes": 4 * math.prod(domain),
        "host_working_allowance_bytes": 20 * math.prod(domain),
        "available_host_bytes_at_start": context["available_host_memory_bytes_at_start"],
        "available_fraction_limit": 0.9,
        "host_measurement": "/proc/meminfo MemAvailable",
        "estimate_scope": "Payload estimate; tile flags and allocator overhead may add memory.",
    }
    unsupported = [
        resource
        for resource, required, available in (
            ("device-memory", estimate, context["available_memory_bytes_at_start"]),
            ("host-memory", host_estimate, context["available_host_memory_bytes_at_start"]),
        )
        if required >= 0.9 * available
    ]
    if unsupported:
        result = {
            "q": q,
            "domain": list(domain),
            "I": depth,
            "status": "unsupported-memory",
            "estimated_bytes": estimate,
            "memory_preflight": memory,
            "unsupported_resources": unsupported,
            "method": "Full GPU/host DDF snapshot preflight; no smaller case substituted",
        }
        write_json(folder / "comparison.json", result)
        return result
    make_input(initial, domain, kind)
    batch = steps // 2 if steps % (2 * depth) == 0 else steps
    for mode in modes:
        destination = folder / mode
        command = solver_command(args, q, domain, mode, tile, depth, batch, steps, initial, destination, True, [])
        successful(
            command,
            destination,
            {
                "q": q,
                "domain": list(domain),
                "B": list(tile),
                "I": depth,
                "actual_steps": steps,
                "batch_steps": batch,
                "mode": mode,
                "input_kind": kind,
            },
            dict(os.environ),
        )
    checks = {
        f"{mode}-t{step}": {
            "status": "passed",
            "ddf": compare_ddf(
                folder / "plain" / f"ddf_{step}.dat", folder / mode / f"ddf_{step}.dat",
                q, domain, initial / "flag.dat",
            ),
            "macroscopic": compare_macroscopic_bits(folder / "plain", folder / mode, step, domain),
        }
        for mode in modes[1:]
        for step in range(batch, steps + 1, batch)
    }
    result = {
        "q": q,
        "domain": list(domain),
        "I": depth,
        "steps": steps,
        "input_kind": kind,
        "status": "passed",
        "memory_preflight": memory,
        "comparisons": checks,
    }
    write_json(folder / "comparison.json", result)
    if not args.keep_ddf:
        for mode in modes:
            for path in (folder / mode).glob("*.dat"):
                path.unlink()
    return result


def run_smoke(args):
    context = gpu_context(args)
    report = tune(args, args.output / "tuning", 27)
    cases = [
        correctness_case(
            args, context, args.output / "channel",
            27, (160, 64, 64), 5, args.steps, "channel", (64, 48, 31),
        )
    ]
    for steps in (3, 4):
        cases.append(
            correctness_case(
                args, context, args.output / f"parity-{steps}",
                27, (64, 48, 64), 1, steps, "paper-correctness", (64, 48, 31),
            )
        )
    return {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "partial-coverage",
        "kind": "getting-started",
        "cases": cases,
        "selected": report["selection"],
        "scope": "single-GPU functional validation",
    }


def run_correctness(args):
    context = gpu_context(args)
    cases = []
    for q in args.q:
        for n in args.sizes:
            for depth in args.depths:
                ring_length, shift = n + 32, 32
                groups = 2 * ring_length // math.gcd(ring_length, shift)
                steps = depth * max(groups, math.ceil(90 / depth))
                cases.append(
                    correctness_case(
                        args, context, args.output / f"q{q}-n{n}-i{depth}",
                        q, (n, n, n), depth, steps, "paper-correctness", (64, 48, 31),
                    )
                )
        cases.append(
            correctness_case(
                args, context, args.output / f"q{q}-partial",
                q, (160, 68, 62), 3, 90, "paper-correctness", (64, 48, 31),
            )
        )
    if args.historical:
        for q, n, tile in ((19, 512, (96, 99, 87)), (27, 128, (96, 82, 74))):
            cases.append(
                correctness_case(
                    args, context, args.output / f"historical-q{q}-n{n}",
                    q, (n, n, n), 9, 18, "paper-correctness", tile,
                )
            )
    return {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "partial-coverage",
        "cases": cases,
        "scope": "same-version active-DDF and full-domain macroscopic bitwise equivalence",
    }


def run_calibrate(args):
    gpu_context(args)
    return {"status": "passed", "report": tune(args, args.output / "tuning", args.q)}


def run_bridge(args):
    gpu_context(args)
    case = json.loads((ROOT / "data/bridge/case.json").read_text(encoding="utf-8"))
    domain = list(reversed(case["shape_zyx"]))
    initial = ROOT / "data/bridge/input"
    args.inv_tau, args.plain_block = case["inv_tau"], (32, 4, 2)
    steps, batch = case["steps"], case["batch_steps"]
    depth, tile = 5, (96, 56, 56)
    grid = [length // edge for length, edge in zip(domain, args.plain_block, strict=True)]
    (args.output / "master-plain/fields").mkdir(parents=True, exist_ok=True)
    legacy = [
        str(args.master_binary.resolve()),
        "--devId", str(args.device),
        "--blockDim", vector(args.plain_block), "--gridDim", vector(grid), "--domainDim", vector(domain),
        "--nstep", str(steps), "--dstep", str(batch), "--innerLoop", str(depth), "--invTau", str(args.inv_tau),
        "--initStateFolder", str(initial), "--dumpFolder", str(args.output / "master-plain/fields"),
        "--streamPolicy", "0", "--optPolicy", "0", "--dumpRho", "--dumpVx", "--dumpVy", "--dumpVz",
    ]
    successful(
        legacy,
        args.output / "master-plain",
        {
            "implementation": "reference-master",
            "binary_sha256": sha256(args.master_binary),
            "case": case,
        },
        dict(os.environ),
    )
    comparisons = {}
    for mode in ("plain", "double", "ring"):
        folder = args.output / mode
        command = solver_command(args, 27, domain, mode, tile, depth, batch, steps, initial, folder, False, [])
        command += ["--output", str(folder), "--dump-rho", "--dump-vx", "--dump-vy", "--dump-vz"]
        successful(
            command,
            folder,
            {"implementation": "packaged-refactor", "case": case, "mode": mode},
            dict(os.environ),
        )
        comparisons[mode] = compare_macroscopic(args.output / "master-plain/fields", folder, steps, domain, case["tolerances"])
    result = {
        "status": "passed" if all(item["status"] == "passed" for item in comparisons.values()) else "failed",
        "case": case,
        "comparisons": comparisons,
        "scope": "Case-specific numerical bridge between reference-master and packaged-refactor",
    }
    write_json(args.output / "summary.json", result)
    assert result["status"] == "passed", result
    if not args.keep_fields:
        for path in args.output.rglob("*.dat"):
            path.unlink()
    return result


def pilot(args, output, q, initial):
    candidates = [
        (32, 1, 1), (64, 1, 1), (128, 1, 1),
        (32, 2, 2), (32, 4, 2), (64, 2, 2), (64, 4, 2), (128, 4, 2), (64, 2, 8),
    ]
    results = []
    for block in candidates:
        args.plain_block = block
        command = solver_command(
            args, q, (256, 256, 256), "plain", (128, 128, 128), 1, 90, 540,
            initial, output, False, [],
        )
        record, stdout, _ = execute(command, output / vector(block), {"block": block, "q": q}, dict(os.environ))
        if record["returncode"] == 0:
            results.append((statistics.median(parse_windows(stdout, 6)[1:]), block))
    assert results, "No Plain CUDA launch candidate passed the pilot"
    args.plain_block = max(results)[1]
    write_json(
        output / "selected.json",
        {
            "block": args.plain_block,
            "pilot_domain": [256, 256, 256],
            "retuned_across_domain_sizes": False,
        },
    )


def fluidx3d_process(args, folder, q, n, batch, process, expected_uuid):
    assert args.opencl_device is not None, "Pass an explicit --opencl-device ordinal"
    environment = dict(os.environ)
    environment.update({"AE_N": str(n), "AE_BATCH": str(batch)})
    binary = args.fluidx3d_dir / f"q{q}/FluidX3D"
    command = [str(binary), str(args.opencl_device)]
    record, stdout = successful(
        command,
        folder / f"fluidx3d-{process}",
        {
            "q": q,
            "N": n,
            "batch_steps": batch,
            "actual_steps": 6 * batch,
            "omega": 0.5,
            "boundary": "six stationary solid faces",
            "binary_sha256": sha256(binary),
        },
        environment,
    )
    values = [float(value) for value in re.findall(r"AE_WINDOW ([0-9.eE+-]+)", stdout)]
    assert len(values) == 6 and all(value > 0 for value in values), stdout[-4000:]
    assert "FP32/FP32" in stdout, "FluidX3D did not confirm FP32 storage/arithmetic"
    uuids = re.findall(r"AE_OPENCL_UUID (GPU-[0-9a-fA-F-]+)", stdout)
    assert uuids == [expected_uuid], ("CUDA/OpenCL device mismatch", expected_uuid, uuids)
    record.update(
        {
            "windows_mlups": values,
            "process_median_mlups": statistics.median(values[1:]),
            "opencl_device": args.opencl_device,
            "uuid": uuids[0],
            "source": "qualification_upstream d8c571f",
        }
    )
    write_json(folder / f"fluidx3d-{process}/run.json", record)
    return record


def run_performance(args):
    if args.fluidx3d_dir is not None:
        assert args.inv_tau == 0.5, "The packaged FluidX3D driver uses omega=0.5"
    context = gpu_context(args)
    summaries = []
    for q in args.q:
        calibration = tune(args, args.output / f"q{q}/tuning", q)
        tile, depth = calibration["selection"]["B"], calibration["selection"]["I"]
        args.performance_depth = depth
        pilot_input = args.output / f"q{q}/pilot-input"
        make_input(pilot_input, (256, 256, 256), "closed")
        pilot(args, args.output / f"q{q}/pilot", q, pilot_input)
        for n in args.sizes:
            domain = (n, n, n)
            folder = args.output / f"q{q}/n{n}"
            estimate = estimated_bytes(q, domain, "plain", tile, False)
            if estimate >= 0.9 * context["available_memory_bytes_at_start"]:
                result = {
                    "q": q,
                    "N": n,
                    "status": "unsupported-memory",
                    "estimated_bytes": estimate,
                    "method": "preflight estimate",
                }
                write_json(folder / "summary.json", result)
                summaries.append(result)
                continue
            initial = folder / "input"
            make_input(initial, domain, "closed")
            windows = {mode: [] for mode in ("plain", "i1", "ring")}
            if args.fluidx3d_dir is not None:
                windows["fluidx3d"] = []
            for process in range(args.processes):
                for label, mode, block, inner in (
                    ("plain", "plain", tile, depth),
                    ("i1", "ring", (128, 128, 128), 1),
                    ("ring", "ring", tile, depth),
                ):
                    record = timed_process(
                        args, folder / f"{label}-{process}", q, domain, mode, block, inner,
                        initial, [], math.prod(domain),
                    )
                    windows[label].append(record["process_median_mlups"])
                if args.fluidx3d_dir is not None:
                    external = fluidx3d_process(args, folder, q, n, math.lcm(90, depth), process, context["gpu"]["uuid"])
                    windows["fluidx3d"].append(external["process_median_mlups"])
            medians = {mode: statistics.median(values) for mode, values in windows.items()}
            cv = {mode: statistics.pstdev(values) / statistics.mean(values) for mode, values in windows.items()}
            result = {
                "q": q,
                "N": n,
                "status": "passed" if max(cv.values()) <= 0.05 else "unstable",
                "B": tile,
                "I": depth,
                "plain_block": args.plain_block,
                "process_estimates": windows,
                "median_mlups": medians,
                "cv": cv,
                "ring_over_plain": medians["ring"] / medians["plain"],
                "i1_over_plain": medians["i1"] / medians["plain"],
                "protocol": {
                    "origin": "fresh validation",
                    "arithmetic_storage": "FP32 BGK",
                    "omega": args.inv_tau,
                    "boundary": "six stationary walls",
                    "initial_density": 1.0,
                    "initial_velocity": [0.0, 0.0, 0.0],
                },
                "fluidx3d": {"status": "not-requested"},
            }
            if args.fluidx3d_dir is not None:
                result["fluidx3d"] = {
                    "median_mlups": medians["fluidx3d"],
                    "cv": cv["fluidx3d"],
                    "process_estimates": windows["fluidx3d"],
                }
                result["ring_over_fluidx3d"] = medians["ring"] / medians["fluidx3d"]
            write_json(folder / "summary.json", result)
            summaries.append(result)
    return {
        "status": "completed",
        "results": summaries,
        "note": "Fresh validation, not replacement metadata for historical Figures 5/6",
    }


def run_selection(args):
    context = gpu_context(args)
    report = tune(args, args.output / "tuning", args.q)
    inputs = report["inputs"]
    assert report["device"] is not None
    total_l2_bytes = int(report["device"]["l2_capacity_bytes"])
    total, candidates = sample_candidates(args.q, int(inputs["capacity_bytes"]), args.samples, args.seed)
    write_json(
        args.output / "candidates.json",
        {
            "count": total,
            "seed": args.seed,
            "sampling": "uniform without replacement; I=2..20, By>=Bz, Bx<=1024",
            "candidates": candidates,
        },
    )
    baseline = timed_process(
        args, args.output / "plain", args.q, (256, 256, 256), "plain", (128, 128, 128), 1,
        None, [], 256**3,
    )["process_median_mlups"]
    results = []
    for index, candidate in enumerate(candidates):
        tile, depth = candidate["B"], candidate["I"]
        domain, cells = full_halo_domain(args.q, tile, depth, total_l2_bytes)
        folder = args.output / f"samples/{index:05d}"
        estimate = estimated_bytes(args.q, domain, "ring", tile, False)
        row = {
            **candidate,
            "domain": domain,
            "cells_per_step": cells,
            "denominator": "full-halo-interior",
            "estimated_bytes": estimate,
            "model_score": model_score(tile, depth, inputs["lambda"], inputs["gamma"]),
        }
        if estimate >= 0.9 * context["available_memory_bytes_at_start"]:
            row["status"] = "unsupported-memory"
        else:
            measurements = [
                timed_process(
                    args, folder / str(process), args.q, domain, "ring", tile, depth,
                    None, ["--interior-tiles-only"], cells,
                )
                for process in range(args.processes)
            ]
            row["processes"] = measurements
            if all(item["status"] == "passed" for item in measurements):
                row["median_mlups"] = statistics.median(item["process_median_mlups"] for item in measurements)
                row["speedup"] = row["median_mlups"] / baseline
                row["status"] = "passed"
            else:
                row["status"] = "failed"
        write_json(folder / "sample.json", row)
        results.append(row)
    chosen = report["selection"]
    domain, cells = full_halo_domain(args.q, chosen["B"], chosen["I"], total_l2_bytes)
    independent = [
        timed_process(
            args, args.output / f"selected/{process}", args.q, domain, "ring", chosen["B"], chosen["I"],
            None, ["--interior-tiles-only"], cells,
        )
        for process in range(5)
    ]
    if not all(item["status"] == "passed" for item in independent):
        return {
            "status": "failed-selected",
            "samples": results,
            "selected_processes": independent,
            "plain_mlups": baseline,
        }
    valid = [row for row in results if row["status"] == "passed"]
    if not valid:
        return {
            "status": "no-valid-candidates",
            "samples": results,
            "selected_processes": independent,
            "plain_mlups": baseline,
        }
    selected_median = statistics.median(item["process_median_mlups"] for item in independent)
    best = max(row["median_mlups"] for row in results if row["status"] == "passed")
    envelope = []
    for depth in range(2, 21):
        points = [row for row in results if row["I"] == depth and row["status"] == "passed"]
        if points:
            model_best = max(points, key=lambda row: row["model_score"])
            envelope.append(
                {
                    "I": depth,
                    "best_measured_mlups": max(row["median_mlups"] for row in points),
                    "model_best_in_sample": model_best,
                    "note": "Sample-restricted predictor; not a full-domain per-I argmax",
                }
            )
    return {
        "status": "completed",
        "samples": results,
        "plain_mlups": baseline,
        "selected_mlups": selected_median,
        "selected_over_best_sampled": selected_median / max(best, selected_median),
        "scope": "quick pipeline check" if args.samples < 10000 else "full new sampling",
        "envelope": envelope,
        "attempted": len(results),
        "successful": len(valid),
    }


def run_profile(args):
    context = gpu_context(args)
    ncu = str(args.ncu.resolve()) if args.ncu is not None else shutil.which("ncu")
    if ncu is None:
        return {"status": "unsupported", "reason": "Nsight Compute ncu is not on PATH"}
    report = tune(args, args.output / "tuning", 27)
    tile, depth = report["selection"]["B"], report["selection"]["I"]
    if estimated_bytes(27, args.domain, "plain", tile, False) >= 0.9 * context["available_memory_bytes_at_start"]:
        return {
            "status": "unsupported-memory",
            "domain": args.domain,
            "note": "Use an explicitly labelled smaller --domain for new validation",
        }
    initial = args.output / "input"
    make_input(initial, args.domain, "closed")
    assert report["device"] is not None
    multiplier = int(report["device"]["l2_capacity_bytes"]) // (108 * math.prod(tile)) + 1
    cases = (
        ("plain", "plain", tile, []),
        ("on", "ring", tile, []),
        ("off", "ring", tile, ["--no-persistent-l2"]),
        ("over", "ring", [tile[0], tile[1] * multiplier, tile[2]], ["--no-persistent-l2"]),
    )
    records = []
    for label, mode, block, extra in cases:
        folder = args.output / label
        timing = [
            timed_process(
                args, folder / f"timing-{process}", 27, args.domain, mode, block, depth,
                initial, extra, math.prod(args.domain),
            )
            for process in range(5)
        ]
        counters = []
        for repeat in range(3):
            target = folder / f"ncu-{repeat}"
            command = solver_command(
                args, 27, args.domain, mode, block, depth, 2 * depth, 2 * depth,
                initial, target, False, extra,
            )
            profile = [
                ncu, "--target-processes", "all", "--csv",
                "--replay-mode", "application", "--cache-control", "none",
                "--app-replay-match", "grid", "--app-replay-mode", "strict",
                "--kernel-name-base", "function", "--kernel-name", "regex:stream_collide_kernel",
                "--metrics", NCU_METRICS, "--log-file", str(target / "metrics.csv"),
                *command,
            ]
            record, stdout, stderr = execute(
                profile,
                target,
                {
                    "case": label,
                    "actual_steps": 2 * depth,
                    "useful_lup": math.prod(args.domain) * 2 * depth,
                    "metrics": NCU_METRICS,
                },
                dict(os.environ),
            )
            profiler_log = (target / "metrics.csv").read_text(encoding="utf-8")
            if "ERR_NVGPUCTRPERM" in stdout + stderr + profiler_log:
                return {
                    "status": "unsupported",
                    "reason": "GPU counter permission denied",
                    "diagnostic": str(target.relative_to(ROOT)),
                }
            assert record["returncode"] == 0, record
            record["counters"] = aggregate_ncu(target / "metrics.csv", math.prod(args.domain) * 2 * depth)
            counters.append(record["counters"])
            records.append(record)
        write_json(
            folder / "timing-summary.json",
            {
                "processes": timing,
                "median_mlups": statistics.median(item["process_median_mlups"] for item in timing),
                "median_hit_rate": statistics.median(item["hit_rate"] for item in counters),
                "median_dram_sectors_per_lup": statistics.median(item["dram_sectors_per_lup"] for item in counters),
            },
        )
    return {
        "status": "completed",
        "profiles": records,
        "note": "NCU counters and independent non-NCU timing are stored separately",
    }


def aggregate_ncu(path, useful_lup):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith('"ID"'))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    names = [
        "dram__sectors.sum",
        "lts__t_requests_aperture_device_lookup_hit.sum",
        "lts__t_requests_aperture_device_lookup_miss.sum",
    ]
    totals = {
        name: sum(float(row["Metric Value"].replace(",", "")) for row in rows if row["Metric Name"] == name)
        for name in names
    }
    hit, miss = totals[names[1]], totals[names[2]]
    assert useful_lup > 0 and hit + miss > 0
    return {
        "metric_sums": totals,
        "useful_lup": useful_lup,
        "hit_rate": hit / (hit + miss),
        "dram_sectors_per_lup": totals[names[0]] / useful_lup,
    }


def run_build_fluidx3d(args):
    upstream = ROOT / "code/third_party/fluidx3d/qualification_upstream"
    builds = []
    for q in args.q:
        target = args.build_dir / f"q{q}"
        shutil.copytree(upstream / "src", target / "src", dirs_exist_ok=True)
        defines = target / "src/defines.hpp"
        text = defines.read_text(encoding="utf-8")
        text = re.sub(
            r"^(//)?#define (D2Q9|D3Q15|D3Q19|D3Q27|FP16S|FP16C)\b", r"//#define \2", text, flags=re.MULTILINE,
        )
        text = text.replace(f"//#define D3Q{q}", f"#define D3Q{q}", 1)
        defines.write_text(text, encoding="utf-8")
        shutil.copy2(ROOT / "scripts/fluidx3d_setup.cpp", target / "src/setup.cpp")
        sources = sorted(str(path) for path in (target / "src").glob("*.cpp"))
        command = [
            args.compiler, "-std=c++17", "-O3", "-pthread", "-Wno-comment",
            "-I" + str(target / "src/OpenCL/include"),
            *sources,
            "-L" + str(target / "src/OpenCL/lib"), "-lOpenCL", "-o", str(target / "FluidX3D"),
        ]
        successful(
            command,
            args.output / f"q{q}",
            {
                "q": q,
                "source": "d8c571f",
                "driver_sha256": sha256(ROOT / "scripts/fluidx3d_setup.cpp"),
            },
            dict(os.environ),
        )
        builds.append({"q": q, "binary": str(target / "FluidX3D"), "sha256": sha256(target / "FluidX3D")})
    return {"status": "passed", "builds": builds}


def parser():
    app = argparse.ArgumentParser(description=__doc__)
    commands = app.add_subparsers(dest="command", required=True)
    handlers = {
        "reference": run_reference,
        "build": run_build,
        "smoke": run_smoke,
        "correctness": run_correctness,
        "calibrate": run_calibrate,
        "performance": run_performance,
        "selection": run_selection,
        "profile": run_profile,
        "build-fluidx3d": run_build_fluidx3d,
        "bridge": run_bridge,
    }
    for name, handler in handlers.items():
        sub = commands.add_parser(name)
        sub.set_defaults(handler=handler)
        sub.add_argument("--output", type=Path, required=True)
        if name not in ("reference", "build", "build-fluidx3d"):
            sub.add_argument("--device", type=int, required=True)
            sub.add_argument("--binary-dir", type=Path, default=ROOT / "build/bin")
            sub.add_argument("--plain-block", nargs=3, type=int, default=(32, 2, 2))
            sub.add_argument("--inv-tau", type=float, default=0.5)
        if name in ("smoke", "correctness"):
            sub.add_argument("--keep-ddf", action="store_true")
        if name == "smoke":
            sub.add_argument("--steps", type=int, default=1500)
            sub.set_defaults(inv_tau=1.6)
        if name in ("performance", "correctness"):
            sub.add_argument("--quick", action="store_true")
            sub.add_argument("--q", nargs="+", type=int, choices=(15, 19, 27), default=[15, 19, 27])
            sub.add_argument(
                "--sizes", nargs="+", type=int,
                default=[128, 256, 384] if name == "correctness" else [128, 256, 384, 512, 640],
            )
        if name == "correctness":
            sub.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 5, 6, 9])
            sub.add_argument(
                "--historical", action="store_true",
                help="Also test the two historically failing launch/domain geometries",
            )
        if name == "performance":
            sub.add_argument("--processes", type=int, default=5)
            sub.add_argument("--fluidx3d-dir", type=Path)
            sub.add_argument("--opencl-device", type=int)
        if name == "bridge":
            sub.add_argument("--master-binary", type=Path, required=True)
            sub.add_argument("--keep-fields", action="store_true")
        if name in ("calibrate", "selection"):
            sub.add_argument("--q", type=int, choices=(15, 19, 27), default=27)
        if name == "selection":
            sub.add_argument("--samples", type=int, default=10000)
            sub.add_argument("--seed", type=int, default=20261110)
            sub.add_argument("--processes", type=int, default=1)
        if name == "profile":
            sub.add_argument("--domain", nargs=3, type=int, default=[512, 512, 512])
            sub.add_argument(
                "--ncu", type=Path, help="Nsight Compute executable; otherwise resolve ncu from PATH",
            )
        if name == "build":
            sub.add_argument("--build-dir", type=Path, default=ROOT / "build")
            sub.add_argument("--toolchain-prefix", type=Path)
            sub.add_argument("--arch", default="native")
            sub.add_argument("--jobs", type=int, default=2)
        if name == "build-fluidx3d":
            sub.add_argument("--q", nargs="+", type=int, choices=(15, 19, 27), default=[15, 19, 27])
            sub.add_argument("--build-dir", type=Path, default=ROOT / "build/fluidx3d")
            sub.add_argument("--compiler", default="g++")
    return app


def main():
    args = parser().parse_args()
    if args.command in ("correctness", "performance") and args.quick:
        args.q, args.sizes = [27], [128]
        if args.command == "correctness":
            args.depths = [1, 3]
    args.output = args.output.resolve()
    if args.command not in ("reference", "build", "build-fluidx3d"):
        args.binary_dir = args.binary_dir.resolve()
    if args.command in ("build", "build-fluidx3d"):
        args.build_dir = args.build_dir.resolve()
    if args.command == "performance" and args.fluidx3d_dir is not None:
        args.fluidx3d_dir = args.fluidx3d_dir.resolve()
    if args.command == "build" and args.toolchain_prefix is not None:
        args.toolchain_prefix = args.toolchain_prefix.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    result = args.handler(args)
    result["schema_version"] = 2
    result["model_id"] = MODEL_ID
    write_json(args.output / "summary.json", result)
    print(f"Completed {args.command}. Results: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
