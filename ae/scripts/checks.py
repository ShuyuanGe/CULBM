"""Input and output contracts shared by the artifact commands."""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

MODEL_ID = "calibrated_service_cost_v1"

LOAD, EQU, COLLIDE, BOUNCE, STORE = (1 << bit for bit in range(5))
DUMP = sum(1 << bit for bit in range(27, 31))
FLUID = LOAD | COLLIDE | STORE | DUMP
WALL = LOAD | BOUNCE | STORE
EQUILIBRIUM = EQU | STORE


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_input(folder: Path, domain, kind: str):
    nx, ny, nz = domain
    assert min(domain) >= 3
    folder.mkdir(parents=True, exist_ok=True)
    shape = (nz, ny, nx)
    flags = np.memmap(folder / "flag.dat", dtype="<u4", mode="w+", shape=shape)
    flags[:] = FLUID
    flags[:, :, (0, -1)] = WALL
    flags[:, (0, -1), :] = WALL
    flags[(0, -1), :, :] = WALL
    if kind in ("paper-correctness", "channel"):
        flags[:, :, (0, -1)] = EQUILIBRIUM
        flags[:, (0, -1), :] = WALL
        flags[(0, -1), :, :] = WALL
    if kind == "channel":
        assert tuple(domain) == (160, 64, 64)
        flags[24:40, 24:40, 48:64] = WALL
    assert kind in ("paper-correctness", "channel", "closed")
    flags.flush()
    for field in ("rho", "vx", "vy", "vz"):
        values = np.memmap(folder / f"{field}.dat", dtype="<f4", mode="w+", shape=shape)
        values[:] = float(field == "rho")
        if field == "vx" and kind in ("paper-correctness", "channel"):
            values[1:-1, 1:-1, 0] = {"paper-correctness": 0.005, "channel": 0.04}[kind]
        values.flush()
    record = {
        "domain_xyz": list(domain),
        "array_shape_zyx": list(shape),
        "kind": kind,
        "dtype_fields": "<f4",
        "dtype_flags": "<u4",
        "files": {path.name: sha256(path) for path in sorted(folder.glob("*.dat"))},
    }
    write_json(folder / "input.json", record)
    return record


def stencil_directions(q: int):
    """Match core/stencil.hpp: x changes fastest, with opposite index Q-1-q."""
    assert q in (15, 19, 27)
    return tuple(
        (x, y, z)
        for z in (-1, 0, 1)
        for y in (-1, 0, 1)
        for x in (-1, 0, 1)
        if q == 27
        or (q == 19 and (x != 0) + (y != 0) + (z != 0) <= 2)
        or (q == 15 and (x != 0) + (y != 0) + (z != 0) != 2)
    )


def ddf_activity(flags):
    """Select the proven boundary rule, otherwise require the complete raw state."""
    fluid_ops = LOAD | COLLIDE | STORE
    fluid = (flags & (LOAD | EQU | COLLIDE | BOUNCE | STORE)) == fluid_ops
    standard_fluid = (flags & (0xFFFFFFFF ^ DUMP)) == fluid_ops
    standard = np.all(standard_fluid | (flags == WALL) | (flags == EQUILIBRIUM))
    exterior_fluid = fluid[(0, -1), :, :].any() or fluid[:, (0, -1), :].any() or fluid[:, :, (0, -1)].any()
    return fluid, bool(standard and not exterior_fluid)


def active_ddf_mask(fluid, direction, standard_boundaries: bool):
    if standard_boundaries:
        dx, dy, dz = direction
        # Post-collision f_q(x) feeds q at x+e_q. Fluid mixes all inputs;
        # pure bounce-back returns other wall links to their source, while
        # prescribed equilibrium overwrites its inputs. Exterior fluid is excluded
        # before this rule is used, so it also covers the ring's clamped reads.
        return fluid | np.roll(fluid, shift=(-dz, -dy, -dx), axis=(0, 1, 2))
    return np.ones_like(fluid, dtype=bool)


def outward_ddf_mask(shape, direction):
    outward = np.zeros(shape, dtype=bool)
    for axis, delta in enumerate(reversed(direction)):
        if delta:
            face = [slice(None)] * 3
            face[axis] = 0 if delta < 0 else -1
            outward[tuple(face)] = True
    return outward


def bit_difference(left, right, index: int, domain):
    nx, ny, _ = domain
    z, remainder = divmod(index, nx * ny)
    y, x = divmod(remainder, nx)
    return {
        "coordinate_zyx": [z, y, x],
        "reference_bits": hex(int(left[index])),
        "actual_bits": hex(int(right[index])),
    }


def compare_ddf(reference: Path, actual: Path, q: int, domain, flags: Path):
    nx, ny, nz = domain
    cells = nx * ny * nz
    shape = (nz, ny, nx)
    expected_bytes = q * cells * 4
    assert reference.stat().st_size == actual.stat().st_size == expected_bytes
    assert flags.stat().st_size == cells * 4
    left = np.memmap(reference, dtype="<u4", mode="r", shape=(q, cells))
    right = np.memmap(actual, dtype="<u4", mode="r", shape=(q, cells))
    flag = np.memmap(flags, dtype="<u4", mode="r", shape=shape)
    fluid, standard_boundaries = ddf_activity(flag)
    active_count = inactive_count = inactive_differences = 0
    inactive_nonfinite = {"reference": 0, "actual": 0}
    unused_differences = {"outward-domain": 0, "other-boundary": 0}
    first_unused = None
    for direction, velocity in enumerate(stencil_directions(q)):
        active = active_ddf_mask(fluid, velocity, standard_boundaries).reshape(-1)
        active_count += int(np.count_nonzero(active))
        inactive_count += cells - int(np.count_nonzero(active))
        changed = left[direction] != right[direction]
        relevant = changed & active
        if relevant.any():
            index = int(np.argmax(relevant))
            raise AssertionError(
                {
                    "scope": "active-ddf",
                    "direction": direction,
                    "velocity_xyz": list(velocity),
                    "flag": hex(int(flag.reshape(-1)[index])),
                    **bit_difference(left[direction], right[direction], index, domain),
                }
            )
        for label, values in (("reference", left[direction]), ("actual", right[direction])):
            nonfinite = ~np.isfinite(values.view("<f4"))
            assert not (nonfinite & active).any(), ("nonfinite-active-ddf", label, direction)
            inactive_nonfinite[label] += int(np.count_nonzero(nonfinite & ~active))
        unused = changed & ~active
        count = int(np.count_nonzero(unused))
        inactive_differences += count
        if count:
            outward = outward_ddf_mask(shape, velocity).reshape(-1)
            outward_count = int(np.count_nonzero(unused & outward))
            unused_differences["outward-domain"] += outward_count
            unused_differences["other-boundary"] += count - outward_count
            if first_unused is None:
                index = int(np.argmax(unused))
                first_unused = {
                    "direction": direction,
                    "velocity_xyz": list(velocity),
                    "kind": "outward-domain" if outward[index] else "other-boundary",
                    "flag": hex(int(flag.reshape(-1)[index])),
                    **bit_difference(left[direction], right[direction], index, domain),
                }
    return {
        "status": "passed",
        "scope": "active-ddf-bitwise",
        "classification": "standard-bounce-equilibrium" if standard_boundaries else "conservative-full-state",
        "bytes": expected_bytes,
        "active_population_count": active_count,
        "inactive_population_count": inactive_count,
        "active_difference_count": 0,
        "inactive_difference_count": inactive_differences,
        "inactive_difference_counts_by_kind": unused_differences,
        "inactive_nonfinite_count": inactive_nonfinite,
        "raw_equal": inactive_differences == 0,
        "raw_sha256": {"reference": sha256(reference), "actual": sha256(actual)},
        "first_unused_difference": first_unused,
    }


def compare_macroscopic_bits(reference: Path, actual: Path, step: int, domain):
    cells = math.prod(domain)
    hashes = {}
    for field in ("rho", "vx", "vy", "vz"):
        left_path, right_path = reference / f"{field}_{step}.dat", actual / f"{field}_{step}.dat"
        assert left_path.stat().st_size == right_path.stat().st_size == 4 * cells
        left = np.memmap(left_path, dtype="<u4", mode="r", shape=(cells,))
        right = np.memmap(right_path, dtype="<u4", mode="r", shape=(cells,))
        changed = left != right
        if changed.any():
            raise AssertionError(
                {
                    "scope": "full-domain-macroscopic",
                    "field": field,
                    **bit_difference(left, right, int(np.argmax(changed)), domain),
                }
            )
        assert np.isfinite(left.view("<f4")).all(), ("nonfinite-macroscopic", field)
        hashes[field] = {"reference": sha256(left_path), "actual": sha256(right_path)}
    return {"status": "passed", "scope": "full-domain-bitwise", "field_sha256": hashes}


def parse_windows(stdout: str, count: int):
    values = [float(value) for value in re.findall(r"speed = ([0-9.eE+-]+) \(MLUPS\)", stdout)]
    assert len(values) == count, {"expected_windows": count, "observed": values}
    assert all(math.isfinite(value) and value > 0 for value in values), values
    return values


def sample_candidates(q: int, capacity_bytes: int, count: int, seed: int):
    # Choosing unique ranks over feasible Bz slices is uniform without materializing
    # millions of full candidate tuples.
    slices = []
    cumulative = []
    total = 0
    for depth in range(2, 21):
        low = 2 * (depth - 1) + 1
        for bx in range(32, 1025, 32):
            if bx < low:
                continue
            by_max = capacity_bytes // (4 * q * bx * low)
            for by in range(low, by_max + 1):
                high = min(by, capacity_bytes // (4 * q * bx * by))
                width = high - low + 1
                if width > 0:
                    total += width
                    slices.append((depth, bx, by, low))
                    cumulative.append(total)
    assert 0 < count <= total
    ranks = np.random.default_rng(seed).choice(total, size=count, replace=False)
    indices = np.searchsorted(cumulative, ranks, side="right")
    starts = (0, *cumulative)
    candidates = []
    for rank, index in zip(ranks, indices, strict=True):
        depth, bx, by, low = slices[int(index)]
        bz = low + int(rank) - starts[int(index)]
        candidates.append({"B": [bx, by, bz], "I": depth, "candidate_rank": int(rank)})
    return total, candidates


def full_halo_domain(q: int, tile, depth: int, total_l2_bytes: int):
    radius = depth - 1
    interior = [edge - 2 * radius for edge in tile]
    assert min(interior) > 0
    tiles = math.floor((2 * total_l2_bytes / (4 * q * math.prod(interior))) ** (1 / 3)) + 1
    while 4 * q * math.prod(interior) * tiles**3 <= 2 * total_l2_bytes:
        tiles += 1
    domain = [2 * (edge - radius) + tiles * length for edge, length in zip(tile, interior)]
    cells_per_step = math.prod(interior) * tiles**3
    assert 4 * q * cells_per_step > 2 * total_l2_bytes
    return domain, cells_per_step


def model_score(tile, depth: int, bandwidth_ratio: float, gamma: float):
    """Evaluate the calibrated service-cost score; global selection stays in C++."""
    bx = tile[0]
    radius = depth - 1
    length = bx - 2 * radius
    stride = math.gcd(length, 8)
    phases = 8 // stride
    sectors = edges = 0
    for phase in range(0, 8, stride):
        for warp in range(0, bx, 32):
            low, high = max(radius, warp), min(bx - radius, warp + 32)
            if high > low:
                sectors += (phase + high - 1) // 8 + 1 - (phase + low) // 8
        edges += int((phase + radius) % 8 != 0) + int((phase + bx - radius) % 8 != 0)
    beta_sectors = 8 * sectors / (phases * length)
    beta_edges = 4 * edges / (phases * length)
    halo_efficiency = math.prod(1 - 2 * radius / edge for edge in tile)
    alpha = 1 + bandwidth_ratio * (beta_sectors - 1) + (2 * gamma - (1 + bandwidth_ratio)) * beta_edges
    return halo_efficiency / ((1 + alpha * halo_efficiency) / (2 * depth) + bandwidth_ratio * (1 - 1 / depth))


def compare_macroscopic(reference: Path, actual: Path, step: int, domain, tolerances):
    cells = math.prod(domain)
    metrics = {}
    squared_error = squared_reference = 0.0
    hashes = {}
    for field in ("rho", "vx", "vy", "vz"):
        left_path, right_path = reference / f"{field}_{step}.dat", actual / f"{field}_{step}.dat"
        assert left_path.stat().st_size == right_path.stat().st_size == 4 * cells
        left = np.memmap(left_path, dtype="<f4", mode="r").astype(np.float64)
        right = np.memmap(right_path, dtype="<f4", mode="r").astype(np.float64)
        assert np.isfinite(left).all() and np.isfinite(right).all()
        delta = right - left
        metrics[field] = float(np.max(np.abs(delta)))
        hashes[field] = {"reference": sha256(left_path), "actual": sha256(right_path)}
        if field != "rho":
            squared_error += float(np.dot(delta, delta))
            squared_reference += float(np.dot(left, left))
    assert squared_reference > 0
    velocity_abs = max(metrics[field] for field in ("vx", "vy", "vz"))
    relative_l2 = math.sqrt(squared_error / squared_reference)
    passed = (
        metrics["rho"] <= tolerances["density_abs"]
        and velocity_abs <= tolerances["velocity_abs"]
        and relative_l2 <= tolerances["velocity_relative_l2"]
    )
    return {
        "status": "passed" if passed else "failed",
        "maximum_absolute_difference": metrics,
        "velocity_absolute_difference": velocity_abs,
        "velocity_relative_l2": relative_l2,
        "norm_region": "all physical domain cells",
        "tolerances": tolerances,
        "field_sha256": hashes,
    }
