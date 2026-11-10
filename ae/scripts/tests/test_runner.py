import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ae import (
    aggregate_ncu,
    correctness_case,
    estimated_bytes,
    estimated_host_bytes,
    host_available_bytes,
    run_build,
    solver_command,
    timed_process,
)
from checks import (
    BOUNCE,
    DUMP,
    EQUILIBRIUM,
    FLUID,
    LOAD,
    MODEL_ID,
    STORE,
    WALL,
    active_ddf_mask,
    compare_ddf,
    compare_macroscopic,
    compare_macroscopic_bits,
    ddf_activity,
    full_halo_domain,
    make_input,
    model_score,
    parse_windows,
    sample_candidates,
    sha256,
    stencil_directions,
)


def independent_directions(q):
    # Literal Cartesian order independently checked against core/stencil.hpp.
    full = (
        (-1, -1, -1),
        (0, -1, -1),
        (1, -1, -1),
        (-1, 0, -1),
        (0, 0, -1),
        (1, 0, -1),
        (-1, 1, -1),
        (0, 1, -1),
        (1, 1, -1),
        (-1, -1, 0),
        (0, -1, 0),
        (1, -1, 0),
        (-1, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
        (-1, 1, 0),
        (0, 1, 0),
        (1, 1, 0),
        (-1, -1, 1),
        (0, -1, 1),
        (1, -1, 1),
        (-1, 0, 1),
        (0, 0, 1),
        (1, 0, 1),
        (-1, 1, 1),
        (0, 1, 1),
        (1, 1, 1),
    )
    indices = {
        15: (0, 2, 4, 6, 8, 10, 12, 13, 14, 16, 18, 20, 22, 24, 26),
        19: (1, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 25),
        27: tuple(range(27)),
    }
    return tuple(full[index] for index in indices[q])


def population_dependencies(flags, directions, mapping):
    """Independent graph of LOAD, EQU, BOUNCE, COLLIDE, STORE in kernel.cu."""
    shape = flags.shape
    q = len(directions)
    nodes = np.arange(q * flags.size).reshape(q, *shape)
    parents = [() for _ in range(nodes.size)]
    for position in np.ndindex(shape):
        incoming = []
        reflected = []
        for direction, velocity in enumerate(directions):
            pull = []
            bounce = []
            for axis, delta in enumerate(reversed(velocity)):
                low, high = position[axis] - delta, position[axis] + delta
                if mapping == "wrap":
                    pull.append(low % shape[axis])
                    bounce.append(high % shape[axis])
                else:
                    pull.append(min(max(low, 0), shape[axis] - 1))
                    bounce.append(min(max(high, 0), shape[axis] - 1))
            incoming.append(int(nodes[(direction, *pull)]))
            reflected.append(int(nodes[(q - 1 - direction, *bounce)]))
        for direction in range(q):
            node = int(nodes[(direction, *position)])
            if flags[position] == FLUID:
                parents[node] = tuple(incoming)
            elif flags[position] == WALL:
                parents[node] = (reflected[direction],)
            else:
                assert flags[position] == EQUILIBRIUM
    return parents, nodes[:, flags == FLUID].reshape(-1)


class RunnerContracts(unittest.TestCase):
    def test_default_build_preserves_generic_toolchain_discovery(self):
        args = SimpleNamespace(
            toolchain_prefix=None,
            build_dir=Path("build"),
            output=Path("results/build"),
            arch="native",
            jobs=2,
        )
        environment = {
            "CXX": "/host/bin/c++",
            "CUDACXX": "/cuda/bin/nvcc",
            "CUDAHOSTCXX": "/host/bin/c++",
            "PATH": "/host/bin:/cuda/bin",
        }
        with patch("ae.successful") as execute, patch.dict("ae.os.environ", environment):
            result = run_build(args)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(execute.call_count, 2)
        configure = execute.call_args_list[0].args
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=native", configure[0])
        for option in ("PREFIX_PATH", "CXX_COMPILER", "CUDA_COMPILER", "CUDA_HOST_COMPILER"):
            self.assertFalse(any(part.startswith(f"-DCMAKE_{option}=") for part in configure[0]))
        for key, value in environment.items():
            self.assertEqual(configure[3][key], value)

    def test_case_specific_macroscopic_budget(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "old").mkdir()
            (root / "new").mkdir()
            for field in ("rho", "vx", "vy", "vz"):
                values = np.full(64, 1.0 if field == "rho" else 0.1, dtype="<f4")
                values.tofile(root / f"old/{field}_1500.dat")
                values.tofile(root / f"new/{field}_1500.dat")
            budgets = {"density_abs": 5e-5, "velocity_abs": 5e-6, "velocity_relative_l2": 1e-3}
            result = compare_macroscopic(root / "old", root / "new", 1500, (4, 4, 4), budgets)
            self.assertEqual(result["status"], "passed")
            np.full(64, 1.01, dtype="<f4").tofile(root / "new/rho_1500.dat")
            result = compare_macroscopic(root / "old", root / "new", 1500, (4, 4, 4), budgets)
            self.assertEqual(result["status"], "failed")

    def test_failed_selection_launch_is_retained(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            binary = root / "culbm-single"
            binary.write_text("#!/bin/sh\necho 'too many resources requested' >&2\nexit 1\n")
            binary.chmod(0o755)
            args = SimpleNamespace(
                binary_dir=root,
                device=0,
                inv_tau=0.5,
                plain_block=(32, 2, 2),
                command="selection",
            )
            record = timed_process(
                args, root / "run", 27, (128, 128, 128), "ring", (1024, 40, 40), 3, None, [], 128**3
            )
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["returncode"], 1)
            self.assertIn("too many resources", (root / "run/stderr.log").read_text())

    def test_ncu_uses_summed_requests(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "metrics.csv"
            rows = ['"ID","Metric Name","Metric Value"']
            for metric, values in (
                ("dram__sectors.sum", [1000, 2000]),
                ("lts__t_requests_aperture_device_lookup_hit.sum", [90, 20]),
                ("lts__t_requests_aperture_device_lookup_miss.sum", [10, 180]),
            ):
                rows.extend(f'"{index}","{metric}","{value}"' for index, value in enumerate(values))
            path.write_text("\n".join(rows))
            result = aggregate_ncu(path, 100)
            self.assertAlmostEqual(result["hit_rate"], 110 / 300)
            self.assertEqual(result["dram_sectors_per_lup"], 30)

    def test_nonzero_edge_service_cost_score(self):
        score = model_score((96, 71, 70), 5, 0.20792, 1.0088)
        self.assertAlmostEqual(score, 2.0879221358116813, places=12)
        self.assertLess(model_score((96, 71, 70), 5, 0.20792, 2.0088), score)
        self.assertEqual(
            model_score((96, 82, 74), 9, 0.18668, 1.0), model_score((96, 82, 74), 9, 0.18668, 20.0)
        )
        self.assertEqual(MODEL_ID, "calibrated_service_cost_v1")

    def test_candidate_counts_and_replay(self):
        expected = [(49.5, 998178), (60, 1307805), (66, 1494509), (80, 1948562)]
        for mib, count in expected:
            total, rows = sample_candidates(27, int(mib * 1024**2), 32, 1234)
            self.assertEqual(total, count)
            self.assertEqual(len({row["candidate_rank"] for row in rows}), 32)
            self.assertEqual((total, rows), sample_candidates(27, int(mib * 1024**2), 32, 1234))
            for row in rows:
                bx, by, bz = row["B"]
                self.assertTrue(by >= bz > 2 * (row["I"] - 1))
                self.assertLessEqual(108 * bx * by * bz, int(mib * 1024**2))

    def test_full_halo_denominator(self):
        tile, depth, capacity = (96, 82, 74), 9, 96 * 1024**2
        domain, cells = full_halo_domain(27, tile, depth, capacity)
        counts = [
            2 + (n - 2 * (b + 1 - depth)) // (b + 2 - 2 * depth)
            for n, b in zip(domain, tile, strict=True)
        ]
        actual = np.prod([count - 2 for count in counts]) * np.prod([b - 16 for b in tile])
        self.assertEqual(cells, actual)
        self.assertGreater(cells * 108, 2 * capacity)

    def test_fluid_and_consumed_boundary_bit_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            make_input(root / "input", (4, 4, 4), "closed")
            for q in (15, 19, 27):
                left = np.zeros((q, 4, 4, 4), dtype="<u4")
                left.tofile(root / "left")
                plus_x = independent_directions(q).index((1, 0, 0))
                for coordinate in ((q // 2, 1, 1, 1), (plus_x, 1, 1, 0)):
                    right = left.copy()
                    right[coordinate] = 1
                    right.tofile(root / "right")
                    with self.assertRaisesRegex(AssertionError, "'scope': 'active-ddf'"):
                        compare_ddf(
                            root / "left", root / "right", q, (4, 4, 4), root / "input/flag.dat"
                        )
            # A D3Q27 corner body diagonal enters fluid, unlike a face-only rule.
            left = np.zeros((27, 4, 4, 4), dtype="<u4")
            left.tofile(root / "left")
            left[26, 0, 0, 0] = 1
            left.tofile(root / "right")
            with self.assertRaisesRegex(AssertionError, "'direction': 26"):
                compare_ddf(root / "left", root / "right", 27, (4, 4, 4), root / "input/flag.dat")

    def test_unused_boundary_bits_are_reported_without_changing_dumps(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            make_input(root / "input", (4, 4, 4), "closed")
            for q in (15, 19, 27):
                left = np.zeros((q, 4, 4, 4), dtype="<u4")
                left.tofile(root / "left")
                right = left.copy()
                minus_x = independent_directions(q).index((-1, 0, 0))
                right[minus_x, 1, 1, 0] = 1
                right.tofile(root / "right")
                before = (sha256(root / "left"), sha256(root / "right"))
                result = compare_ddf(
                    root / "left", root / "right", q, (4, 4, 4), root / "input/flag.dat"
                )
                self.assertEqual(result["status"], "passed")
                self.assertFalse(result["raw_equal"])
                self.assertEqual(result["inactive_difference_count"], 1)
                self.assertEqual(result["first_unused_difference"]["kind"], "outward-domain")
                self.assertEqual(result["inactive_difference_counts_by_kind"]["outward-domain"], 1)
                self.assertEqual(
                    result["active_population_count"] + result["inactive_population_count"], q * 64
                )
                self.assertEqual(result["raw_sha256"]["reference"], before[0])
                self.assertEqual(result["raw_sha256"]["actual"], before[1])
                self.assertEqual(before, (sha256(root / "left"), sha256(root / "right")))
                # Rest populations on solid cells are unused but are not outward.
                right = left.copy()
                right[q // 2, 1, 1, 0] = 1
                right.tofile(root / "right")
                result = compare_ddf(
                    root / "left", root / "right", q, (4, 4, 4), root / "input/flag.dat"
                )
                self.assertEqual(result["first_unused_difference"]["kind"], "other-boundary")
                self.assertEqual(result["inactive_difference_counts_by_kind"]["other-boundary"], 1)

    def test_macroscopic_bits_must_match_at_each_snapshot(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for directory in ("left", "right"):
                (root / directory).mkdir()
                for step in (3, 4):
                    for field in ("rho", "vx", "vy", "vz"):
                        np.zeros(64, dtype="<u4").tofile(root / directory / f"{field}_{step}.dat")
            for step in (3, 4):
                result = compare_macroscopic_bits(root / "left", root / "right", step, (4, 4, 4))
                self.assertEqual(result["status"], "passed")
                for field in ("rho", "vx", "vy", "vz"):
                    path = root / "right" / f"{field}_{step}.dat"
                    bits = np.zeros(64, dtype="<u4")
                    bits[63] = 1
                    bits.tofile(path)
                    with self.assertRaisesRegex(AssertionError, f"'field': '{field}'"):
                        compare_macroscopic_bits(root / "left", root / "right", step, (4, 4, 4))
                    bits[63] = 0
                    bits.tofile(path)

    def test_correctness_case_rejects_macro_mismatch_with_equal_ddf(self):
        def solver_outputs(command, output, metadata, environment):
            output.mkdir()
            for step in (2, 4):
                np.zeros(15 * 64, dtype="<u4").tofile(output / f"ddf_{step}.dat")
                for field in ("rho", "vx", "vy", "vz"):
                    bits = np.zeros(64, dtype="<u4")
                    bits[1] = int(metadata["mode"] == "twist" and field == "vx")
                    bits.tofile(output / f"{field}_{step}.dat")

        with tempfile.TemporaryDirectory() as name:
            args = SimpleNamespace(
                binary_dir=Path("build/bin"),
                device=0,
                inv_tau=0.5,
                plain_block=(32, 2, 2),
                keep_ddf=True,
            )
            with patch("ae.successful", side_effect=solver_outputs):
                with self.assertRaisesRegex(AssertionError, "'field': 'vx'"):
                    correctness_case(
                        args,
                        {
                            "available_memory_bytes_at_start": 1024**3,
                            "available_host_memory_bytes_at_start": 1024**3,
                        },
                        Path(name),
                        15,
                        (4, 4, 4),
                        1,
                        4,
                        "closed",
                        (32, 8, 8),
                    )

    def test_upstream_full_snapshot_memory_is_counted(self):
        domain, tile = (128, 96, 64), (64, 48, 31)
        cells = 128 * 96 * 64
        for q in (15, 19, 27):
            for mode in ("plain", "twist", "double", "ring"):
                without_dump = estimated_bytes(q, domain, mode, tile, False)
                with_dump = estimated_bytes(q, domain, mode, tile, True)
                self.assertEqual(with_dump - without_dump, 4 * q * cells)
            self.assertEqual(estimated_host_bytes(q, domain, True), 4 * (q + 1 + 5) * cells)
            self.assertEqual(estimated_host_bytes(q, domain, False), 20 * cells)
        self.assertGreater(estimated_bytes(19, (512, 512, 512), "plain", tile, True), 24 * 1024**3)
        self.assertLess(
            estimated_bytes(27, (384, 384, 384), "plain", tile, True), 0.9 * 24 * 1024**3
        )

    def test_device_and_host_preflight_skip_without_substituting_cases(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for q, domain, gpu_bytes, host_bytes, resource in (
                (19, (512, 512, 512), 24 * 1024**3, 64 * 1024**3, "device-memory"),
                (27, (128, 128, 128), 24 * 1024**3, 1024**2, "host-memory"),
            ):
                with patch("ae.make_input") as generate, patch("ae.successful") as execute:
                    result = correctness_case(
                        SimpleNamespace(),
                        {
                            "available_memory_bytes_at_start": gpu_bytes,
                            "available_host_memory_bytes_at_start": host_bytes,
                        },
                        root / resource,
                        q,
                        domain,
                        9,
                        18,
                        "paper-correctness",
                        (96, 99, 87),
                    )
                    generate.assert_not_called()
                    execute.assert_not_called()
                self.assertEqual(result["status"], "unsupported-memory")
                self.assertEqual(result["unsupported_resources"], [resource])
                self.assertEqual(result["domain"], list(domain))
                self.assertEqual(
                    result["memory_preflight"]["device_ddf_snapshot_bytes"],
                    4 * q * np.prod(domain),
                )
                self.assertEqual(
                    result["memory_preflight"]["host_macro_snapshot_bytes"], 4 * np.prod(domain)
                )

    def test_memavailable_uses_kibibytes_not_memfree(self):
        with patch("ae.Path.read_text", return_value="MemFree: 1 kB\nMemAvailable: 123456 kB\n"):
            self.assertEqual(host_available_bytes(), 123456 * 1024)

    def test_stencil_order_and_boundary_dependency_closure(self):
        self.assertEqual((LOAD, BOUNCE, STORE, DUMP), (1, 8, 16, 15 << 27))
        for q in (15, 19, 27):
            directions = independent_directions(q)
            self.assertEqual(stencil_directions(q), directions)
            self.assertEqual(directions[q // 2], (0, 0, 0))
            for index, velocity in enumerate(directions):
                self.assertEqual(directions[q - 1 - index], tuple(-axis for axis in velocity))
            for equilibrium_face in (False, True):
                flags = np.full((6, 7, 8), WALL, dtype="<u4")
                flags[1:-1, 1:-1, 1:-1] = FLUID
                flags[1:4, 2:5, 2:5] = WALL
                if equilibrium_face:
                    flags[1:-1, 1:-1, 0] = EQUILIBRIUM
                fluid, standard = ddf_activity(flags)
                self.assertTrue(standard)
                masks = np.stack(
                    [active_ddf_mask(fluid, velocity, standard) for velocity in directions]
                )
                for mapping in ("wrap", "clamp"):
                    parents, roots = population_dependencies(flags, directions, mapping)
                    closure = set(int(node) for node in roots)
                    pending = list(closure)
                    while pending:
                        for parent in parents[pending.pop()]:
                            if parent not in closure:
                                closure.add(parent)
                                pending.append(parent)
                    self.assertEqual(set(np.flatnonzero(masks)), closure)
                    # Simultaneously perturb every excluded slot and propagate
                    # through nine updates, exceeding wall reflection cycles.
                    perturbations = ~masks.reshape(-1)
                    for _ in range(9):
                        perturbations = np.array(
                            [
                                any(perturbations[parent] for parent in sources)
                                for sources in parents
                            ]
                        )
                        self.assertFalse(perturbations[roots].any())

    def test_unproven_flag_layouts_require_all_raw_populations(self):
        flags = np.full((4, 4, 4), WALL, dtype="<u4")
        flags[1:-1, 1:-1, 1:-1] = FLUID
        for position, replacement in (((1, 1, 0), FLUID), ((1, 0, 1), WALL | DUMP)):
            changed = flags.copy()
            changed[position] = replacement
            fluid, standard = ddf_activity(changed)
            self.assertFalse(standard)
            self.assertTrue(active_ddf_mask(fluid, (-1, 0, 0), standard).all())

    def test_launch_contract_and_exact_steps(self):
        args = SimpleNamespace(
            binary_dir=Path("build/bin"),
            device=0,
            inv_tau=0.5,
            plain_block=(32, 2, 2),
        )
        command = solver_command(
            args,
            27,
            (128, 128, 128),
            "ring",
            (96, 82, 74),
            9,
            90,
            540,
            None,
            Path("results"),
            False,
            [],
        )
        self.assertEqual(command[command.index("--block") + 1], "[96,1,1]")
        self.assertEqual(command[command.index("--grid") + 1], "[1,82,74]")
        self.assertNotIn("--tile-allocation", command)
        plain = solver_command(
            args,
            27,
            (157, 67, 61),
            "plain",
            (96, 82, 74),
            1,
            90,
            540,
            None,
            Path("results"),
            False,
            [],
        )
        self.assertEqual(plain[plain.index("--block") + 1], "[1,1,1]")
        self.assertEqual(plain[plain.index("--grid") + 1], "[157,67,61]")
        with self.assertRaises(AssertionError):
            solver_command(
                args,
                27,
                (128, 128, 128),
                "ring",
                (96, 82, 74),
                9,
                100,
                600,
                None,
                Path("results"),
                False,
                [],
            )

    def test_window_count(self):
        stdout = "\n".join(f"speed = {i + 1:.4f} (MLUPS)" for i in range(6))
        self.assertEqual(parse_windows(stdout, 6), [1, 2, 3, 4, 5, 6])
        with self.assertRaises(AssertionError):
            parse_windows(stdout, 5)


if __name__ == "__main__":
    unittest.main()
