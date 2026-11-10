import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("ae_package", Path(__file__).resolve().parents[1] / "package.py")
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


class ReviewerPackaging(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "artifact"
        self.output = self.base / "reviewer.tar.gz"

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def complete_fixture(self):
        for name in (
            "README.md",
            "pyproject.toml",
            "uv.lock",
            "paper/main.pdf",
            "paper/supplementary.pdf",
            "paper/SHA256SUMS",
            "paper/README.md",
            "data/refactor-validation/summary.json",
            "data/refactor-validation/inventory.json",
            "code/CMakeLists.txt",
        ):
            self.write(name, "final\n")
        freeze = {"model_id": PACKAGE.MODEL_ID, "source_commit": "frozen", "source_files": [{"path": "CMakeLists.txt"}]}
        self.write("data/provenance/freeze.json", json.dumps(freeze))

    def prebuilt_fixture(self):
        for name in PACKAGE.FILE_ALLOWLIST["prebuilt"]:
            self.write(f"prebuilt/{name}", "fallback\n")
        binaries = []
        for name in PACKAGE.FILE_ALLOWLIST["prebuilt/bin"]:
            path = self.write(f"prebuilt/bin/{name}", f"binary {name}\n")
            path.chmod(0o755)
            binaries.append({"path": f"bin/{name}", "sha256": PACKAGE.sha256(path), "bytes": path.stat().st_size})
        manifest = {"model_id": PACKAGE.MODEL_ID, "source_commit": "frozen", "binaries": binaries}
        self.write("prebuilt/manifest.json", json.dumps(manifest))
        return manifest

    def test_nested_allowlists_preserve_scientific_evidence(self):
        included = (
            "paper/source/main.tex",
            "paper/source/figures/d2q9_pull.pdf",
            "data/provenance/claims.json",
            "data/bridge/reference_protocol.json",
            "data/bridge/master_identity.json",
            "data/bridge/input/flag.dat",
            "data/bridge/master_source/src/kernel.cu",
            "code/README.md",
            "code/cmake/remove_rpath.cmake",
            "code/env/build.yml",
            "code/tools/tune/selector.cpp",
            "data/reference/baselines/qualification_5090/culbm_source_diff.patch",
            "data/reference/baselines/qualification_5090/historical_protocol.json",
            "data/reference/baselines/formal_20260717/local/source/build/raw.log",
            "data/reference/baselines/formal_20260717/local/source/input.dat",
            "data/refactor-validation/selection/q27/run.json",
            "data/refactor-validation/build/compile/stderr.log",
            "data/refactor-validation/environment/ncu-version.txt",
            "data/refactor-validation/geometry-q19-512/experiment.py",
            "data/refactor-validation/geometry-q27-128/experiment.py",
        )
        excluded = (
            "VALIDATION.md",
            "environment/README.md",
            "environment/build.yml",
            "environment/conda-linux-64.lock",
            "environment/profile.yml",
            "environment/profile-linux-64.lock",
            "environment/toolchain-packages.json",
            "CHANGES.md",
            "paper/CHANGES.md",
            "paper/ORIGINAL_SHA256SUMS",
            "paper/source/figures/old.pdf",
            "paper/build/main.log",
            "data/provenance/freeze-initial.json",
            "data/provenance/ae-changes.patch",
            "data/provenance/refactor-source-original.tar.gz",
            "data/bridge/refs.json",
            "data/bridge/commands/current-ring/run.json",
            "data/bridge/original_scripts/run.py",
            "data/reference/fig7/historical_figure.pdf",
            "data/reference/fig7/earlier_selected_measurements.json",
            "data/reference/calibration/repeatability/historical_provenance.json",
            "data/validation-history/native-default/summary.json",
            "data/refactor-validation/physical-smoke/summary.json",
            "data/refactor-validation/selection/input/flag.dat",
            "data/refactor-validation/environment/before.json",
            "data/refactor-validation/environment/after.json",
            "data/refactor-validation/environment/install.log",
            "data/refactor-validation/smoke/snapshot/summary.json",
            "data/refactor-validation/selection/comparison.pdf",
            "data/refactor-validation/correctness/experiment.py",
            "code/src/__pycache__/cached.pyc",
            "code/cmake/private.cmake",
            "scripts/author-journal.py",
            "prebuilt/bin/device_prop",
            "prebuilt/static-link.patch",
            "prebuilt/build/run.json",
        )
        for name in included + excluded:
            self.write(name, "fixture\n")
        executable = self.write("data/refactor-validation/model/executable.txt", "executable\n")
        executable.chmod(0o755)
        actual = {str(path.relative_to(self.root)) for path in PACKAGE.collect_files(self.root)}
        self.assertEqual(actual, set(included))

    def test_selected_symlink_is_rejected(self):
        source = self.write("outside.txt", "outside\n")
        link = self.root / "code/src/kernel.cu"
        link.parent.mkdir(parents=True)
        link.symlink_to(source)
        with self.assertRaises(AssertionError):
            PACKAGE.collect_files(self.root)

    def test_prebuilt_membership_integrity_and_executable_modes(self):
        self.complete_fixture()
        self.prebuilt_fixture()
        licenses = ("prebuilt/licenses/GPL.txt", "prebuilt/licenses/runtime/NOTICE.rst")
        for name in licenses:
            self.write(name, "license\n")
        for name in (
            "prebuilt/static-link.patch",
            "prebuilt/bin/device_prop",
            "prebuilt/build/compiler.log",
            "prebuilt/licenses/ignored.md",
            "prebuilt/licenses/build/ignored.txt",
        ):
            self.write(name, "excluded\n")
        with patch.object(PACKAGE.subprocess, "check_output", return_value="frozen\n"):
            PACKAGE.package_artifact(self.root, self.output, self.base, False)
        with tarfile.open(self.output) as archive:
            expected = {f"prebuilt/{name}" for name in PACKAGE.FILE_ALLOWLIST["prebuilt"]}
            expected |= {f"prebuilt/bin/{name}" for name in PACKAGE.FILE_ALLOWLIST["prebuilt/bin"]}
            expected |= set(licenses)
            actual = {name.removeprefix("l2lbm-ae/") for name in archive.getnames() if "/prebuilt/" in name}
            self.assertEqual(actual, expected)
            for name in PACKAGE.FILE_ALLOWLIST["prebuilt/bin"]:
                self.assertEqual(archive.getmember(f"l2lbm-ae/prebuilt/bin/{name}").mode, 0o755)

    def test_incomplete_prebuilt_fallback_is_rejected(self):
        self.complete_fixture()
        self.write("prebuilt/README.md", "incomplete\n")
        with patch.object(PACKAGE.subprocess, "check_output", return_value="frozen\n"):
            with self.assertRaisesRegex(AssertionError, "Incomplete prebuilt"):
                PACKAGE.package_artifact(self.root, self.output, self.base, True)
        self.assertFalse((self.root / "SHA256SUMS").exists())

    def test_prebuilt_stale_identity_or_bad_binary_is_rejected(self):
        self.complete_fixture()
        valid = self.prebuilt_fixture()
        for key, value, reason in (
            ("model_id", "obsolete", "model identity"),
            ("source_commit", "previous", "source commit"),
            ("binaries", valid["binaries"] + valid["binaries"][:1], "exactly"),
        ):
            manifest = json.loads(json.dumps(valid))
            manifest[key] = value
            self.write("prebuilt/manifest.json", json.dumps(manifest))
            with patch.object(PACKAGE.subprocess, "check_output", return_value="frozen\n"):
                with self.assertRaisesRegex(AssertionError, reason):
                    PACKAGE.package_artifact(self.root, self.output, self.base, True)
        for key, value, reason in (
            ("sha256", "0" * 64, "hash mismatch"),
            ("bytes", -1, "size mismatch"),
        ):
            manifest = json.loads(json.dumps(valid))
            manifest["binaries"][0][key] = value
            self.write("prebuilt/manifest.json", json.dumps(manifest))
            with patch.object(PACKAGE.subprocess, "check_output", return_value="frozen\n"):
                with self.assertRaisesRegex(AssertionError, reason):
                    PACKAGE.package_artifact(self.root, self.output, self.base, True)
        self.write("prebuilt/manifest.json", json.dumps(valid))
        (self.root / "prebuilt/bin/culbm-single").chmod(0o644)
        with patch.object(PACKAGE.subprocess, "check_output", return_value="frozen\n"):
            with self.assertRaisesRegex(AssertionError, "not executable"):
                PACKAGE.package_artifact(self.root, self.output, self.base, True)
        self.assertFalse((self.root / "SHA256SUMS").exists())
        self.assertFalse(self.output.exists())

    def test_dry_run_writes_nothing_and_checks_head_twice(self):
        self.complete_fixture()
        before = {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        with patch.object(PACKAGE.subprocess, "check_output", return_value="frozen\n") as head:
            result = PACKAGE.package_artifact(self.root, self.output, self.base, True)
        after = {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertEqual(head.call_count, 2)
        self.assertFalse(self.output.exists())
        self.assertIn("results/", result["members"])
        self.assertIn("SHA256SUMS", result["members"])
        self.assertEqual(result["model_id"], PACKAGE.MODEL_ID)

    def test_fixture_archive_has_final_identity_and_empty_results(self):
        self.complete_fixture()
        self.write("data/validation-history/private/summary.json", "private")
        self.write("results/old-output.json", "private")
        with patch.object(PACKAGE.subprocess, "check_output", return_value="frozen\n"):
            PACKAGE.package_artifact(self.root, self.output, self.base, False)
        with tarfile.open(self.output) as archive:
            names = set(archive.getnames())
            manifest = json.load(archive.extractfile("l2lbm-ae/data/provenance/artifact-source-manifest.json"))
            checksums = archive.extractfile("l2lbm-ae/SHA256SUMS").read().decode().splitlines()
            for line in checksums:
                expected, relative = line.split("  ", 1)
                content = archive.extractfile(f"l2lbm-ae/{relative}").read()
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected)
        self.assertIn("l2lbm-ae/results", names)
        self.assertFalse(any(name.startswith("l2lbm-ae/results/") for name in names))
        self.assertFalse(any("validation-history" in name for name in names))
        self.assertEqual(manifest["model_id"], PACKAGE.MODEL_ID)
        self.assertEqual(manifest["source_commit"], "frozen")
        self.assertTrue(self.output.with_suffix(".gz.sha256").is_file())

    def test_wrong_model_or_initial_head_prevents_writes(self):
        self.complete_fixture()
        with patch.object(PACKAGE.subprocess, "check_output", return_value="changed\n"):
            with self.assertRaisesRegex(AssertionError, "before packaging"):
                PACKAGE.package_artifact(self.root, self.output, self.base, False)
        self.assertFalse((self.root / "SHA256SUMS").exists())
        freeze_path = self.root / "data/provenance/freeze.json"
        freeze = json.loads(freeze_path.read_text())
        freeze["model_id"] = "superseded_model"
        freeze_path.write_text(json.dumps(freeze))
        with self.assertRaisesRegex(AssertionError, "final unified model"):
            PACKAGE.package_artifact(self.root, self.output, self.base, False)
        self.assertFalse((self.root / "SHA256SUMS").exists())
        self.assertFalse(self.output.exists())

    def test_head_changes_or_size_overflow_prevent_final_archive(self):
        self.complete_fixture()
        with patch.object(PACKAGE.subprocess, "check_output", side_effect=["frozen\n", "changed\n"]):
            with self.assertRaisesRegex(AssertionError, "during packaging"):
                PACKAGE.package_artifact(self.root, self.output, self.base, False)
        self.assertFalse(self.output.exists())
        with patch.object(PACKAGE.subprocess, "check_output", return_value="frozen\n"):
            with patch.object(PACKAGE, "MAX_ARCHIVE_BYTES", 1):
                with self.assertRaisesRegex(AssertionError, "upload limit"):
                    PACKAGE.package_artifact(self.root, self.output, self.base, False)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
