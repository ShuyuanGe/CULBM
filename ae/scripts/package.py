"""Package only the final reviewer-facing source, paper, and evidence."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

MODEL_ID = "calibrated_service_cost_v1"
MAX_ARCHIVE_BYTES = 600_000_000
PROVENANCE = "data/provenance"
SOURCE_MANIFEST = f"{PROVENANCE}/artifact-source-manifest.json"
FILE_ALLOWLIST = {
    "": ("README.md", "pyproject.toml", "uv.lock"),
    "paper": ("main.pdf", "supplementary.pdf", "SHA256SUMS", "README.md"),
    "paper/source": ("main.tex", "supplementary.tex", "ref.bib", "acmart.cls", "ACM-Reference-Format.bst"),
    "paper/source/figures": (
        "overview.pdf",
        "d2q9_pull.pdf",
        "ideal_speedup.pdf",
        "r4_fixed_domain_performance.pdf",
        "roofline_average_gpu_d3q27_complete.pdf",
        "d3q27_joint_tile_depth_validation.pdf",
        "CrossVelocitySetApplicability.pdf",
    ),
    "code": ("CMakeLists.txt", "README.md"),
    "code/cmake": ("remove_rpath.cmake",),
    "code/env": ("build.yml",),
    "code/third_party/fluidx3d": ("VERSIONS.json", "historical_N128_nu1.patch"),
    "prebuilt": ("README.md", "manifest.json", "build.sh", "NOTICES.txt"),
    "prebuilt/bin": ("culbm-single", "culbm-tune"),
    "scripts": ("ae.py", "checks.py", "reference.py", "package.py", "fluidx3d_setup.cpp"),
    PROVENANCE: (
        "freeze.json",
        "artifact-source-manifest.json",
        "physical-upstream-check.json",
        "claims.json",
        "gaps.json",
        "plot_sources.json",
        "reference_inventory.json",
    ),
    f"{PROVENANCE}/gpu-cache": (
        "README.md",
        "4090d.raw.log",
        "5090.raw.log",
        "5090dv2.raw.log",
        "l20.raw.log",
        "l40.raw.log",
        "rtx6000ada.raw.log",
        "raw_bytes.cpp",
        "raw_bytes.sha256",
    ),
    "data/bridge": ("README.md", "case.json", "reference_protocol.json", "master_identity.json"),
    "data/reference/fig5": ("fixed_domain_speedup.csv", "rtxpro6000_n896_fallback.csv"),
    "data/reference/fig6": ("cross_velocity_speedup.json", "raw_samples.json"),
    "data/reference/fig7": (
        "run.log",
        "rtx6000ada.json",
        "rtx5090dv2.json",
        "rtx4090d.json",
        "l40.json",
        "l20.json",
        "rtxpro6000.json",
        "selected_outcomes.json",
        "selected_argmax_measurements.json",
    ),
    "data/reference/roofline": ("kernel_ops.json",),
    "data/reference/calibration": ("legacy_lambda.csv", "legacy_gamma.csv"),
    "data/reference/ncu": (
        "binary_sha256.txt",
        "gpu_identity.txt",
        "source_sha256.txt",
        "summary.md",
        "raw_ncu_sha256.txt",
        "manifest.json",
        "summary.csv",
        "timing_per_repeat.csv",
        "analysis_scripts_sha256.txt",
        "per_repeat.csv",
        "timing_summary.csv",
        "raw_timing_sha256.txt",
    ),
    "data/refactor-validation": ("summary.json", "inventory.json", "README.md"),
}
TREE_ALLOWLIST = (
    "code/src",
    "code/tools",
    "code/third_party/fluidx3d/upstream",
    "code/third_party/fluidx3d/qualification_upstream",
    "scripts/plots",
    "scripts/tests",
    "data/bridge/input",
    "data/bridge/master_source",
    "data/reference/baselines/plain_cuda",
    "data/reference/baselines/formal_20260717",
    "data/reference/baselines/qualification_5090",
    "data/reference/calibration/repeatability",
    "data/reference/calibration/source",
    "data/reference/calibration/middle_efficiency",
    "data/reference/ncu/raw_ncu",
    "data/reference/ncu/raw_timing",
    "data/reference/ncu/original_scripts",
)
VALIDATION_ALLOWLIST = (
    "build",
    "model",
    "smoke",
    "correctness",
    "correctness-256",
    "correctness-384",
    "geometry-q19-512",
    "geometry-q27-128",
    "bridge",
    "performance",
    "selection",
    "profile",
    "reference",
    "environment",
    "clean-extracted",
)
CACHE_DIRECTORIES = {".git", ".venv", ".ruff_cache", ".pytest_cache", "__pycache__"}
SOURCE_EXCLUSIONS = CACHE_DIRECTORIES | {"build", "build-physical", "results", "dist"}
REPORT_EXCLUSIONS = CACHE_DIRECTORIES | {
    "bin",
    "lib",
    "CMakeFiles",
    "package",
    "source",
    "snapshot",
    "snapshots",
    "validation-history",
    "install-history",
    "before",
    "after",
}
REPORT_SUFFIXES = {".json", ".jsonl", ".log", ".csv", ".txt"}
GEOMETRY_SCRIPTS = {
    Path("data/refactor-validation/geometry-q19-512/experiment.py"),
    Path("data/refactor-validation/geometry-q27-128/experiment.py"),
}
AUTHOR_ENVIRONMENT_FILES = {
    "before.json",
    "after.json",
    "install.log",
    "persistent-install.log",
    "install-history.json",
}
PRIVATE_REFERENCE_FILES = {
    Path("data/reference/calibration/repeatability/historical_provenance.json")
}


def sha256(path):
    with path.open("rb") as content:
        return hashlib.file_digest(content, "sha256").hexdigest()


def collect_files(root):
    files = {
        root / folder / name
        for folder, names in FILE_ALLOWLIST.items()
        for name in names
        if (root / folder / name).is_file()
    }
    trees = [
        (name, CACHE_DIRECTORIES if name.startswith("data/") else SOURCE_EXCLUSIONS)
        for name in TREE_ALLOWLIST
    ]
    trees.extend((f"data/refactor-validation/{name}", REPORT_EXCLUSIONS) for name in VALIDATION_ALLOWLIST)
    trees.append(("prebuilt/licenses", SOURCE_EXCLUSIONS | {"bin", "lib"}))
    for name, excluded in trees:
        tree = root / name
        assert not tree.is_symlink(), tree
        for folder, directories, names in os.walk(tree):
            directories[:] = sorted(set(directories) - excluded)
            for directory in directories:
                assert not (Path(folder) / directory).is_symlink(), Path(folder) / directory
            for filename in sorted(names):
                path = Path(folder) / filename
                if path.name == ".DS_Store" or path.relative_to(root) in PRIVATE_REFERENCE_FILES:
                    continue
                if name == "prebuilt/licenses" and path.suffix not in {".txt", ".rst"}:
                    continue
                if name.startswith("data/refactor-validation/"):
                    geometry_script = path.relative_to(root) in GEOMETRY_SCRIPTS
                    if not geometry_script and (path.suffix not in REPORT_SUFFIXES or path.stat().st_mode & 0o111):
                        continue
                    if path.name in AUTHOR_ENVIRONMENT_FILES:
                        continue
                files.add(path)
    for path in files:
        assert not path.is_symlink(), path
        assert path.resolve().is_relative_to(root.resolve()), path
    return sorted(files)


def validate_prebuilt(root, files, freeze):
    members = {path.relative_to(root) for path in files if path.is_relative_to(root / "prebuilt")}
    if not members:
        return
    required = {Path("prebuilt") / name for name in FILE_ALLOWLIST["prebuilt"]}
    required |= {Path("prebuilt/bin") / name for name in FILE_ALLOWLIST["prebuilt/bin"]}
    assert required <= members, f"Incomplete prebuilt fallback: {sorted(required - members)}"
    manifest = json.loads((root / "prebuilt/manifest.json").read_text())
    assert manifest["model_id"] == MODEL_ID, "Prebuilt model identity does not match."
    assert manifest["source_commit"] == freeze["source_commit"], "Prebuilt source commit is stale."
    binaries = manifest["binaries"]
    expected = {f"bin/{name}" for name in FILE_ALLOWLIST["prebuilt/bin"]}
    assert len(binaries) == len(expected) and {row["path"] for row in binaries} == expected, (
        "Prebuilt manifest must identify exactly culbm-single and culbm-tune."
    )
    for row in binaries:
        path = root / "prebuilt" / row["path"]
        assert path.stat().st_size == row["bytes"], f"Prebuilt size mismatch: {path}"
        assert sha256(path) == row["sha256"], f"Prebuilt hash mismatch: {path}"
        assert path.stat().st_mode & 0o100, f"Prebuilt binary is not executable: {path}"


def check_head(upstream_repo, expected, phase):
    actual = subprocess.check_output(["git", "-C", str(upstream_repo), "rev-parse", "HEAD"], text=True).strip()
    assert actual == expected, f"Upstream changed {phase}; refresh and validate the source snapshot."


def package_artifact(root, output, upstream_repo, dry_run):
    freeze = json.loads((root / PROVENANCE / "freeze.json").read_text())
    assert freeze["model_id"] == MODEL_ID, "Only the final unified model can be packaged."
    commit = freeze["source_commit"]
    check_head(upstream_repo, commit, "before packaging")
    assert not output.resolve().is_relative_to(root.resolve()), "Write the archive outside its input tree."
    files = collect_files(root)
    validate_prebuilt(root, files, freeze)
    required = {
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "paper/main.pdf",
        "paper/supplementary.pdf",
        "paper/SHA256SUMS",
        "paper/README.md",
        f"{PROVENANCE}/freeze.json",
        "data/refactor-validation/summary.json",
        "data/refactor-validation/inventory.json",
    }
    present = {str(path.relative_to(root)) for path in files}
    assert required <= present, f"Missing final submission files: {sorted(required - present)}"
    expected_sources = {f"code/{row['path']}" for row in freeze["source_files"]}
    assert expected_sources <= present, f"Missing frozen source files: {sorted(expected_sources - present)}"
    manifest_path = root / SOURCE_MANIFEST
    checksums = root / "SHA256SUMS"
    membership = sorted(set(files) | {manifest_path, checksums})
    result = {"model_id": MODEL_ID, "source_commit": commit, "files": len(membership)}
    if dry_run:
        check_head(upstream_repo, commit, "during the dry run")
        return result | {
            "mode": "dry_run",
            "members": [str(path.relative_to(root)) for path in membership] + ["results/"],
            "archive_size_limit_bytes": MAX_ARCHIVE_BYTES,
        }
    source_files = [
        {"path": str(path.relative_to(root / "code")), "sha256": sha256(path)}
        for path in files
        if path.is_relative_to(root / "code")
    ]
    manifest_path.write_text(
        json.dumps({"model_id": MODEL_ID, "source_commit": commit, "files": source_files}, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = sorted(set(files) | {manifest_path})
    checksums.write_text("".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in payload), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".partial")
    with pending.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in membership:
                    info = archive.gettarinfo(str(path), arcname=f"l2lbm-ae/{path.relative_to(root)}")
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as content:
                        archive.addfile(info, content)
                info = tarfile.TarInfo("l2lbm-ae/results/")
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                archive.addfile(info)
    assert pending.stat().st_size <= MAX_ARCHIVE_BYTES, "Artifact exceeds the HotCRP upload limit."
    check_head(upstream_repo, commit, "during packaging")
    pending.replace(output)
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return result | {"archive": str(output), "bytes": output.stat().st_size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="List archive membership without writing any files.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = package_artifact(root, args.output.resolve(), args.upstream_repo, args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
