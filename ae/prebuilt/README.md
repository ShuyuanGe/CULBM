# Prebuilt executables

These `culbm-single` and `culbm-tune` executables are a fallback for the source build.
They include the CUDA and C++ runtimes, so no compiler or CUDA Toolkit is needed.
They require Linux x86-64 with glibc ≥2.17 and an NVIDIA driver compatible with
CUDA 12.x and your GPU. The Python scripts still use `uv` as described in the main README.

From the unpacked artifact directory:

```bash
uv sync
mkdir -p build/bin
cp prebuilt/bin/culbm-single prebuilt/bin/culbm-tune build/bin/
uv run scripts/ae.py smoke --device 0 --output results/smoke
```

After the smoke test passes, use the remaining commands in the [main README](../README.md).

Embedded GPU architectures are `sm_80`, `sm_86`, `sm_87`, `sm_89`, `sm_90`, `sm_100`,
`sm_101`, `sm_103`, `sm_120` and `sm_121`. These contain native SASS and do not require
PTX JIT. Source builds can target your GPU using `CMAKE_CUDA_ARCHITECTURES=native`.

[manifest.json](manifest.json) records source identity, binary hashes, build versions
and validation. Third-party notices are in [NOTICES.txt](NOTICES.txt).

To rebuild the full architecture bundle with a suitable toolchain, run
`bash prebuilt/build.sh /path/to/build-directory`.
