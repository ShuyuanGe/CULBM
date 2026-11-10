# L2LBM artifact

[Paper](paper/main.pdf) · [Supplementary material](paper/supplementary.pdf).
Requested badges are **Reusable** and **Results Reproduced**.

## 1. Build and check — about 30 minutes

Use Linux x86-64 and an **NVIDIA GPU supporting persisting L2**. Source builds need a C++23
host compiler (e.g. `g++ >= 14.1`) and a CUDA device compiler supporting at least C++20
(e.g. `nvcc >= 12.0`). Choose mutually compatible compilers supporting your GPU;
see [build dependencies](code/README.md#build).
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for the Python scripts.
Allow 2 GiB free GPU memory, 8 GiB RAM and 8 GiB disk for this step.

| GPU | Total L2 (MiB) |
| --- | ---: |
| A100 | 40 |
| H100 PCIe | 50 |
| GB200 GPU | 129.25 |
| RTX 3090 | 6 |
| RTX 4090 / **RTX 4090 D** | 72 |
| RTX 5090 / **RTX 5090 D v2** | 96 |
| **L20** / **L40** / **RTX 6000 Ada** | 96 |
| **RTX PRO 6000** | 128 |

Bold names mark the paper's GPUs; prefer these for direct comparison. As a rough reference,
maximum persisting L2 is around 10/16 of total L2, though it varies by device; `culbm-tune` queries it.

Run from the unpacked directory.

```bash
uv sync
cmake -S code -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel 2 --target culbm-single culbm-tune
```

If building is unavailable, copy the [prebuilt binaries](prebuilt/README.md) instead.

```bash
mkdir -p build/bin
cp prebuilt/bin/culbm-single prebuilt/bin/culbm-tune build/bin/
```

Then run the quick test using either installation.

```bash
uv run scripts/ae.py smoke --device 0 --output results/smoke
```

Expect `results/smoke/summary.json` with `status=passed` and three cases. Correctness
checks require bitwise equality with Plain CUDA for density, velocity and active DDF
populations. Unused boundary-population differences are reported separately.

## 2. Recompute supplied results — 1–5 minutes, no GPU

```bash
uv run scripts/ae.py reference --output results/reference
```

This reanalyzes archived data for Figures 3–7, Table 3 and supplementary Table D.1,
and checks Table 2 calibration inputs. See `results/reference/summary.json`,
tables in the same directory and plots in `results/reference/figures/`.

| Paper result | Expected result from supplied data |
| --- | --- |
| Figure 5 | Geometric mean speedup **2.25×** over FluidX3D |
| Figure 6 | Geometric mean speedups **2.43× / 2.41× / 2.37×** over Plain CUDA for D3Q15/19/27 |
| Figure 7 | Geometric mean selection quality **96.8%** of best observed throughput |
| Table 3 | L2 request hit rate **18.32% → 90.82%**, about **5.9×** less DRAM traffic |

## 3. Validate blocking on your GPU

Use an idle GPU for timing. Performance, selection and profiling automatically
calibrate and select parameters; `calibrate` exposes the inputs separately.
Allow 80 GiB disk and 32 GiB RAM for full correctness. GPU memory is checked per case.

| Command | Paper check and expected output | Estimated time |
| --- | --- | --- |
| `correctness` | §7.2 / D.4; supported cases pass the bitwise checks above | 15–60 min |
| `calibrate` | Table 2 / C.1; inputs and selected tile/depth in `tuning/calibration.json` | 1–5 min |
| `performance` | Figures 5–6 / D.5; `ring_over_plain`, `ring_over_fluidx3d`, timing CV ≤5% | 10–60 min |
| `selection` | Figure 7 method; `selected_over_best_sampled` near one, successful/attempted counts | 4–7 h for 10,000 attempts |
| `profile` | Table 3 mechanism; L2 request hit rate and DRAM traffic for plain, on, off and oversized cases | 5–30 min |

```bash
uv run scripts/ae.py correctness --device 0 --output results/correctness
uv run scripts/ae.py calibrate --device 0 --output results/calibrate
uv run scripts/ae.py selection --device 0 --output results/selection
uv run scripts/ae.py profile --device 0 --domain 256 256 256 --output results/profile
```

For quick checks, add `--quick` to `correctness` or `--samples 32` to `selection`
(workflow only; the full run uses 10,000 attempts). Profiling requires Nsight Compute
(`ncu`) and GPU counter permission. Omit `--domain 256 256 256` to use the paper's
512-cubed domain when memory permits.

For the FluidX3D comparison, install the OpenCL loader and NVIDIA OpenCL runtime.
Select CUDA and OpenCL IDs for the same physical GPU; leave `CUDA_DISABLE_PTX_JIT` unset.

```bash
uv run scripts/ae.py build-fluidx3d --compiler c++ --output results/baseline-build
uv run scripts/ae.py performance --device 0 --q 15 19 27 --sizes 128 256 384 \
  --fluidx3d-dir build/fluidx3d --opencl-device 0 --output results/performance
```

This uses the FP32 BGK protocol in D.5 and five independent processes. Ratios above
one indicate acceleration. Without FluidX3D, `performance` compares internal CUDA paths only.

Inspect `summary.json` in your output directory and the case logs. `completed`
does not mean all cases passed; `unsupported-memory` marks a skipped case,
and `unstable` means timing CV exceeds 5%.

## 4. Scope and reuse

GPU checks cover single-GPU FP32 BGK. On another GPU, compare trends and ratios;
absolute throughput and selected tile/depth can differ. Missing historical metadata
and launch differences limit exact reruns of original experiments; see
[coverage details](data/provenance/gaps.json).

Adapt the checks to another velocity set, domain or depth, for example with D3Q19.

```bash
uv run scripts/ae.py correctness --q 19 --sizes 128 --depths 1 3 --device 0 --output results/reuse-q19
```

Use `--help` for options, `sha256sum --check SHA256SUMS` for package integrity,
and HotCRP for questions during evaluation.
