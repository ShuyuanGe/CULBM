# CULBM

CULBM is a CUDA-native lattice Boltzmann toolkit built around cache-aware halo blocking. The repository bundles the core solver sources, configuration helpers, binary dump visualizers, and performance-analysis notebooks required to define initial states, run experiments, and interpret MLUPS trends end to end.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## `single_dev_expt_main` CLI

`single_dev_expt_main` is parameterized entirely through CLI flags. Vectors must be passed in bracket form (e.g., `[32,16,2]`).

| Flag | Purpose | Default |
| --- | --- | --- |
| `--devId` | CUDA device index. | `0` |
| `--blockDim [bx,by,bz]` | CUDA block configuration per kernel launch. | `[32,16,2]` |
| `--gridDim [gx,gy,gz]` | Grid configuration per launch. | `[2,3,19]` |
| `--domainDim [nx,ny,nz]` | Logical simulation domain; can exceed `blockDim * gridDim` when halo blocking is active. | `[256,256,256]` |
| `--innerLoop` | Iterations executed while a tile remains resident (halo blocking depth). | `5` |
| `--streamPolicy {0,1}` | `0`: pull-stream double buffering, `1`: in-place streaming. | `0` |
| `--optPolicy {0…3}` | `0`: none, `1`: static L2 blocking, `2`: L1/L2 mixed, `3`: dynamic L2 blocking. | `0` |
| `--invTau` | Reciprocal relaxation time. | `0.5` |
| `--nstep`, `--dstep` | Total steps and dump cadence. | `1000`, `100` |
| `--initStateFolder <path>` | Optional folder with `flag.dat`, `rho.dat`, `vx.dat`, `vy.dat`, `vz.dat`. Omitting this flag triggers uniform default initialization. | empty |
| `--dumpFolder <path>` | Output directory for binary dumps. | `data/left_inlet_right_outlet_256_256_256_output` |
| `--dumpRho`, `--dumpVx`, `--dumpVy`, `--dumpVz` | When present, dump per-field snapshots every `dstep` steps as `rho_<step>.dat`, `vx_<step>.dat`, … | disabled |

Additional expert knobs include `--streamPolicy`, `--optPolicy`, and `--innerLoop`. Their interactions mirror the implementation in [src/simulator/single_dev_expt/simulator_expt_platform.cu](src/simulator/single_dev_expt/simulator_expt_platform.cu).

### Typical launch

```bash
./build/src/single_dev_expt_main \
	--blockDim [32,16,2] \
	--gridDim [2,6,38] \
	--domainDim [288,272,280] \
	--innerLoop 5 \
	--streamPolicy 1 \
	--optPolicy 3
```

```bash
./build/src/single_dev_expt_main \
	--blockDim [32,16,2] \
	--gridDim [9,17,140] \
	--domainDim [288,272,280] \
	--streamPolicy 0 \
	--optPolicy 0
```

On an RTX 4090D, the halo-blocked configuration above sustains roughly **6810 MLUPS**, whereas the plain pull-stream run reaches **3805 MLUPS**, a >70% throughput uplift.

The solver reports MLUPS after each batch and stores optional `.dat` frames for downstream tools.

## Scenario 1 · Custom domains → snapshots → visualization

1. **Author boundary/obstacle masks.** Use [data/init_state.ipynb](data/init_state.ipynb) to run helpers such as `leftInletRightOutletCubeObs`. Each run writes `flag.dat`, `vx.dat`, and related seeds into a folder like `data/left_inlet_right_outlet_cube_obs_288_272_280_init_state`.
2. **Simulate with dumps enabled.** Point `single_dev_expt_main` at the generated folder via `--initStateFolder`, choose a dump target via `--dumpFolder`, and enable any subset of `--dumpRho/--dumpVx/--dumpVy/--dumpVz`. Snapshot frequency follows `--dstep`, so the example command above emits frames at steps 200, 400, …

```bash
./build/src/single_dev_expt_main \
    --devId 0 \
    --blockDim [32,16,2] \
    --gridDim [2,6,38] \
    --domainDim [288,272,280] \
    --innerLoop 5 \
    --streamPolicy 1 \
    --optPolicy 3 \
    --invTau 0.5 \
    --nstep 1200 \
    --dstep 200 \
    --initStateFolder data/left_inlet_right_outlet_cube_obs_288_272_280_init_state \
    --dumpFolder data/left_inlet_right_outlet_cube_obs_288_272_280_output \
    --dumpRho --dumpVx --dumpVy --dumpVz
```
3. **Inspect results.** Open [data/vis.ipynb](data/vis.ipynb), adjust `nx`, `ny`, `nz`, and `outputFolder` to match the dump location, and run the plotting cells to load `vx_<step>.dat`, `vy_<step>.dat`, `vz_<step>.dat`, and render velocity slices or magnitude heatmaps.

This pipeline keeps everything binary-compatible with the CUDA kernels (no intermediate conversions) and lets you iterate quickly on inlet speeds, obstacle geometries, or dumping cadence.

## Scenario 2 · Batch experiments → curve fitting

1. **Enumerate configurations.** [analysis/batch_experiments.py](analysis/batch_experiments.py) scans combinations of block-count triplets and inner-loop depths. Each configuration spawns `single_dev_expt_main` with arguments assembled from the constants near the top of the script (e.g., `BLOCK_DIM`, `GRID_DIM`, `STREAM_POLICY`, `OPT_POLICY`). Update those tuples to match the hardware you are targeting.
2. **Collect MLUPS logs.** Running the script produces a pickle (`batch_experiment_results_s{stream}o{opt}.pkl`) that stores the raw speeds reported by the executable alongside the exact CLI used.
3. **Fit performance surfaces.** Load the pickle in [analysis/analysis.ipynb](analysis/analysis.ipynb). The notebook already demonstrates how to compute averages, filter high-throughput cases, regress the amortization factor $\,1/\psi(I,\lambda)$, and visualize acceleration contours $\kappa(I,B)$.

Use this workflow when you need principled tuning guidance for `--innerLoop`, `--domainDim`, or different blocking layouts before committing to a single scenario.

---

