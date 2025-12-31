# cuLBM

cuLBM is a CUDA-native lattice Boltzmann toolkit with two complementary GPU drivers: the cache-aware, single-device experimentation path (`single_dev_expt_main`) and the domain-decomposed multi-device path (`multi_dev_main`). The repository bundles the core solver sources, configuration helpers, binary dump visualizers, and performance-analysis notebooks required to define initial states, run experiments, and interpret MLUPS trends end to end.

## Usage

### Build

```bash
cmake -S . -B build
cmake --build build
```

### Runtimes

#### `single_dev_expt_main`

##### CLI reference

`single_dev_expt_main` is parameterized entirely through CLI flags. Vectors must be passed in bracket form (e.g., `[32,16,2]`).

| Flag | Purpose | Default |
| --- | --- | --- |
| `--devId` | CUDA device index. | `0` |
| `--blockDim [bx,by,bz]` | CUDA block configuration for kernel launch. | `[32,16,2]` |
| `--gridDim [gx,gy,gz]` | CUDA grid configuration for kernel launch. | `[2,3,19]` |
| `--domainDim [Nx,Ny,Nz]` | Logical simulation domain; can exceed `blockDim * gridDim` when halo blocking is active. | `[256,256,256]` |
| `--innerLoop` | Iterations executed while a tile remains resident (halo blocking depth). | `5` |
| `--streamPolicy {0,1}` | `0`: pull-stream double buffering, `1`: in-place streaming. | `0` |
| `--optPolicy {0…3}` | `0`: none, `1`: static L2 blocking, `2`: L1/L2 mixed, `3`: dynamic L2 blocking. | `0` |
| `--invTau` | Reciprocal relaxation time. | `0.5` |
| `--nstep`, `--dstep` | Total steps and dump cadence. | `1000`, `100` |
| `--initStateFolder <path>` | Optional folder with `flag.dat`, `rho.dat`, `vx.dat`, `vy.dat`, `vz.dat`. Omitting this flag triggers uniform default initialization. | empty |
| `--dumpFolder <path>` | Output directory for binary dumps. | `data/left_inlet_right_outlet_256_256_256_output` |
| `--dumpRho`, `--dumpVx`, `--dumpVy`, `--dumpVz` | When present, dump per-field snapshots every `dstep` steps as `rho_<step>.dat`, `vx_<step>.dat`, … | disabled |

Additional expert knobs include `--streamPolicy`, `--optPolicy`, and `--innerLoop`. Their interactions mirror the implementation in [src/simulator/single_dev_expt/simulator_expt_platform.cu](src/simulator/single_dev_expt/simulator_expt_platform.cu).

##### Typical launches

The first command shows the `[32,16,2]` block and `[2,6,38]` grid combination that was hand-tuned only for an RTX 4090; expect to retune those vectors for other GPUs before chasing the same MLUPS. 
The second command is a reference pull-stream run with no optimizations and same domain size `[288,272,280]` for comparison.
The reference pull-stream run reaches **3805 MLUPS** and the tuned setup hits **6810 MLUPS** (a 1.8x speedup).

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

#### `multi_dev_main`

##### CLI reference

`multi_dev_main` launches a synchronous decomposition across `devDim.x * devDim.y * devDim.z` GPUs, assigns a tile of size `blkDim * gridDim` to each device, and optionally dumps per-field sub-volumes every `dstep` steps. Unlike `single_dev_expt_main`, it does **not** expose a `--domainDim` override; the global domain is fixed by `blkDim * gridDim * devDim`. It also does **not** accept `--initStateFolder` inputs yet, so the multi-GPU path currently seeds domain data internally rather than consuming obstacle/boundary masks authored via [scripts/state_initialization.ipynb](scripts/state_initialization.ipynb).

| Flag | Purpose | Default |
| --- | --- | --- |
| `--devDim [dx,dy,dz]` | Logical device grid describing how many GPUs participate along each axis. | `[1,1,1]` |
| `--blkDim [bx,by,bz]` | CUDA block geometry for kernel launch on each GPU. | `[32,8,4]` |
| `--gridDim [gx,gy,gz]` | CUDA grid dimensions for kernel launch on each GPU; the per-device domain is `blkDim * gridDim`. | `[16,32,64]` |
| `--invTau` | Reciprocal relaxation time. | `0.5` |
| `--nstep`, `--dstep` | Total steps and dump cadence. | `1000`, `100` |
| `--dumpFolder <path>` | Root folder for tiled dumps; each GPU writes `frame_<step>_dev_{x}_{y}_{z}`. | `data/multi_dev_output` |
| `--dumpRho`, `--dumpVx`, `--dumpVy`, `--dumpVz` | Enable field dumps every `dstep` steps. | disabled |

##### Typical launch (2×2×1 GPUs)

```bash
./build/src/multi_dev_main \
    --devDim [2,2,1] \
    --blkDim [32,8,4] \
    --gridDim [16,32,64] \
    --nstep 800 \
    --dstep 200
```

`multi_dev_main` derives its spatial coverage directly from `blkDim * gridDim` on each device, 
e.g. under the above configuration:

- Per-device span: $(32,8,4) \times (16,32,64) = (256,256,256)$, matching the log entry `Single Device Domain Dimension: [256,256,256]`.
- Global span: $(256,256,256) \times (2,2,1) = (512,512,256)$, so the log reports `All Device Domain Dimension: [512,512,256]`.

Because of this multiplicative relationship, there is no `--domainDim` argument on the multi-GPU binary, so domain customization must happen via the block/grid/device triplets for now.

The executable prints the per-device and global domain sizes, enables peer-to-peer links for adjacent ranks, and emits tiles that can be reassembled with the helper cells in [visualization/dump_visualization.ipynb](visualization/dump_visualization.ipynb).

### Domain Decomposition

Interactive visualization tools are provided to explore domain decomposition and halo exchange patterns.

- **Single Device**: [visualization/single_dev_blocking.html](visualization/single_dev_blocking.html) visualizes the block-level decomposition, including valid regions and halo zones for a single GPU.
- **Multi-Device**: [visualization/multi_dev_blocking.html](visualization/multi_dev_blocking.html) visualizes the global domain decomposition across multiple GPUs, showing how the problem is partitioned.

### Workflows

Scenarios 1–3 cover complementary workflows: Scenario 1 stays on a single GPU, Scenario 2 extends runs across multiple GPUs, and Scenario 3 sweeps single-GPU parameter spaces for tuning.

#### Scenario 1 (Single-GPU): Custom domains → snapshots → visualization

1. **Author boundary/obstacle masks.** Use [scripts/state_initialization.ipynb](scripts/state_initialization.ipynb) to run helpers such as `leftInletRightOutletCubeObs`. Each run writes `flag.dat`, `vx.dat`, and related seeds into a folder like `data/left_inlet_right_outlet_cube_obs_288_272_280_init_state`.
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

3. **Visualize results.** Open [visualization/dump_visualization.ipynb](visualization/dump_visualization.ipynb) and run **Single-GPU Dump**. Set `Nx`, `Ny`, `Nz`, and `outputFolder` so they match the single-device dump folder, then execute the cell to load `vx_<step>.dat`, `vy_<step>.dat`, and `vz_<step>.dat` and render the mid-plane magnitude heatmaps.

This pipeline keeps everything binary-compatible with the CUDA kernels (no intermediate conversions) and lets you iterate quickly on inlet speeds, obstacle geometries, or dumping cadence.

#### Scenario 2 (Multi-GPU): decomposition → tiled dumps → visualization

1. **Choose the device grid.** Decide how many GPUs participate along each axis via `--devDim [dx,dy,dz]`, then size `--blkDim`/`--gridDim` so that each rank owns a reasonable tile. The aggregate domain equals `(blkDim * gridDim) ⊙ devDim`, so validate it matches your target problem size. (Custom initial states from `scripts/state_initialization.ipynb` are not wired up here yet, so geometry tweaks must wait for future multi-GPU support.)
2. **Launch `multi_dev_main`.** Each device writes binary tiles named `frame_<step>_dev_{ix}_{iy}_{iz}` into `--dumpFolder`. Use a command such as:

```bash
./build/src/multi_dev_main \
    --invTau 0.5 \
    --dstep 200 \
    --nstep 2000 \
    --devDim [2,2,1] \
    --blkDim [32,8,4] \
    --gridDim [8,32,64] \
    --dumpFolder data/multi_dev_output \
    --dumpVx --dumpVy --dumpVz
```
    

3. **Visualize results.** Open [visualization/dump_visualization.ipynb](visualization/dump_visualization.ipynb) and run **Multi-GPU Dump**. Set the domain shape (`Nx`, `Ny`, `Nz`), the device grid tuple (`nx`, `ny`, `nz`), and `outputFolder` to mirror your run, then execute the stitching cell to reconstruct a full field. Then plot the slices or magnitude trends.

#### Scenario 3 (Single-GPU): Blocking experiments → curve fitting

1. **Enumerate configurations.** [analysis/blocking_experiments.py](analysis/blocking_experiments.py) scans combinations of block-count triplets and inner-loop depths. Each configuration spawns `single_dev_expt_main` with arguments assembled from the constants near the top of the script (e.g., `BLOCK_DIM`, `GRID_DIM`, `STREAM_POLICY`, `OPT_POLICY`). Update those tuples to match the hardware you are targeting.
2. **Collect MLUPS logs.** Running the script produces a pickle (`blocking_experiment_results_s{stream}o{opt}.pkl`) that stores the raw speeds reported by the executable alongside the exact CLI used.
3. **Fit performance surfaces.** Load the pickle in [analysis/blocking_analysis.ipynb](analysis/blocking_analysis.ipynb). The notebook already demonstrates how to compute averages, filter high-throughput cases, regress the amortization factor $\,1/\psi(I,\lambda)$, and visualize acceleration contours $\kappa(I,B)$.

Use this workflow when you need principled tuning guidance for `--innerLoop`, `--domainDim`, or different blocking layouts before committing to a single scenario.

#### Scenario 4 (Multi-GPU): Weak scaling benchmarks → efficiency analysis

1. **Execute scaling sweeps.** [analysis/scaling_experiments.py](analysis/scaling_experiments.py) orchestrates weak scaling tests across 1 to 4 GPUs. It launches `multi_dev_main` with a constant per-device workload (fixed `blkDim` and `gridDim`) while scaling the device count. The script captures standard output and error streams into log files (`1gpu.log`, `2gpu.log`, etc.).
2. **Assess scaling performance.** Open [analysis/scaling_analysis.ipynb](analysis/scaling_analysis.ipynb) to process the generated logs. The notebook extracts MLUPS throughput, calculates effective memory bandwidth, and renders a scaling plot (`weak_scaling_performance.png`) to benchmark performance against reference solvers like FluidX3D.

---

