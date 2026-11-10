# cuLBM

cuLBM is a CUDA lattice Boltzmann toolkit. Use `culbm-tune` to select a GPU configuration and `culbm-single` to run a single-GPU simulation.

## Usage

### Build

Requires CMake >=4.0.1, CCCL >=3.2, a C++23 host compiler, and a C++20 CUDA compiler.

```bash
cmake -S . -B build
cmake --build build
```

Without a build environment, create one with Conda and [env/build.yml](env/build.yml):

```bash
conda env create -f env/build.yml
conda activate culbm-build
```

To include specific GPU architectures, configure with `-DCMAKE_CUDA_ARCHITECTURES`, for example:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="89-real;120-real"
```

### Single-GPU workflow

Select D3Q27 blocking parameters on an idle GPU:

```bash
./build/bin/culbm-tune --device 0 --q 27
```

Example output for RTX 5090:

```text
[Launch options]
--block '[96,1,1]' --grid '[1,82,74]' --inner-steps 9
```

Use your GPU’s output in the blocking run:

```bash
./build/bin/culbm-single \
    --device 0 --domain '[384,384,384]' \
    --block '[96,1,1]' --grid '[1,82,74]' --inner-steps 9 \
    --stream-policy esotwist --opt-policy blocking --ddf-buffer ring \
    --q 27 --collision bgk \
    --steps 900 --batch-steps 90
```

For exact step counts, use `--batch-steps` divisible by `--inner-steps`, and `--steps` divisible by `--batch-steps`.

For comparison, run plain CUDA on the same domain:

```bash
./build/bin/culbm-single \
    --device 0 --domain '[384,384,384]' \
    --block '[32,4,2]' \
    --stream-policy pull --opt-policy plain --ddf-buffer double \
    --q 27 --collision bgk \
    --steps 900 --batch-steps 90
```

### `culbm-tune`

| Flag | Purpose | Default |
| --- | --- | --- |
| `--device` | CUDA-visible GPU index used for measurements. | `0` |
| `--q {15,19,27}` | Number of lattice velocities, using FP32. | |
| `--lambda` | Previously measured bandwidth ratio. | measured |
| `--gamma` | Previously measured partial-sector penalty. | measured |
| `--capacity` | Available capacity in MiB. | queried |
| `--i-min`, `--i-max` | Minimum and maximum temporal depth. | `2`, `20` |
| `--b-max` | Maximum tile edge. | `160` |

Supply known values to skip their measurements. With all three provided, no GPU is required. Queried capacity is the smaller of the persisting L2 and access-policy window limits.

### `culbm-single`

Pass vectors in brackets, for example `[96,1,1]`.

| Flag | Purpose | Default |
| --- | --- | --- |
| `--device` | CUDA device index. | `0` |
| `--block [bx,by,bz]` | CUDA block configuration for kernel launch. | |
| `--grid [gx,gy,gz]` | CUDA grid configuration; required for blocking and when the plain grid cannot be derived. | derived for plain |
| `--domain [Nx,Ny,Nz]` | Logical simulation domain; can exceed `blockDim * gridDim` when halo blocking is active. | |
| `--inner-steps` | Iterations per tile; required for blocking. | |
| `--stream-policy {pull,esotwist}` | Streaming strategy for the computation buffer. | |
| `--opt-policy {plain,blocking}` | Computation organization. | |
| `--ddf-buffer {single,double,ring}` | Global DDF storage; `single` requires `STORE` on every domain cell. | |
| `--q {15,19,27}` | Number of lattice velocities. | `27` |
| `--collision {bgk,mrt,cumulant}` | Collision operator; cumulant requires D3Q19 or D3Q27. | `bgk` |
| `--inv-tau` | Reciprocal relaxation time. | `0.5` |
| `--steps` | Total steps. | |
| `--batch-steps` | Steps per MLUPS report and field dump. | |
| `--init <path>` | Optional folder with `flag.dat`, `rho.dat`, `vx.dat`, `vy.dat`, `vz.dat`. Omitting this flag triggers uniform default initialization. | empty |
| `--output <path>` | Output directory for binary dumps; required when dumping. | |
| `--dump-rho`, `--dump-vx`, `--dump-vy`, `--dump-vz` | When present, dump per-field snapshots every `--batch-steps` steps as `rho_<step>.dat`, `vx_<step>.dat`, … | disabled |

Supported combinations:

| Stream policy | Optimization | Global DDF storage |
| --- | --- | --- |
| `pull` | `plain` | `double` |
| `esotwist` | `plain` | `single` |
| `esotwist` | `blocking` | `double` |
| `esotwist` | `blocking` | `ring` |

`single` uses one full-domain buffer, `double` uses two, and `ring` uses a rolling buffer.

Blocking requires a compatible GPU and driver on Linux x86-64. Multi-step blocking does not support general periodic fluid-domain halos.

### `culbm-multi`

The multi-GPU solver is under development.

### Visualizations

- [Single-GPU decomposition](https://raw.githack.com/ShuyuanGe/cuLBM/master/visualization/single_dev_blocking.html)
- [Multi-GPU decomposition](https://raw.githack.com/ShuyuanGe/cuLBM/master/visualization/multi_dev_blocking.html)
