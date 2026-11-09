# Lambda

In `env-cu12` or `env-cu13`:

```bash
nvcc -std=c++20 -I"$CONDA_PREFIX/include/cccl" measure.cu -o measure
./measure
```

Measures median `cg` copy bandwidth with arrays of $C_{\mathrm{L2}}/4$ (L2) and
$3C_{\mathrm{L2}}$ (DRAM), where $C_{\mathrm{L2}}$ is the CUDA-reported L2 capacity.
Reports $\lambda = B_{\mathrm{DRAM}} / B_{\mathrm{L2}}$.
