# Gamma

In `env-cu12` or `env-cu13`:

```bash
nvcc -std=c++20 -I"$CONDA_PREFIX/include/cccl" measure.cu -o measure
./measure
```

Measures 101-sample median times for 1 GiB streaming stores at 32 warps per SM.
`4F` writes four full sectors per 128-byte warp unit; `3F+1P` replaces one with a partial sector.

$$
\gamma = 4\frac{T_{\mathrm{3F+1P}}}{T_{\mathrm{4F}}} - 3.
$$

With full-sector demand normalized to one, $\gamma$ is the effective partial-sector demand.
