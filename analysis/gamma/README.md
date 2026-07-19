# Gamma

```bash
nvcc measure.cu -o measure
./measure
```

The program measures the 101-sample median times of two 1 GiB streaming-store controls at 32 active warps per SM. 4F writes four full sectors per 128-byte warp unit, while 3F+1P replaces one with a partial sector.

$$
\gamma = 4\frac{T_{\mathrm{3F+1P}}}{T_{\mathrm{4F}}} - 3.
$$

With one full-sector service demand normalized to one, $\gamma$ is the effective demand of the partial sector, so a larger $\gamma$ means a stronger final-store misalignment penalty.

The program prints both times with five decimal places and $\gamma$ with four decimal places.
