# Lambda

```bash
nvcc measure.cu -o measure
./measure
```

The program uses the same `cg` SM-copy kernel at two footprints, where $C_{\mathrm{L2}}$ is the CUDA-reported L2 capacity. \
For L2 bandwidth, each array is $C_{\mathrm{L2}}/4$. \
For DRAM bandwidth, each array is $3C_{\mathrm{L2}}$. \
It prints both bandwidths with four significant digits and $\lambda = B_{\mathrm{DRAM}} / B_{\mathrm{L2}}$ with five decimal places.
