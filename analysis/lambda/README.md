# Lambda

```bash
nvcc measure.cu -o measure
./measure
```

The program runs the same copy kernel over separately allocated source and destination arrays, where $C_{\mathrm{L2}}$ is the CUDA-reported L2 capacity. \
For L2 bandwidth, each array is $7C_{\mathrm{L2}}/16$. \
For DRAM bandwidth, each array is $C_{\mathrm{L2}}$. \
It prints both bandwidths and $\lambda = B_{\mathrm{DRAM}} / B_{\mathrm{L2}}$.
