# GPU calibration and tile selection

`culbm-tune` uses model `calibrated_service_cost_v1`. Its calibrated final-store
service cost is

$$
\alpha=1+\lambda(\beta_{\mathrm{sec}}-1)
 +(2\gamma-(1+\lambda))\beta_{\mathrm{edge}}.
$$

The score uses tile geometry, temporal depth, lambda and gamma. The persisting
capacity (`--capacity`, MiB) is a hard feasibility bound on tile DDF bytes.
By default the search covers depth 2 through 20, x extents divisible by 32 up
to 1024, y greater than or equal to z, and positive valid interiors. There is
no fixed y or z cap. `--b-max` explicitly restricts all three extents; the JSON
report records the restriction.

This deterministic replay uses no CUDA device queries or measurements:

```bash
./build/bin/culbm-tune --q 27 \
    --lambda .18668 --gamma 10.3660 --capacity 60 \
    --report-json selection.json
```

Expected selection is `B=[96,82,74]`, `I=9`, with `1307805` feasible candidates.
If an input is omitted, only the missing persisting capacity or calibrations
are queried or measured on `--device`. The report parent directory must exist.
Total L2 is a measured device property used for calibration and experiment
sizing, not an input to the score or the offline replay command.

## Calibration protocol

The lambda calibration is a **two-size copy measurement**, rather than the
original full copy sweep. Each buffer uses total-L2/4 bytes for the L2 control
and 3 times total L2 for the DRAM control, rounded down to 256 bytes. Each of
101 samples warms up with two passes, then event-times at least 16 GiB of
useful read-plus-write traffic. Lambda is the ratio of median DRAM and L2
bandwidth. The report retains all elapsed times, bandwidths, sizes, and passes.

The gamma calibration retains the matched 4F and 3F+1P store controls: 128
threads per block, 8 blocks per SM, 101 paired samples, one warm-up launch per
timed control, a 1 GiB buffer, and partial offset 7. Gamma is
`4 * median(partial_ms) / median(full_ms) - 3`. Sample arrays remain in acquisition
order. These constants and kernels are unchanged from the standalone tools.

## JSON contract

`schema_version` is 2 and `model_id` is `calibrated_service_cost_v1`.
`inputs` records q, lambda, gamma, `capacity_mib` and `capacity_bytes`.
`sources` distinguishes provided inputs from measured or queried values.
`search` records depth limits, the x cap of 1024, optional `b_max`, and
`candidate_count`. `selection` contains `B`, `I`, predicted `score`, `block`,
and `grid`.

`device` is null for three-input offline replay. Otherwise it records the
CUDA-visible index, GPU name, driver/runtime versions, compute capability,
SM count and hardware `l2_capacity_bytes`. Experiment runners use this hardware
property to size >2L2 domains and oversized controls.

`calibration.lambda` contains method `two_size_copy`, constants, and `l2`/`dram`
controls with `bytes`, `passes`, `median_gbps`, `elapsed_ms`, and `bandwidth_gbps`.
`calibration.gamma` contains method `matched_pure_store`, constants, medians,
`full_ms`, and `partial_ms`. Each measured control contains 101 samples.
Provided parameters have null calibration records; replay does not create
synthetic measurements.

## Regression check without a GPU

```bash
cmake --build build --target culbm-tune-model-check
./build/bin/culbm-tune-model-check
```

This checks 18 selections for Q15/19/27 using six frozen calibrations, all
42,449,228 candidate visits, a nonzero-edge service-cost score, partial-sector
response, the exact capacity boundary, and an explicitly restricted search.
It does not validate calibration on hardware.
