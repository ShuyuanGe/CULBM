# Correctness

Compare plain CUDA and ring blocking bitwise after 1,539 steps on a `384 × 384 × 384` domain with an obstacle.

Requires a built `culbm-single`, Python with NumPy, over 12.45 GiB of free GPU memory plus CUDA overhead, and at least 2.11 GiB of free disk space.

From the repository root:

```bash
python analysis/correctness/generate.py /tmp/culbm-case
analysis/correctness/run.sh ./build/bin/culbm-single /tmp/culbm-case /tmp/culbm-dumps
analysis/correctness/compare.sh /tmp/culbm-dumps
```

The blocking parameters target RTX 5090. For another GPU, use `culbm-tune` and update `run.sh`. If changing the domain, update `generate.py` and `run.sh`; if changing the step count, update `compare.sh`. Set `--steps` and `--batch-steps` to multiples of `--inner-steps`.

`compare.sh` compares corresponding density and velocity fields between the two runs, not against an external reference.
