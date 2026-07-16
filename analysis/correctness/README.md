# Correctness

Requires Python with NumPy and a built `single_dev_expt_main`.

This case was configured for the NVIDIA RTX PRO 6000 Blackwell Workstation Edition.

| Item | Value |
| --- | ---: |
| GPU memory | 96 GB GDDR7 with ECC; CUDA reports 102,011,043,840 bytes (95.01 GiB) usable |
| L2 cache | 128 MiB total; 80 MiB maximum persisting-L2 set-aside |
| Domain | `768 x 384 x 384` |
| Temporal configuration | `B=(96,94,86)`, `I=9` |
| Resident footprint of `B` | 83,814,912 bytes (83.81 MB, 79.93 MiB) |
| Plain CUDA allocation | 26,726,105,088 bytes (26.73 GB, 24.89 GiB) |
| Temporal CUDA allocation | 17,573,810,688 bytes (17.57 GB, 16.37 GiB) |

The allocation totals sum the solver's explicit device allocations and exclude
the CUDA context and allocator overhead. The two paths run sequentially, so the
case requires more than 24.89 GiB of usable device memory. The resident
footprint of `B` is only 71,168 bytes below the 80 MiB persisting-L2 limit.

For another GPU, adjust the domain-dependent constants consistently across the
three scripts and choose a feasible `B` and `I` in `run.sh`. Bitwise equality
remains the requirement.

```bash
python generate.py case/
./run.sh ../../build/src/single_dev_expt_main case/ dumps/
python compare.py dumps/
```
