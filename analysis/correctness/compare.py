import argparse
from pathlib import Path

import numpy as np

FIELDS = ("rho", "vx", "vy", "vz")
SHAPE = (384, 384, 768)  # z, y, x
STEP = 1539


parser = argparse.ArgumentParser()
parser.add_argument("dump_dir", type=Path)
dump_dir = parser.parse_args().dump_dir

plain = {}
for field in FIELDS:
    runs = tuple(
        np.memmap(
            dump_dir / run / f"{field}_{STEP}.dat",
            dtype="<u4",
            mode="r",
            shape=SHAPE,
        )
        for run in ("plain", "temporal")
    )
    equal = runs[0] == runs[1]
    if not np.all(equal):
        z, y, x = np.argwhere(~equal)[0]
        values = tuple(int(run[z, y, x]) for run in runs)
        raise AssertionError(f"{field}: xyz=({x}, {y}, {z}), uint32={values}")
    plain[field] = runs[0]
    print(f"{field}: bitwise equal")

one = int(np.asarray(1.0, dtype="<f4").view("<u4"))
initial = {"rho": one, "vx": 0, "vy": 0, "vz": 0}
assert all(np.any(plain[field] != initial[field]) for field in FIELDS)
assert np.any(plain["rho"][1:-1, 1:-1, 766] != one) or np.any(plain["vx"][1:-1, 1:-1, 766] != 0)
shell = np.s_[127:257, 127:257, 127:257]
assert np.any(plain["vy"][shell] != 0) or np.any(plain["vz"][shell] != 0)
print("nontrivial propagation: pass")
