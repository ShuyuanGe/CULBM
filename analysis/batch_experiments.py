import datetime as dt
import pickle
import re
import subprocess
from pathlib import Path
import itertools
from tqdm import tqdm

def mul3(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])

work_dir = Path(__file__).parent.parent.resolve()
print(f"Working directory: {work_dir}")

BINARY = work_dir / "build/src/single_dev_expt_main"    
BLOCK_DIM = (32, 16, 2)
GRID_DIM = (2, 6, 38)
STREAM_POLICY = 1
OPT_POLICY = 3
INNER_LOOPS = range(1, 16)

DSTEP = 100
NSTEP = 1000

MAX_BLOCK_COUNT = 66
BLOCK_COUNTS = [
    (a, b, c)
    for a in range(1, MAX_BLOCK_COUNT + 1)
    for b in range(1, MAX_BLOCK_COUNT // a + 1)
    for c in range(1, MAX_BLOCK_COUNT // (a * b) + 1)
    if a * b * c <= MAX_BLOCK_COUNT
]
print(f"Total block count configurations: {len(BLOCK_COUNTS)}")

OUTPUT = Path(__file__).with_name(f"batch_experiment_results_s{STREAM_POLICY}o{OPT_POLICY}.pkl")


TB_BLOCK_SHAPE = mul3(BLOCK_DIM, GRID_DIM)

SPEED_RE = re.compile(r"speed = ([0-9]+(?:\.[0-9]+)?) \(MLUPS\)")


def fmt_triplet(values: tuple[int, int, int]) -> str:
    return f"[{values[0]},{values[1]},{values[2]}]"

BASE_ARGS = [
    "--blockDim", fmt_triplet(BLOCK_DIM),
    "--gridDim", fmt_triplet(GRID_DIM),
    "--streamPolicy", str(STREAM_POLICY),
    "--dstep", str(DSTEP),
    "--nstep", str(NSTEP),
    "--optPolicy", str(OPT_POLICY),
]

def get_domain_dim(M: tuple[int, int, int], B: tuple[int, int, int], I: int) -> tuple[int, int, int]:
    shrink = I - 1

    def dimension(m: int, b: int) -> int:
        if m < 2:
            return b
        core = b - (2 * shrink)
        face = b - shrink
        return core * (m - 2) + face * 2

    mx, my, mz = M
    bx, by, bz = B
    return (
        dimension(mx, bx),
        dimension(my, by),
        dimension(mz, bz),
    )


def load_existing_results():
    if not OUTPUT.exists():
        return {"metadata": {}, "records": []}
    with OUTPUT.open("rb") as fh:
        return pickle.load(fh)


def save_results(payload):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as fh:
        pickle.dump(payload, fh)


if not BINARY.exists():
    raise FileNotFoundError(f"solver binary not found: {BINARY}")

payload = load_existing_results()
records = payload["records"]

tested_configs = {(r["block_count"], r["inner_loop"]) for r in records}
all_configs = set(itertools.product(BLOCK_COUNTS, INNER_LOOPS))
completed_configs = tested_configs & all_configs
remaining_configs = all_configs - tested_configs

try:
    with tqdm(total=len(all_configs), initial=len(completed_configs), desc="Running experiments", unit="config") as pbar:
        for block_count in BLOCK_COUNTS:
            for inner_loop in INNER_LOOPS:
                if (block_count, inner_loop) in tested_configs:
                    continue

                domain_dim = get_domain_dim(M=block_count, B=TB_BLOCK_SHAPE, I=inner_loop)
                cmd = [
                    str(BINARY),
                    *BASE_ARGS,
                    "--innerLoop", str(inner_loop),
                    "--domainDim", fmt_triplet(domain_dim),
                ]

                pbar.set_postfix({"block": str(block_count), "inner_loop": inner_loop})
                completed = subprocess.run(cmd, text=True, capture_output=True, cwd=BINARY.parent.parent)
                speeds = [float(v) for v in SPEED_RE.findall(completed.stdout)][1:] # skip the first step

                record = {
                    "inner_loop": inner_loop,
                    "block_count": block_count,
                    "domain_size": domain_dim,
                    "block_size": TB_BLOCK_SHAPE,
                    "speeds": speeds,
                }
                records.append(record)
                tested_configs.add((block_count, inner_loop))
                completed_configs.add((block_count, inner_loop))
                remaining_configs.remove((block_count, inner_loop))

                if completed.returncode:
                    tqdm.write(f"Error: {completed.stderr}")

                payload["metadata"] = {
                    "last_updated": dt.datetime.now().isoformat(),
                    "binary": str(BINARY),
                    "block_dim": BLOCK_DIM,
                    "grid_dim": GRID_DIM,
                    "tb_block_shape": TB_BLOCK_SHAPE,
                    "stream_policy": STREAM_POLICY,
                    "opt_policy": OPT_POLICY,
                    "dstep": DSTEP,
                    "nstep": NSTEP,
                }
                payload["records"] = records
                save_results(payload)
                pbar.update(1)

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    save_results(payload)
    print(f"Stored {len(records)} records to {OUTPUT}")


