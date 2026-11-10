import os
import subprocess
from pathlib import Path


def run_experiment():
    working_dir = Path(__file__).parent.parent.resolve()
    executable = (working_dir / "build" / "bin" / "culbm-multi").resolve()

    if not os.path.exists(executable):
        print(f"Error: Executable not found at {executable}")
        return

    # Common parameters for Weak Scaling (Constant work per GPU)
    blk_dim = "[32,32,1]"
    grid_dim = "[8,8,228]"
    nstep = 500
    dstep = 50
    inv_tau = 0.5

    # Experiments configuration: (Number of GPUs, Topology)
    experiments = [
        (1, "[1,1,1]"),
        (2, "[2,1,1]"),
        (3, "[3,1,1]"),
        (4, "[2,2,1]"),
    ]

    print(f"{'GPUs':<5} | {'Topology':<10} | {'Status':<10}")
    print("-" * 40)

    for n_gpu, dev_dim in experiments:
        log_file = f"{n_gpu}gpu.log"

        cmd = [
            executable,
            "--device-grid",
            dev_dim,
            "--grid",
            grid_dim,
            "--block",
            blk_dim,
            "--steps",
            str(nstep),
            "--batch-steps",
            str(dstep),
            "--inv-tau",
            str(inv_tau),
        ]

        print(f"Running {n_gpu} GPU(s)...", end="\r")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        with open(log_file, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

        if result.returncode != 0:
            print(f"{n_gpu:<5} | {dev_dim:<10} | {'FAILED':<10} | {'See ' + log_file}")
        else:
            print(f"{n_gpu:<5} | {dev_dim:<10} | {'SUCCESS':<10} | {'Log saved'}")


if __name__ == "__main__":
    run_experiment()
