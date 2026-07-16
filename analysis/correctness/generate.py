import argparse
from pathlib import Path

import numpy as np

SHAPE = (384, 384, 768)  # z, y, x
CUBE = np.s_[128:256, 128:256, 128:256]

LOAD = 1 << 0
EQU = 1 << 1
COLLIDE = 1 << 2
BOUNCE_BACK = 1 << 3
STORE = 1 << 4
DUMP_RHO = 1 << 27
DUMP_VX = 1 << 28
DUMP_VY = 1 << 29
DUMP_VZ = 1 << 30

WALL = LOAD | BOUNCE_BACK | STORE
EQUILIBRIUM = EQU | STORE
FLUID = LOAD | COLLIDE | STORE | DUMP_RHO | DUMP_VX | DUMP_VY | DUMP_VZ


parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
output = parser.parse_args().output
output.mkdir(parents=True, exist_ok=True)

flag = np.memmap(output / "flag.dat", dtype="<u4", mode="w+", shape=SHAPE)
flag[:] = FLUID
flag[:, :, (0, -1)] = EQUILIBRIUM
flag[CUBE] = WALL
flag[:, (0, -1), :] = WALL
flag[(0, -1), :, :] = WALL
flag.flush()

vx = np.memmap(output / "vx.dat", dtype="<f4", mode="w+", shape=SHAPE)
vx[:] = 0.0
vx[:, :, 0] = 0.05
vx[:, (0, -1), :] = 0.0
vx[(0, -1), :, :] = 0.0
vx.flush()
