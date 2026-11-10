import argparse

import numpy as np


def interior_fraction(Bx, By, Bz, I):
    r = I - 1
    return (1 - 2 * r / Bx) * (1 - 2 * r / By) * (1 - 2 * r / Bz)


def beta(Bx, I, b_max):
    r = I - 1
    L = Bx - 2 * r
    warp = np.arange(0, b_max, 32)
    x_lo = np.maximum(r[..., None], warp)
    x_hi = np.minimum(Bx[..., None] - r[..., None], warp + 32)
    phase = np.arange(8)
    sector_lo = (phase[:, None] + x_lo[..., None, :]) // 8
    sector_hi = (phase[:, None] + x_hi[..., None, :] - 1) // 8 + 1
    sectors = np.where(x_hi[..., None, :] > x_lo[..., None, :], sector_hi - sector_lo, 0)
    gcd = np.gcd(L, 8)
    P = 8 // gcd
    orbit = phase % gcd[..., None] == 0
    n_sec = (sectors.sum(axis=-1) * orbit).sum(axis=-1)
    left_edge = (phase + r[..., None]) % 8 != 0
    right_edge = (phase + Bx[..., None] - r[..., None]) % 8 != 0
    n_edge = (left_edge & orbit).sum(axis=-1) + (right_edge & orbit).sum(axis=-1)
    return (
        np.divide(8 * n_sec, P * L, out=np.ones(n_sec.shape), where=L > 0),
        np.divide(4 * n_edge, P * L, out=np.zeros(n_edge.shape), where=L > 0),
    )


def store_penalty(Bx, By, Bz, I, capacity, cell_bytes, lambda_, gamma, b_max):
    volume = cell_bytes * Bx * By * Bz
    beta_sec, beta_edge = beta(Bx, I, b_max)
    exponent = np.zeros(volume.shape)
    np.divide(-volume, capacity - volume, out=exponent, where=volume < capacity)
    probability = np.where(volume == capacity, 1, 1 - np.exp(exponent))
    return 1 + lambda_ * (beta_sec - 1) + probability * (2 * gamma - (1 + lambda_)) * beta_edge


def score(Bx, By, Bz, I, capacity, cell_bytes, lambda_, gamma, b_max):
    halo = interior_fraction(Bx, By, Bz, I)
    penalty = store_penalty(Bx, By, Bz, I, capacity, cell_bytes, lambda_, gamma, b_max)
    return halo / ((1 + penalty * halo) / (2 * I) + lambda_ * (1 - 1 / I))


def main():
    parser = argparse.ArgumentParser(description="Select a tile and temporal depth for a GPU.")
    parser.add_argument("--lambda", dest="lambda_", type=float, required=True, help="Measured bandwidth ratio")
    parser.add_argument("--gamma", type=float, required=True, help="Measured partial-sector penalty")
    parser.add_argument("--capacity", type=float, required=True, metavar="MiB", help="Available capacity in MiB")
    parser.add_argument("--q", type=int, choices=(15, 19, 27), required=True, help="Velocity count (FP32)")
    parser.add_argument("--i-min", type=int, default=2)
    parser.add_argument("--i-max", type=int, default=20)
    parser.add_argument("--b-max", type=int, default=160)
    args = parser.parse_args()
    assert np.isfinite(args.lambda_) and args.lambda_ >= 0
    assert np.isfinite(args.gamma) and args.gamma >= 0
    assert np.isfinite(args.capacity) and args.capacity > 0
    assert 2 <= args.i_min <= args.i_max and args.b_max >= 32
    capacity = args.capacity * 2**20
    cell_bytes = args.q * 4
    I, Bx, By, Bz = np.ogrid[args.i_min : args.i_max + 1, 32 : args.b_max + 1 : 32, 1 : args.b_max + 1, 1 : args.b_max + 1]
    valid = (Bx * By * Bz * cell_bytes <= capacity) & (Bx > 2 * (I - 1)) & (By >= Bz) & (Bz > 2 * (I - 1))
    assert np.any(valid), "no valid tile fits the specified capacity and search bounds"
    scores = np.where(valid, score(Bx, By, Bz, I, capacity, cell_bytes, args.lambda_, args.gamma, args.b_max), -np.inf)
    i, bx, by, bz = np.unravel_index(np.argmax(scores), scores.shape)
    bx, by, bz = 32 * (bx + 1), by + 1, bz + 1
    print(f"B*: [{bx},{by},{bz}]")
    print(f"I*: {i + args.i_min}")
    print(f"--block '[{bx},1,1]' --grid '[1,{by},{bz}]' --inner-steps {i + args.i_min}")


if __name__ == "__main__":
    main()
