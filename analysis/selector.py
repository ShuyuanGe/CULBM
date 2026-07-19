import numpy as np


LAMBDA = 0.19103
GAMMA = 10.1228
L2 = 128 * 2**20
PERSISTING_L2 = 80 * 2**20

I_MIN = 2
I_MAX = 20
B_MAX = 160
CELL_BYTES = 27 * 4


def h(Bx, By, Bz, I):
    r = I - 1
    return (1 - 2 * r / Bx) * (1 - 2 * r / By) * (1 - 2 * r / Bz)


def beta(Bx, I):
    r = I - 1
    L = Bx - 2 * r
    warp = np.arange(0, B_MAX, 32)
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


def alpha(Bx, By, Bz, I):
    V = CELL_BYTES * Bx * By * Bz
    beta_sec, beta_edge = beta(Bx, I)
    q = np.zeros(V.shape)
    np.divide(-V, L2 - V, out=q, where=V < L2)
    p = 1 - np.exp(q)
    return 1 + LAMBDA * (beta_sec - 1) + p * (2 * GAMMA - (1 + LAMBDA)) * beta_edge


def S0(Bx, By, Bz, I):
    H = h(Bx, By, Bz, I)
    return H / ((1 + H) / (2 * I) + LAMBDA * (1 - 1 / I))


def S(Bx, By, Bz, I):
    H = h(Bx, By, Bz, I)
    return H / ((1 + alpha(Bx, By, Bz, I) * H) / (2 * I) + LAMBDA * (1 - 1 / I))


I, Bx, By, Bz = np.ogrid[I_MIN : I_MAX + 1, 32 : B_MAX + 1 : 32, 1 : B_MAX + 1, 1 : B_MAX + 1]
valid = (Bx * By * Bz <= PERSISTING_L2 / CELL_BYTES) & (Bx > 2 * (I - 1)) & (By >= Bz) & (Bz > 2 * (I - 1))

score = np.where(valid, S(Bx, By, Bz, I), -np.inf)
i, bx, by, bz = np.unravel_index(np.argmax(score), score.shape)
bx, by, bz = 32 * (bx + 1), by + 1, bz + 1
print(f"B*: {bx}x{by}x{bz}")
print(f"I*: {i + I_MIN}")
