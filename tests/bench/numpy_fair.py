#!/usr/bin/env python3
"""NumPy side of fair full-surface bench — same generation rules as bench_all_fair.rs."""

from __future__ import annotations

import argparse
import time

import numpy as np


def median(xs: list[float]) -> float:
    # True median: average the middle pair on even counts (xs[n//2] alone
    # returns the worse of 2 samples - one contention stall became the cell).
    xs = sorted(xs)
    m = len(xs) // 2
    if len(xs) % 2 == 0 and len(xs) >= 2:
        return (xs[m - 1] + xs[m]) / 2
    return xs[m]


def time_ms(iters: int, warm: int, fn) -> float:
    for _ in range(warm):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return median(samples)


def dense_n(n: int) -> np.ndarray:
    data = np.empty((n, n), dtype=np.float64)
    x = 0.001
    for i in range(n * n):
        data.flat[i] = x
        x += 0.000017
    return data


def dense2_n(n: int) -> np.ndarray:
    data = np.empty((n, n), dtype=np.float64)
    x = 0.002
    for i in range(n * n):
        data.flat[i] = x
        x += 0.000013
    return data


def vec_n(n: int) -> np.ndarray:
    return 0.5 + np.arange(n, dtype=np.float64) * 0.01


def vec2_n(n: int) -> np.ndarray:
    return 0.25 + np.arange(n, dtype=np.float64) * 0.007


def spd_n(n: int) -> np.ndarray:
    if n >= 1024:
        ii, jj = np.indices((n, n))
        s = (0.01 * ((ii + 2 * jj) % 7)).astype(np.float64)
        s = s + np.eye(n) * (n + 1)
        return s
    a = dense_n(n)
    return a.T @ a + np.eye(n)


def budget(n: int, heavy: bool) -> tuple[int, int]:
    # >=5 odd samples at large n: real median, robust to shared-host stalls.
    if heavy:
        if n >= 4096:
            return 5, 1
        if n >= 1024:
            return 5, 1
        if n >= 256:
            return 6, 2
        return 15, 3
    if n >= 4096:
        return 5, 2
    if n >= 1024:
        return 8, 2
    if n >= 256:
        return 15, 3
    return 40, 5



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024,4096")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    print("face\top\tn\tms")
    for n in sizes:
        a = dense_n(n)
        b = dense2_n(n)
        v = vec_n(n)
        w = vec2_n(n)
        s = spd_n(n)
        rhs = vec_n(n)

        it, wrm = budget(n, False)
        print(f"numpy\tzeros\t{n}\t{time_ms(it, wrm, lambda: np.zeros((n, n), dtype=np.float64)):.6f}")
        print(f"numpy\tones\t{n}\t{time_ms(it, wrm, lambda: np.ones((n, n), dtype=np.float64)):.6f}")
        print(f"numpy\tfull\t{n}\t{time_ms(it, wrm, lambda: np.full((n, n), 1.5, dtype=np.float64)):.6f}")
        print(f"numpy\teye\t{n}\t{time_ms(it, wrm, lambda: np.eye(n, dtype=np.float64)):.6f}")
        print(f"numpy\tarange\t{n}\t{time_ms(it, wrm, lambda: np.arange(0.0, float(n), dtype=np.float64)):.6f}")
        print(f"numpy\tcopy\t{n}\t{time_ms(it, wrm, lambda: a.copy()):.6f}")
        if n % 2 == 0:
            shape = (n // 2, n * 2)
            print(f"numpy\treshape\t{n}\t{time_ms(it, wrm, lambda: a.reshape(shape)):.6f}")
        # in-place fill on a dedicated buffer (clone outside timer)
        tfill = a.copy()
        print(f"numpy\tfill\t{n}\t{time_ms(it, wrm, lambda: tfill.fill(3.0)):.6f}")
        print(f"numpy\telem_add\t{n}\t{time_ms(it, wrm, lambda: a + b):.6f}")
        print(f"numpy\telem_sub\t{n}\t{time_ms(it, wrm, lambda: a - b):.6f}")
        print(f"numpy\telem_mul\t{n}\t{time_ms(it, wrm, lambda: a * b):.6f}")
        print(f"numpy\telem_div\t{n}\t{time_ms(it, wrm, lambda: a / b):.6f}")
        print(f"numpy\telem_add_scalar\t{n}\t{time_ms(it, wrm, lambda: a + 2.5):.6f}")
        print(f"numpy\tsum\t{n}\t{time_ms(it, wrm, lambda: a.sum()):.6f}")
        print(f"numpy\tmean\t{n}\t{time_ms(it, wrm, lambda: a.mean()):.6f}")
        print(f"numpy\tmin\t{n}\t{time_ms(it, wrm, lambda: a.min()):.6f}")
        print(f"numpy\tmax\t{n}\t{time_ms(it, wrm, lambda: a.max()):.6f}")
        print(f"numpy\ttranspose\t{n}\t{time_ms(it, wrm, lambda: a.T.copy()):.6f}")
        print(f"numpy\tdot\t{n}\t{time_ms(it, wrm, lambda: float(np.dot(v, w))):.6f}")
        print(f"numpy\tnorm\t{n}\t{time_ms(it, wrm, lambda: float(np.linalg.norm(a, 'fro'))):.6f}")

        it, wrm = budget(n, True)
        print(f"numpy\tmatmul\t{n}\t{time_ms(it, wrm, lambda: a @ b):.6f}")
        print(f"numpy\tsolve\t{n}\t{time_ms(it, wrm, lambda: np.linalg.solve(s, rhs)):.6f}")
        print(f"numpy\tcholesky\t{n}\t{time_ms(it, wrm, lambda: np.linalg.cholesky(s)):.6f}")
        # NumPy has no cho_solve (SciPy does); the honest NumPy-user reference
        # for an SPD solve is np.linalg.solve on the same system.
        print(f"numpy\tcholesky_solve\t{n}\t{time_ms(it, wrm, lambda: np.linalg.solve(s, rhs)):.6f}")
        print(f"numpy\tqr\t{n}\t{time_ms(it, wrm, lambda: np.linalg.qr(a, mode='reduced')):.6f}")
        print(f"numpy\tsvd\t{n}\t{time_ms(it, wrm, lambda: np.linalg.svd(a, full_matrices=False)):.6f}")


if __name__ == "__main__":
    main()
