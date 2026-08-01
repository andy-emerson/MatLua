#!/usr/bin/env python3
"""NumPy side of the P5 dense desk-math comparison.

Emits TSV lines: face\\top\\tn\\tms  (median wall ms), matching bench_dense.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


def time_ms(iters: int, warm: int, fn) -> float:
    for _ in range(warm):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return median(samples)


def dense(n: int) -> np.ndarray:
    data = np.empty((n, n), dtype=np.float64)
    x = 0.001
    for i in range(n * n):
        data.flat[i] = x
        x += 0.000017
    return data


def spd(n: int) -> np.ndarray:
    a = dense(n)
    return a.T @ a + np.eye(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    print("face\top\tn\tms")
    for n in sizes:
        if n >= 1024:
            iters, warm = 5, 2
        elif n >= 256:
            iters, warm = 15, 3
        else:
            iters, warm = 40, 5

        a = dense(n)
        b = dense(n)
        ms = time_ms(iters, warm, lambda: a @ b)
        print(f"numpy\tmatmul\t{n}\t{ms:.6f}")

        if n >= 1024:
            iters, warm = 5, 2
        elif n >= 256:
            iters, warm = 12, 3
        else:
            iters, warm = 30, 5
        a = spd(n)
        rhs = 0.5 + np.arange(n, dtype=np.float64) * 0.01
        ms = time_ms(iters, warm, lambda: np.linalg.solve(a, rhs))
        print(f"numpy\tsolve\t{n}\t{ms:.6f}")

        if n >= 1024:
            iters, warm = 20, 3
        else:
            iters, warm = 50, 5
        x = dense(n)
        y = dense(n)
        ms = time_ms(iters, warm, lambda: x + y)
        print(f"numpy\telem_add\t{n}\t{ms:.6f}")


if __name__ == "__main__":
    main()
