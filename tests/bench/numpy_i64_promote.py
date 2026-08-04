#!/usr/bin/env python3
"""NumPy twin for i64→f64 promote-out (int64 inputs, float results / LA)."""

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
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return median(samples)


def dense(n: int) -> np.ndarray:
    # Vectorized arithmetic progression (mod 2^64 as int64 wrap).
    i = np.arange(n * n, dtype=np.uint64)
    data = (1 + i * np.uint64(17)).astype(np.int64)
    return data.reshape(n, n)


def budget(n: int, heavy: bool) -> tuple[int, int]:
    # Match tests/bench/i64_promote.rs budget()
    if heavy:
        if n >= 4096:
            return 5, 1
        if n >= 1024:
            return 5, 1
        if n >= 256:
            return 5, 2
        return 11, 2
    if n >= 4096:
        return 5, 2
    if n >= 1024:
        return 6, 2
    if n >= 256:
        return 12, 3
    return 30, 5



def emit(op: str, n: int, ms: float) -> None:
    print(f"numpy\t{op}\t{n}\t{ms:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024,4096")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print("face\top\tn\tms")
    for n in sizes:
        a = dense(n)
        it, wrm = budget(n, False)
        ith, wrmh = budget(n, True)
        emit("mean", n, time_ms(it, wrm, lambda: float(a.mean())))
        emit("std", n, time_ms(it, wrm, lambda: float(a.std(ddof=0))))
        emit("median", n, time_ms(it, wrm, lambda: float(np.median(a))))
        emit("quantile", n, time_ms(it, wrm, lambda: float(np.quantile(a, 0.75))))
        emit("norm", n, time_ms(it, wrm, lambda: float(np.linalg.norm(a))))
        # Same integer SPD as i64_promote.rs spd_i64(); int64 in, so NumPy
        # pays the int64->f64 promotion inside the clock like MatLua does.
        i = np.arange(n)
        j = np.arange(n)
        s = ((i[:, None] + 2 * j[None, :]) % 7).astype(np.int64)
        s[np.diag_indices(n)] += n + 1
        v = (np.arange(n, dtype=np.int64) * 3 + 1)
        emit("solve", n, time_ms(ith, wrmh, lambda: np.linalg.solve(s, v)))
        emit("cholesky", n, time_ms(ith, wrmh, lambda: np.linalg.cholesky(s)))
        # NumPy has no cho_solve (SciPy does); the honest NumPy-user reference
        # for an SPD solve is np.linalg.solve on the same system.
        emit("cholesky_solve", n, time_ms(ith, wrmh, lambda: np.linalg.solve(s, v)))
        emit("qr", n, time_ms(ith, wrmh, lambda: np.linalg.qr(a)))



if __name__ == "__main__":
    main()
