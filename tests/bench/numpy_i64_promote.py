#!/usr/bin/env python3
"""NumPy twin for i64→f64 promote-out (int64 inputs, float results / LA)."""

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
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return median(samples)


def dense(n: int) -> np.ndarray:
    data = np.empty(n * n, dtype=np.int64)
    x = 1
    for i in range(n * n):
        data[i] = np.int64(x if x < 2**63 else x - 2**64)
        x = (x + 17) & ((1 << 64) - 1)
    return data.reshape(n, n)


def budget(n: int, heavy: bool) -> tuple[int, int]:
    # Match tests/bench/i64_promote.rs budget()
    if heavy:
        if n >= 1024:
            return 2, 1
        if n >= 256:
            return 4, 1
        return 10, 2
    if n >= 1024:
        return 6, 2
    if n >= 256:
        return 12, 3
    return 30, 5


def emit(op: str, n: int, ms: float) -> None:
    print(f"numpy\t{op}\t{n}\t{ms:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print("face\top\tn\tms")
    for n in sizes:
        a = dense(n)
        af = a.astype(np.float64)
        s = (a.T @ a) + n * np.eye(n, dtype=np.int64)
        sf = s.astype(np.float64)
        v = (np.arange(n, dtype=np.int64) * 3 + 1).astype(np.float64)
        it, wrm = budget(n, False)
        ith, wrmh = budget(n, True)
        emit("mean", n, time_ms(it, wrm, lambda: float(a.mean())))
        emit("std", n, time_ms(it, wrm, lambda: float(a.std(ddof=0))))
        emit("median", n, time_ms(it, wrm, lambda: float(np.median(a))))
        emit("quantile", n, time_ms(it, wrm, lambda: float(np.quantile(a, 0.75))))
        emit("norm", n, time_ms(it, wrm, lambda: float(np.linalg.norm(af))))
        emit("solve", n, time_ms(ith, wrmh, lambda: np.linalg.solve(sf, v)))
        emit("cholesky", n, time_ms(ith, wrmh, lambda: np.linalg.cholesky(sf)))
        emit("qr", n, time_ms(ith, wrmh, lambda: np.linalg.qr(af)))


if __name__ == "__main__":
    main()
