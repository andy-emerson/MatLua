#!/usr/bin/env python3
"""NumPy int64 twin for tests/bench/i64_surface.rs generation rules."""

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
    # Match Rust: x = 1; then x = x.wrapping_add(17) per element (row-major).
    data = np.empty(n * n, dtype=np.int64)
    x = 1
    for i in range(n * n):
        data[i] = np.int64(x if x < 2**63 else x - 2**64)
        x = (x + 17) & ((1 << 64) - 1)
    return data.reshape(n, n)


def vec_n(n: int) -> np.ndarray:
    # (i as i64).wrapping_mul(3).wrapping_add(1)
    i = np.arange(n, dtype=np.int64)
    return i * np.int64(3) + np.int64(1)


def budget(n: int, heavy: bool) -> tuple[int, int]:
    if heavy:
        if n >= 1024:
            return 3, 1
        if n >= 256:
            return 6, 2
        return 15, 3
    if n >= 1024:
        return 8, 2
    if n >= 256:
        return 20, 4
    return 50, 8


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
        b = dense(n)
        v = vec_n(n)
        it, wrm = budget(n, False)
        ith, wrmh = budget(n, True)

        emit("elem_add", n, time_ms(it, wrm, lambda: a + b))
        emit("elem_mul", n, time_ms(it, wrm, lambda: a * b))
        emit("sum", n, time_ms(it, wrm, lambda: int(a.sum())))
        emit("min", n, time_ms(it, wrm, lambda: int(a.min())))
        emit("transpose", n, time_ms(it, wrm, lambda: a.T.copy()))
        emit("dot", n, time_ms(it, wrm, lambda: int(v @ v)))
        emit("matmul", n, time_ms(ith, wrmh, lambda: a @ b))
        u = v  # length n
        emit("unique", n, time_ms(it, wrm, lambda: np.unique(u)))
        emit("isin", n, time_ms(it, wrm, lambda: np.isin(a, v)))


if __name__ == "__main__":
    main()
