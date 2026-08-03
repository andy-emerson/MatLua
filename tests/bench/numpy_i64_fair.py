#!/usr/bin/env python3
"""NumPy twin for i64_surface.

Most ops: NumPy **int64** (same dtype).

**matmul** is special: NumPy has int64 arrays but **no integer BLAS**
(OpenBLAS/MKL are float-only; see numpy#14556). int64@int64 is a slow
fallback and is **not** a product performance bar. Matmul reference is
therefore **float64 BLAS** on the **same integer-valued** data (e.g. 3.0),
i.e. the ceiling a package like BLAS.wasm competes with. Exact wrapping
i64 product remains MatLua's job; the ratio is "how far from BLAS GEMM".
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
    # Vectorized arithmetic progression (mod 2^64 as int64 wrap).
    i = np.arange(n * n, dtype=np.uint64)
    data = (1 + i * np.uint64(17)).astype(np.int64)
    return data.reshape(n, n)


def dense2(n: int) -> np.ndarray:
    i = np.arange(n * n, dtype=np.uint64)
    data = (2 + i * np.uint64(13)).astype(np.int64)
    return data.reshape(n, n)


def vec_n(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.int64)
    return i * np.int64(3) + np.int64(1)


def budget(n: int, heavy: bool) -> tuple[int, int]:
    if heavy:
        if n >= 4096:
            return 1, 0
        if n >= 1024:
            return 3, 1
        if n >= 256:
            return 6, 2
        return 15, 3
    if n >= 4096:
        return 2, 1
    if n >= 1024:
        return 8, 2
    if n >= 256:
        return 20, 4
    return 50, 8



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
        b = dense2(n)
        v = vec_n(n)
        it, wrm = budget(n, False)
        ith, wrmh = budget(n, True)
        emit("zeros", n, time_ms(it, wrm, lambda: np.zeros((n, n), dtype=np.int64)))
        emit("ones", n, time_ms(it, wrm, lambda: np.ones((n, n), dtype=np.int64)))
        emit("full", n, time_ms(it, wrm, lambda: np.full((n, n), 7, dtype=np.int64)))
        emit("eye", n, time_ms(it, wrm, lambda: np.eye(n, dtype=np.int64)))
        emit("arange", n, time_ms(it, wrm, lambda: np.arange(0, n, dtype=np.int64)))
        emit("copy", n, time_ms(it, wrm, lambda: a.copy()))
        if n % 2 == 0:
            emit("reshape", n, time_ms(it, wrm, lambda: a.reshape(n // 2, n * 2)))
        t = a.copy()
        emit("fill", n, time_ms(it, wrm, lambda: t.fill(3)))
        emit("elem_add", n, time_ms(it, wrm, lambda: a + b))
        emit("elem_sub", n, time_ms(it, wrm, lambda: a - b))
        emit("elem_mul", n, time_ms(it, wrm, lambda: a * b))
        # avoid div0: b never 0 with dense2 starting at 2
        emit("elem_div", n, time_ms(it, wrm, lambda: a // b))
        emit("sum", n, time_ms(it, wrm, lambda: int(a.sum())))
        emit("min", n, time_ms(it, wrm, lambda: int(a.min())))
        emit("max", n, time_ms(it, wrm, lambda: int(a.max())))
        emit("transpose", n, time_ms(it, wrm, lambda: a.T.copy()))
        emit("dot", n, time_ms(it, wrm, lambda: int(v @ v)))
        # BLAS reference: f64 with integer-valued entries (not int64@int64 fallback).
        af = a.astype(np.float64, copy=False)
        bf = b.astype(np.float64, copy=False)
        emit("matmul", n, time_ms(ith, wrmh, lambda: af @ bf))
        emit("unique", n, time_ms(it, wrm, lambda: np.unique(v)))
        emit("isin", n, time_ms(it, wrm, lambda: np.isin(a, v)))


if __name__ == "__main__":
    main()
