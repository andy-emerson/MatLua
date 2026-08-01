#!/usr/bin/env python3
"""NumPy side of compose_chain (XᵀX and normal equations)."""
import argparse
import sys
import time

import numpy as np

def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]

def time_ms(iters, warm, fn):
    for _ in range(warm):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return median(samples)

def design(k):
    m = 4 * k
    x = np.empty((m, k), dtype=np.float64)
    v = 0.001
    for i in range(m):
        for j in range(k):
            x[i, j] = v
            v += 0.000017
    y = 0.1 + np.arange(m, dtype=np.float64) * 0.01
    return x, y

def budget(n):
    if n <= 64:
        return 40, 8
    if n <= 256:
        return 12, 3
    return 5, 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s]
    print("face\top\tn\tms")
    for k in sizes:
        x, y = design(k)
        iters, warm = budget(k)
        # NumPy path is already "short" (view transpose)
        ms = time_ms(iters, warm, lambda: x.T @ x)
        print(f"numpy\txtx_short\t{k}\t{ms:.6f}")
        # long-style: force copy of transpose
        ms = time_ms(iters, warm, lambda: np.asarray(x.T.copy()) @ x)
        print(f"numpy\txtx_long\t{k}\t{ms:.6f}")
        ms = time_ms(iters, warm, lambda: np.linalg.solve(x.T @ x, x.T @ y))
        print(f"numpy\tnormal_eq_short\t{k}\t{ms:.6f}")
        ms = time_ms(
            iters,
            warm,
            lambda: np.linalg.solve(x.T.copy() @ x, x.T.copy() @ y),
        )
        print(f"numpy\tnormal_eq_long\t{k}\t{ms:.6f}")

if __name__ == "__main__":
    main()
