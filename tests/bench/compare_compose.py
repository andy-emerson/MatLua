#!/usr/bin/env python3
"""Join compose_chain + numpy_compose TSVs; ratios vs NumPy short path."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def load_tsv(text):
    d = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("face"):
            continue
        face, op, n, ms = line.split("\t")
        d[(face, op, int(n))] = float(ms)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024")
    args = ap.parse_args()
    sizes = args.sizes

    rust = subprocess.check_output(
        [
            "cargo",
            "test",
            "--release",
            "--features",
            "lua",
            "--test",
            "compose_chain",
            "--",
            "--run",
            "--sizes",
            sizes,
        ],
        cwd=ROOT,
        text=True,
    )
    # cargo prints noise; keep TSV lines
    rust_lines = "\n".join(
        ln for ln in rust.splitlines() if ln.startswith(("rust", "lua", "face"))
    )
    numpy = subprocess.check_output(
        [sys.executable, str(ROOT / "tests/bench/numpy_compose.py"), f"--sizes={sizes}"],
        text=True,
    )
    data = load_tsv(rust_lines + "\n" + numpy)
    out = ROOT / "tests/bench/last_compose.tsv"
    with out.open("w") as f:
        f.write("face\top\tn\tms\n")
        for k, v in sorted(data.items()):
            f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{v:.6f}\n")

    print("| op | k | NumPy short (ms) | Rust long | Rust short | Lua long | Lua short | Rshort/N | Lshort/N | L/R short |")
    print("|----|--:|-----------------:|----------:|-----------:|---------:|----------:|---------:|---------:|----------:|")
    ops = ["xtx_short", "normal_eq_short"]
    # also show long for context via paired names
    for op in ["xtx", "normal_eq"]:
        for n in [int(s) for s in sizes.split(",")]:
            ns = data.get(("numpy", f"{op}_short", n))
            rl = data.get(("rust", f"{op}_long", n))
            rs = data.get(("rust", f"{op}_short", n))
            ll = data.get(("lua", f"{op}_long", n))
            ls = data.get(("lua", f"{op}_short", n))
            if ns is None:
                continue
            def f(x):
                return f"{x:.4f}" if x is not None else "—"
            def r(a, b):
                return f"{a/b:.2f}×" if a is not None and b else "—"
            print(
                f"| {op} | {n} | {f(ns)} | {f(rl)} | {f(rs)} | {f(ll)} | {f(ls)} | {r(rs,ns)} | {r(ls,ns)} | {r(ls,rs)} |"
            )

if __name__ == "__main__":
    main()
