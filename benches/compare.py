#!/usr/bin/env python3
"""Three-way dense desk compare: NumPy baseline vs MatLua Rust vs MatLua Lua.

**How to read the table**

NumPy is the baseline: **1.00×** on every row.
MatLua columns are *wall time relative to NumPy on the same op and size*:

- **1.00×** — same wall time as NumPy
- **2.00×** — twice as slow (worse)
- **0.50×** — twice as fast (better)

Absolute milliseconds are printed in a second table for calibration only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def load_tsv(text: str) -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("face"):
            continue
        face, op, n, ms = line.split("\t")
        out[(face, op, int(n))] = float(ms)
    return out


def fmt_rel(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}×"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="NumPy = 1.00× baseline; MatLua Rust and Lua as relative wall time."
    )
    ap.add_argument("--sizes", default="64,256,1024")
    ap.add_argument("--skip-lua", action="store_true")
    args = ap.parse_args()

    if args.skip_lua:
        rust_cmd = [
            "cargo",
            "run",
            "--release",
            "--example",
            "bench_dense",
            "--",
            "--sizes",
            args.sizes,
        ]
    else:
        rust_cmd = [
            "cargo",
            "run",
            "--release",
            "--features",
            "lua",
            "--example",
            "bench_dense",
            "--",
            "--sizes",
            args.sizes,
        ]

    print("## Running MatLua bench…", file=sys.stderr)
    r = subprocess.run(
        rust_cmd,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    rust_tsv = r.stdout
    if r.stderr:
        print(r.stderr, file=sys.stderr, end="")

    print("## Running NumPy bench…", file=sys.stderr)
    n = subprocess.run(
        [sys.executable, str(root / "benches" / "numpy_bench.py"), "--sizes", args.sizes],
        check=True,
        capture_output=True,
        text=True,
    )
    np_tsv = n.stdout

    data = load_tsv(rust_tsv)
    data.update(load_tsv(np_tsv))

    ops = ["matmul", "solve", "elem_add"]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    faces_order = ["numpy", "rust", "lua"]

    print()
    print(
        "Relative wall time (**NumPy = 1.00×** baseline). "
        "Lower is better. >1 means slower than NumPy."
    )
    print()
    print("| op | n | NumPy | MatLua Rust | MatLua Lua |")
    print("|----|---:|------:|------------:|-----------:|")

    gaps_rust: list[str] = []
    gaps_lua: list[str] = []
    for op in ops:
        for nsz in sizes:
            rust = data.get(("rust", op, nsz))
            lua = data.get(("lua", op, nsz))
            numpy = data.get(("numpy", op, nsz))
            if numpy is None or numpy <= 0:
                continue
            ratio_r = (rust / numpy) if rust is not None else None
            ratio_l = (lua / numpy) if lua is not None else None
            print(
                f"| {op} | {nsz} | 1.00× | {fmt_rel(ratio_r)} | {fmt_rel(ratio_l)} |"
            )
            if op in ("matmul", "solve") and nsz >= 256:
                if ratio_r is not None and ratio_r > 2.0:
                    gaps_rust.append(
                        f"{op} n={nsz}: MatLua Rust = {ratio_r:.2f}× NumPy (above ~2× bar)"
                    )
                if ratio_l is not None and ratio_l > 2.0:
                    gaps_lua.append(
                        f"{op} n={nsz}: MatLua Lua = {ratio_l:.2f}× NumPy (above ~2× bar)"
                    )

    print()
    print("Absolute wall time (ms) — same runs, for calibration only:")
    print()
    print("| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |")
    print("|----|---:|-----------:|-----------------:|----------------:|")
    for op in ops:
        for nsz in sizes:
            rust = data.get(("rust", op, nsz))
            lua = data.get(("lua", op, nsz))
            numpy = data.get(("numpy", op, nsz))
            if numpy is None:
                continue
            rs = f"{rust:.4f}" if rust is not None else "—"
            ls = f"{lua:.4f}" if lua is not None else "—"
            print(f"| {op} | {nsz} | {numpy:.4f} | {rs} | {ls} |")

    print()
    if gaps_rust or gaps_lua:
        if gaps_rust:
            print("### Gaps vs §7.2 bar (Rust critical path, medium+ matmul/solve)")
            for g in gaps_rust:
                print(f"- {g}")
        if gaps_lua:
            print("### Gaps vs §7.2 bar (Lua product face, medium+ matmul/solve)")
            for g in gaps_lua:
                print(f"- {g}")
    else:
        print(
            "### Bar check\n"
            "MatLua Rust and Lua matmul/solve at n≥256 within ~2× NumPy on this machine "
            "(or no medium+ rows)."
        )

    out = root / "benches" / "last_results.tsv"
    lines = ["face\top\tn\tms"]
    for face in faces_order:
        for op in ops:
            for nsz in sizes:
                k = (face, op, nsz)
                if k in data:
                    lines.append(f"{face}\t{op}\t{nsz}\t{data[k]:.6f}")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out.relative_to(root)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
