#!/usr/bin/env python3
"""Merge MatLua + NumPy TSV benches and print a comparison table (P5)."""

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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024")
    ap.add_argument("--skip-lua", action="store_true")
    args = ap.parse_args()

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
    if not args.skip_lua:
        # insert features before --
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
    faces_order = ["rust", "lua", "numpy"]

    print()
    print("| op | n | MatLua Rust (ms) | MatLua Lua (ms) | NumPy (ms) | Rust/NumPy | Lua/NumPy |")
    print("|----|---:|-----------------:|----------------:|-----------:|-----------:|----------:|")

    gaps: list[str] = []
    for op in ops:
        for nsz in sizes:
            rust = data.get(("rust", op, nsz))
            lua = data.get(("lua", op, nsz))
            numpy = data.get(("numpy", op, nsz))
            if rust is None or numpy is None:
                continue
            ratio_r = rust / numpy if numpy > 0 else float("inf")
            ratio_l = (lua / numpy) if (lua is not None and numpy > 0) else None
            lua_s = f"{lua:.4f}" if lua is not None else "—"
            ratio_l_s = f"{ratio_l:.2f}×" if ratio_l is not None else "—"
            print(
                f"| {op} | {nsz} | {rust:.4f} | {lua_s} | {numpy:.4f} | {ratio_r:.2f}× | {ratio_l_s} |"
            )
            # Contract: medium+ matmul/solve within ~1–2× (rust face)
            if op in ("matmul", "solve") and nsz >= 256 and ratio_r > 2.0:
                gaps.append(
                    f"{op} n={nsz}: Rust/NumPy = {ratio_r:.2f}× (above ~2× bar)"
                )

    print()
    # Product-face gaps (Lua) — same soft bar; this is what users feel.
    lua_gaps: list[str] = []
    for op in ("matmul", "solve"):
        for nsz in sizes:
            if nsz < 256:
                continue
            lua = data.get(("lua", op, nsz))
            numpy = data.get(("numpy", op, nsz))
            if lua is None or numpy is None or numpy <= 0:
                continue
            ratio_l = lua / numpy
            if ratio_l > 2.0:
                lua_gaps.append(
                    f"{op} n={nsz}: Lua/NumPy = {ratio_l:.2f}× (above ~2× bar)"
                )

    if gaps or lua_gaps:
        if gaps:
            print("### Gaps vs §7.2 bar (Rust critical path, medium+ matmul/solve)")
            for g in gaps:
                print(f"- {g}")
        if lua_gaps:
            print("### Gaps vs §7.2 bar (Lua product face, medium+ matmul/solve)")
            for g in lua_gaps:
                print(f"- {g}")
    else:
        print(
            "### Bar check\n"
            "Rust and Lua matmul/solve at n≥256 within ~2× of NumPy on this machine "
            "(or no medium+ rows)."
        )

    # Write machine-readable dump next to harness
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
