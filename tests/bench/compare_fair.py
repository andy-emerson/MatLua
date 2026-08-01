#!/usr/bin/env python3
"""Run MatLua fair_all + NumPy fair benches and print a comparison table.

Measurement standard:
  - Same input generation rule on all faces
  - Setup outside the timer; one op call inside
  - Median wall time (ms)
  - Faces: numpy, rust (MatLua core), lua (product face)
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


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024")
    ap.add_argument("--skip-run", action="store_true", help="use last_results.tsv only")
    args = ap.parse_args()

    out_tsv = root / "tests" / "bench" / "last_results.tsv"

    if not args.skip_run:
        print("## MatLua fair_all…", file=sys.stderr)
        r = subprocess.run(
            [
                "cargo",
                "test",
                "--release",
                "--features",
                "lua",
                "--test",
                "fair_all",
                "--",
                "--run",
                "--sizes",
                args.sizes,
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        # harness=false binary prints TSV on stdout; cargo may wrap — use stdout
        matlua_tsv = r.stdout
        if r.stderr:
            print(r.stderr, file=sys.stderr, end="")

        print("## NumPy fair…", file=sys.stderr)
        n = subprocess.run(
            [sys.executable, str(root / "tests" / "bench" / "numpy_fair.py"), "--sizes", args.sizes],
            check=True,
            capture_output=True,
            text=True,
        )
        data = load_tsv(matlua_tsv)
        data.update(load_tsv(n.stdout))
    else:
        data = load_tsv(out_tsv.read_text())

    # Preserve op order from rust rows when present
    ops: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for (face, op, nsz), _ in sorted(data.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        if face != "rust":
            continue
        key = (op, nsz)
        if key not in seen:
            seen.add(key)
            ops.append(key)
    if not ops:
        for (face, op, nsz) in sorted(data.keys(), key=lambda k: (k[1], k[2])):
            key = (op, nsz)
            if key not in seen:
                seen.add(key)
                ops.append(key)

    print()
    print("| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) | Rust/NumPy | Lua/NumPy | Lua/Rust |")
    print("|----|---:|-----------:|-----------------:|----------------:|-----------:|----------:|---------:|")
    for op, nsz in ops:
        rust = data.get(("rust", op, nsz))
        lua = data.get(("lua", op, nsz))
        numpy = data.get(("numpy", op, nsz))
        if rust is None or numpy is None:
            continue
        ls = f"{lua:.4f}" if lua is not None else "—"
        r_np = rust / numpy if numpy > 0 else float("inf")
        l_np = (lua / numpy) if lua is not None and numpy > 0 else None
        l_r = (lua / rust) if lua is not None and rust > 0 else None
        l_np_s = f"{l_np:.2f}×" if l_np is not None else "—"
        l_r_s = f"{l_r:.2f}×" if l_r is not None else "—"
        print(
            f"| {op} | {nsz} | {numpy:.4f} | {rust:.4f} | {ls} | {r_np:.2f}× | {l_np_s} | {l_r_s} |"
        )

    lines = ["face\top\tn\tms"]
    for face in ("numpy", "rust", "lua"):
        for op, nsz in ops:
            k = (face, op, nsz)
            if k in data:
                lines.append(f"{face}\t{op}\t{nsz}\t{data[k]:.6f}")
    out_tsv.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_tsv.relative_to(root)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
