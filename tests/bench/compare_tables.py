#!/usr/bin/env python3
"""Build four performance tables (f64/i64 x absolute/relative) from TSV faces.

TSV lines: face \\t op \\t n \\t ms
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def load_tsv(path: Path) -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("face"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        face, op, n_s, ms_s = parts
        try:
            out[(face, op, int(n_s))] = float(ms_s)
        except ValueError:
            continue
    return out


def fmt_ms(x: float | None) -> str:
    if x is None:
        return "—"
    if x >= 10:
        return f"{x:.3f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def fmt_x(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}x"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    aligns = ["---" if h in ("op", "face") else "---:" for h in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def build_f64(data: dict) -> tuple[str, str]:
    keys = sorted({(op, n) for (face, op, n) in data if face in ("numpy", "rust", "lua")})
    abs_rows = []
    rel_rows = []
    for op, n in keys:
        np_ms = data.get(("numpy", op, n))
        ru_ms = data.get(("rust", op, n))
        lu_ms = data.get(("lua", op, n))
        abs_rows.append([op, str(n), fmt_ms(np_ms), fmt_ms(ru_ms), fmt_ms(lu_ms)])
        r_np = (ru_ms / np_ms) if np_ms and ru_ms and np_ms > 0 else None
        l_np = (lu_ms / np_ms) if np_ms and lu_ms and np_ms > 0 else None
        l_r = (lu_ms / ru_ms) if ru_ms and lu_ms and ru_ms > 0 else None
        rel_rows.append([op, str(n), "1.00x", fmt_x(r_np), fmt_x(l_np), fmt_x(l_r)])
    abs_t = md_table(
        ["op", "n", "NumPy (ms)", "MatLua Rust (ms)", "MatLua Lua (ms)"],
        abs_rows,
    )
    rel_t = md_table(
        ["op", "n", "NumPy", "Rust/NumPy", "Lua/NumPy", "Lua/Rust"],
        rel_rows,
    )
    return abs_t, rel_t


def build_i64(data: dict) -> tuple[str, str]:
    # faces: numpy, rust, lua (rust/lua from i64_surface; accept legacy face "i64" as rust)
    faces_r = ("rust", "i64")
    keys = sorted(
        {
            (op, n)
            for (face, op, n) in data
            if face in ("numpy", "rust", "lua", "i64")
        }
    )
    abs_rows = []
    rel_rows = []
    for op, n in keys:
        np_ms = data.get(("numpy", op, n))
        ru_ms = data.get(("rust", op, n))
        if ru_ms is None:
            ru_ms = data.get(("i64", op, n))  # legacy
        lu_ms = data.get(("lua", op, n))
        abs_rows.append([op, str(n), fmt_ms(np_ms), fmt_ms(ru_ms), fmt_ms(lu_ms)])
        r_np = (ru_ms / np_ms) if np_ms and ru_ms and np_ms > 0 else None
        l_np = (lu_ms / np_ms) if np_ms and lu_ms and np_ms > 0 else None
        l_r = (lu_ms / ru_ms) if ru_ms and lu_ms and ru_ms > 0 else None
        rel_rows.append([op, str(n), "1.00x", fmt_x(r_np), fmt_x(l_np), fmt_x(l_r)])
    abs_t = md_table(
        ["op", "n", "NumPy int64 (ms)", "MatLua Rust i64 (ms)", "MatLua Lua i64 (ms)"],
        abs_rows,
    )
    rel_t = md_table(
        ["op", "n", "NumPy", "Rust/NumPy", "Lua/NumPy", "Lua/Rust"],
        rel_rows,
    )
    return abs_t, rel_t



def build_promote(data: dict) -> tuple[str, str]:
    """i64→f64 promote-out three-way (numpy / rust / lua)."""
    keys = sorted({(op, n) for (face, op, n) in data if face in ("numpy", "rust", "lua")})
    abs_rows = []
    rel_rows = []
    for op, n in keys:
        np_ms = data.get(("numpy", op, n))
        ru_ms = data.get(("rust", op, n))
        lu_ms = data.get(("lua", op, n))
        abs_rows.append([op, str(n), fmt_ms(np_ms), fmt_ms(ru_ms), fmt_ms(lu_ms)])
        r_np = (ru_ms / np_ms) if np_ms and ru_ms and np_ms > 0 else None
        l_np = (lu_ms / np_ms) if np_ms and lu_ms and np_ms > 0 else None
        l_r = (lu_ms / ru_ms) if ru_ms and lu_ms and ru_ms > 0 else None
        rel_rows.append([op, str(n), "1.00x", fmt_x(r_np), fmt_x(l_np), fmt_x(l_r)])
    abs_t = md_table(
        ["op", "n", "NumPy (ms)", "MatLua Rust (ms)", "MatLua Lua (ms)"],
        abs_rows,
    )
    rel_t = md_table(
        ["op", "n", "NumPy", "Rust/NumPy", "Lua/NumPy", "Lua/Rust"],
        rel_rows,
    )
    return abs_t, rel_t

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", type=Path, default=Path("tests/bench/last_f64.tsv"))
    ap.add_argument("--i64", type=Path, default=Path("tests/bench/last_i64.tsv"))
    ap.add_argument("--promote", type=Path, default=Path("tests/bench/last_i64_promote.tsv"))
    ap.add_argument("--write-readme", type=Path, default=None)
    args = ap.parse_args()

    f64 = load_tsv(args.f64)
    i64 = load_tsv(args.i64)
    promo = load_tsv(args.promote)
    f_abs, f_rel = build_f64(f64)
    i_abs, i_rel = build_i64(i64)
    p_abs, p_rel = build_promote(promo)

    body = f"""### Table A — f64 absolute (ms)

{f_abs}

### Table B — f64 relative (NumPy = 1.00x)

{f_rel}

### Table C — i64 absolute (ms)

{i_abs}

### Table D — i64 relative (NumPy = 1.00x)

{i_rel}

### Table E — i64→f64 promote-out absolute (ms)

{p_abs}

### Table F — i64→f64 promote-out relative (NumPy = 1.00x)

{p_rel}
"""
    print(body)
    if args.write_readme:
        readme = args.write_readme.read_text()
        start = "<!-- PERF_TABLES_START -->"
        end = "<!-- PERF_TABLES_END -->"
        if start in readme and end in readme:
            pre, rest = readme.split(start, 1)
            _, post = rest.split(end, 1)
            readme = pre + start + "\n\n" + body + "\n" + end + post
        else:
            m = re.search(r"## Latest fair results.*", readme, re.S)
            if m:
                head = readme[: m.start()]
                readme = (
                    head
                    + "## Latest results\n\n"
                    + start
                    + "\n\n"
                    + body
                    + "\n"
                    + end
                    + "\n"
                )
            else:
                readme = readme.rstrip() + "\n\n" + start + "\n\n" + body + "\n" + end + "\n"
        args.write_readme.write_text(readme)
        print(f"updated {args.write_readme}", file=sys.stderr)


if __name__ == "__main__":
    main()
