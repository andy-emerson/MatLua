# Performance comparison harness (P5)

Three-way compare on ordinary dense `f64` desk ops (`matmul`, `solve`,
elementwise add):

| Face | Role |
|------|------|
| **MatLua Lua** | Product surface (what users call) |
| **MatLua Rust** | Critical path under Lua |
| **NumPy** | External “worth using” bar |

## Contract

See [DESIGN.md §7.2](../DESIGN.md#72-performance-program-p0p5): sizes
64 / 256 / 1024, faces measured separately, ~1–2× NumPy on medium+ matmul/solve
for the **Rust** face.

## Run

```bash
# Full comparison (Rust + Lua + NumPy), writes benches/last_results.tsv
python3 benches/compare.py

# Rust only
cargo run --release --example bench_dense

# Rust + Lua
cargo run --release --features lua --example bench_dense

# NumPy only
python3 benches/numpy_bench.py
```

Optional: `python3 benches/compare.py --sizes 64,256`

## Files

| File | Role |
|------|------|
| `examples/bench_dense.rs` | MatLua timers (TSV on stdout) |
| `benches/numpy_bench.py` | NumPy timers (TSV on stdout) |
| `benches/compare.py` | Runs both, prints markdown table + ratios |
| `benches/last_results.tsv` | Last machine-local dump (gitignored if desired) |
