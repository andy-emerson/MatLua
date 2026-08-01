# P5 comparison snapshot

Recorded on the CI/agent sandbox used for the P5 harness (Linux x86_64,
NumPy 2.x + OpenBLAS, MatLua release). Re-run with `python3 benches/compare.py`
on your machine; numbers will differ.

## Table (median wall ms)

| op | n | MatLua Rust (ms) | MatLua Lua (ms) | NumPy (ms) | Rust/NumPy | Lua/NumPy |
|----|---:|-----------------:|----------------:|-----------:|-----------:|----------:|
| matmul | 64 | 0.0483 | 0.0780 | 0.0112 | 4.30× | 6.95× |
| matmul | 256 | 0.8245 | 0.9100 | 0.5242 | 1.57× | 1.74× |
| matmul | 1024 | 38.6288 | 67.0350 | 17.0864 | 2.26× | 3.92× |
| solve | 64 | 0.0693 | 0.0870 | 0.0358 | 1.93× | 2.43× |
| solve | 256 | 1.0729 | 1.5010 | 0.6416 | 1.67× | 2.34× |
| solve | 1024 | 22.9954 | 44.4760 | 25.1010 | 0.92× | 1.77× |
| elem_add | 64 | 0.0015 | 0.0030 | 0.0028 | 0.54× | 1.07× |
| elem_add | 256 | 0.0330 | 0.0710 | 0.0355 | 0.93× | 2.00× |
| elem_add | 1024 | 1.1727 | 2.0940 | 1.1619 | 1.01× | 1.80× |

## Bar check (§7.2)

- **Rust matmul/solve at n ≥ 256:** mostly within ~1–2× NumPy.  
  **Residual gap:** `matmul` n=1024 ≈ **2.26×** NumPy on this host (slightly over the soft 2× ceiling). Likely remaining cost: row-major↔faer kernel boundary and result pack-out; OpenBLAS GEMM is extremely tuned at large n.
- **Rust solve n=1024:** **faster than NumPy** on this host (0.92×).
- **Elementwise add:** at parity or better for Rust.
- **Lua face:** bulk matmul/solve at medium sizes stays near the Rust line; small-n and Lua call overhead are larger (expected; not the §7.2 primary bar).

## How to reproduce

```bash
python3 benches/compare.py
```
