# Performance comparison snapshot (3-way)

## How to read this

**NumPy is always 1.00×** — the baseline we compare against.

MatLua columns are **wall time ÷ NumPy wall time** on the same op and size:

| Value | Meaning |
|------:|---------|
| **1.00×** | same speed as NumPy |
| **2.00×** | twice as slow |
| **0.50×** | twice as fast |

| Face | Role |
|------|------|
| **NumPy** | Baseline (always 1.00× in the relative table) |
| **MatLua Rust** | Critical path under the Lua face |
| **MatLua Lua** | Product surface — what users call |

Recorded on the agent sandbox (Linux x86_64, 2 CPUs, NumPy + OpenBLAS,
MatLua **release**). Wall clock. Re-run: `python3 benches/compare.py`.

## P6 snapshot — relative to NumPy

| op | n | NumPy | MatLua Rust | MatLua Lua |
|----|---:|------:|------------:|-----------:|
| matmul | 64 | 1.00× | 1.06× | 1.71× |
| matmul | 256 | 1.00× | 0.77× | 0.77× |
| matmul | 1024 | 1.00× | 1.25× | 1.39× |
| solve | 64 | 1.00× | 2.21× | 1.79× |
| solve | 256 | 1.00× | 2.14× | 1.68× |
| solve | 1024 | 1.00× | 0.96× | 1.19× |
| elem_add | 64 | 1.00× | 0.96× | 2.65× |
| elem_add | 256 | 1.00× | 0.74× | 1.27× |
| elem_add | 1024 | 1.00× | 1.23× | 3.48× |

## Same runs — absolute ms (calibration only)

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
|----|---:|-----------:|-----------------:|----------------:|
| matmul | 64 | 0.0096 | 0.0102 | 0.0164 |
| matmul | 256 | 0.5234 | 0.4022 | 0.4024 |
| matmul | 1024 | 17.1390 | 21.3654 | 23.7975 |
| solve | 64 | 0.0357 | 0.0787 | 0.0638 |
| solve | 256 | 0.6631 | 1.4207 | 1.1119 |
| solve | 1024 | 27.6425 | 26.6354 | 32.9699 |
| elem_add | 64 | 0.0017 | 0.0017 | 0.0046 |
| elem_add | 256 | 0.0424 | 0.0314 | 0.0540 |
| elem_add | 1024 | 1.0764 | 1.3279 | 3.7482 |

### vs pre-P6 (matmul, MatLua Rust relative to NumPy)

| n | before | after P6 |
|---:|-------:|---------:|
| 64 | 4.30× | 1.06× |
| 256 | 1.57× | 0.77× |
| 1024 | 2.26× | 1.25× |

### Bar check (§7.2)

- **matmul medium+:** MatLua Rust and Lua within ~1–2× NumPy (often under 1× at n=256 here).
- **solve:** large-n competitive; n=256 Rust ~2.1× (soft edge on this 2-core host).
- **elem_add:** Rust near 1×; Lua higher at large n (userdata alloc).
- **Lua tracks Rust** on bulk matmul/solve.

### What P6 changed

1. GEMM writes **directly** into a pre-sized row-major buffer (no intermediate faer `Mat` + pack-out).
2. Large products use faer **global parallelism** (Rayon); tiny products stay sequential.
3. Benches use **wall clock**; Lua uses `call_global` (no reparse in the sample).

## How to reproduce

```bash
python3 benches/compare.py
```
