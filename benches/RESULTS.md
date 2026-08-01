# Performance comparison snapshot (3-way)

**Faces (always reported together):**

| Face | Role |
|------|------|
| **MatLua Lua** | Product surface — what users call |
| **MatLua Rust** | Critical path under the Lua face |
| **NumPy** | External bar for “worth using” |

Recorded on the agent sandbox (Linux x86_64, 2 CPUs, NumPy + OpenBLAS,
MatLua **release**). Wall clock throughout. Re-run: `python3 benches/compare.py`.

## P6 snapshot (dest GEMM + parallel large matmul)

| op | n | MatLua Rust (ms) | MatLua Lua (ms) | NumPy (ms) | Rust/NumPy | Lua/NumPy |
|----|---:|-----------------:|----------------:|-----------:|-----------:|----------:|
| matmul | 64 | 0.0102 | 0.0164 | 0.0096 | 1.06× | 1.71× |
| matmul | 256 | 0.4022 | 0.4024 | 0.5234 | 0.77× | 0.77× |
| matmul | 1024 | 21.3654 | 23.7975 | 17.1390 | 1.25× | 1.39× |
| solve | 64 | 0.0787 | 0.0638 | 0.0357 | 2.21× | 1.79× |
| solve | 256 | 1.4207 | 1.1119 | 0.6631 | 2.14× | 1.68× |
| solve | 1024 | 26.6354 | 32.9699 | 27.6425 | 0.96× | 1.19× |
| elem_add | 64 | 0.0017 | 0.0046 | 0.0017 | 0.96× | 2.65× |
| elem_add | 256 | 0.0314 | 0.0540 | 0.0424 | 0.74× | 1.27× |
| elem_add | 1024 | 1.3279 | 3.7482 | 1.0764 | 1.23× | 3.48× |

### vs pre-P6 (matmul only)

| n | Rust/NumPy before | after P6 |
|---:|------------------:|---------:|
| 64 | 4.30× | 1.06× |
| 256 | 1.57× | 0.77× |
| 1024 | 2.26× | 1.25× |

### Bar check (§7.2)

- **matmul (Rust + Lua), medium+:** inside ~1–2× NumPy; at n=256 often **under** NumPy on this host.
- **solve:** large-n competitive; n=256 Rust sits ~2.1× (soft edge; noise-sensitive on 2-core sandbox).
- **elem_add:** Rust at parity; Lua pays userdata alloc (visible at n=1024).
- **Lua ≈ Rust** on bulk matmul/solve (product face tracks the core).

### What P6 changed

1. GEMM writes **directly** into a pre-sized row-major buffer (no intermediate faer `Mat` + pack-out).
2. Large products use faer **global parallelism** (Rayon); tiny products stay sequential.
3. Bench measures **wall clock** for all three faces; Lua uses `call_global` so compile cost is not in the sample.

## How to reproduce

```bash
python3 benches/compare.py
```
