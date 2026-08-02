# Tests

| Path | Job |
|------|-----|
| [`correctness/`](correctness/) | Public API + Lua face correctness (`--features lua`). |
| [`bench/`](bench/) | Microbenches: f64 three-way (NumPy / Rust / Lua) and i64 two-way (NumPy int64 / MatLua Rust i64). |

## How measurement works

For each op and size \(n \in \{64, 256, 1024\}\):

1. Shared generation rule (dtype-specific; see harness sources).
2. Time **one call** (setup outside the clock).
3. Report **median** wall time in milliseconds after warmup.

**f64 faces:** NumPy (`float64` + OpenBLAS where applicable) · MatLua Rust · MatLua Lua.  
**i64 faces:** NumPy `int64` · MatLua Rust wrapping `i64` · MatLua Lua i64 (same three-way shape as f64).  
**i64→f64 promote-out:** mean/std/median/quantile/norm/solve/cholesky/qr (Tables E–F).

**Relative tables:** NumPy is always **1.00x**. Other columns are `time / NumPy_time`.

### How many trials? Is it deterministic?

**Not a single deterministic run.** Each cell is:

1. **Warmup:** `warm` untimed calls (fill caches / GC steady state).
2. **Timed trials:** `iters` wall-clock samples of **one call** each.
3. **Report:** **median** of those samples (ms), not mean — less sensitive to a single GC spike.

Budget (same idea as `fair_all` / `i64_surface`):

| n | light ops (iters, warm) | heavy LA (iters, warm) |
|---|-------------------------|------------------------|
| 64 | 30–50, 5–8 | 10–15, 2–3 |
| 256 | 12–20, 3–4 | 4–6, 1–2 |
| 1024 | 6–8, 2 | 2–3, 1 |

So rows are **reproducible in distribution**, not bit-identical across machines/loads. Re-run on the same host for A/B opts.

Promote-out sizes default: **64, 256, 1024**.

```bash
# Correctness
cargo test
cargo test --features lua

# Full refresh of the four tables below
cargo test --release --features lua --test fair_all -- --run --sizes 64,256,1024,1024 \
  | awk -F'\t' 'NF==4 && ($1=="rust"||$1=="lua"){print}' > tests/bench/last_f64.tsv
python3 tests/bench/numpy_fair.py --sizes 64,256,1024,1024 \
  | awk -F'\t' 'NF==4 && $1=="numpy"{print}' >> tests/bench/last_f64.tsv

cargo test --release --features lua --test i64_surface -- --run --sizes 64,256,1024,1024 \
  | awk -F'\t' 'NF==4 && ($1=="rust"||$1=="lua"){print}' > tests/bench/last_i64.tsv
python3 tests/bench/numpy_i64_fair.py --sizes 64,256,1024,1024 \
  | awk -F'\t' 'NF==4 && $1=="numpy"{print}' >> tests/bench/last_i64.tsv

cargo test --release --features lua --test i64_promote -- --run --sizes 64,256,1024 \
  | awk -F'\t' 'NF==4 && ($1=="rust"||$1=="lua"){print}' > tests/bench/last_i64_promote.tsv
python3 tests/bench/numpy_i64_promote.py --sizes 64,256,1024 \
  | awk -F'\t' 'NF==4 && $1=="numpy"{print}' >> tests/bench/last_i64_promote.tsv

python3 tests/bench/compare_tables.py --write-readme tests/README.md
```

Or: `python3 tests/bench/compare_fair.py` still builds the old single-table path from `last_results.tsv` if you need it; **prefer `compare_tables.py`**.

### Tracking

| Item | Status |
|------|--------|
| in-place `out=` full surface | [#21](https://github.com/andy-emerson/MatLua/issues/21) **deferred past M7.c** (partial `*_out` remains) |
| M7.c optimize (f64 + i64) | in progress — see DESIGN §7.1 / §7.1.2 |
| Explicit i64 SIMD GEMM | researched: AVX2 lacks i64 mul; AVX-512DQ / unstable `std::simd` deferred |

### Caveats

- Sub-0.01 ms cells are **noisy**; ratios can swing.
- f64 matmul/solve/decompositions use **faer** (+ OpenBLAS on the NumPy side for large GEMM).
- i64 matmul is **packed GEBP + Rayon** (wrapping); NumPy `int64 @ int64` is not BLAS GEMM — Table D is a product reference, not "beat MKL".
- MatLua always **materializes** transpose; NumPy may return a view (we force `.T.copy()` on the NumPy i64 side).

## Latest results

Host: Linux x86_64, 2 CPUs, MatLua **release**, NumPy + OpenBLAS.  
Run date: **2026-08-02** (M7.c: n=4096 added; **Strassen removed**; tables refreshed).  
Re-run: commands above.

<!-- PERF_TABLES_START -->

### Table A — f64 absolute wall time (ms)

Median wall time. Setup outside the clock. Smaller is better.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000602 | 0.000167 | 0.000394 |
| arange | 256 | 0.000799 | 0.000371 | 0.000654 |
| arange | 1024 | 0.001206 | 0.001188 | 0.001816 |
| arange | 4096 | 0.003177 | 0.005565 | — |
| cholesky | 64 | 0.022694 | 0.015273 | 0.016207 |
| cholesky | 256 | 0.815386 | 0.591251 | 0.618983 |
| cholesky | 1024 | 21.315 | 19.149 | 19.323 |
| cholesky | 4096 | 749.631 | 936.856 | — |
| copy | 64 | 0.001367 | 0.001015 | 0.004129 |
| copy | 256 | 0.013447 | 0.014980 | 0.014295 |
| copy | 1024 | 0.643857 | 0.604653 | 0.656669 |
| copy | 4096 | 34.715 | 79.277 | — |
| dot | 64 | 0.000714 | 0.000052 | 0.000384 |
| dot | 256 | 0.000767 | 0.000118 | 0.000426 |
| dot | 1024 | 0.000880 | 0.000382 | 0.000743 |
| dot | 4096 | 0.002669 | 0.001465 | — |
| elem_add | 64 | 0.002803 | 0.001327 | 0.005608 |
| elem_add | 256 | 0.038455 | 0.024794 | 0.024963 |
| elem_add | 1024 | 0.967327 | 0.883850 | 1.0720 |
| elem_add | 4096 | 50.625 | 67.255 | — |
| elem_add_scalar | 64 | 0.001704 | 0.001023 | 0.005253 |
| elem_add_scalar | 256 | 0.015633 | 0.015250 | 0.015886 |
| elem_add_scalar | 1024 | 0.697182 | 0.624724 | 0.737881 |
| elem_add_scalar | 4096 | 31.780 | 60.080 | — |
| elem_div | 64 | 0.003383 | 0.003182 | 0.004879 |
| elem_div | 256 | 0.045009 | 0.044438 | 0.045444 |
| elem_div | 1024 | 0.966113 | 0.887022 | 1.0170 |
| elem_div | 4096 | 49.849 | 67.425 | — |
| elem_mul | 64 | 0.002799 | 0.001317 | 0.004936 |
| elem_mul | 256 | 0.037938 | 0.023671 | 0.025438 |
| elem_mul | 1024 | 0.962047 | 0.885087 | 1.1224 |
| elem_mul | 4096 | 49.103 | 66.370 | — |
| elem_sub | 64 | 0.002777 | 0.001493 | 0.004803 |
| elem_sub | 256 | 0.037615 | 0.023895 | 0.025360 |
| elem_sub | 1024 | 0.964023 | 0.884657 | 0.997705 |
| elem_sub | 4096 | 49.277 | 66.287 | — |
| eye | 64 | 0.002520 | 0.000740 | 0.003460 |
| eye | 256 | 0.014400 | 0.011254 | 0.013024 |
| eye | 1024 | 0.382700 | 0.306077 | 0.348158 |
| eye | 4096 | 13.390 | 57.604 | — |
| fill | 64 | 0.001724 | 0.000394 | 0.000597 |
| fill | 256 | 0.022433 | 0.011557 | 0.011884 |
| fill | 1024 | 0.383886 | 0.354795 | 0.380317 |
| fill | 4096 | 15.898 | 16.645 | — |
| full | 64 | 0.002711 | 0.000858 | 0.004473 |
| full | 256 | 0.023402 | 0.011669 | 0.013287 |
| full | 1024 | 0.377249 | 0.329763 | 0.423879 |
| full | 4096 | 21.317 | 50.588 | — |
| matmul | 64 | 0.011325 | 0.011022 | 0.016467 |
| matmul | 256 | 0.532442 | 0.379446 | 0.534160 |
| matmul | 1024 | 17.276 | 20.037 | 22.161 |
| matmul | 4096 | 1059.356 | 1385.326 | — |
| max | 64 | 0.001952 | 0.001556 | 0.001664 |
| max | 256 | 0.009606 | 0.023955 | 0.024208 |
| max | 1024 | 0.297066 | 0.304769 | 0.277304 |
| max | 4096 | 9.8072 | 8.6076 | — |
| mean | 64 | 0.004358 | 0.000720 | 0.000819 |
| mean | 256 | 0.016683 | 0.010996 | 0.011220 |
| mean | 1024 | 0.385627 | 0.301839 | 0.342621 |
| mean | 4096 | 12.440 | 10.790 | — |
| min | 64 | 0.001947 | 0.001564 | 0.001667 |
| min | 256 | 0.009615 | 0.023889 | 0.024249 |
| min | 1024 | 0.313118 | 0.303662 | 0.643197 |
| min | 4096 | 9.9607 | 8.2759 | — |
| norm | 64 | 0.002627 | 0.000723 | 0.000842 |
| norm | 256 | 0.009410 | 0.010994 | 0.011236 |
| norm | 1024 | 0.191305 | 0.324944 | 0.384663 |
| norm | 4096 | 4.4022 | 11.861 | — |
| ones | 64 | 0.002832 | 0.000858 | 0.004586 |
| ones | 256 | 0.023504 | 0.011609 | 0.012509 |
| ones | 1024 | 0.379341 | 0.358708 | 0.421860 |
| ones | 4096 | 23.889 | 52.164 | — |
| qr | 64 | 0.121041 | 0.274030 | 0.273456 |
| qr | 256 | 4.8880 | 3.0273 | 2.5877 |
| qr | 1024 | 117.065 | 68.757 | 67.295 |
| reshape | 64 | 0.000307 | 0.000084 | 0.000295 |
| reshape | 256 | 0.000311 | 0.000083 | 0.000324 |
| reshape | 1024 | 0.000329 | 0.000083 | 0.000584 |
| reshape | 4096 | 0.001214 | 0.000397 | — |
| solve | 64 | 0.037012 | 0.076557 | 0.077921 |
| solve | 256 | 0.632791 | 1.1917 | 1.2009 |
| solve | 1024 | 25.985 | 31.095 | 26.538 |
| solve | 4096 | 1107.119 | 931.672 | — |
| sum | 64 | 0.002284 | 0.000716 | 0.000816 |
| sum | 256 | 0.014546 | 0.010978 | 0.011211 |
| sum | 1024 | 0.371332 | 0.309749 | 0.345639 |
| sum | 4096 | 12.639 | 13.026 | — |
| svd | 64 | 0.293963 | 0.484457 | 0.456273 |
| svd | 256 | 9.4609 | 10.669 | 12.746 |
| svd | 1024 | 345.985 | 419.184 | 410.335 |
| transpose | 64 | 0.002551 | 0.003118 | 0.006486 |
| transpose | 256 | 0.052708 | 0.049352 | 0.051550 |
| transpose | 1024 | 4.6962 | 1.7827 | 2.0117 |
| transpose | 4096 | 190.566 | 92.395 | — |
| zeros | 64 | 0.000861 | 0.000765 | 0.003489 |
| zeros | 256 | 0.011355 | 0.010895 | 0.012628 |
| zeros | 1024 | 0.351175 | 0.303399 | 0.365042 |
| zeros | 4096 | 0.011429 | 62.377 | — |

### Table B — f64 vs NumPy (relative)

**NumPy is always 1.00x** (baseline). Values are wall_time / NumPy wall_time.

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.28x | 0.65x | 2.36x |
| arange | 256 | 1.00x | 0.46x | 0.82x | 1.76x |
| arange | 1024 | 1.00x | 0.99x | 1.51x | 1.53x |
| arange | 4096 | 1.00x | 1.75x | — | — |
| cholesky | 64 | 1.00x | 0.67x | 0.71x | 1.06x |
| cholesky | 256 | 1.00x | 0.73x | 0.76x | 1.05x |
| cholesky | 1024 | 1.00x | 0.90x | 0.91x | 1.01x |
| cholesky | 4096 | 1.00x | 1.25x | — | — |
| copy | 64 | 1.00x | 0.74x | 3.02x | 4.07x |
| copy | 256 | 1.00x | 1.11x | 1.06x | 0.95x |
| copy | 1024 | 1.00x | 0.94x | 1.02x | 1.09x |
| copy | 4096 | 1.00x | 2.28x | — | — |
| dot | 64 | 1.00x | 0.07x | 0.54x | 7.38x |
| dot | 256 | 1.00x | 0.15x | 0.56x | 3.61x |
| dot | 1024 | 1.00x | 0.43x | 0.84x | 1.95x |
| dot | 4096 | 1.00x | 0.55x | — | — |
| elem_add | 64 | 1.00x | 0.47x | 2.00x | 4.23x |
| elem_add | 256 | 1.00x | 0.64x | 0.65x | 1.01x |
| elem_add | 1024 | 1.00x | 0.91x | 1.11x | 1.21x |
| elem_add | 4096 | 1.00x | 1.33x | — | — |
| elem_add_scalar | 64 | 1.00x | 0.60x | 3.08x | 5.13x |
| elem_add_scalar | 256 | 1.00x | 0.98x | 1.02x | 1.04x |
| elem_add_scalar | 1024 | 1.00x | 0.90x | 1.06x | 1.18x |
| elem_add_scalar | 4096 | 1.00x | 1.89x | — | — |
| elem_div | 64 | 1.00x | 0.94x | 1.44x | 1.53x |
| elem_div | 256 | 1.00x | 0.99x | 1.01x | 1.02x |
| elem_div | 1024 | 1.00x | 0.92x | 1.05x | 1.15x |
| elem_div | 4096 | 1.00x | 1.35x | — | — |
| elem_mul | 64 | 1.00x | 0.47x | 1.76x | 3.75x |
| elem_mul | 256 | 1.00x | 0.62x | 0.67x | 1.07x |
| elem_mul | 1024 | 1.00x | 0.92x | 1.17x | 1.27x |
| elem_mul | 4096 | 1.00x | 1.35x | — | — |
| elem_sub | 64 | 1.00x | 0.54x | 1.73x | 3.22x |
| elem_sub | 256 | 1.00x | 0.64x | 0.67x | 1.06x |
| elem_sub | 1024 | 1.00x | 0.92x | 1.03x | 1.13x |
| elem_sub | 4096 | 1.00x | 1.35x | — | — |
| eye | 64 | 1.00x | 0.29x | 1.37x | 4.68x |
| eye | 256 | 1.00x | 0.78x | 0.90x | 1.16x |
| eye | 1024 | 1.00x | 0.80x | 0.91x | 1.14x |
| eye | 4096 | 1.00x | 4.30x | — | — |
| fill | 64 | 1.00x | 0.23x | 0.35x | 1.52x |
| fill | 256 | 1.00x | 0.52x | 0.53x | 1.03x |
| fill | 1024 | 1.00x | 0.92x | 0.99x | 1.07x |
| fill | 4096 | 1.00x | 1.05x | — | — |
| full | 64 | 1.00x | 0.32x | 1.65x | 5.21x |
| full | 256 | 1.00x | 0.50x | 0.57x | 1.14x |
| full | 1024 | 1.00x | 0.87x | 1.12x | 1.29x |
| full | 4096 | 1.00x | 2.37x | — | — |
| matmul | 64 | 1.00x | 0.97x | 1.45x | 1.49x |
| matmul | 256 | 1.00x | 0.71x | 1.00x | 1.41x |
| matmul | 1024 | 1.00x | 1.16x | 1.28x | 1.11x |
| matmul | 4096 | 1.00x | 1.31x | — | — |
| max | 64 | 1.00x | 0.80x | 0.85x | 1.07x |
| max | 256 | 1.00x | 2.49x | 2.52x | 1.01x |
| max | 1024 | 1.00x | 1.03x | 0.93x | 0.91x |
| max | 4096 | 1.00x | 0.88x | — | — |
| mean | 64 | 1.00x | 0.17x | 0.19x | 1.14x |
| mean | 256 | 1.00x | 0.66x | 0.67x | 1.02x |
| mean | 1024 | 1.00x | 0.78x | 0.89x | 1.14x |
| mean | 4096 | 1.00x | 0.87x | — | — |
| min | 64 | 1.00x | 0.80x | 0.86x | 1.07x |
| min | 256 | 1.00x | 2.48x | 2.52x | 1.02x |
| min | 1024 | 1.00x | 0.97x | 2.05x | 2.12x |
| min | 4096 | 1.00x | 0.83x | — | — |
| norm | 64 | 1.00x | 0.28x | 0.32x | 1.16x |
| norm | 256 | 1.00x | 1.17x | 1.19x | 1.02x |
| norm | 1024 | 1.00x | 1.70x | 2.01x | 1.18x |
| norm | 4096 | 1.00x | 2.69x | — | — |
| ones | 64 | 1.00x | 0.30x | 1.62x | 5.34x |
| ones | 256 | 1.00x | 0.49x | 0.53x | 1.08x |
| ones | 1024 | 1.00x | 0.95x | 1.11x | 1.18x |
| ones | 4096 | 1.00x | 2.18x | — | — |
| qr | 64 | 1.00x | 2.26x | 2.26x | 1.00x |
| qr | 256 | 1.00x | 0.62x | 0.53x | 0.85x |
| qr | 1024 | 1.00x | 0.59x | 0.57x | 0.98x |
| reshape | 64 | 1.00x | 0.27x | 0.96x | 3.51x |
| reshape | 256 | 1.00x | 0.27x | 1.04x | 3.90x |
| reshape | 1024 | 1.00x | 0.25x | 1.78x | 7.04x |
| reshape | 4096 | 1.00x | 0.33x | — | — |
| solve | 64 | 1.00x | 2.07x | 2.11x | 1.02x |
| solve | 256 | 1.00x | 1.88x | 1.90x | 1.01x |
| solve | 1024 | 1.00x | 1.20x | 1.02x | 0.85x |
| solve | 4096 | 1.00x | 0.84x | — | — |
| sum | 64 | 1.00x | 0.31x | 0.36x | 1.14x |
| sum | 256 | 1.00x | 0.75x | 0.77x | 1.02x |
| sum | 1024 | 1.00x | 0.83x | 0.93x | 1.12x |
| sum | 4096 | 1.00x | 1.03x | — | — |
| svd | 64 | 1.00x | 1.65x | 1.55x | 0.94x |
| svd | 256 | 1.00x | 1.13x | 1.35x | 1.19x |
| svd | 1024 | 1.00x | 1.21x | 1.19x | 0.98x |
| transpose | 64 | 1.00x | 1.22x | 2.54x | 2.08x |
| transpose | 256 | 1.00x | 0.94x | 0.98x | 1.04x |
| transpose | 1024 | 1.00x | 0.38x | 0.43x | 1.13x |
| transpose | 4096 | 1.00x | 0.48x | — | — |
| zeros | 64 | 1.00x | 0.89x | 4.05x | 4.56x |
| zeros | 256 | 1.00x | 0.96x | 1.11x | 1.16x |
| zeros | 1024 | 1.00x | 0.86x | 1.04x | 1.20x |
| zeros | 4096 | 1.00x | 5457.83x | — | — |

### Table C — i64 absolute wall time (ms)

Three faces: NumPy **int64** · MatLua Rust wrapping **i64** · MatLua **Lua** i64.
Same generation as `i64_surface` / `numpy_i64_fair.py`.
NumPy integer matmul is not OpenBLAS DGEMM; useful reference, not an MKL peer.

| op | n | NumPy int64 (ms) | MatLua Rust i64 (ms) | MatLua Lua i64 (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000530 | 0.000143 | 0.000317 |
| arange | 256 | 0.000677 | 0.000323 | 0.000528 |
| arange | 1024 | 0.001004 | 0.000976 | 0.001740 |
| arange | 4096 | 0.002611 | 0.003712 | — |
| copy | 64 | 0.001343 | 0.000931 | 0.001651 |
| copy | 256 | 0.013141 | 0.012697 | 0.014203 |
| copy | 1024 | 0.677079 | 0.610821 | 0.616676 |
| copy | 4096 | 39.722 | 12.101 | — |
| dot | 64 | 0.001113 | 0.000053 | 0.000222 |
| dot | 256 | 0.001118 | 0.000144 | 0.000294 |
| dot | 1024 | 0.001556 | 0.000422 | 0.000626 |
| dot | 4096 | 0.004364 | 0.001461 | — |
| elem_add | 64 | 0.001775 | 0.002510 | 0.011311 |
| elem_add | 256 | 0.024043 | 0.040040 | 0.041991 |
| elem_add | 1024 | 1.0014 | 0.892204 | 0.991803 |
| elem_add | 4096 | 47.800 | 30.224 | — |
| elem_div | 64 | 0.016426 | 0.008316 | 0.008720 |
| elem_div | 256 | 0.222805 | 0.131533 | 0.132691 |
| elem_div | 1024 | 3.6588 | 2.2057 | 2.2307 |
| elem_div | 4096 | 76.514 | 37.784 | — |
| elem_mul | 64 | 0.002271 | 0.002520 | 0.002911 |
| elem_mul | 256 | 0.026879 | 0.038760 | 0.039670 |
| elem_mul | 1024 | 0.992050 | 0.890917 | 1.0258 |
| elem_mul | 4096 | 47.138 | 29.396 | — |
| elem_sub | 64 | 0.001751 | 0.002537 | 0.002873 |
| elem_sub | 256 | 0.021717 | 0.039287 | 0.040304 |
| elem_sub | 1024 | 1.0016 | 0.893407 | 1.0106 |
| elem_sub | 4096 | 47.503 | 29.604 | — |
| eye | 64 | 0.002301 | 0.000295 | 0.001489 |
| eye | 256 | 0.014505 | 0.011241 | 0.011685 |
| eye | 1024 | 0.385887 | 0.305712 | 0.319102 |
| eye | 4096 | 15.107 | 8.9539 | — |
| fill | 64 | 0.001721 | 0.000380 | 0.000494 |
| fill | 256 | 0.022816 | 0.011592 | 0.011817 |
| fill | 1024 | 0.401988 | 0.334283 | 0.355755 |
| fill | 4096 | 16.863 | 16.202 | — |
| full | 64 | 0.002690 | 0.000455 | 0.001616 |
| full | 256 | 0.023399 | 0.011614 | 0.012703 |
| full | 1024 | 0.414420 | 0.335068 | 0.361943 |
| full | 4096 | 23.415 | 14.193 | — |
| isin | 64 | 0.021868 | 0.003718 | 0.012740 |
| isin | 256 | 0.111063 | 0.066247 | 0.056145 |
| isin | 1024 | 1.8814 | 0.981969 | 0.990230 |
| isin | 4096 | 44.995 | 27.167 | — |
| matmul | 64 | 0.180787 | 0.106064 | 0.106554 |
| matmul | 256 | 14.692 | 3.2937 | 3.3652 |
| matmul | 1024 | 6984.509 | 213.644 | 213.030 |
| matmul | 4096 | — | 13763.991 | — |
| max | 64 | 0.002072 | 0.000930 | 0.001034 |
| max | 256 | 0.008313 | 0.014332 | 0.014517 |
| max | 1024 | 0.351596 | 0.386298 | 0.375182 |
| max | 4096 | 10.016 | 15.257 | — |
| min | 64 | 0.002076 | 0.000936 | 0.001033 |
| min | 256 | 0.008253 | 0.014439 | 0.014626 |
| min | 1024 | 0.350920 | 0.389458 | 0.416977 |
| min | 4096 | 10.359 | 15.427 | — |
| ones | 64 | 0.002672 | 0.000452 | 0.001387 |
| ones | 256 | 0.023490 | 0.011643 | 0.012905 |
| ones | 1024 | 0.413227 | 0.358801 | 0.350341 |
| ones | 4096 | 25.526 | 14.531 | — |
| reshape | 64 | 0.000315 | 0.000082 | 0.000274 |
| reshape | 256 | 0.000337 | 0.000083 | 0.000265 |
| reshape | 1024 | 0.000366 | 0.000082 | 0.000492 |
| reshape | 4096 | 0.001130 | 0.000220 | — |
| sum | 64 | 0.002405 | 0.000336 | 0.000455 |
| sum | 256 | 0.014640 | 0.007666 | 0.007850 |
| sum | 1024 | 0.386414 | 0.297773 | 0.333941 |
| sum | 4096 | 11.687 | 11.862 | — |
| transpose | 64 | 0.002814 | 0.003146 | 0.003704 |
| transpose | 256 | 0.052162 | 0.049726 | 0.050096 |
| transpose | 1024 | 6.8580 | 1.9198 | 2.0882 |
| transpose | 4096 | 197.423 | 65.835 | — |
| unique | 64 | 0.003700 | 0.000146 | 0.000309 |
| unique | 256 | 0.004566 | 0.000593 | 0.000803 |
| unique | 1024 | 0.009645 | 0.001235 | 0.002780 |
| unique | 4096 | 0.034082 | 0.028569 | — |
| zeros | 64 | 0.000754 | 0.000281 | 0.001525 |
| zeros | 256 | 0.011295 | 0.010885 | 0.011372 |
| zeros | 1024 | 0.390960 | 0.332275 | 0.333322 |
| zeros | 4096 | 0.018843 | 10.879 | — |

### Table D — i64 vs NumPy int64 (relative)

**NumPy is always 1.00x**. Same columns as Table B (Rust/NumPy, Lua/NumPy, Lua/Rust).

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.27x | 0.60x | 2.22x |
| arange | 256 | 1.00x | 0.48x | 0.78x | 1.63x |
| arange | 1024 | 1.00x | 0.97x | 1.73x | 1.78x |
| arange | 4096 | 1.00x | 1.42x | — | — |
| copy | 64 | 1.00x | 0.69x | 1.23x | 1.77x |
| copy | 256 | 1.00x | 0.97x | 1.08x | 1.12x |
| copy | 1024 | 1.00x | 0.90x | 0.91x | 1.01x |
| copy | 4096 | 1.00x | 0.30x | — | — |
| dot | 64 | 1.00x | 0.05x | 0.20x | 4.19x |
| dot | 256 | 1.00x | 0.13x | 0.26x | 2.04x |
| dot | 1024 | 1.00x | 0.27x | 0.40x | 1.48x |
| dot | 4096 | 1.00x | 0.33x | — | — |
| elem_add | 64 | 1.00x | 1.41x | 6.37x | 4.51x |
| elem_add | 256 | 1.00x | 1.67x | 1.75x | 1.05x |
| elem_add | 1024 | 1.00x | 0.89x | 0.99x | 1.11x |
| elem_add | 4096 | 1.00x | 0.63x | — | — |
| elem_div | 64 | 1.00x | 0.51x | 0.53x | 1.05x |
| elem_div | 256 | 1.00x | 0.59x | 0.60x | 1.01x |
| elem_div | 1024 | 1.00x | 0.60x | 0.61x | 1.01x |
| elem_div | 4096 | 1.00x | 0.49x | — | — |
| elem_mul | 64 | 1.00x | 1.11x | 1.28x | 1.16x |
| elem_mul | 256 | 1.00x | 1.44x | 1.48x | 1.02x |
| elem_mul | 1024 | 1.00x | 0.90x | 1.03x | 1.15x |
| elem_mul | 4096 | 1.00x | 0.62x | — | — |
| elem_sub | 64 | 1.00x | 1.45x | 1.64x | 1.13x |
| elem_sub | 256 | 1.00x | 1.81x | 1.86x | 1.03x |
| elem_sub | 1024 | 1.00x | 0.89x | 1.01x | 1.13x |
| elem_sub | 4096 | 1.00x | 0.62x | — | — |
| eye | 64 | 1.00x | 0.13x | 0.65x | 5.05x |
| eye | 256 | 1.00x | 0.77x | 0.81x | 1.04x |
| eye | 1024 | 1.00x | 0.79x | 0.83x | 1.04x |
| eye | 4096 | 1.00x | 0.59x | — | — |
| fill | 64 | 1.00x | 0.22x | 0.29x | 1.30x |
| fill | 256 | 1.00x | 0.51x | 0.52x | 1.02x |
| fill | 1024 | 1.00x | 0.83x | 0.88x | 1.06x |
| fill | 4096 | 1.00x | 0.96x | — | — |
| full | 64 | 1.00x | 0.17x | 0.60x | 3.55x |
| full | 256 | 1.00x | 0.50x | 0.54x | 1.09x |
| full | 1024 | 1.00x | 0.81x | 0.87x | 1.08x |
| full | 4096 | 1.00x | 0.61x | — | — |
| isin | 64 | 1.00x | 0.17x | 0.58x | 3.43x |
| isin | 256 | 1.00x | 0.60x | 0.51x | 0.85x |
| isin | 1024 | 1.00x | 0.52x | 0.53x | 1.01x |
| isin | 4096 | 1.00x | 0.60x | — | — |
| matmul | 64 | 1.00x | 0.59x | 0.59x | 1.00x |
| matmul | 256 | 1.00x | 0.22x | 0.23x | 1.02x |
| matmul | 1024 | 1.00x | 0.03x | 0.03x | 1.00x |
| matmul | 4096 | 1.00x | — | — | — |
| max | 64 | 1.00x | 0.45x | 0.50x | 1.11x |
| max | 256 | 1.00x | 1.72x | 1.75x | 1.01x |
| max | 1024 | 1.00x | 1.10x | 1.07x | 0.97x |
| max | 4096 | 1.00x | 1.52x | — | — |
| min | 64 | 1.00x | 0.45x | 0.50x | 1.10x |
| min | 256 | 1.00x | 1.75x | 1.77x | 1.01x |
| min | 1024 | 1.00x | 1.11x | 1.19x | 1.07x |
| min | 4096 | 1.00x | 1.49x | — | — |
| ones | 64 | 1.00x | 0.17x | 0.52x | 3.07x |
| ones | 256 | 1.00x | 0.50x | 0.55x | 1.11x |
| ones | 1024 | 1.00x | 0.87x | 0.85x | 0.98x |
| ones | 4096 | 1.00x | 0.57x | — | — |
| reshape | 64 | 1.00x | 0.26x | 0.87x | 3.34x |
| reshape | 256 | 1.00x | 0.25x | 0.79x | 3.19x |
| reshape | 1024 | 1.00x | 0.22x | 1.34x | 6.00x |
| reshape | 4096 | 1.00x | 0.19x | — | — |
| sum | 64 | 1.00x | 0.14x | 0.19x | 1.35x |
| sum | 256 | 1.00x | 0.52x | 0.54x | 1.02x |
| sum | 1024 | 1.00x | 0.77x | 0.86x | 1.12x |
| sum | 4096 | 1.00x | 1.01x | — | — |
| transpose | 64 | 1.00x | 1.12x | 1.32x | 1.18x |
| transpose | 256 | 1.00x | 0.95x | 0.96x | 1.01x |
| transpose | 1024 | 1.00x | 0.28x | 0.30x | 1.09x |
| transpose | 4096 | 1.00x | 0.33x | — | — |
| unique | 64 | 1.00x | 0.04x | 0.08x | 2.12x |
| unique | 256 | 1.00x | 0.13x | 0.18x | 1.35x |
| unique | 1024 | 1.00x | 0.13x | 0.29x | 2.25x |
| unique | 4096 | 1.00x | 0.84x | — | — |
| zeros | 64 | 1.00x | 0.37x | 2.02x | 5.43x |
| zeros | 256 | 1.00x | 0.96x | 1.01x | 1.04x |
| zeros | 1024 | 1.00x | 0.85x | 0.85x | 1.00x |
| zeros | 4096 | 1.00x | 577.33x | — | — |

### Table E — i64→f64 promote-out absolute wall time (ms)

Integer inputs, floating / LA outputs (mean, std, median, quantile, norm, solve, cholesky, qr).
NumPy uses int64 stats where natural, else float64 after cast for LA.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 64 | 0.019083 | 0.015257 | 0.016814 |
| cholesky | 256 | 0.906579 | 0.725783 | 0.968087 |
| cholesky | 1024 | 30.238 | 31.250 | 25.775 |
| mean | 64 | 0.006400 | 0.000353 | 0.000476 |
| mean | 256 | 0.043551 | 0.007666 | 0.007924 |
| mean | 1024 | 0.736529 | 0.405788 | 0.374152 |
| mean | 4096 | 19.148 | 12.419 | — |
| median | 64 | 0.014966 | 0.003483 | 0.003755 |
| median | 256 | 0.099034 | 0.149859 | 0.054920 |
| median | 1024 | 1.9173 | 1.4271 | 1.3541 |
| norm | 64 | 0.002373 | 0.001454 | 0.001528 |
| norm | 256 | 0.008174 | 0.021938 | 0.022192 |
| norm | 1024 | 0.173264 | 0.534429 | 0.481324 |
| norm | 4096 | 5.6759 | 15.330 | — |
| qr | 64 | 0.107815 | 0.301220 | 0.319826 |
| qr | 256 | 5.4606 | 4.2982 | 4.6534 |
| qr | 1024 | 137.149 | 98.186 | 87.721 |
| quantile | 64 | 0.049881 | 0.003477 | 0.004422 |
| quantile | 256 | 0.220493 | 0.162025 | 0.062992 |
| quantile | 1024 | 3.6532 | 1.5380 | 1.5220 |
| solve | 64 | 0.031780 | 0.072825 | 0.095425 |
| solve | 256 | 0.733339 | 1.4994 | 1.3696 |
| solve | 1024 | 28.643 | 31.609 | 30.586 |
| std | 64 | 0.017440 | 0.003110 | 0.003208 |
| std | 256 | 0.124176 | 0.051350 | 0.051570 |
| std | 1024 | 2.5630 | 1.1812 | 1.1365 |
| std | 4096 | 92.392 | 29.986 | — |

### Table F — i64→f64 promote-out vs NumPy (relative)

**NumPy is always 1.00x**.

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky | 64 | 1.00x | 0.80x | 0.88x | 1.10x |
| cholesky | 256 | 1.00x | 0.80x | 1.07x | 1.33x |
| cholesky | 1024 | 1.00x | 1.03x | 0.85x | 0.82x |
| mean | 64 | 1.00x | 0.06x | 0.07x | 1.35x |
| mean | 256 | 1.00x | 0.18x | 0.18x | 1.03x |
| mean | 1024 | 1.00x | 0.55x | 0.51x | 0.92x |
| mean | 4096 | 1.00x | 0.65x | — | — |
| median | 64 | 1.00x | 0.23x | 0.25x | 1.08x |
| median | 256 | 1.00x | 1.51x | 0.55x | 0.37x |
| median | 1024 | 1.00x | 0.74x | 0.71x | 0.95x |
| norm | 64 | 1.00x | 0.61x | 0.64x | 1.05x |
| norm | 256 | 1.00x | 2.68x | 2.71x | 1.01x |
| norm | 1024 | 1.00x | 3.08x | 2.78x | 0.90x |
| norm | 4096 | 1.00x | 2.70x | — | — |
| qr | 64 | 1.00x | 2.79x | 2.97x | 1.06x |
| qr | 256 | 1.00x | 0.79x | 0.85x | 1.08x |
| qr | 1024 | 1.00x | 0.72x | 0.64x | 0.89x |
| quantile | 64 | 1.00x | 0.07x | 0.09x | 1.27x |
| quantile | 256 | 1.00x | 0.73x | 0.29x | 0.39x |
| quantile | 1024 | 1.00x | 0.42x | 0.42x | 0.99x |
| solve | 64 | 1.00x | 2.29x | 3.00x | 1.31x |
| solve | 256 | 1.00x | 2.04x | 1.87x | 0.91x |
| solve | 1024 | 1.00x | 1.10x | 1.07x | 0.97x |
| std | 64 | 1.00x | 0.18x | 0.18x | 1.03x |
| std | 256 | 1.00x | 0.41x | 0.42x | 1.00x |
| std | 1024 | 1.00x | 0.46x | 0.44x | 0.96x |
| std | 4096 | 1.00x | 0.32x | — | — |

<!-- PERF_TABLES_END -->
