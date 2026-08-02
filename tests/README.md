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
**i64 faces:** NumPy `int64` · MatLua Rust wrapping `i64` (Lua i64 face exists; not in this harness yet).

**Relative tables:** NumPy is always **1.00x**. Other columns are `time / NumPy_time`.

```bash
# Correctness
cargo test
cargo test --features lua

# Full refresh of the four tables below
cargo test --release --features lua --test fair_all -- --run --sizes 64,256,1024 \
  | awk -F'\t' 'NF==4 && ($1=="rust"||$1=="lua"){print}' > tests/bench/last_f64.tsv
python3 tests/bench/numpy_fair.py --sizes 64,256,1024 \
  | awk -F'\t' 'NF==4 && $1=="numpy"{print}' >> tests/bench/last_f64.tsv

cargo test --release --test i64_surface -- --run --sizes 64,256,1024 \
  | awk -F'\t' 'NF==4 && $1=="i64"{print}' > tests/bench/last_i64.tsv
python3 tests/bench/numpy_i64_fair.py --sizes 64,256,1024 \
  | awk -F'\t' 'NF==4 && $1=="numpy"{print}' >> tests/bench/last_i64.tsv

python3 tests/bench/compare_tables.py --write-readme tests/README.md
```

Or: `python3 tests/bench/compare_fair.py` still builds the old single-table path from `last_results.tsv` if you need it; **prefer `compare_tables.py`**.

### Tracking

| Item | Status |
|------|--------|
| in-place `out=` full surface | [#21](https://github.com/andy-emerson/MatLua/issues/21) open |
| M7.c optimize (f64 + i64) | in progress — see DESIGN §7.1 / §7.1.2 |
| Explicit i64 SIMD GEMM | researched: AVX2 lacks i64 mul; AVX-512DQ / unstable `std::simd` deferred |

### Caveats

- Sub-0.01 ms cells are **noisy**; ratios can swing.
- f64 matmul/solve/decompositions use **faer** (+ OpenBLAS on the NumPy side for large GEMM).
- i64 matmul is **packed GEBP + Rayon** (wrapping); NumPy `int64 @ int64` is not BLAS GEMM — Table D is a product reference, not "beat MKL".
- MatLua always **materializes** transpose; NumPy may return a view (we force `.T.copy()` on the NumPy i64 side).

## Latest results

Host: Linux x86_64, 2 CPUs, MatLua **release**, NumPy + OpenBLAS.  
Run date: **2026-08-02** (M7.c wave 4 + four-table layout).  
Re-run: commands above.

<!-- PERF_TABLES_START -->

### Table A — f64 absolute wall time (ms)

Median wall time. Setup outside the clock. Smaller is better.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000597 | 0.000159 | 0.000380 |
| arange | 256 | 0.000896 | 0.000355 | 0.000556 |
| arange | 1024 | 0.001246 | 0.001089 | 0.002143 |
| cholesky | 64 | 0.022723 | 0.015602 | 0.016589 |
| cholesky | 256 | 0.794056 | 0.584784 | 0.583642 |
| cholesky | 1024 | 23.421 | 19.946 | 19.735 |
| copy | 64 | 0.001383 | 0.000987 | 0.003826 |
| copy | 256 | 0.014995 | 0.013513 | 0.017179 |
| copy | 1024 | 0.604413 | 0.623065 | 0.650492 |
| dot | 64 | 0.000710 | 0.000053 | 0.000393 |
| dot | 256 | 0.000730 | 0.000111 | 0.000434 |
| dot | 1024 | 0.000914 | 0.000369 | 0.000812 |
| elem_add | 64 | 0.002785 | 0.001342 | 0.005165 |
| elem_add | 256 | 0.035480 | 0.022014 | 0.026664 |
| elem_add | 1024 | 0.897466 | 0.918545 | 1.0858 |
| elem_add_scalar | 64 | 0.001682 | 0.001027 | 0.004756 |
| elem_add_scalar | 256 | 0.013634 | 0.014393 | 0.016413 |
| elem_add_scalar | 1024 | 0.609078 | 0.661803 | 0.752491 |
| elem_div | 64 | 0.003385 | 0.003020 | 0.004641 |
| elem_div | 256 | 0.044545 | 0.044391 | 0.045613 |
| elem_div | 1024 | 0.875145 | 0.949891 | 0.989700 |
| elem_mul | 64 | 0.002772 | 0.001483 | 0.004449 |
| elem_mul | 256 | 0.035342 | 0.022542 | 0.028297 |
| elem_mul | 1024 | 0.877843 | 0.922675 | 0.958317 |
| elem_sub | 64 | 0.002778 | 0.001340 | 0.003845 |
| elem_sub | 256 | 0.035471 | 0.021913 | 0.025220 |
| elem_sub | 1024 | 0.874425 | 0.920814 | 0.980728 |
| eye | 64 | 0.002479 | 0.000729 | 0.003335 |
| eye | 256 | 0.018327 | 0.012507 | 0.011667 |
| eye | 1024 | 0.333963 | 0.346295 | 0.345033 |
| fill | 64 | 0.001713 | 0.000383 | 0.000835 |
| fill | 256 | 0.025442 | 0.011530 | 0.011710 |
| fill | 1024 | 0.373959 | 0.374495 | 0.359752 |
| full | 64 | 0.002687 | 0.000863 | 0.004202 |
| full | 256 | 0.025459 | 0.011780 | 0.012291 |
| full | 1024 | 0.369765 | 0.362892 | 0.371387 |
| matmul | 64 | 0.011089 | 0.011125 | 0.014067 |
| matmul | 256 | 0.521153 | 0.366506 | 0.532847 |
| matmul | 1024 | 17.543 | 20.294 | 22.728 |
| max | 64 | 0.001898 | 0.001555 | 0.001850 |
| max | 256 | 0.009311 | 0.023992 | 0.024064 |
| max | 1024 | 0.270049 | 0.505820 | 0.446426 |
| mean | 64 | 0.004274 | 0.000720 | 0.000906 |
| mean | 256 | 0.016562 | 0.010998 | 0.011185 |
| mean | 1024 | 0.334586 | 0.314180 | 0.289340 |
| min | 64 | 0.001889 | 0.001559 | 0.001675 |
| min | 256 | 0.009244 | 0.023950 | 0.024161 |
| min | 1024 | 0.269541 | 0.505141 | 0.482939 |
| norm | 64 | 0.002688 | 0.000724 | 0.000944 |
| norm | 256 | 0.009358 | 0.011001 | 0.011268 |
| norm | 1024 | 0.165912 | 0.350982 | 0.347890 |
| ones | 64 | 0.002807 | 0.000852 | 0.004225 |
| ones | 256 | 0.025738 | 0.011624 | 0.012321 |
| ones | 1024 | 0.364844 | 0.370847 | 0.363060 |
| qr | 64 | 0.114445 | 0.253005 | 0.254171 |
| qr | 256 | 5.0142 | 2.5864 | 2.5520 |
| qr | 1024 | 113.121 | 66.331 | 60.553 |
| reshape | 64 | 0.000296 | 0.000082 | 0.000275 |
| reshape | 256 | 0.000341 | 0.000093 | 0.000425 |
| reshape | 1024 | 0.000332 | 0.000083 | 0.000622 |
| solve | 64 | 0.036106 | 0.074490 | 0.074890 |
| solve | 256 | 0.676272 | 1.2959 | 1.1063 |
| solve | 1024 | 25.387 | 28.517 | 34.267 |
| sum | 64 | 0.002247 | 0.000716 | 0.000902 |
| sum | 256 | 0.014510 | 0.010990 | 0.011180 |
| sum | 1024 | 0.324233 | 0.316667 | 0.305014 |
| svd | 64 | 0.302632 | 0.458540 | 0.435121 |
| svd | 256 | 10.005 | 11.038 | 11.453 |
| svd | 1024 | 345.692 | 409.670 | 419.166 |
| transpose | 64 | 0.002562 | 0.003133 | 0.006428 |
| transpose | 256 | 0.052147 | 0.049085 | 0.049853 |
| transpose | 1024 | 4.4588 | 1.8999 | 1.9815 |
| zeros | 64 | 0.000853 | 0.000760 | 0.004036 |
| zeros | 256 | 0.012901 | 0.012975 | 0.011380 |
| zeros | 1024 | 0.325926 | 0.325737 | 0.346323 |

### Table B — f64 vs NumPy (relative)

**NumPy is always 1.00x** (baseline). Values are wall_time / NumPy wall_time.

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.27x | 0.64x | 2.39x |
| arange | 256 | 1.00x | 0.40x | 0.62x | 1.57x |
| arange | 1024 | 1.00x | 0.87x | 1.72x | 1.97x |
| cholesky | 64 | 1.00x | 0.69x | 0.73x | 1.06x |
| cholesky | 256 | 1.00x | 0.74x | 0.74x | 1.00x |
| cholesky | 1024 | 1.00x | 0.85x | 0.84x | 0.99x |
| copy | 64 | 1.00x | 0.71x | 2.77x | 3.88x |
| copy | 256 | 1.00x | 0.90x | 1.15x | 1.27x |
| copy | 1024 | 1.00x | 1.03x | 1.08x | 1.04x |
| dot | 64 | 1.00x | 0.07x | 0.55x | 7.42x |
| dot | 256 | 1.00x | 0.15x | 0.59x | 3.91x |
| dot | 1024 | 1.00x | 0.40x | 0.89x | 2.20x |
| elem_add | 64 | 1.00x | 0.48x | 1.85x | 3.85x |
| elem_add | 256 | 1.00x | 0.62x | 0.75x | 1.21x |
| elem_add | 1024 | 1.00x | 1.02x | 1.21x | 1.18x |
| elem_add_scalar | 64 | 1.00x | 0.61x | 2.83x | 4.63x |
| elem_add_scalar | 256 | 1.00x | 1.06x | 1.20x | 1.14x |
| elem_add_scalar | 1024 | 1.00x | 1.09x | 1.24x | 1.14x |
| elem_div | 64 | 1.00x | 0.89x | 1.37x | 1.54x |
| elem_div | 256 | 1.00x | 1.00x | 1.02x | 1.03x |
| elem_div | 1024 | 1.00x | 1.09x | 1.13x | 1.04x |
| elem_mul | 64 | 1.00x | 0.53x | 1.60x | 3.00x |
| elem_mul | 256 | 1.00x | 0.64x | 0.80x | 1.26x |
| elem_mul | 1024 | 1.00x | 1.05x | 1.09x | 1.04x |
| elem_sub | 64 | 1.00x | 0.48x | 1.38x | 2.87x |
| elem_sub | 256 | 1.00x | 0.62x | 0.71x | 1.15x |
| elem_sub | 1024 | 1.00x | 1.05x | 1.12x | 1.07x |
| eye | 64 | 1.00x | 0.29x | 1.35x | 4.57x |
| eye | 256 | 1.00x | 0.68x | 0.64x | 0.93x |
| eye | 1024 | 1.00x | 1.04x | 1.03x | 1.00x |
| fill | 64 | 1.00x | 0.22x | 0.49x | 2.18x |
| fill | 256 | 1.00x | 0.45x | 0.46x | 1.02x |
| fill | 1024 | 1.00x | 1.00x | 0.96x | 0.96x |
| full | 64 | 1.00x | 0.32x | 1.56x | 4.87x |
| full | 256 | 1.00x | 0.46x | 0.48x | 1.04x |
| full | 1024 | 1.00x | 0.98x | 1.00x | 1.02x |
| matmul | 64 | 1.00x | 1.00x | 1.27x | 1.26x |
| matmul | 256 | 1.00x | 0.70x | 1.02x | 1.45x |
| matmul | 1024 | 1.00x | 1.16x | 1.30x | 1.12x |
| max | 64 | 1.00x | 0.82x | 0.97x | 1.19x |
| max | 256 | 1.00x | 2.58x | 2.58x | 1.00x |
| max | 1024 | 1.00x | 1.87x | 1.65x | 0.88x |
| mean | 64 | 1.00x | 0.17x | 0.21x | 1.26x |
| mean | 256 | 1.00x | 0.66x | 0.68x | 1.02x |
| mean | 1024 | 1.00x | 0.94x | 0.86x | 0.92x |
| min | 64 | 1.00x | 0.83x | 0.89x | 1.07x |
| min | 256 | 1.00x | 2.59x | 2.61x | 1.01x |
| min | 1024 | 1.00x | 1.87x | 1.79x | 0.96x |
| norm | 64 | 1.00x | 0.27x | 0.35x | 1.30x |
| norm | 256 | 1.00x | 1.18x | 1.20x | 1.02x |
| norm | 1024 | 1.00x | 2.12x | 2.10x | 0.99x |
| ones | 64 | 1.00x | 0.30x | 1.51x | 4.96x |
| ones | 256 | 1.00x | 0.45x | 0.48x | 1.06x |
| ones | 1024 | 1.00x | 1.02x | 1.00x | 0.98x |
| qr | 64 | 1.00x | 2.21x | 2.22x | 1.00x |
| qr | 256 | 1.00x | 0.52x | 0.51x | 0.99x |
| qr | 1024 | 1.00x | 0.59x | 0.54x | 0.91x |
| reshape | 64 | 1.00x | 0.28x | 0.93x | 3.35x |
| reshape | 256 | 1.00x | 0.27x | 1.25x | 4.57x |
| reshape | 1024 | 1.00x | 0.25x | 1.87x | 7.49x |
| solve | 64 | 1.00x | 2.06x | 2.07x | 1.01x |
| solve | 256 | 1.00x | 1.92x | 1.64x | 0.85x |
| solve | 1024 | 1.00x | 1.12x | 1.35x | 1.20x |
| sum | 64 | 1.00x | 0.32x | 0.40x | 1.26x |
| sum | 256 | 1.00x | 0.76x | 0.77x | 1.02x |
| sum | 1024 | 1.00x | 0.98x | 0.94x | 0.96x |
| svd | 64 | 1.00x | 1.52x | 1.44x | 0.95x |
| svd | 256 | 1.00x | 1.10x | 1.14x | 1.04x |
| svd | 1024 | 1.00x | 1.19x | 1.21x | 1.02x |
| transpose | 64 | 1.00x | 1.22x | 2.51x | 2.05x |
| transpose | 256 | 1.00x | 0.94x | 0.96x | 1.02x |
| transpose | 1024 | 1.00x | 0.43x | 0.44x | 1.04x |
| zeros | 64 | 1.00x | 0.89x | 4.73x | 5.31x |
| zeros | 256 | 1.00x | 1.01x | 0.88x | 0.88x |
| zeros | 1024 | 1.00x | 1.00x | 1.06x | 1.06x |

### Table C — i64 absolute wall time (ms)

MatLua **wrapping i64** vs NumPy **int64** (same generation rule as `i64_surface`).
NumPy integer matmul is not OpenBLAS DGEMM; useful reference, not an MKL peer.

| op | n | NumPy int64 (ms) | MatLua Rust i64 (ms) |
| --- | ---: | ---: | ---: |
| dot | 64 | 0.001103 | 0.000053 |
| dot | 256 | 0.001252 | 0.000142 |
| dot | 1024 | 0.001516 | 0.000392 |
| elem_add | 64 | 0.001785 | 0.002550 |
| elem_add | 256 | 0.024721 | 0.039883 |
| elem_add | 1024 | 1.0444 | 1.0200 |
| elem_mul | 64 | 0.002369 | 0.002510 |
| elem_mul | 256 | 0.028610 | 0.039440 |
| elem_mul | 1024 | 0.987595 | 0.992077 |
| isin | 64 | 0.021786 | 0.044336 |
| isin | 256 | 0.111368 | 0.713782 |
| isin | 1024 | 1.7800 | 11.495 |
| matmul | 64 | 0.129852 | 0.109687 |
| matmul | 256 | 14.893 | 3.2863 |
| matmul | 1024 | 5196.266 | 206.415 |
| min | 64 | 0.002080 | 0.000937 |
| min | 256 | 0.009123 | 0.014417 |
| min | 1024 | 0.330090 | 0.418728 |
| sum | 64 | 0.002403 | 0.000336 |
| sum | 256 | 0.014729 | 0.007652 |
| sum | 1024 | 0.341873 | 0.347830 |
| transpose | 64 | 0.002840 | 0.003208 |
| transpose | 256 | 0.054769 | 0.053261 |
| transpose | 1024 | 6.3738 | 4.6462 |
| unique | 64 | 0.003984 | 0.000138 |
| unique | 256 | 0.004707 | 0.000534 |
| unique | 1024 | 0.009719 | 0.000927 |

### Table D — i64 vs NumPy int64 (relative)

**NumPy is always 1.00x**.

| op | n | NumPy | MatLua i64 / NumPy |
| --- | ---: | ---: | ---: |
| dot | 64 | 1.00x | 0.05x |
| dot | 256 | 1.00x | 0.11x |
| dot | 1024 | 1.00x | 0.26x |
| elem_add | 64 | 1.00x | 1.43x |
| elem_add | 256 | 1.00x | 1.61x |
| elem_add | 1024 | 1.00x | 0.98x |
| elem_mul | 64 | 1.00x | 1.06x |
| elem_mul | 256 | 1.00x | 1.38x |
| elem_mul | 1024 | 1.00x | 1.00x |
| isin | 64 | 1.00x | 2.04x |
| isin | 256 | 1.00x | 6.41x |
| isin | 1024 | 1.00x | 6.46x |
| matmul | 64 | 1.00x | 0.84x |
| matmul | 256 | 1.00x | 0.22x |
| matmul | 1024 | 1.00x | 0.04x |
| min | 64 | 1.00x | 0.45x |
| min | 256 | 1.00x | 1.58x |
| min | 1024 | 1.00x | 1.27x |
| sum | 64 | 1.00x | 0.14x |
| sum | 256 | 1.00x | 0.52x |
| sum | 1024 | 1.00x | 1.02x |
| transpose | 64 | 1.00x | 1.13x |
| transpose | 256 | 1.00x | 0.97x |
| transpose | 1024 | 1.00x | 0.73x |
| unique | 64 | 1.00x | 0.03x |
| unique | 256 | 1.00x | 0.11x |
| unique | 1024 | 1.00x | 0.10x |

<!-- PERF_TABLES_END -->
