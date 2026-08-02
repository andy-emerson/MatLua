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

```bash
# Correctness
cargo test
cargo test --features lua

# Full refresh of the four tables below
cargo test --release --features lua --test fair_all -- --run --sizes 64,256,1024 \
  | awk -F'\t' 'NF==4 && ($1=="rust"||$1=="lua"){print}' > tests/bench/last_f64.tsv
python3 tests/bench/numpy_fair.py --sizes 64,256,1024 \
  | awk -F'\t' 'NF==4 && $1=="numpy"{print}' >> tests/bench/last_f64.tsv

cargo test --release --features lua --test i64_surface -- --run --sizes 64,256,1024 \
  | awk -F'\t' 'NF==4 && ($1=="rust"||$1=="lua"){print}' > tests/bench/last_i64.tsv
python3 tests/bench/numpy_i64_fair.py --sizes 64,256,1024 \
  | awk -F'\t' 'NF==4 && $1=="numpy"{print}' >> tests/bench/last_i64.tsv

cargo test --release --features lua --test i64_promote -- --run --sizes 64,256 \
  | awk -F'\t' 'NF==4 && ($1=="rust"||$1=="lua"){print}' > tests/bench/last_i64_promote.tsv
python3 tests/bench/numpy_i64_promote.py --sizes 64,256 \
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
Run date: **2026-08-02** (M7.c: hybrid isin, i64 GC debt, expanded harness, Tables A–F).  
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

Three faces: NumPy **int64** · MatLua Rust wrapping **i64** · MatLua **Lua** i64.
Same generation as `i64_surface` / `numpy_i64_fair.py`.
NumPy integer matmul is not OpenBLAS DGEMM; useful reference, not an MKL peer.

| op | n | NumPy int64 (ms) | MatLua Rust i64 (ms) | MatLua Lua i64 (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000527 | 0.000145 | 0.000295 |
| arange | 256 | 0.000680 | 0.000325 | 0.000557 |
| arange | 1024 | 0.000971 | 0.000982 | 0.001720 |
| copy | 64 | 0.001335 | 0.000943 | 0.001702 |
| copy | 256 | 0.013136 | 0.012713 | 0.013653 |
| copy | 1024 | 0.648106 | 0.610872 | 0.625100 |
| dot | 64 | 0.001075 | 0.000054 | 0.000247 |
| dot | 256 | 0.001115 | 0.000117 | 0.000325 |
| dot | 1024 | 0.001515 | 0.000398 | 0.000633 |
| elem_add | 64 | 0.001797 | 0.002510 | 0.011964 |
| elem_add | 256 | 0.023454 | 0.040557 | 0.039612 |
| elem_add | 1024 | 1.0291 | 0.897564 | 0.961748 |
| elem_div | 64 | 0.016525 | 0.008317 | 0.008806 |
| elem_div | 256 | 0.224362 | 0.131433 | 0.132279 |
| elem_div | 1024 | 3.6491 | 2.2228 | 2.2667 |
| elem_mul | 64 | 0.002374 | 0.002612 | 0.003027 |
| elem_mul | 256 | 0.026910 | 0.044189 | 0.039478 |
| elem_mul | 1024 | 0.971227 | 0.900460 | 0.980359 |
| elem_sub | 64 | 0.001742 | 0.002526 | 0.002993 |
| elem_sub | 256 | 0.019994 | 0.039502 | 0.039422 |
| elem_sub | 1024 | 0.991655 | 0.897798 | 1.0367 |
| eye | 64 | 0.002258 | 0.000289 | 0.001529 |
| eye | 256 | 0.016184 | 0.011235 | 0.011657 |
| eye | 1024 | 0.380120 | 0.311409 | 0.315790 |
| fill | 64 | 0.001724 | 0.000382 | 0.000527 |
| fill | 256 | 0.022399 | 0.011577 | 0.011838 |
| fill | 1024 | 0.376212 | 0.364736 | 0.363622 |
| full | 64 | 0.002649 | 0.000787 | 0.001728 |
| full | 256 | 0.023416 | 0.011566 | 0.012510 |
| full | 1024 | 0.379855 | 0.348201 | 0.348772 |
| isin | 64 | 0.021623 | 0.004377 | 0.013379 |
| isin | 256 | 0.110742 | 0.055370 | 0.067065 |
| isin | 1024 | 1.8258 | 1.0686 | 1.1102 |
| matmul | 64 | 0.174588 | 0.104751 | 0.106219 |
| matmul | 256 | 14.561 | 3.3832 | 3.3867 |
| matmul | 1024 | 6883.607 | 209.573 | 210.602 |
| max | 64 | 0.002071 | 0.000928 | 0.001049 |
| max | 256 | 0.008220 | 0.014438 | 0.014525 |
| max | 1024 | 0.317649 | 0.318331 | 0.332875 |
| min | 64 | 0.002048 | 0.000925 | 0.001040 |
| min | 256 | 0.008268 | 0.014455 | 0.014620 |
| min | 1024 | 0.348003 | 0.335465 | 0.365521 |
| ones | 64 | 0.002678 | 0.000460 | 0.001385 |
| ones | 256 | 0.023416 | 0.011632 | 0.011790 |
| ones | 1024 | 0.380358 | 0.342863 | 0.364060 |
| reshape | 64 | 0.000315 | 0.000082 | 0.000283 |
| reshape | 256 | 0.000328 | 0.000083 | 0.000264 |
| reshape | 1024 | 0.000337 | 0.000090 | 0.000442 |
| sum | 64 | 0.002396 | 0.000364 | 0.000459 |
| sum | 256 | 0.013295 | 0.007684 | 0.007857 |
| sum | 1024 | 0.331874 | 0.259828 | 0.319055 |
| transpose | 64 | 0.002854 | 0.003151 | 0.003792 |
| transpose | 256 | 0.052220 | 0.049610 | 0.050984 |
| transpose | 1024 | 5.7102 | 1.9169 | 2.0330 |
| unique | 64 | 0.003651 | 0.000134 | 0.000319 |
| unique | 256 | 0.004502 | 0.000627 | 0.000852 |
| unique | 1024 | 0.009880 | 0.000935 | 0.002374 |
| zeros | 64 | 0.000731 | 0.000281 | 0.001442 |
| zeros | 256 | 0.011294 | 0.010885 | 0.011346 |
| zeros | 1024 | 0.391828 | 0.314795 | 0.340455 |

### Table D — i64 vs NumPy int64 (relative)

**NumPy is always 1.00x**. Same columns as Table B (Rust/NumPy, Lua/NumPy, Lua/Rust).

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.28x | 0.56x | 2.03x |
| arange | 256 | 1.00x | 0.48x | 0.82x | 1.71x |
| arange | 1024 | 1.00x | 1.01x | 1.77x | 1.75x |
| copy | 64 | 1.00x | 0.71x | 1.27x | 1.80x |
| copy | 256 | 1.00x | 0.97x | 1.04x | 1.07x |
| copy | 1024 | 1.00x | 0.94x | 0.96x | 1.02x |
| dot | 64 | 1.00x | 0.05x | 0.23x | 4.57x |
| dot | 256 | 1.00x | 0.10x | 0.29x | 2.78x |
| dot | 1024 | 1.00x | 0.26x | 0.42x | 1.59x |
| elem_add | 64 | 1.00x | 1.40x | 6.66x | 4.77x |
| elem_add | 256 | 1.00x | 1.73x | 1.69x | 0.98x |
| elem_add | 1024 | 1.00x | 0.87x | 0.93x | 1.07x |
| elem_div | 64 | 1.00x | 0.50x | 0.53x | 1.06x |
| elem_div | 256 | 1.00x | 0.59x | 0.59x | 1.01x |
| elem_div | 1024 | 1.00x | 0.61x | 0.62x | 1.02x |
| elem_mul | 64 | 1.00x | 1.10x | 1.28x | 1.16x |
| elem_mul | 256 | 1.00x | 1.64x | 1.47x | 0.89x |
| elem_mul | 1024 | 1.00x | 0.93x | 1.01x | 1.09x |
| elem_sub | 64 | 1.00x | 1.45x | 1.72x | 1.18x |
| elem_sub | 256 | 1.00x | 1.98x | 1.97x | 1.00x |
| elem_sub | 1024 | 1.00x | 0.91x | 1.05x | 1.15x |
| eye | 64 | 1.00x | 0.13x | 0.68x | 5.29x |
| eye | 256 | 1.00x | 0.69x | 0.72x | 1.04x |
| eye | 1024 | 1.00x | 0.82x | 0.83x | 1.01x |
| fill | 64 | 1.00x | 0.22x | 0.31x | 1.38x |
| fill | 256 | 1.00x | 0.52x | 0.53x | 1.02x |
| fill | 1024 | 1.00x | 0.97x | 0.97x | 1.00x |
| full | 64 | 1.00x | 0.30x | 0.65x | 2.20x |
| full | 256 | 1.00x | 0.49x | 0.53x | 1.08x |
| full | 1024 | 1.00x | 0.92x | 0.92x | 1.00x |
| isin | 64 | 1.00x | 0.20x | 0.62x | 3.06x |
| isin | 256 | 1.00x | 0.50x | 0.61x | 1.21x |
| isin | 1024 | 1.00x | 0.59x | 0.61x | 1.04x |
| matmul | 64 | 1.00x | 0.60x | 0.61x | 1.01x |
| matmul | 256 | 1.00x | 0.23x | 0.23x | 1.00x |
| matmul | 1024 | 1.00x | 0.03x | 0.03x | 1.00x |
| max | 64 | 1.00x | 0.45x | 0.51x | 1.13x |
| max | 256 | 1.00x | 1.76x | 1.77x | 1.01x |
| max | 1024 | 1.00x | 1.00x | 1.05x | 1.05x |
| min | 64 | 1.00x | 0.45x | 0.51x | 1.12x |
| min | 256 | 1.00x | 1.75x | 1.77x | 1.01x |
| min | 1024 | 1.00x | 0.96x | 1.05x | 1.09x |
| ones | 64 | 1.00x | 0.17x | 0.52x | 3.01x |
| ones | 256 | 1.00x | 0.50x | 0.50x | 1.01x |
| ones | 1024 | 1.00x | 0.90x | 0.96x | 1.06x |
| reshape | 64 | 1.00x | 0.26x | 0.90x | 3.45x |
| reshape | 256 | 1.00x | 0.25x | 0.80x | 3.18x |
| reshape | 1024 | 1.00x | 0.27x | 1.31x | 4.91x |
| sum | 64 | 1.00x | 0.15x | 0.19x | 1.26x |
| sum | 256 | 1.00x | 0.58x | 0.59x | 1.02x |
| sum | 1024 | 1.00x | 0.78x | 0.96x | 1.23x |
| transpose | 64 | 1.00x | 1.10x | 1.33x | 1.20x |
| transpose | 256 | 1.00x | 0.95x | 0.98x | 1.03x |
| transpose | 1024 | 1.00x | 0.34x | 0.36x | 1.06x |
| unique | 64 | 1.00x | 0.04x | 0.09x | 2.38x |
| unique | 256 | 1.00x | 0.14x | 0.19x | 1.36x |
| unique | 1024 | 1.00x | 0.09x | 0.24x | 2.54x |
| zeros | 64 | 1.00x | 0.38x | 1.97x | 5.13x |
| zeros | 256 | 1.00x | 0.96x | 1.00x | 1.04x |
| zeros | 1024 | 1.00x | 0.80x | 0.87x | 1.08x |

### Table E — i64→f64 promote-out absolute wall time (ms)

Integer inputs, floating / LA outputs (mean, std, median, quantile, norm, solve, cholesky, qr).
NumPy uses int64 stats where natural, else float64 after cast for LA.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 64 | 0.019610 | 0.014993 | 0.015883 |
| cholesky | 256 | 0.918360 | 0.791629 | 0.898380 |
| mean | 64 | 0.006354 | 0.000370 | 0.000511 |
| mean | 256 | 0.040647 | 0.007668 | 0.007916 |
| median | 64 | 0.014928 | 0.003545 | 0.003553 |
| median | 256 | 0.097106 | 0.149280 | 0.152249 |
| norm | 64 | 0.002616 | 0.000028 | 0.001538 |
| norm | 256 | 0.008491 | 0.000029 | 0.022193 |
| qr | 64 | 0.115955 | 0.312135 | 0.309646 |
| qr | 256 | 5.4881 | 3.9545 | 4.7887 |
| quantile | 64 | 0.049379 | 0.004038 | 0.004247 |
| quantile | 256 | 0.230521 | 0.161495 | 0.163386 |
| solve | 64 | 0.031955 | 0.079574 | 0.080077 |
| solve | 256 | 0.690720 | 1.5591 | 1.7635 |
| std | 64 | 0.018222 | 0.003109 | 0.003206 |
| std | 256 | 0.124542 | 0.051363 | 0.051895 |

### Table F — i64→f64 promote-out vs NumPy (relative)

**NumPy is always 1.00x**.

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky | 64 | 1.00x | 0.76x | 0.81x | 1.06x |
| cholesky | 256 | 1.00x | 0.86x | 0.98x | 1.13x |
| mean | 64 | 1.00x | 0.06x | 0.08x | 1.38x |
| mean | 256 | 1.00x | 0.19x | 0.19x | 1.03x |
| median | 64 | 1.00x | 0.24x | 0.24x | 1.00x |
| median | 256 | 1.00x | 1.54x | 1.57x | 1.02x |
| norm | 64 | 1.00x | 0.01x | 0.59x | 54.93x |
| norm | 256 | 1.00x | 0.00x | 2.61x | 765.28x |
| qr | 64 | 1.00x | 2.69x | 2.67x | 0.99x |
| qr | 256 | 1.00x | 0.72x | 0.87x | 1.21x |
| quantile | 64 | 1.00x | 0.08x | 0.09x | 1.05x |
| quantile | 256 | 1.00x | 0.70x | 0.71x | 1.01x |
| solve | 64 | 1.00x | 2.49x | 2.51x | 1.01x |
| solve | 256 | 1.00x | 2.26x | 2.55x | 1.13x |
| std | 64 | 1.00x | 0.17x | 0.18x | 1.03x |
| std | 256 | 1.00x | 0.41x | 0.42x | 1.01x |

<!-- PERF_TABLES_END -->
