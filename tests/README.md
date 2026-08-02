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
| arange | 64 | 0.000523 | 0.000153 | 0.000312 |
| arange | 256 | 0.000666 | 0.000322 | 0.000625 |
| arange | 1024 | 0.000980 | 0.000990 | 0.001740 |
| copy | 64 | 0.001349 | 0.000940 | 0.001710 |
| copy | 256 | 0.015064 | 0.012721 | 0.014447 |
| copy | 1024 | 0.614864 | 0.624320 | 0.637005 |
| dot | 64 | 0.001098 | 0.000054 | 0.000230 |
| dot | 256 | 0.001132 | 0.000126 | 0.000316 |
| dot | 1024 | 0.001520 | 0.000400 | 0.000628 |
| elem_add | 64 | 0.001785 | 0.002527 | 0.013264 |
| elem_add | 256 | 0.024568 | 0.039567 | 0.040873 |
| elem_add | 1024 | 0.894746 | 0.929146 | 1.0267 |
| elem_div | 64 | 0.016488 | 0.008324 | 0.008864 |
| elem_div | 256 | 0.222635 | 0.131386 | 0.139823 |
| elem_div | 1024 | 3.6393 | 2.2191 | 2.2533 |
| elem_mul | 64 | 0.002353 | 0.002529 | 0.003033 |
| elem_mul | 256 | 0.028114 | 0.038631 | 0.051160 |
| elem_mul | 1024 | 0.891278 | 0.931190 | 0.976326 |
| elem_sub | 64 | 0.001748 | 0.002526 | 0.002987 |
| elem_sub | 256 | 0.022469 | 0.038940 | 0.039668 |
| elem_sub | 1024 | 0.889325 | 0.928898 | 0.990635 |
| eye | 64 | 0.002303 | 0.000289 | 0.001586 |
| eye | 256 | 0.014463 | 0.011234 | 0.013200 |
| eye | 1024 | 0.329581 | 0.317440 | 0.349882 |
| fill | 64 | 0.001720 | 0.000721 | 0.000539 |
| fill | 256 | 0.022407 | 0.011561 | 0.011845 |
| fill | 1024 | 0.365074 | 0.356100 | 0.381970 |
| full | 64 | 0.002698 | 0.000451 | 0.001686 |
| full | 256 | 0.023407 | 0.011666 | 0.012320 |
| full | 1024 | 0.401198 | 0.358025 | 0.391439 |
| isin | 64 | 0.021866 | 0.003713 | 0.015475 |
| isin | 256 | 0.112041 | 0.056667 | 0.067035 |
| isin | 1024 | 1.7806 | 1.0715 | 1.1147 |
| matmul | 64 | 0.132825 | 0.105147 | 0.107465 |
| matmul | 256 | 14.590 | 3.2694 | 3.2963 |
| matmul | 1024 | 6134.993 | 207.357 | 215.994 |
| max | 64 | 0.002077 | 0.000934 | 0.001043 |
| max | 256 | 0.008248 | 0.014296 | 0.014627 |
| max | 1024 | 0.281244 | 0.388303 | 0.442397 |
| min | 64 | 0.002062 | 0.000937 | 0.001045 |
| min | 256 | 0.008540 | 0.014460 | 0.014577 |
| min | 1024 | 0.280476 | 0.389246 | 0.453462 |
| ones | 64 | 0.002652 | 0.000452 | 0.001398 |
| ones | 256 | 0.023402 | 0.011704 | 0.011890 |
| ones | 1024 | 0.379174 | 0.357296 | 0.389833 |
| reshape | 64 | 0.000315 | 0.000082 | 0.000280 |
| reshape | 256 | 0.000328 | 0.000083 | 0.000276 |
| reshape | 1024 | 0.000341 | 0.000082 | 0.000541 |
| sum | 64 | 0.002383 | 0.000323 | 0.000455 |
| sum | 256 | 0.014311 | 0.007672 | 0.007848 |
| sum | 1024 | 0.305004 | 0.312731 | 0.317322 |
| transpose | 64 | 0.002836 | 0.003126 | 0.003869 |
| transpose | 256 | 0.052489 | 0.049394 | 0.050482 |
| transpose | 1024 | 4.8933 | 1.9202 | 2.0046 |
| unique | 64 | 0.003612 | 0.000135 | 0.000314 |
| unique | 256 | 0.004565 | 0.000547 | 0.000821 |
| unique | 1024 | 0.012409 | 0.000960 | 0.002673 |
| zeros | 64 | 0.000778 | 0.000281 | 0.001564 |
| zeros | 256 | 0.011288 | 0.010889 | 0.012854 |
| zeros | 1024 | 0.365153 | 0.340832 | 0.352434 |

### Table D — i64 vs NumPy int64 (relative)

**NumPy is always 1.00x**. Same columns as Table B (Rust/NumPy, Lua/NumPy, Lua/Rust).

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.29x | 0.60x | 2.04x |
| arange | 256 | 1.00x | 0.48x | 0.94x | 1.94x |
| arange | 1024 | 1.00x | 1.01x | 1.78x | 1.76x |
| copy | 64 | 1.00x | 0.70x | 1.27x | 1.82x |
| copy | 256 | 1.00x | 0.84x | 0.96x | 1.14x |
| copy | 1024 | 1.00x | 1.02x | 1.04x | 1.02x |
| dot | 64 | 1.00x | 0.05x | 0.21x | 4.26x |
| dot | 256 | 1.00x | 0.11x | 0.28x | 2.51x |
| dot | 1024 | 1.00x | 0.26x | 0.41x | 1.57x |
| elem_add | 64 | 1.00x | 1.42x | 7.43x | 5.25x |
| elem_add | 256 | 1.00x | 1.61x | 1.66x | 1.03x |
| elem_add | 1024 | 1.00x | 1.04x | 1.15x | 1.10x |
| elem_div | 64 | 1.00x | 0.50x | 0.54x | 1.06x |
| elem_div | 256 | 1.00x | 0.59x | 0.63x | 1.06x |
| elem_div | 1024 | 1.00x | 0.61x | 0.62x | 1.02x |
| elem_mul | 64 | 1.00x | 1.07x | 1.29x | 1.20x |
| elem_mul | 256 | 1.00x | 1.37x | 1.82x | 1.32x |
| elem_mul | 1024 | 1.00x | 1.04x | 1.10x | 1.05x |
| elem_sub | 64 | 1.00x | 1.45x | 1.71x | 1.18x |
| elem_sub | 256 | 1.00x | 1.73x | 1.77x | 1.02x |
| elem_sub | 1024 | 1.00x | 1.04x | 1.11x | 1.07x |
| eye | 64 | 1.00x | 0.13x | 0.69x | 5.49x |
| eye | 256 | 1.00x | 0.78x | 0.91x | 1.18x |
| eye | 1024 | 1.00x | 0.96x | 1.06x | 1.10x |
| fill | 64 | 1.00x | 0.42x | 0.31x | 0.75x |
| fill | 256 | 1.00x | 0.52x | 0.53x | 1.02x |
| fill | 1024 | 1.00x | 0.98x | 1.05x | 1.07x |
| full | 64 | 1.00x | 0.17x | 0.62x | 3.74x |
| full | 256 | 1.00x | 0.50x | 0.53x | 1.06x |
| full | 1024 | 1.00x | 0.89x | 0.98x | 1.09x |
| isin | 64 | 1.00x | 0.17x | 0.71x | 4.17x |
| isin | 256 | 1.00x | 0.51x | 0.60x | 1.18x |
| isin | 1024 | 1.00x | 0.60x | 0.63x | 1.04x |
| matmul | 64 | 1.00x | 0.79x | 0.81x | 1.02x |
| matmul | 256 | 1.00x | 0.22x | 0.23x | 1.01x |
| matmul | 1024 | 1.00x | 0.03x | 0.04x | 1.04x |
| max | 64 | 1.00x | 0.45x | 0.50x | 1.12x |
| max | 256 | 1.00x | 1.73x | 1.77x | 1.02x |
| max | 1024 | 1.00x | 1.38x | 1.57x | 1.14x |
| min | 64 | 1.00x | 0.45x | 0.51x | 1.12x |
| min | 256 | 1.00x | 1.69x | 1.71x | 1.01x |
| min | 1024 | 1.00x | 1.39x | 1.62x | 1.16x |
| ones | 64 | 1.00x | 0.17x | 0.53x | 3.09x |
| ones | 256 | 1.00x | 0.50x | 0.51x | 1.02x |
| ones | 1024 | 1.00x | 0.94x | 1.03x | 1.09x |
| reshape | 64 | 1.00x | 0.26x | 0.89x | 3.41x |
| reshape | 256 | 1.00x | 0.25x | 0.84x | 3.33x |
| reshape | 1024 | 1.00x | 0.24x | 1.59x | 6.60x |
| sum | 64 | 1.00x | 0.14x | 0.19x | 1.41x |
| sum | 256 | 1.00x | 0.54x | 0.55x | 1.02x |
| sum | 1024 | 1.00x | 1.03x | 1.04x | 1.01x |
| transpose | 64 | 1.00x | 1.10x | 1.36x | 1.24x |
| transpose | 256 | 1.00x | 0.94x | 0.96x | 1.02x |
| transpose | 1024 | 1.00x | 0.39x | 0.41x | 1.04x |
| unique | 64 | 1.00x | 0.04x | 0.09x | 2.33x |
| unique | 256 | 1.00x | 0.12x | 0.18x | 1.50x |
| unique | 1024 | 1.00x | 0.08x | 0.22x | 2.78x |
| zeros | 64 | 1.00x | 0.36x | 2.01x | 5.57x |
| zeros | 256 | 1.00x | 0.96x | 1.14x | 1.18x |
| zeros | 1024 | 1.00x | 0.93x | 0.97x | 1.03x |

### Table E — i64→f64 promote-out absolute wall time (ms)

Integer inputs, floating / LA outputs (mean, std, median, quantile, norm, solve, cholesky, qr).
NumPy uses int64 stats where natural, else float64 after cast for LA.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 64 | 0.018924 | 0.015160 | — |
| cholesky | 256 | 0.819506 | 0.639453 | — |
| mean | 64 | 0.006196 | 0.000372 | — |
| mean | 256 | 0.040957 | 0.007667 | — |
| mean | 1024 | 0.672370 | 0.339691 | — |
| median | 64 | 0.014910 | 0.003514 | — |
| median | 256 | 0.098730 | 0.148421 | — |
| median | 1024 | 1.8540 | 1.4269 | — |
| norm | 64 | 0.002616 | 0.001406 | — |
| norm | 256 | 0.007628 | 0.021930 | — |
| norm | 1024 | 0.152144 | 0.457871 | — |
| qr | 64 | 0.113361 | 0.281095 | — |
| qr | 256 | 4.8013 | 3.8716 | — |
| quantile | 64 | 0.049570 | 0.003402 | — |
| quantile | 256 | 0.238343 | 0.148436 | — |
| quantile | 1024 | 3.7403 | 1.2553 | — |
| solve | 64 | 0.032174 | 0.072677 | — |
| solve | 256 | 0.691679 | 1.3877 | — |
| solve | 1024 | 31.271 | 40.341 | — |
| std | 64 | 0.018180 | 0.003116 | — |
| std | 256 | 0.123106 | 0.051364 | — |
| std | 1024 | 2.4054 | 1.0485 | — |

### Table F — i64→f64 promote-out vs NumPy (relative)

**NumPy is always 1.00x**.

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky | 64 | 1.00x | 0.80x | — | — |
| cholesky | 256 | 1.00x | 0.78x | — | — |
| mean | 64 | 1.00x | 0.06x | — | — |
| mean | 256 | 1.00x | 0.19x | — | — |
| mean | 1024 | 1.00x | 0.51x | — | — |
| median | 64 | 1.00x | 0.24x | — | — |
| median | 256 | 1.00x | 1.50x | — | — |
| median | 1024 | 1.00x | 0.77x | — | — |
| norm | 64 | 1.00x | 0.54x | — | — |
| norm | 256 | 1.00x | 2.87x | — | — |
| norm | 1024 | 1.00x | 3.01x | — | — |
| qr | 64 | 1.00x | 2.48x | — | — |
| qr | 256 | 1.00x | 0.81x | — | — |
| quantile | 64 | 1.00x | 0.07x | — | — |
| quantile | 256 | 1.00x | 0.62x | — | — |
| quantile | 1024 | 1.00x | 0.34x | — | — |
| solve | 64 | 1.00x | 2.26x | — | — |
| solve | 256 | 1.00x | 2.01x | — | — |
| solve | 1024 | 1.00x | 1.29x | — | — |
| std | 64 | 1.00x | 0.17x | — | — |
| std | 256 | 1.00x | 0.42x | — | — |
| std | 1024 | 1.00x | 0.44x | — | — |

<!-- PERF_TABLES_END -->
