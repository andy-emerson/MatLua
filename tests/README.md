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
Run date: **2026-08-02** (M7.c wave 7 retune: Strassen leaf 4096, full table refresh).  
Re-run: commands above.

<!-- PERF_TABLES_START -->

### Table A — f64 absolute wall time (ms)

Median wall time. Setup outside the clock. Smaller is better.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000604 | 0.000164 | 0.000420 |
| arange | 256 | 0.000822 | 0.000353 | 0.000611 |
| arange | 1024 | 0.001223 | 0.001203 | 0.001893 |
| cholesky | 64 | 0.022770 | 0.015361 | 0.016552 |
| cholesky | 256 | 0.827808 | 0.600331 | 0.629789 |
| cholesky | 1024 | 23.665 | 18.946 | 19.510 |
| copy | 64 | 0.001378 | 0.001011 | 0.003730 |
| copy | 256 | 0.013557 | 0.013291 | 0.014417 |
| copy | 1024 | 0.648106 | 0.594045 | 0.667680 |
| dot | 64 | 0.000717 | 0.000058 | 0.000395 |
| dot | 256 | 0.000917 | 0.000128 | 0.000427 |
| dot | 1024 | 0.000846 | 0.000378 | 0.000789 |
| elem_add | 64 | 0.002805 | 0.001438 | 0.005080 |
| elem_add | 256 | 0.037154 | 0.022624 | 0.024694 |
| elem_add | 1024 | 1.2213 | 0.894933 | 0.992365 |
| elem_add_scalar | 64 | 0.001688 | 0.001022 | 0.004773 |
| elem_add_scalar | 256 | 0.013905 | 0.014189 | 0.015816 |
| elem_add_scalar | 1024 | 0.675368 | 0.607637 | 0.715400 |
| elem_div | 64 | 0.003383 | 0.003181 | 0.004771 |
| elem_div | 256 | 0.044998 | 0.044368 | 0.045430 |
| elem_div | 1024 | 1.0686 | 0.899489 | 0.968099 |
| elem_mul | 64 | 0.002817 | 0.001496 | 0.004527 |
| elem_mul | 256 | 0.037074 | 0.022816 | 0.025946 |
| elem_mul | 1024 | 1.1887 | 0.905443 | 0.947708 |
| elem_sub | 64 | 0.002797 | 0.001486 | 0.004187 |
| elem_sub | 256 | 0.037144 | 0.022275 | 0.024697 |
| elem_sub | 1024 | 1.1986 | 0.904160 | 0.967027 |
| eye | 64 | 0.002351 | 0.000738 | 0.003125 |
| eye | 256 | 0.015961 | 0.011250 | 0.011661 |
| eye | 1024 | 0.338616 | 0.331914 | 0.312409 |
| fill | 64 | 0.001734 | 0.000395 | 0.000546 |
| fill | 256 | 0.022392 | 0.011558 | 0.011815 |
| fill | 1024 | 0.357372 | 0.342175 | 0.360033 |
| full | 64 | 0.002600 | 0.000858 | 0.004590 |
| full | 256 | 0.023347 | 0.011620 | 0.012497 |
| full | 1024 | 0.370596 | 0.358799 | 0.355955 |
| matmul | 64 | 0.010946 | 0.010907 | 0.014299 |
| matmul | 256 | 0.526253 | 0.354936 | 0.511167 |
| matmul | 1024 | 17.360 | 19.749 | 21.630 |
| max | 64 | 0.001920 | 0.001575 | 0.001661 |
| max | 256 | 0.012500 | 0.023930 | 0.024274 |
| max | 1024 | 0.333526 | 0.284882 | 0.297580 |
| mean | 64 | 0.004290 | 0.000720 | 0.000904 |
| mean | 256 | 0.016564 | 0.011002 | 0.011218 |
| mean | 1024 | 0.391250 | 0.270154 | 0.307940 |
| min | 64 | 0.001913 | 0.001558 | 0.001677 |
| min | 256 | 0.012509 | 0.024194 | 0.024350 |
| min | 1024 | 0.324982 | 0.273156 | 0.273696 |
| norm | 64 | 0.002393 | 0.000723 | 0.000943 |
| norm | 256 | 0.009120 | 0.011005 | 0.011253 |
| norm | 1024 | 0.168368 | 0.264961 | 0.314075 |
| ones | 64 | 0.002861 | 0.000863 | 0.004492 |
| ones | 256 | 0.023574 | 0.011660 | 0.012260 |
| ones | 1024 | 0.367146 | 0.361602 | 0.357042 |
| qr | 64 | 0.123195 | 0.255050 | 0.368914 |
| qr | 256 | 4.7759 | 3.1391 | 2.6449 |
| qr | 1024 | 111.551 | 60.400 | 65.057 |
| reshape | 64 | 0.000309 | 0.000085 | 0.000293 |
| reshape | 256 | 0.000312 | 0.000085 | 0.000368 |
| reshape | 1024 | 0.000320 | 0.000083 | 0.000602 |
| solve | 64 | 0.037695 | 0.075979 | 0.076796 |
| solve | 256 | 0.700279 | 1.4088 | 1.0701 |
| solve | 1024 | 25.328 | 28.087 | 24.624 |
| sum | 64 | 0.002276 | 0.000716 | 0.000912 |
| sum | 256 | 0.014966 | 0.010983 | 0.011161 |
| sum | 1024 | 0.383410 | 0.275139 | 0.306840 |
| svd | 64 | 0.260066 | 0.469490 | 0.597106 |
| svd | 256 | 9.4294 | 11.959 | 12.658 |
| svd | 1024 | 324.085 | 401.242 | 411.825 |
| transpose | 64 | 0.002580 | 0.003160 | 0.006394 |
| transpose | 256 | 0.052246 | 0.049760 | 0.050533 |
| transpose | 1024 | 6.5637 | 1.8466 | 2.2098 |
| zeros | 64 | 0.000865 | 0.000770 | 0.004153 |
| zeros | 256 | 0.011301 | 0.010897 | 0.011350 |
| zeros | 1024 | 0.356871 | 0.319619 | 0.337365 |

### Table B — f64 vs NumPy (relative)

**NumPy is always 1.00x** (baseline). Values are wall_time / NumPy wall_time.

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.27x | 0.70x | 2.56x |
| arange | 256 | 1.00x | 0.43x | 0.74x | 1.73x |
| arange | 1024 | 1.00x | 0.98x | 1.55x | 1.57x |
| cholesky | 64 | 1.00x | 0.67x | 0.73x | 1.08x |
| cholesky | 256 | 1.00x | 0.73x | 0.76x | 1.05x |
| cholesky | 1024 | 1.00x | 0.80x | 0.82x | 1.03x |
| copy | 64 | 1.00x | 0.73x | 2.71x | 3.69x |
| copy | 256 | 1.00x | 0.98x | 1.06x | 1.08x |
| copy | 1024 | 1.00x | 0.92x | 1.03x | 1.12x |
| dot | 64 | 1.00x | 0.08x | 0.55x | 6.81x |
| dot | 256 | 1.00x | 0.14x | 0.47x | 3.34x |
| dot | 1024 | 1.00x | 0.45x | 0.93x | 2.09x |
| elem_add | 64 | 1.00x | 0.51x | 1.81x | 3.53x |
| elem_add | 256 | 1.00x | 0.61x | 0.66x | 1.09x |
| elem_add | 1024 | 1.00x | 0.73x | 0.81x | 1.11x |
| elem_add_scalar | 64 | 1.00x | 0.61x | 2.83x | 4.67x |
| elem_add_scalar | 256 | 1.00x | 1.02x | 1.14x | 1.11x |
| elem_add_scalar | 1024 | 1.00x | 0.90x | 1.06x | 1.18x |
| elem_div | 64 | 1.00x | 0.94x | 1.41x | 1.50x |
| elem_div | 256 | 1.00x | 0.99x | 1.01x | 1.02x |
| elem_div | 1024 | 1.00x | 0.84x | 0.91x | 1.08x |
| elem_mul | 64 | 1.00x | 0.53x | 1.61x | 3.03x |
| elem_mul | 256 | 1.00x | 0.62x | 0.70x | 1.14x |
| elem_mul | 1024 | 1.00x | 0.76x | 0.80x | 1.05x |
| elem_sub | 64 | 1.00x | 0.53x | 1.50x | 2.82x |
| elem_sub | 256 | 1.00x | 0.60x | 0.66x | 1.11x |
| elem_sub | 1024 | 1.00x | 0.75x | 0.81x | 1.07x |
| eye | 64 | 1.00x | 0.31x | 1.33x | 4.23x |
| eye | 256 | 1.00x | 0.70x | 0.73x | 1.04x |
| eye | 1024 | 1.00x | 0.98x | 0.92x | 0.94x |
| fill | 64 | 1.00x | 0.23x | 0.31x | 1.38x |
| fill | 256 | 1.00x | 0.52x | 0.53x | 1.02x |
| fill | 1024 | 1.00x | 0.96x | 1.01x | 1.05x |
| full | 64 | 1.00x | 0.33x | 1.77x | 5.35x |
| full | 256 | 1.00x | 0.50x | 0.54x | 1.08x |
| full | 1024 | 1.00x | 0.97x | 0.96x | 0.99x |
| matmul | 64 | 1.00x | 1.00x | 1.31x | 1.31x |
| matmul | 256 | 1.00x | 0.67x | 0.97x | 1.44x |
| matmul | 1024 | 1.00x | 1.14x | 1.25x | 1.10x |
| max | 64 | 1.00x | 0.82x | 0.87x | 1.05x |
| max | 256 | 1.00x | 1.91x | 1.94x | 1.01x |
| max | 1024 | 1.00x | 0.85x | 0.89x | 1.04x |
| mean | 64 | 1.00x | 0.17x | 0.21x | 1.26x |
| mean | 256 | 1.00x | 0.66x | 0.68x | 1.02x |
| mean | 1024 | 1.00x | 0.69x | 0.79x | 1.14x |
| min | 64 | 1.00x | 0.81x | 0.88x | 1.08x |
| min | 256 | 1.00x | 1.93x | 1.95x | 1.01x |
| min | 1024 | 1.00x | 0.84x | 0.84x | 1.00x |
| norm | 64 | 1.00x | 0.30x | 0.39x | 1.30x |
| norm | 256 | 1.00x | 1.21x | 1.23x | 1.02x |
| norm | 1024 | 1.00x | 1.57x | 1.87x | 1.19x |
| ones | 64 | 1.00x | 0.30x | 1.57x | 5.21x |
| ones | 256 | 1.00x | 0.49x | 0.52x | 1.05x |
| ones | 1024 | 1.00x | 0.98x | 0.97x | 0.99x |
| qr | 64 | 1.00x | 2.07x | 2.99x | 1.45x |
| qr | 256 | 1.00x | 0.66x | 0.55x | 0.84x |
| qr | 1024 | 1.00x | 0.54x | 0.58x | 1.08x |
| reshape | 64 | 1.00x | 0.28x | 0.95x | 3.45x |
| reshape | 256 | 1.00x | 0.27x | 1.18x | 4.33x |
| reshape | 1024 | 1.00x | 0.26x | 1.88x | 7.25x |
| solve | 64 | 1.00x | 2.02x | 2.04x | 1.01x |
| solve | 256 | 1.00x | 2.01x | 1.53x | 0.76x |
| solve | 1024 | 1.00x | 1.11x | 0.97x | 0.88x |
| sum | 64 | 1.00x | 0.31x | 0.40x | 1.27x |
| sum | 256 | 1.00x | 0.73x | 0.75x | 1.02x |
| sum | 1024 | 1.00x | 0.72x | 0.80x | 1.12x |
| svd | 64 | 1.00x | 1.81x | 2.30x | 1.27x |
| svd | 256 | 1.00x | 1.27x | 1.34x | 1.06x |
| svd | 1024 | 1.00x | 1.24x | 1.27x | 1.03x |
| transpose | 64 | 1.00x | 1.22x | 2.48x | 2.02x |
| transpose | 256 | 1.00x | 0.95x | 0.97x | 1.02x |
| transpose | 1024 | 1.00x | 0.28x | 0.34x | 1.20x |
| zeros | 64 | 1.00x | 0.89x | 4.80x | 5.39x |
| zeros | 256 | 1.00x | 0.96x | 1.00x | 1.04x |
| zeros | 1024 | 1.00x | 0.90x | 0.95x | 1.06x |

### Table C — i64 absolute wall time (ms)

Three faces: NumPy **int64** · MatLua Rust wrapping **i64** · MatLua **Lua** i64.
Same generation as `i64_surface` / `numpy_i64_fair.py`.
NumPy integer matmul is not OpenBLAS DGEMM; useful reference, not an MKL peer.

| op | n | NumPy int64 (ms) | MatLua Rust i64 (ms) | MatLua Lua i64 (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000540 | 0.000144 | 0.000299 |
| arange | 256 | 0.000663 | 0.000323 | 0.000517 |
| arange | 1024 | 0.000998 | 0.000980 | 0.001882 |
| copy | 64 | 0.001359 | 0.000942 | 0.001648 |
| copy | 256 | 0.013494 | 0.012711 | 0.014510 |
| copy | 1024 | 0.644443 | 0.598466 | 0.637199 |
| dot | 64 | 0.001130 | 0.000061 | 0.000229 |
| dot | 256 | 0.001118 | 0.000142 | 0.000332 |
| dot | 1024 | 0.001501 | 0.000380 | 0.000634 |
| elem_add | 64 | 0.001829 | 0.002511 | 0.011057 |
| elem_add | 256 | 0.020590 | 0.038448 | 0.039576 |
| elem_add | 1024 | 0.955725 | 0.873726 | 1.0014 |
| elem_div | 64 | 0.016035 | 0.008315 | 0.008733 |
| elem_div | 256 | 0.223074 | 0.131332 | 0.134811 |
| elem_div | 1024 | 3.6357 | 2.1964 | 2.2584 |
| elem_mul | 64 | 0.002394 | 0.002849 | 0.002931 |
| elem_mul | 256 | 0.026269 | 0.038386 | 0.039586 |
| elem_mul | 1024 | 0.947346 | 0.875859 | 0.985688 |
| elem_sub | 64 | 0.001797 | 0.002523 | 0.002954 |
| elem_sub | 256 | 0.019906 | 0.038452 | 0.039233 |
| elem_sub | 1024 | 0.952889 | 0.872277 | 0.979253 |
| eye | 64 | 0.002347 | 0.000296 | 0.001467 |
| eye | 256 | 0.016203 | 0.011242 | 0.011682 |
| eye | 1024 | 0.348445 | 0.300307 | 0.307532 |
| fill | 64 | 0.001727 | 0.000380 | 0.000542 |
| fill | 256 | 0.022429 | 0.011537 | 0.011877 |
| fill | 1024 | 0.365993 | 0.349751 | 0.352995 |
| full | 64 | 0.002697 | 0.000451 | 0.001580 |
| full | 256 | 0.023530 | 0.011615 | 0.012775 |
| full | 1024 | 0.365675 | 0.324095 | 0.338404 |
| isin | 64 | 0.021876 | 0.004374 | 0.012228 |
| isin | 256 | 0.112236 | 0.055383 | 0.056169 |
| isin | 1024 | 1.7494 | 1.0088 | 1.0844 |
| matmul | 64 | 0.141803 | 0.106330 | 0.106417 |
| matmul | 256 | 14.634 | 3.2621 | 3.2794 |
| matmul | 1024 | 5784.915 | 206.241 | 206.658 |
| max | 64 | 0.002052 | 0.000937 | 0.001033 |
| max | 256 | 0.008255 | 0.014463 | 0.014497 |
| max | 1024 | 0.314256 | 0.334585 | 0.350045 |
| min | 64 | 0.002049 | 0.000936 | 0.001031 |
| min | 256 | 0.008293 | 0.014441 | 0.014602 |
| min | 1024 | 0.298244 | 0.358098 | 0.383567 |
| ones | 64 | 0.002678 | 0.000451 | 0.001355 |
| ones | 256 | 0.023545 | 0.011664 | 0.012454 |
| ones | 1024 | 0.371055 | 0.326136 | 0.337461 |
| reshape | 64 | 0.000315 | 0.000080 | 0.000269 |
| reshape | 256 | 0.000323 | 0.000083 | 0.000265 |
| reshape | 1024 | 0.000353 | 0.000083 | 0.000528 |
| sum | 64 | 0.002399 | 0.000295 | 0.000456 |
| sum | 256 | 0.013227 | 0.007640 | 0.007887 |
| sum | 1024 | 0.306872 | 0.274348 | 0.299371 |
| transpose | 64 | 0.002792 | 0.003140 | 0.003684 |
| transpose | 256 | 0.052132 | 0.050105 | 0.050701 |
| transpose | 1024 | 5.5270 | 1.9268 | 1.9940 |
| unique | 64 | 0.003742 | 0.000148 | 0.000313 |
| unique | 256 | 0.004680 | 0.000655 | 0.000890 |
| unique | 1024 | 0.009488 | 0.001226 | 0.002877 |
| zeros | 64 | 0.000795 | 0.000282 | 0.001477 |
| zeros | 256 | 0.011343 | 0.010879 | 0.011385 |
| zeros | 1024 | 0.387076 | 0.309417 | 0.327955 |

### Table D — i64 vs NumPy int64 (relative)

**NumPy is always 1.00x**. Same columns as Table B (Rust/NumPy, Lua/NumPy, Lua/Rust).

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.27x | 0.55x | 2.08x |
| arange | 256 | 1.00x | 0.49x | 0.78x | 1.60x |
| arange | 1024 | 1.00x | 0.98x | 1.89x | 1.92x |
| copy | 64 | 1.00x | 0.69x | 1.21x | 1.75x |
| copy | 256 | 1.00x | 0.94x | 1.08x | 1.14x |
| copy | 1024 | 1.00x | 0.93x | 0.99x | 1.06x |
| dot | 64 | 1.00x | 0.05x | 0.20x | 3.75x |
| dot | 256 | 1.00x | 0.13x | 0.30x | 2.34x |
| dot | 1024 | 1.00x | 0.25x | 0.42x | 1.67x |
| elem_add | 64 | 1.00x | 1.37x | 6.05x | 4.40x |
| elem_add | 256 | 1.00x | 1.87x | 1.92x | 1.03x |
| elem_add | 1024 | 1.00x | 0.91x | 1.05x | 1.15x |
| elem_div | 64 | 1.00x | 0.52x | 0.54x | 1.05x |
| elem_div | 256 | 1.00x | 0.59x | 0.60x | 1.03x |
| elem_div | 1024 | 1.00x | 0.60x | 0.62x | 1.03x |
| elem_mul | 64 | 1.00x | 1.19x | 1.22x | 1.03x |
| elem_mul | 256 | 1.00x | 1.46x | 1.51x | 1.03x |
| elem_mul | 1024 | 1.00x | 0.92x | 1.04x | 1.13x |
| elem_sub | 64 | 1.00x | 1.40x | 1.64x | 1.17x |
| elem_sub | 256 | 1.00x | 1.93x | 1.97x | 1.02x |
| elem_sub | 1024 | 1.00x | 0.92x | 1.03x | 1.12x |
| eye | 64 | 1.00x | 0.13x | 0.63x | 4.96x |
| eye | 256 | 1.00x | 0.69x | 0.72x | 1.04x |
| eye | 1024 | 1.00x | 0.86x | 0.88x | 1.02x |
| fill | 64 | 1.00x | 0.22x | 0.31x | 1.43x |
| fill | 256 | 1.00x | 0.51x | 0.53x | 1.03x |
| fill | 1024 | 1.00x | 0.96x | 0.96x | 1.01x |
| full | 64 | 1.00x | 0.17x | 0.59x | 3.50x |
| full | 256 | 1.00x | 0.49x | 0.54x | 1.10x |
| full | 1024 | 1.00x | 0.89x | 0.93x | 1.04x |
| isin | 64 | 1.00x | 0.20x | 0.56x | 2.80x |
| isin | 256 | 1.00x | 0.49x | 0.50x | 1.01x |
| isin | 1024 | 1.00x | 0.58x | 0.62x | 1.07x |
| matmul | 64 | 1.00x | 0.75x | 0.75x | 1.00x |
| matmul | 256 | 1.00x | 0.22x | 0.22x | 1.01x |
| matmul | 1024 | 1.00x | 0.04x | 0.04x | 1.00x |
| max | 64 | 1.00x | 0.46x | 0.50x | 1.10x |
| max | 256 | 1.00x | 1.75x | 1.76x | 1.00x |
| max | 1024 | 1.00x | 1.06x | 1.11x | 1.05x |
| min | 64 | 1.00x | 0.46x | 0.50x | 1.10x |
| min | 256 | 1.00x | 1.74x | 1.76x | 1.01x |
| min | 1024 | 1.00x | 1.20x | 1.29x | 1.07x |
| ones | 64 | 1.00x | 0.17x | 0.51x | 3.00x |
| ones | 256 | 1.00x | 0.50x | 0.53x | 1.07x |
| ones | 1024 | 1.00x | 0.88x | 0.91x | 1.03x |
| reshape | 64 | 1.00x | 0.25x | 0.85x | 3.36x |
| reshape | 256 | 1.00x | 0.26x | 0.82x | 3.19x |
| reshape | 1024 | 1.00x | 0.24x | 1.50x | 6.36x |
| sum | 64 | 1.00x | 0.12x | 0.19x | 1.55x |
| sum | 256 | 1.00x | 0.58x | 0.60x | 1.03x |
| sum | 1024 | 1.00x | 0.89x | 0.98x | 1.09x |
| transpose | 64 | 1.00x | 1.12x | 1.32x | 1.17x |
| transpose | 256 | 1.00x | 0.96x | 0.97x | 1.01x |
| transpose | 1024 | 1.00x | 0.35x | 0.36x | 1.03x |
| unique | 64 | 1.00x | 0.04x | 0.08x | 2.11x |
| unique | 256 | 1.00x | 0.14x | 0.19x | 1.36x |
| unique | 1024 | 1.00x | 0.13x | 0.30x | 2.35x |
| zeros | 64 | 1.00x | 0.35x | 1.86x | 5.24x |
| zeros | 256 | 1.00x | 0.96x | 1.00x | 1.05x |
| zeros | 1024 | 1.00x | 0.80x | 0.85x | 1.06x |

### Table E — i64→f64 promote-out absolute wall time (ms)

Integer inputs, floating / LA outputs (mean, std, median, quantile, norm, solve, cholesky, qr).
NumPy uses int64 stats where natural, else float64 after cast for LA.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 64 | 0.019365 | 0.015247 | 0.016200 |
| cholesky | 256 | 0.823410 | 0.620288 | 0.688599 |
| cholesky | 1024 | 22.912 | 22.583 | 27.762 |
| mean | 64 | 0.006387 | 0.000345 | 0.000473 |
| mean | 256 | 0.041313 | 0.007676 | 0.007886 |
| mean | 1024 | 0.705920 | 0.297316 | 0.302337 |
| median | 64 | 0.015942 | 0.003523 | 0.003722 |
| median | 256 | 0.097571 | 0.148489 | 0.054117 |
| median | 1024 | 1.9518 | 1.2445 | 1.2359 |
| norm | 64 | 0.002707 | 0.001454 | 0.001549 |
| norm | 256 | 0.008869 | 0.021937 | 0.022332 |
| norm | 1024 | 0.153472 | 0.404779 | 0.450777 |
| qr | 64 | 0.111062 | 0.278053 | 0.273226 |
| qr | 256 | 4.9143 | 3.9019 | 4.2380 |
| qr | 1024 | 114.068 | 83.262 | 86.086 |
| quantile | 64 | 0.052135 | 0.003506 | 0.003565 |
| quantile | 256 | 0.219299 | 0.149035 | 0.055171 |
| quantile | 1024 | 3.6623 | 1.2372 | 1.2464 |
| solve | 64 | 0.030620 | 0.104248 | 0.063388 |
| solve | 256 | 0.664828 | 1.2227 | 1.1605 |
| solve | 1024 | 25.849 | 51.134 | 28.627 |
| std | 64 | 0.017611 | 0.003110 | 0.003218 |
| std | 256 | 0.124318 | 0.051363 | 0.051583 |
| std | 1024 | 2.3123 | 1.0080 | 1.0592 |

### Table F — i64→f64 promote-out vs NumPy (relative)

**NumPy is always 1.00x**.

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky | 64 | 1.00x | 0.79x | 0.84x | 1.06x |
| cholesky | 256 | 1.00x | 0.75x | 0.84x | 1.11x |
| cholesky | 1024 | 1.00x | 0.99x | 1.21x | 1.23x |
| mean | 64 | 1.00x | 0.05x | 0.07x | 1.37x |
| mean | 256 | 1.00x | 0.19x | 0.19x | 1.03x |
| mean | 1024 | 1.00x | 0.42x | 0.43x | 1.02x |
| median | 64 | 1.00x | 0.22x | 0.23x | 1.06x |
| median | 256 | 1.00x | 1.52x | 0.55x | 0.36x |
| median | 1024 | 1.00x | 0.64x | 0.63x | 0.99x |
| norm | 64 | 1.00x | 0.54x | 0.57x | 1.07x |
| norm | 256 | 1.00x | 2.47x | 2.52x | 1.02x |
| norm | 1024 | 1.00x | 2.64x | 2.94x | 1.11x |
| qr | 64 | 1.00x | 2.50x | 2.46x | 0.98x |
| qr | 256 | 1.00x | 0.79x | 0.86x | 1.09x |
| qr | 1024 | 1.00x | 0.73x | 0.75x | 1.03x |
| quantile | 64 | 1.00x | 0.07x | 0.07x | 1.02x |
| quantile | 256 | 1.00x | 0.68x | 0.25x | 0.37x |
| quantile | 1024 | 1.00x | 0.34x | 0.34x | 1.01x |
| solve | 64 | 1.00x | 3.40x | 2.07x | 0.61x |
| solve | 256 | 1.00x | 1.84x | 1.75x | 0.95x |
| solve | 1024 | 1.00x | 1.98x | 1.11x | 0.56x |
| std | 64 | 1.00x | 0.18x | 0.18x | 1.03x |
| std | 256 | 1.00x | 0.41x | 0.41x | 1.00x |
| std | 1024 | 1.00x | 0.44x | 0.46x | 1.05x |

<!-- PERF_TABLES_END -->
