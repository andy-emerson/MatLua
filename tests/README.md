# Tests

| Path | Job |
|------|-----|
| [`correctness/`](correctness/) | Public API + Lua face (`cargo test`, `cargo test --features lua`). |
| [`bench/`](bench/) | Performance microbenches vs NumPy (results below). |

## Measurement

### Method

Sizes: **64, 256, 1024, 4096**. Each cell is the **median** of several timed
single calls after a short warmup (not one-shot); setup sits outside the
clock. A face that was not measured shows **—** — cells are never invented,
and summary ranges are derived from measured cells only.

**M7.c plan (durable):** keep exact i64 matmul (plan A); complete honest
numbers before any competitiveness threshold. M7.c is **not closed** by
publishing these tables.

### Yardsticks

- **NumPy is the product bar** (= 1.00x): the product is the Lua face, so
  the summary tables read Lua vs NumPy; the Rust face is developer
  diagnostics in the appendix.
- **i64 matmul family** (`matmul` / `matmul_at` / `matmul_bt`): the NumPy
  reference is **float64 BLAS** on the same integer-valued inputs (not
  `int64@int64` — NumPy has no integer BLAS, so that fallback is not a
  product bar). MatLua times are **exact wrapping i64**. See DESIGN §7.1.2.
- **Machine roofline** (engineering yardstick): `i64_roofline` measures the
  running host's achievable wrapping i64 multiply-add throughput, so i64
  GEMM is also judged as **% of machine ceiling** — the BLAS ratio alone
  mixes kernel quality with ISA physics (no 64-bit vector multiply below
  AVX-512DQ). See the Roofline section.

### Provenance

Every table names the host that produced it; all faces of one table come
from one host and one session. Run-to-run noise on shared cloud hosts is
real (±10–20% observed); treat small deltas accordingly. Occasional wider
spreads between the Rust and Lua faces of the same op are contention — both
faces call the same kernel.

## Results

**Host:** 4 vCPU Intel Xeon @ 2.10 GHz (shared cloud container), rustc
1.94.1 at default codegen, NumPy 2.4.6 (bundled OpenBLAS), 2026-08-03.

<!-- PERF_TABLES_START -->

### Summary — Lua face vs NumPy

One row per op, user point of view: the Lua face (the product) against NumPy
(the bar). Absolute times at the largest measured n; the ratio column is the
**min–max of Lua/NumPy across all measured n** (below 1.00x = MatLua faster;
a wide range means size-dependent). Derived from the appendix cells — nothing
new is measured for this table. Rust-face and per-n detail: appendix below.

#### f64

| op | largest n | NumPy (ms) | MatLua Lua (ms) | Lua/NumPy, all n |
| --- | ---: | ---: | ---: | ---: |
| arange | 4096 | 0.003118 | 0.006333 | 0.34x–2.03x |
| cholesky | 4096 | 879.691 | 550.450 | 0.60x–1.07x |
| copy | 4096 | 78.985 | 592.903 | 0.85x–7.51x |
| dot | 4096 | 0.002211 | 0.001695 | 0.45x–0.77x |
| elem_add | 4096 | 72.766 | 625.955 | 0.56x–8.60x |
| elem_add_scalar | 4096 | 65.429 | 25.504 | 0.39x–1.26x |
| elem_div | 4096 | 76.558 | 34.133 | 0.45x–1.19x |
| elem_mul | 4096 | 74.262 | 32.698 | 0.44x–2.38x |
| elem_sub | 4096 | 72.780 | 33.322 | 0.46x–1.66x |
| eye | 4096 | 6.1249 | 85.996 | 0.30x–14.04x |
| fill | 4096 | 22.499 | 20.015 | 0.55x–0.90x |
| full | 4096 | 59.841 | 19.135 | 0.32x–0.99x |
| matmul | 4096 | 771.017 | 803.897 | 0.67x–1.11x |
| max | 4096 | 11.202 | 22.518 | 1.24x–3.44x |
| mean | 4096 | 16.604 | 13.670 | 0.24x–0.82x |
| min | 4096 | 11.511 | 18.771 | 1.14x–3.31x |
| norm | 4096 | 3.3668 | 12.420 | 0.50x–3.69x |
| ones | 4096 | 67.075 | 430.262 | 0.39x–6.41x |
| qr | 4096 | 4834.037 | 2231.960 | 0.20x–3.31x |
| reshape | 4096 | 0.000815 | 0.040975 | 0.85x–50.28x |
| solve | 4096 | 559.387 | 1654.838 | 1.37x–5.77x |
| sum | 4096 | 18.181 | 11.547 | 0.44x–0.71x |
| svd | 4096 | 12578.454 | 17989.051 | 1.43x–2.65x |
| transpose | 4096 | 272.905 | 79.744 | 0.24x–1.45x |
| zeros | 4096 | 0.013027 | 0.005207 | 0.16x–61.76x |

#### i64

`matmul` / `matmul_at` / `matmul_bt` reference is NumPy **f64 BLAS** on
integer-valued data (see Yardsticks); MatLua times are exact wrapping i64.

| op | largest n | NumPy (ms) | MatLua Lua (ms) | Lua/NumPy, all n |
| --- | ---: | ---: | ---: | ---: |
| arange | 4096 | 0.002385 | 0.006321 | 0.63x–2.65x |
| copy | 4096 | 75.227 | 90.488 | 0.85x–1.20x |
| dot | 4096 | 0.003856 | 0.001784 | 0.23x–0.46x |
| elem_add | 4096 | 72.404 | 77.408 | 1.07x–2.07x |
| elem_div | 4096 | 147.753 | 112.702 | 0.56x–0.76x |
| elem_mul | 4096 | 74.510 | 78.822 | 1.06x–1.96x |
| elem_sub | 4096 | 69.999 | 78.394 | 1.12x–2.10x |
| eye | 4096 | 6.5028 | 6.9000 | 0.21x–1.06x |
| fill | 4096 | 22.943 | 20.637 | 0.81x–1.02x |
| full | 4096 | 55.969 | 63.400 | 0.43x–1.13x |
| isin | 4096 | 74.665 | 37.025 | 0.28x–0.97x |
| matmul | 4096 | 801.538 | 3945.952 | 4.92x–8.91x |
| matmul_at | 4096 | 753.077 | 3750.200 | 4.98x–10.26x |
| matmul_bt | 4096 | 778.012 | 3777.489 | 4.86x–7.28x |
| max | 4096 | 7.0320 | 21.779 | 0.71x–3.10x |
| min | 4096 | 9.0915 | 24.635 | 0.96x–2.95x |
| ones | 4096 | 59.139 | 57.752 | 0.44x–1.00x |
| reshape | 4096 | 0.000784 | 0.000640 | 0.82x–1.18x |
| sum | 4096 | 11.719 | 14.664 | 0.49x–1.25x |
| transpose | 4096 | 266.504 | 103.264 | 0.24x–2.03x |
| unique | 4096 | 0.447670 | 0.008696 | 0.02x–0.05x |
| zeros | 4096 | 0.011488 | 0.008146 | 0.69x–1.18x |

#### i64→f64 promote-out

| op | largest n | NumPy (ms) | MatLua Lua (ms) | Lua/NumPy, all n |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 4096 | 1477.145 | 1035.662 | 0.70x–1.03x |
| mean | 4096 | 21.005 | 13.709 | 0.12x–0.65x |
| median | 4096 | 122.721 | 84.302 | 0.29x–0.83x |
| norm | 4096 | 7.2317 | 18.702 | 0.76x–5.34x |
| qr | 4096 | 5100.289 | 3943.379 | 0.51x–2.76x |
| quantile | 4096 | 136.378 | 86.623 | 0.08x–0.64x |
| solve | 4096 | 1125.655 | 949.086 | 0.84x–6.79x |
| std | 4096 | 124.349 | 49.597 | 0.18x–0.42x |

### Appendix — full three-face tables

<details>
<summary>Table A — f64 absolute (ms)</summary>

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000453 | 0.000119 | 0.000226 |
| arange | 256 | 0.000878 | 0.000196 | 0.000299 |
| arange | 1024 | 0.001004 | 0.000406 | 0.000563 |
| arange | 4096 | 0.003118 | 0.005278 | 0.006333 |
| cholesky | 64 | 0.024133 | 0.014285 | 0.014487 |
| cholesky | 256 | 0.993523 | 1.2606 | 1.0582 |
| cholesky | 1024 | 33.599 | 20.129 | 21.625 |
| cholesky | 4096 | 879.691 | 612.657 | 550.450 |
| copy | 64 | 0.001156 | 0.000938 | 0.000988 |
| copy | 256 | 0.012354 | 0.013345 | 0.014725 |
| copy | 1024 | 0.726600 | 0.739543 | 0.719048 |
| copy | 4096 | 78.985 | 23.104 | 592.903 |
| dot | 64 | 0.000608 | 0.000048 | 0.000274 |
| dot | 256 | 0.000668 | 0.000110 | 0.000332 |
| dot | 1024 | 0.001211 | 0.000323 | 0.000614 |
| dot | 4096 | 0.002211 | 0.001509 | 0.001695 |
| elem_add | 64 | 0.002524 | 0.001360 | 0.001408 |
| elem_add | 256 | 0.018188 | 0.035639 | 0.034156 |
| elem_add | 1024 | 1.3684 | 1.0281 | 1.0821 |
| elem_add | 4096 | 72.766 | 34.189 | 625.955 |
| elem_add_scalar | 64 | 0.001630 | 0.000998 | 0.001038 |
| elem_add_scalar | 256 | 0.013831 | 0.015777 | 0.017412 |
| elem_add_scalar | 1024 | 0.739524 | 0.741208 | 0.727840 |
| elem_add_scalar | 4096 | 65.429 | 25.633 | 25.504 |
| elem_div | 64 | 0.002986 | 0.002902 | 0.002715 |
| elem_div | 256 | 0.041424 | 0.040698 | 0.049136 |
| elem_div | 1024 | 1.2981 | 0.995151 | 0.990335 |
| elem_div | 4096 | 76.558 | 34.296 | 34.133 |
| elem_mul | 64 | 0.002535 | 0.001357 | 0.001277 |
| elem_mul | 256 | 0.018315 | 0.028278 | 0.043533 |
| elem_mul | 1024 | 1.3254 | 1.0247 | 1.0254 |
| elem_mul | 4096 | 74.262 | 34.523 | 32.698 |
| elem_sub | 64 | 0.002527 | 0.001355 | 0.001321 |
| elem_sub | 256 | 0.018373 | 0.027836 | 0.030574 |
| elem_sub | 1024 | 1.3796 | 1.0531 | 1.0286 |
| elem_sub | 4096 | 72.780 | 34.476 | 33.322 |
| eye | 64 | 0.001977 | 0.000286 | 0.000589 |
| eye | 256 | 0.013386 | 0.010300 | 0.010215 |
| eye | 1024 | 0.448483 | 0.475302 | 0.449369 |
| eye | 4096 | 6.1249 | 10.140 | 85.996 |
| fill | 64 | 0.001226 | 0.000347 | 0.000674 |
| fill | 256 | 0.015380 | 0.009931 | 0.009877 |
| fill | 1024 | 0.545401 | 0.508196 | 0.491537 |
| fill | 4096 | 22.499 | 20.483 | 20.015 |
| full | 64 | 0.001875 | 0.000706 | 0.000773 |
| full | 256 | 0.017018 | 0.009944 | 0.010137 |
| full | 1024 | 0.507796 | 0.470368 | 0.500686 |
| full | 4096 | 59.841 | 18.838 | 19.135 |
| matmul | 64 | 0.009695 | 0.008698 | 0.010767 |
| matmul | 256 | 0.490592 | 0.321777 | 0.330195 |
| matmul | 1024 | 13.167 | 12.710 | 12.889 |
| matmul | 4096 | 771.017 | 860.782 | 803.897 |
| max | 64 | 0.001443 | 0.001623 | 0.001790 |
| max | 256 | 0.007514 | 0.025684 | 0.025875 |
| max | 1024 | 0.407043 | 0.565983 | 0.649142 |
| max | 4096 | 11.202 | 21.797 | 22.518 |
| mean | 64 | 0.003146 | 0.000656 | 0.000761 |
| mean | 256 | 0.015760 | 0.009981 | 0.010120 |
| mean | 1024 | 0.545041 | 0.313272 | 0.401684 |
| mean | 4096 | 16.604 | 12.781 | 13.670 |
| min | 64 | 0.001494 | 0.001687 | 0.001703 |
| min | 256 | 0.007550 | 0.027498 | 0.024991 |
| min | 1024 | 0.419681 | 0.551123 | 0.637280 |
| min | 4096 | 11.511 | 21.722 | 18.771 |
| norm | 64 | 0.001612 | 0.000659 | 0.000801 |
| norm | 256 | 0.007490 | 0.009984 | 0.010135 |
| norm | 1024 | 0.133549 | 0.360083 | 0.438664 |
| norm | 4096 | 3.3668 | 13.383 | 12.420 |
| ones | 64 | 0.001957 | 0.000695 | 0.000767 |
| ones | 256 | 0.017863 | 0.010459 | 0.014142 |
| ones | 1024 | 0.508872 | 0.560301 | 0.625051 |
| ones | 4096 | 67.075 | 918.484 | 430.262 |
| qr | 64 | 0.120766 | 0.794518 | 0.399939 |
| qr | 256 | 5.6738 | 6.2122 | 3.8545 |
| qr | 1024 | 239.196 | 56.784 | 48.610 |
| qr | 4096 | 4834.037 | 3618.492 | 2231.960 |
| reshape | 64 | 0.000206 | 0.000071 | 0.000227 |
| reshape | 256 | 0.000225 | 0.000071 | 0.000229 |
| reshape | 1024 | 0.000307 | 0.000072 | 0.000262 |
| reshape | 4096 | 0.000815 | 0.000382 | 0.040975 |
| solve | 64 | 0.039865 | 0.139488 | 0.202083 |
| solve | 256 | 0.663548 | 2.6102 | 3.8283 |
| solve | 1024 | 23.706 | 38.998 | 32.398 |
| solve | 4096 | 559.387 | 2129.724 | 1654.838 |
| sum | 64 | 0.001720 | 0.000652 | 0.000759 |
| sum | 256 | 0.014279 | 0.009978 | 0.010098 |
| sum | 1024 | 0.607097 | 0.344989 | 0.404038 |
| sum | 4096 | 18.181 | 13.034 | 11.547 |
| svd | 64 | 0.352279 | 0.786292 | 0.933470 |
| svd | 256 | 10.312 | 16.045 | 21.275 |
| svd | 1024 | 359.192 | 503.654 | 747.168 |
| svd | 4096 | 12578.454 | 14803.643 | 17989.051 |
| transpose | 64 | 0.002001 | 0.002743 | 0.002899 |
| transpose | 256 | 0.069276 | 0.049969 | 0.048686 |
| transpose | 1024 | 11.253 | 2.2603 | 2.6646 |
| transpose | 4096 | 272.905 | 85.038 | 79.744 |
| zeros | 64 | 0.000619 | 0.000290 | 0.000602 |
| zeros | 256 | 0.011330 | 0.002138 | 0.699738 |
| zeros | 1024 | 0.408301 | 0.003695 | 0.065060 |
| zeros | 4096 | 0.013027 | 0.059163 | 0.005207 |

</details>

<details>
<summary>Table B — f64 relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.26x | 0.50x | 1.90x |
| arange | 256 | 1.00x | 0.22x | 0.34x | 1.53x |
| arange | 1024 | 1.00x | 0.40x | 0.56x | 1.39x |
| arange | 4096 | 1.00x | 1.69x | 2.03x | 1.20x |
| cholesky | 64 | 1.00x | 0.59x | 0.60x | 1.01x |
| cholesky | 256 | 1.00x | 1.27x | 1.07x | 0.84x |
| cholesky | 1024 | 1.00x | 0.60x | 0.64x | 1.07x |
| cholesky | 4096 | 1.00x | 0.70x | 0.63x | 0.90x |
| copy | 64 | 1.00x | 0.81x | 0.85x | 1.05x |
| copy | 256 | 1.00x | 1.08x | 1.19x | 1.10x |
| copy | 1024 | 1.00x | 1.02x | 0.99x | 0.97x |
| copy | 4096 | 1.00x | 0.29x | 7.51x | 25.66x |
| dot | 64 | 1.00x | 0.08x | 0.45x | 5.71x |
| dot | 256 | 1.00x | 0.16x | 0.50x | 3.02x |
| dot | 1024 | 1.00x | 0.27x | 0.51x | 1.90x |
| dot | 4096 | 1.00x | 0.68x | 0.77x | 1.12x |
| elem_add | 64 | 1.00x | 0.54x | 0.56x | 1.04x |
| elem_add | 256 | 1.00x | 1.96x | 1.88x | 0.96x |
| elem_add | 1024 | 1.00x | 0.75x | 0.79x | 1.05x |
| elem_add | 4096 | 1.00x | 0.47x | 8.60x | 18.31x |
| elem_add_scalar | 64 | 1.00x | 0.61x | 0.64x | 1.04x |
| elem_add_scalar | 256 | 1.00x | 1.14x | 1.26x | 1.10x |
| elem_add_scalar | 1024 | 1.00x | 1.00x | 0.98x | 0.98x |
| elem_add_scalar | 4096 | 1.00x | 0.39x | 0.39x | 0.99x |
| elem_div | 64 | 1.00x | 0.97x | 0.91x | 0.94x |
| elem_div | 256 | 1.00x | 0.98x | 1.19x | 1.21x |
| elem_div | 1024 | 1.00x | 0.77x | 0.76x | 1.00x |
| elem_div | 4096 | 1.00x | 0.45x | 0.45x | 1.00x |
| elem_mul | 64 | 1.00x | 0.54x | 0.50x | 0.94x |
| elem_mul | 256 | 1.00x | 1.54x | 2.38x | 1.54x |
| elem_mul | 1024 | 1.00x | 0.77x | 0.77x | 1.00x |
| elem_mul | 4096 | 1.00x | 0.46x | 0.44x | 0.95x |
| elem_sub | 64 | 1.00x | 0.54x | 0.52x | 0.97x |
| elem_sub | 256 | 1.00x | 1.52x | 1.66x | 1.10x |
| elem_sub | 1024 | 1.00x | 0.76x | 0.75x | 0.98x |
| elem_sub | 4096 | 1.00x | 0.47x | 0.46x | 0.97x |
| eye | 64 | 1.00x | 0.14x | 0.30x | 2.06x |
| eye | 256 | 1.00x | 0.77x | 0.76x | 0.99x |
| eye | 1024 | 1.00x | 1.06x | 1.00x | 0.95x |
| eye | 4096 | 1.00x | 1.66x | 14.04x | 8.48x |
| fill | 64 | 1.00x | 0.28x | 0.55x | 1.94x |
| fill | 256 | 1.00x | 0.65x | 0.64x | 0.99x |
| fill | 1024 | 1.00x | 0.93x | 0.90x | 0.97x |
| fill | 4096 | 1.00x | 0.91x | 0.89x | 0.98x |
| full | 64 | 1.00x | 0.38x | 0.41x | 1.09x |
| full | 256 | 1.00x | 0.58x | 0.60x | 1.02x |
| full | 1024 | 1.00x | 0.93x | 0.99x | 1.06x |
| full | 4096 | 1.00x | 0.31x | 0.32x | 1.02x |
| matmul | 64 | 1.00x | 0.90x | 1.11x | 1.24x |
| matmul | 256 | 1.00x | 0.66x | 0.67x | 1.03x |
| matmul | 1024 | 1.00x | 0.97x | 0.98x | 1.01x |
| matmul | 4096 | 1.00x | 1.12x | 1.04x | 0.93x |
| max | 64 | 1.00x | 1.12x | 1.24x | 1.10x |
| max | 256 | 1.00x | 3.42x | 3.44x | 1.01x |
| max | 1024 | 1.00x | 1.39x | 1.59x | 1.15x |
| max | 4096 | 1.00x | 1.95x | 2.01x | 1.03x |
| mean | 64 | 1.00x | 0.21x | 0.24x | 1.16x |
| mean | 256 | 1.00x | 0.63x | 0.64x | 1.01x |
| mean | 1024 | 1.00x | 0.57x | 0.74x | 1.28x |
| mean | 4096 | 1.00x | 0.77x | 0.82x | 1.07x |
| min | 64 | 1.00x | 1.13x | 1.14x | 1.01x |
| min | 256 | 1.00x | 3.64x | 3.31x | 0.91x |
| min | 1024 | 1.00x | 1.31x | 1.52x | 1.16x |
| min | 4096 | 1.00x | 1.89x | 1.63x | 0.86x |
| norm | 64 | 1.00x | 0.41x | 0.50x | 1.22x |
| norm | 256 | 1.00x | 1.33x | 1.35x | 1.02x |
| norm | 1024 | 1.00x | 2.70x | 3.28x | 1.22x |
| norm | 4096 | 1.00x | 3.97x | 3.69x | 0.93x |
| ones | 64 | 1.00x | 0.36x | 0.39x | 1.10x |
| ones | 256 | 1.00x | 0.59x | 0.79x | 1.35x |
| ones | 1024 | 1.00x | 1.10x | 1.23x | 1.12x |
| ones | 4096 | 1.00x | 13.69x | 6.41x | 0.47x |
| qr | 64 | 1.00x | 6.58x | 3.31x | 0.50x |
| qr | 256 | 1.00x | 1.09x | 0.68x | 0.62x |
| qr | 1024 | 1.00x | 0.24x | 0.20x | 0.86x |
| qr | 4096 | 1.00x | 0.75x | 0.46x | 0.62x |
| reshape | 64 | 1.00x | 0.34x | 1.10x | 3.20x |
| reshape | 256 | 1.00x | 0.32x | 1.02x | 3.23x |
| reshape | 1024 | 1.00x | 0.23x | 0.85x | 3.64x |
| reshape | 4096 | 1.00x | 0.47x | 50.28x | 107.26x |
| solve | 64 | 1.00x | 3.50x | 5.07x | 1.45x |
| solve | 256 | 1.00x | 3.93x | 5.77x | 1.47x |
| solve | 1024 | 1.00x | 1.65x | 1.37x | 0.83x |
| solve | 4096 | 1.00x | 3.81x | 2.96x | 0.78x |
| sum | 64 | 1.00x | 0.38x | 0.44x | 1.16x |
| sum | 256 | 1.00x | 0.70x | 0.71x | 1.01x |
| sum | 1024 | 1.00x | 0.57x | 0.67x | 1.17x |
| sum | 4096 | 1.00x | 0.72x | 0.64x | 0.89x |
| svd | 64 | 1.00x | 2.23x | 2.65x | 1.19x |
| svd | 256 | 1.00x | 1.56x | 2.06x | 1.33x |
| svd | 1024 | 1.00x | 1.40x | 2.08x | 1.48x |
| svd | 4096 | 1.00x | 1.18x | 1.43x | 1.22x |
| transpose | 64 | 1.00x | 1.37x | 1.45x | 1.06x |
| transpose | 256 | 1.00x | 0.72x | 0.70x | 0.97x |
| transpose | 1024 | 1.00x | 0.20x | 0.24x | 1.18x |
| transpose | 4096 | 1.00x | 0.31x | 0.29x | 0.94x |
| zeros | 64 | 1.00x | 0.47x | 0.97x | 2.08x |
| zeros | 256 | 1.00x | 0.19x | 61.76x | 327.29x |
| zeros | 1024 | 1.00x | 0.01x | 0.16x | 17.61x |
| zeros | 4096 | 1.00x | 4.54x | 0.40x | 0.09x |

</details>

<details>
<summary>Table C — i64 absolute (ms) — matmul* NumPy column is f64 BLAS on integer-valued data</summary>

| op | n | NumPy int64 (ms) | MatLua Rust i64 (ms) | MatLua Lua i64 (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000570 | 0.000119 | 0.000372 |
| arange | 256 | 0.000527 | 0.000187 | 0.000478 |
| arange | 1024 | 0.000827 | 0.000400 | 0.000521 |
| arange | 4096 | 0.002385 | 0.006053 | 0.006321 |
| copy | 64 | 0.001270 | 0.000870 | 0.001083 |
| copy | 256 | 0.013367 | 0.012352 | 0.014308 |
| copy | 1024 | 0.721960 | 0.718143 | 0.780308 |
| copy | 4096 | 75.227 | 80.413 | 90.488 |
| dot | 64 | 0.000836 | 0.000051 | 0.000232 |
| dot | 256 | 0.001308 | 0.000119 | 0.000300 |
| dot | 1024 | 0.001771 | 0.000377 | 0.000629 |
| dot | 4096 | 0.003856 | 0.001445 | 0.001784 |
| elem_add | 64 | 0.001651 | 0.002159 | 0.002329 |
| elem_add | 256 | 0.027597 | 0.034391 | 0.056997 |
| elem_add | 1024 | 1.0463 | 1.1599 | 1.1950 |
| elem_add | 4096 | 72.404 | 83.218 | 77.408 |
| elem_div | 64 | 0.015148 | 0.007576 | 0.008457 |
| elem_div | 256 | 0.209624 | 0.120139 | 0.143326 |
| elem_div | 1024 | 3.5169 | 2.1945 | 2.2204 |
| elem_div | 4096 | 147.753 | 132.858 | 112.702 |
| elem_mul | 64 | 0.002120 | 0.002150 | 0.002707 |
| elem_mul | 256 | 0.028276 | 0.034921 | 0.055458 |
| elem_mul | 1024 | 1.0620 | 1.0970 | 1.1325 |
| elem_mul | 4096 | 74.510 | 85.445 | 78.822 |
| elem_sub | 64 | 0.001643 | 0.002148 | 0.003405 |
| elem_sub | 256 | 0.027511 | 0.034135 | 0.057655 |
| elem_sub | 1024 | 1.0304 | 1.1508 | 1.1959 |
| elem_sub | 4096 | 69.999 | 86.751 | 78.394 |
| eye | 64 | 0.002838 | 0.000283 | 0.000591 |
| eye | 256 | 0.013268 | 0.010246 | 0.011505 |
| eye | 1024 | 0.455391 | 0.420030 | 0.447287 |
| eye | 4096 | 6.5028 | 6.6006 | 6.9000 |
| fill | 64 | 0.001205 | 0.000345 | 0.000978 |
| fill | 256 | 0.015973 | 0.009872 | 0.013061 |
| fill | 1024 | 0.517837 | 0.500680 | 0.529832 |
| fill | 4096 | 22.943 | 20.262 | 20.637 |
| full | 64 | 0.002585 | 0.000700 | 0.001124 |
| full | 256 | 0.016129 | 0.009963 | 0.013548 |
| full | 1024 | 0.543909 | 0.518205 | 0.530998 |
| full | 4096 | 55.969 | 57.846 | 63.400 |
| isin | 64 | 0.017731 | 0.002755 | 0.005025 |
| isin | 256 | 0.074356 | 0.046940 | 0.071844 |
| isin | 1024 | 1.6055 | 0.902582 | 1.5568 |
| isin | 4096 | 74.665 | 83.488 | 37.025 |
| matmul | 64 | 0.009671 | 0.070614 | 0.086143 |
| matmul | 256 | 0.443698 | 2.9071 | 2.3562 |
| matmul | 1024 | 11.069 | 69.045 | 66.592 |
| matmul | 4096 | 801.538 | 4200.647 | 3945.952 |
| matmul_at | 64 | 0.009697 | 0.070860 | 0.099481 |
| matmul_at | 256 | 0.417440 | 2.7000 | 2.4419 |
| matmul_at | 1024 | 10.800 | 63.447 | 60.094 |
| matmul_at | 4096 | 753.077 | 4053.275 | 3750.200 |
| matmul_bt | 64 | 0.013894 | 0.068405 | 0.101207 |
| matmul_bt | 256 | 0.416893 | 2.5071 | 2.3166 |
| matmul_bt | 1024 | 10.220 | 57.612 | 66.595 |
| matmul_bt | 4096 | 778.012 | 4031.814 | 3777.489 |
| max | 64 | 0.001474 | 0.000834 | 0.001044 |
| max | 256 | 0.006895 | 0.013502 | 0.017098 |
| max | 1024 | 0.379846 | 0.585552 | 0.587565 |
| max | 4096 | 7.0320 | 23.625 | 21.779 |
| min | 64 | 0.001471 | 0.000850 | 0.001409 |
| min | 256 | 0.006953 | 0.013507 | 0.020505 |
| min | 1024 | 0.397823 | 0.602694 | 0.576086 |
| min | 4096 | 9.0915 | 22.951 | 24.635 |
| ones | 64 | 0.002521 | 0.000703 | 0.001115 |
| ones | 256 | 0.016289 | 0.012141 | 0.010136 |
| ones | 1024 | 0.539005 | 0.515250 | 0.538040 |
| ones | 4096 | 59.139 | 58.049 | 57.752 |
| reshape | 64 | 0.000225 | 0.000072 | 0.000266 |
| reshape | 256 | 0.000262 | 0.000072 | 0.000300 |
| reshape | 1024 | 0.000310 | 0.000079 | 0.000263 |
| reshape | 4096 | 0.000784 | 0.000148 | 0.000640 |
| sum | 64 | 0.001578 | 0.000333 | 0.000771 |
| sum | 256 | 0.008674 | 0.006989 | 0.007147 |
| sum | 1024 | 0.378861 | 0.388984 | 0.405126 |
| sum | 4096 | 11.719 | 15.807 | 14.664 |
| transpose | 64 | 0.002193 | 0.002750 | 0.004462 |
| transpose | 256 | 0.072306 | 0.047284 | 0.069957 |
| transpose | 1024 | 10.818 | 2.5764 | 2.5566 |
| transpose | 4096 | 266.504 | 113.559 | 103.264 |
| unique | 64 | 0.006023 | 0.000151 | 0.000316 |
| unique | 256 | 0.019520 | 0.000310 | 0.000477 |
| unique | 1024 | 0.082348 | 0.000827 | 0.001362 |
| unique | 4096 | 0.447670 | 0.003574 | 0.008696 |
| zeros | 64 | 0.000966 | 0.000266 | 0.000662 |
| zeros | 256 | 0.010285 | 0.010619 | 0.012152 |
| zeros | 1024 | 0.410120 | 0.439133 | 0.434347 |
| zeros | 4096 | 0.011488 | 0.007466 | 0.008146 |

</details>

<details>
<summary>Table D — i64 relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.21x | 0.65x | 3.13x |
| arange | 256 | 1.00x | 0.35x | 0.91x | 2.56x |
| arange | 1024 | 1.00x | 0.48x | 0.63x | 1.30x |
| arange | 4096 | 1.00x | 2.54x | 2.65x | 1.04x |
| copy | 64 | 1.00x | 0.69x | 0.85x | 1.24x |
| copy | 256 | 1.00x | 0.92x | 1.07x | 1.16x |
| copy | 1024 | 1.00x | 0.99x | 1.08x | 1.09x |
| copy | 4096 | 1.00x | 1.07x | 1.20x | 1.13x |
| dot | 64 | 1.00x | 0.06x | 0.28x | 4.55x |
| dot | 256 | 1.00x | 0.09x | 0.23x | 2.52x |
| dot | 1024 | 1.00x | 0.21x | 0.36x | 1.67x |
| dot | 4096 | 1.00x | 0.37x | 0.46x | 1.23x |
| elem_add | 64 | 1.00x | 1.31x | 1.41x | 1.08x |
| elem_add | 256 | 1.00x | 1.25x | 2.07x | 1.66x |
| elem_add | 1024 | 1.00x | 1.11x | 1.14x | 1.03x |
| elem_add | 4096 | 1.00x | 1.15x | 1.07x | 0.93x |
| elem_div | 64 | 1.00x | 0.50x | 0.56x | 1.12x |
| elem_div | 256 | 1.00x | 0.57x | 0.68x | 1.19x |
| elem_div | 1024 | 1.00x | 0.62x | 0.63x | 1.01x |
| elem_div | 4096 | 1.00x | 0.90x | 0.76x | 0.85x |
| elem_mul | 64 | 1.00x | 1.01x | 1.28x | 1.26x |
| elem_mul | 256 | 1.00x | 1.24x | 1.96x | 1.59x |
| elem_mul | 1024 | 1.00x | 1.03x | 1.07x | 1.03x |
| elem_mul | 4096 | 1.00x | 1.15x | 1.06x | 0.92x |
| elem_sub | 64 | 1.00x | 1.31x | 2.07x | 1.59x |
| elem_sub | 256 | 1.00x | 1.24x | 2.10x | 1.69x |
| elem_sub | 1024 | 1.00x | 1.12x | 1.16x | 1.04x |
| elem_sub | 4096 | 1.00x | 1.24x | 1.12x | 0.90x |
| eye | 64 | 1.00x | 0.10x | 0.21x | 2.09x |
| eye | 256 | 1.00x | 0.77x | 0.87x | 1.12x |
| eye | 1024 | 1.00x | 0.92x | 0.98x | 1.06x |
| eye | 4096 | 1.00x | 1.02x | 1.06x | 1.05x |
| fill | 64 | 1.00x | 0.29x | 0.81x | 2.83x |
| fill | 256 | 1.00x | 0.62x | 0.82x | 1.32x |
| fill | 1024 | 1.00x | 0.97x | 1.02x | 1.06x |
| fill | 4096 | 1.00x | 0.88x | 0.90x | 1.02x |
| full | 64 | 1.00x | 0.27x | 0.43x | 1.61x |
| full | 256 | 1.00x | 0.62x | 0.84x | 1.36x |
| full | 1024 | 1.00x | 0.95x | 0.98x | 1.02x |
| full | 4096 | 1.00x | 1.03x | 1.13x | 1.10x |
| isin | 64 | 1.00x | 0.16x | 0.28x | 1.82x |
| isin | 256 | 1.00x | 0.63x | 0.97x | 1.53x |
| isin | 1024 | 1.00x | 0.56x | 0.97x | 1.72x |
| isin | 4096 | 1.00x | 1.12x | 0.50x | 0.44x |
| matmul | 64 | 1.00x | 7.30x | 8.91x | 1.22x |
| matmul | 256 | 1.00x | 6.55x | 5.31x | 0.81x |
| matmul | 1024 | 1.00x | 6.24x | 6.02x | 0.96x |
| matmul | 4096 | 1.00x | 5.24x | 4.92x | 0.94x |
| matmul_at | 64 | 1.00x | 7.31x | 10.26x | 1.40x |
| matmul_at | 256 | 1.00x | 6.47x | 5.85x | 0.90x |
| matmul_at | 1024 | 1.00x | 5.87x | 5.56x | 0.95x |
| matmul_at | 4096 | 1.00x | 5.38x | 4.98x | 0.93x |
| matmul_bt | 64 | 1.00x | 4.92x | 7.28x | 1.48x |
| matmul_bt | 256 | 1.00x | 6.01x | 5.56x | 0.92x |
| matmul_bt | 1024 | 1.00x | 5.64x | 6.52x | 1.16x |
| matmul_bt | 4096 | 1.00x | 5.18x | 4.86x | 0.94x |
| max | 64 | 1.00x | 0.57x | 0.71x | 1.25x |
| max | 256 | 1.00x | 1.96x | 2.48x | 1.27x |
| max | 1024 | 1.00x | 1.54x | 1.55x | 1.00x |
| max | 4096 | 1.00x | 3.36x | 3.10x | 0.92x |
| min | 64 | 1.00x | 0.58x | 0.96x | 1.66x |
| min | 256 | 1.00x | 1.94x | 2.95x | 1.52x |
| min | 1024 | 1.00x | 1.51x | 1.45x | 0.96x |
| min | 4096 | 1.00x | 2.52x | 2.71x | 1.07x |
| ones | 64 | 1.00x | 0.28x | 0.44x | 1.59x |
| ones | 256 | 1.00x | 0.75x | 0.62x | 0.83x |
| ones | 1024 | 1.00x | 0.96x | 1.00x | 1.04x |
| ones | 4096 | 1.00x | 0.98x | 0.98x | 0.99x |
| reshape | 64 | 1.00x | 0.32x | 1.18x | 3.69x |
| reshape | 256 | 1.00x | 0.27x | 1.15x | 4.17x |
| reshape | 1024 | 1.00x | 0.25x | 0.85x | 3.33x |
| reshape | 4096 | 1.00x | 0.19x | 0.82x | 4.32x |
| sum | 64 | 1.00x | 0.21x | 0.49x | 2.32x |
| sum | 256 | 1.00x | 0.81x | 0.82x | 1.02x |
| sum | 1024 | 1.00x | 1.03x | 1.07x | 1.04x |
| sum | 4096 | 1.00x | 1.35x | 1.25x | 0.93x |
| transpose | 64 | 1.00x | 1.25x | 2.03x | 1.62x |
| transpose | 256 | 1.00x | 0.65x | 0.97x | 1.48x |
| transpose | 1024 | 1.00x | 0.24x | 0.24x | 0.99x |
| transpose | 4096 | 1.00x | 0.43x | 0.39x | 0.91x |
| unique | 64 | 1.00x | 0.03x | 0.05x | 2.09x |
| unique | 256 | 1.00x | 0.02x | 0.02x | 1.54x |
| unique | 1024 | 1.00x | 0.01x | 0.02x | 1.65x |
| unique | 4096 | 1.00x | 0.01x | 0.02x | 2.43x |
| zeros | 64 | 1.00x | 0.28x | 0.69x | 2.49x |
| zeros | 256 | 1.00x | 1.03x | 1.18x | 1.14x |
| zeros | 1024 | 1.00x | 1.07x | 1.06x | 0.99x |
| zeros | 4096 | 1.00x | 0.65x | 0.71x | 1.09x |

</details>

<details>
<summary>Table E — i64→f64 promote-out absolute (ms)</summary>

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 64 | 0.020093 | 0.015209 | 0.016742 |
| cholesky | 256 | 1.0281 | 1.6664 | 1.0629 |
| cholesky | 1024 | 36.664 | 32.434 | 26.431 |
| cholesky | 4096 | 1477.145 | 631.274 | 1035.662 |
| mean | 64 | 0.004937 | 0.000457 | 0.000588 |
| mean | 256 | 0.042473 | 0.006981 | 0.007246 |
| mean | 1024 | 0.910436 | 0.522004 | 0.379455 |
| mean | 4096 | 21.005 | 13.558 | 13.709 |
| median | 64 | 0.011139 | 0.003696 | 0.003220 |
| median | 256 | 0.068510 | 0.150412 | 0.051408 |
| median | 1024 | 1.7946 | 2.3273 | 1.4821 |
| median | 4096 | 122.721 | 87.931 | 84.302 |
| norm | 64 | 0.001851 | 0.001279 | 0.001413 |
| norm | 256 | 0.007510 | 0.020026 | 0.022159 |
| norm | 1024 | 0.106887 | 0.636760 | 0.570743 |
| norm | 4096 | 7.2317 | 18.392 | 18.702 |
| qr | 64 | 0.131271 | 0.753971 | 0.361778 |
| qr | 256 | 5.4732 | 7.7588 | 3.7979 |
| qr | 1024 | 150.474 | 92.287 | 77.013 |
| qr | 4096 | 5100.289 | 4945.552 | 3943.379 |
| quantile | 64 | 0.038493 | 0.003036 | 0.003228 |
| quantile | 256 | 0.156399 | 0.160841 | 0.051541 |
| quantile | 1024 | 3.1327 | 1.5396 | 1.3706 |
| quantile | 4096 | 136.378 | 233.632 | 86.623 |
| solve | 64 | 0.033843 | 0.115233 | 0.229902 |
| solve | 256 | 0.774210 | 2.9946 | 2.2366 |
| solve | 1024 | 25.573 | 41.622 | 37.617 |
| solve | 4096 | 1125.655 | 2161.786 | 949.086 |
| std | 64 | 0.016673 | 0.002833 | 0.003074 |
| std | 256 | 0.125670 | 0.047077 | 0.052666 |
| std | 1024 | 3.1455 | 1.2512 | 1.1930 |
| std | 4096 | 124.349 | 51.858 | 49.597 |

</details>

<details>
<summary>Table F — i64→f64 promote-out relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky | 64 | 1.00x | 0.76x | 0.83x | 1.10x |
| cholesky | 256 | 1.00x | 1.62x | 1.03x | 0.64x |
| cholesky | 1024 | 1.00x | 0.88x | 0.72x | 0.81x |
| cholesky | 4096 | 1.00x | 0.43x | 0.70x | 1.64x |
| mean | 64 | 1.00x | 0.09x | 0.12x | 1.29x |
| mean | 256 | 1.00x | 0.16x | 0.17x | 1.04x |
| mean | 1024 | 1.00x | 0.57x | 0.42x | 0.73x |
| mean | 4096 | 1.00x | 0.65x | 0.65x | 1.01x |
| median | 64 | 1.00x | 0.33x | 0.29x | 0.87x |
| median | 256 | 1.00x | 2.20x | 0.75x | 0.34x |
| median | 1024 | 1.00x | 1.30x | 0.83x | 0.64x |
| median | 4096 | 1.00x | 0.72x | 0.69x | 0.96x |
| norm | 64 | 1.00x | 0.69x | 0.76x | 1.10x |
| norm | 256 | 1.00x | 2.67x | 2.95x | 1.11x |
| norm | 1024 | 1.00x | 5.96x | 5.34x | 0.90x |
| norm | 4096 | 1.00x | 2.54x | 2.59x | 1.02x |
| qr | 64 | 1.00x | 5.74x | 2.76x | 0.48x |
| qr | 256 | 1.00x | 1.42x | 0.69x | 0.49x |
| qr | 1024 | 1.00x | 0.61x | 0.51x | 0.83x |
| qr | 4096 | 1.00x | 0.97x | 0.77x | 0.80x |
| quantile | 64 | 1.00x | 0.08x | 0.08x | 1.06x |
| quantile | 256 | 1.00x | 1.03x | 0.33x | 0.32x |
| quantile | 1024 | 1.00x | 0.49x | 0.44x | 0.89x |
| quantile | 4096 | 1.00x | 1.71x | 0.64x | 0.37x |
| solve | 64 | 1.00x | 3.40x | 6.79x | 2.00x |
| solve | 256 | 1.00x | 3.87x | 2.89x | 0.75x |
| solve | 1024 | 1.00x | 1.63x | 1.47x | 0.90x |
| solve | 4096 | 1.00x | 1.92x | 0.84x | 0.44x |
| std | 64 | 1.00x | 0.17x | 0.18x | 1.09x |
| std | 256 | 1.00x | 0.37x | 0.42x | 1.12x |
| std | 1024 | 1.00x | 0.40x | 0.38x | 0.95x |
| std | 4096 | 1.00x | 0.42x | 0.40x | 0.96x |

</details>

<!-- PERF_TABLES_END -->

## Roofline (i64 engineering yardstick)

Host: 4 vCPU Intel Xeon @ 2.10 GHz (shared cloud container, AVX-512DQ
available), rustc 1.94.1, 2026-08-03. Gops = 2 × MACs / s. Median of 5
samples; ±10–20% run-to-run noise observed on this shared host.

All rows are **default codegen** builds; the shipped GEMM selects its ISA
path at runtime (`is_x86_feature_detected`), so no build flags are needed.

| kernel | Gops | note |
| --- | ---: | --- |
| scalar_chain | 2.29 | dependent MAC latency floor (context) |
| scalar_ilp8 | 3.85 | 8 independent accumulators |
| vec_mac_i64 | 3.83 | flat `c[j]+=a[j]*b[j]`, baseline codegen |
| vec_mac_f64 | 8.89 | ISA-physics context vs vec_mac_i64 |
| tile_4x8_i64 | 4.9–6.1 | GEBP register tile, baseline codegen, 1 thread |
| tile_4x8_i64_par | 20.0–23.1 | aggregate, 4 threads |
| tile_4x8_i64_isa | 9.0 | same tile, AVX-512DQ codegen (`vpmullq`), 1 thread |
| tile_4x8_i64_isa_par | 42.8 | aggregate, 4 threads |
| gemm_1024, pre-rework kernel | 17.3 | named-scalar 8×8, NC=64 structure |
| gemm_1024, portable profile | 21.0 | Goto order + flat 4×8 tile |
| gemm_1024 (shipped, dispatched) | 27–34 | AVX-512DQ profile selected at runtime; ≈80% of isa_tile_par |

Readings: (1) the shipped GEMM sits near the measured ceiling of whichever
ISA path dispatch selects (~88% of the baseline tile ceiling on non-AVX-512DQ
CPUs, ~80% of the ISA tile ceiling here); (2) micro-kernel source shape
decides everything — under `#[target_feature]` only the 32-lane 4×8 tile
stays register-clean (6×16 = 7.3 Gops, 8×8 = 13.8, 4×16 = 9.8, all
spill-bound; recorded in `linalg/i64_ops.rs`); (3) exact-i64 ceilings remain
well under BLAS f64 GEMM (register-blocked FMA) — the residual ~5× matmul gap
in Table D is dominated by that ISA physics, not kernel quality.

## Refreshing the tables

```bash
cargo test
cargo test --features lua

# Refresh result tables (release)
cargo test --release --features lua --test fair_all -- --run --sizes 64,256,1024,4096 \
  | awk -F'\t' 'NF==4 && ($1=="rust"||$1=="lua"){print}' > tests/bench/last_f64.tsv
python3 tests/bench/numpy_fair.py --sizes 64,256,1024,4096 \
  | awk -F'\t' 'NF==4 && $1=="numpy"{print}' >> tests/bench/last_f64.tsv

cargo test --release --features lua --test i64_surface -- --run --sizes 64,256,1024,4096 \
  | awk -F'\t' 'NF==4 && ($1=="rust"||$1=="lua"){print}' > tests/bench/last_i64.tsv
python3 tests/bench/numpy_i64_fair.py --sizes 64,256,1024,4096 \
  | awk -F'\t' 'NF==4 && $1=="numpy"{print}' >> tests/bench/last_i64.tsv

cargo test --release --features lua --test i64_promote -- --run --sizes 64,256,1024,4096 \
  | awk -F'\t' 'NF==4 && ($1=="rust"||$1=="lua"){print}' > tests/bench/last_i64_promote.tsv
python3 tests/bench/numpy_i64_promote.py --sizes 64,256,1024,4096 \
  | awk -F'\t' 'NF==4 && $1=="numpy"{print}' >> tests/bench/last_i64_promote.tsv

python3 tests/bench/compare_tables.py --write-readme tests/README.md

# Machine ceiling for exact wrapping i64 MACs (optionally with
# RUSTFLAGS="-C target-cpu=native" for the ISA-enabled ceiling)
cargo test --release --test i64_roofline -- --run
```
