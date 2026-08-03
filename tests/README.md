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

**Calibration gate:** before publishing a refresh, run `i64_roofline` and
compare the register-resident tile ceilings against the Roofline section's
recorded values for the host. If they deviate beyond noise (±20%), the
session is unfit: cells would be honestly measured but incomparable — to the
recorded roofline, to other sessions, and often to each other within the
run. Discard the run and refresh later; do not publish it. (Motivating
case: 2026-08-03 evening, this container measured ~3× degraded on pure
register-resident compute — hypervisor co-tenancy, no in-container load; a
completed full refresh was discarded rather than published.)

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
| arange | 4096 | 0.002804 | 0.001881 | 0.46x–0.67x |
| cholesky | 4096 | 627.969 | 501.373 | 0.80x–2.79x |
| copy | 4096 | 78.258 | 22.122 | 0.28x–1.09x |
| dot | 4096 | 0.001526 | 0.001816 | 0.46x–1.19x |
| elem_add | 4096 | 70.024 | 32.779 | 0.47x–0.94x |
| elem_add_scalar | 4096 | 62.254 | 25.548 | 0.41x–1.27x |
| elem_div | 4096 | 69.903 | 33.703 | 0.48x–0.99x |
| elem_mul | 4096 | 69.014 | 33.424 | 0.48x–0.75x |
| elem_sub | 4096 | 69.881 | 33.948 | 0.49x–0.90x |
| eye | 4096 | 5.0909 | 5.8790 | 0.29x–1.15x |
| fill | 4096 | 18.706 | 20.410 | 0.63x–1.09x |
| full | 4096 | 57.877 | 20.639 | 0.36x–0.92x |
| matmul | 4096 | 525.096 | 674.572 | 0.56x–1.28x |
| max | 4096 | 5.2957 | 3.2275 | 0.61x–2.22x |
| mean | 4096 | 5.5977 | 1.9605 | 0.23x–0.90x |
| min | 4096 | 5.0018 | 2.7505 | 0.55x–2.51x |
| norm | 4096 | 1.6572 | 2.4897 | 0.44x–4.80x |
| ones | 4096 | 57.086 | 20.394 | 0.36x–1.12x |
| qr | 4096 | 3418.611 | 1283.633 | 0.32x–2.38x |
| reshape | 4096 | 0.000303 | 0.000301 | 0.90x–1.09x |
| solve | 4096 | 463.860 | 606.536 | 1.05x–3.13x |
| sum | 4096 | 11.970 | 2.3238 | 0.19x–0.90x |
| svd | 4096 | 9075.689 | 13640.806 | 1.50x–3.11x |
| transpose | 4096 | 272.944 | 56.902 | 0.21x–1.44x |
| zeros | 4096 | 0.010040 | 0.004437 | 0.44x–42.10x |

#### i64

`matmul` / `matmul_at` / `matmul_bt` reference is NumPy **f64 BLAS** on
integer-valued data (see Yardsticks); MatLua times are exact wrapping i64.

| op | largest n | NumPy (ms) | MatLua Lua (ms) | Lua/NumPy, all n |
| --- | ---: | ---: | ---: | ---: |
| arange | 4096 | 0.002057 | 0.001517 | 0.56x–0.74x |
| copy | 4096 | 83.201 | 76.344 | 0.89x–1.06x |
| dot | 4096 | 0.003062 | 0.001760 | 0.30x–0.57x |
| elem_add | 4096 | 72.381 | 65.389 | 0.81x–1.22x |
| elem_div | 4096 | 116.070 | 92.580 | 0.55x–0.80x |
| elem_mul | 4096 | 75.974 | 67.305 | 0.71x–1.06x |
| elem_sub | 4096 | 71.416 | 75.201 | 0.85x–1.23x |
| eye | 4096 | 5.2306 | 4.8259 | 0.32x–0.94x |
| fill | 4096 | 19.249 | 18.763 | 0.64x–1.22x |
| full | 4096 | 60.505 | 58.125 | 0.46x–1.02x |
| isin | 4096 | 53.286 | 32.050 | 0.28x–1.54x |
| matmul | 4096 | 521.965 | 3591.910 | 5.95x–9.18x |
| matmul_at | 4096 | 506.037 | 3554.662 | 5.72x–9.17x |
| matmul_bt | 4096 | 486.950 | 3489.273 | 6.27x–7.40x |
| max | 4096 | 5.5552 | 2.9360 | 0.50x–1.40x |
| min | 4096 | 5.5873 | 3.1797 | 0.51x–1.40x |
| ones | 4096 | 62.256 | 56.433 | 0.40x–1.02x |
| reshape | 4096 | 0.000317 | 0.000300 | 0.89x–1.08x |
| sum | 4096 | 11.280 | 4.0137 | 0.36x–1.17x |
| transpose | 4096 | 264.398 | 92.284 | 0.35x–1.37x |
| unique | 4096 | 0.488477 | 0.005539 | 0.01x–0.05x |
| zeros | 4096 | 0.011178 | 0.004606 | 0.41x–1.08x |

#### i64→f64 promote-out

| op | largest n | NumPy (ms) | MatLua Lua (ms) | Lua/NumPy, all n |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 4096 | 883.006 | 809.168 | 0.88x–1.31x |
| mean | 4096 | 23.687 | 3.1615 | 0.13x–0.40x |
| median | 4096 | 103.015 | 88.240 | 0.14x–0.86x |
| norm | 4096 | 1.4172 | 2.9528 | 2.08x–7.39x |
| qr | 4096 | 3642.083 | 3308.234 | 0.63x–1.71x |
| quantile | 4096 | 118.958 | 89.009 | 0.08x–0.75x |
| solve | 4096 | 572.642 | 1396.753 | 0.83x–3.84x |
| std | 4096 | 151.886 | 23.343 | 0.15x–0.44x |

### Appendix — full three-face tables

<details>
<summary>Table A — f64 absolute (ms)</summary>

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000530 | 0.000144 | 0.000275 |
| arange | 256 | 0.000773 | 0.000214 | 0.000356 |
| arange | 1024 | 0.001145 | 0.000478 | 0.000677 |
| arange | 4096 | 0.002804 | 0.001530 | 0.001881 |
| cholesky | 64 | 0.022014 | 0.020552 | 0.061454 |
| cholesky | 256 | 0.956171 | 0.976746 | 0.883104 |
| cholesky | 1024 | 19.219 | 17.022 | 17.740 |
| cholesky | 4096 | 627.969 | 465.454 | 501.373 |
| copy | 64 | 0.001344 | 0.000986 | 0.001118 |
| copy | 256 | 0.015115 | 0.014645 | 0.016504 |
| copy | 1024 | 0.690180 | 0.676993 | 0.711860 |
| copy | 4096 | 78.258 | 21.607 | 22.122 |
| dot | 64 | 0.000671 | 0.000056 | 0.000307 |
| dot | 256 | 0.000763 | 0.000115 | 0.000437 |
| dot | 1024 | 0.001353 | 0.000374 | 0.000651 |
| dot | 4096 | 0.001526 | 0.001423 | 0.001816 |
| elem_add | 64 | 0.001633 | 0.001340 | 0.001527 |
| elem_add | 256 | 0.046049 | 0.028665 | 0.029300 |
| elem_add | 1024 | 1.1920 | 0.985842 | 1.0206 |
| elem_add | 4096 | 70.024 | 33.010 | 32.779 |
| elem_add_scalar | 64 | 0.001615 | 0.001132 | 0.001198 |
| elem_add_scalar | 256 | 0.015963 | 0.017527 | 0.020321 |
| elem_add_scalar | 1024 | 0.657768 | 0.699883 | 0.752667 |
| elem_add_scalar | 4096 | 62.254 | 28.267 | 25.548 |
| elem_div | 64 | 0.003444 | 0.003185 | 0.003108 |
| elem_div | 256 | 0.046373 | 0.045365 | 0.045949 |
| elem_div | 1024 | 1.2889 | 0.984303 | 1.0171 |
| elem_div | 4096 | 69.903 | 33.309 | 33.703 |
| elem_mul | 64 | 0.002123 | 0.001339 | 0.001465 |
| elem_mul | 256 | 0.046460 | 0.027416 | 0.027872 |
| elem_mul | 1024 | 1.3324 | 0.963834 | 1.0022 |
| elem_mul | 4096 | 69.014 | 32.950 | 33.424 |
| elem_sub | 64 | 0.001626 | 0.001337 | 0.001458 |
| elem_sub | 256 | 0.043129 | 0.027568 | 0.029377 |
| elem_sub | 1024 | 1.4951 | 0.974135 | 0.999238 |
| elem_sub | 4096 | 69.881 | 32.971 | 33.948 |
| eye | 64 | 0.002255 | 0.000320 | 0.000653 |
| eye | 256 | 0.016010 | 0.012517 | 0.011880 |
| eye | 1024 | 0.389392 | 0.359303 | 0.405936 |
| eye | 4096 | 5.0909 | 5.5152 | 5.8790 |
| fill | 64 | 0.001396 | 0.000394 | 0.000877 |
| fill | 256 | 0.018794 | 0.012270 | 0.012232 |
| fill | 1024 | 0.375987 | 0.368842 | 0.375065 |
| fill | 4096 | 18.706 | 18.864 | 20.410 |
| full | 64 | 0.002185 | 0.000882 | 0.000878 |
| full | 256 | 0.019710 | 0.012291 | 0.012739 |
| full | 1024 | 0.399874 | 0.385211 | 0.366987 |
| full | 4096 | 57.877 | 19.075 | 20.639 |
| matmul | 64 | 0.008566 | 0.008243 | 0.009997 |
| matmul | 256 | 0.447347 | 0.192036 | 0.252034 |
| matmul | 1024 | 9.0771 | 9.1142 | 9.9454 |
| matmul | 4096 | 525.096 | 655.738 | 674.572 |
| max | 64 | 0.001688 | 0.001468 | 0.001484 |
| max | 256 | 0.009149 | 0.022853 | 0.020317 |
| max | 1024 | 0.344630 | 0.374660 | 0.397425 |
| max | 4096 | 5.2957 | 1.8538 | 3.2275 |
| mean | 64 | 0.003651 | 0.000642 | 0.000857 |
| mean | 256 | 0.017914 | 0.008758 | 0.008857 |
| mean | 1024 | 0.337594 | 0.315085 | 0.305195 |
| mean | 4096 | 5.5977 | 1.7895 | 1.9605 |
| min | 64 | 0.001685 | 0.001388 | 0.001609 |
| min | 256 | 0.009141 | 0.022865 | 0.022942 |
| min | 1024 | 0.329492 | 0.371688 | 0.375200 |
| min | 4096 | 5.0018 | 1.8085 | 2.7505 |
| norm | 64 | 0.001968 | 0.000757 | 0.000859 |
| norm | 256 | 0.005490 | 0.010760 | 0.010856 |
| norm | 1024 | 0.078998 | 0.332299 | 0.379440 |
| norm | 4096 | 1.6572 | 2.3892 | 2.4897 |
| ones | 64 | 0.002304 | 0.000902 | 0.001011 |
| ones | 256 | 0.020788 | 0.012278 | 0.013646 |
| ones | 1024 | 0.421217 | 0.373539 | 0.471751 |
| ones | 4096 | 57.086 | 19.449 | 20.394 |
| qr | 64 | 0.113120 | 0.620965 | 0.268777 |
| qr | 256 | 4.7789 | 3.8459 | 2.2145 |
| qr | 1024 | 111.256 | 37.672 | 35.814 |
| qr | 4096 | 3418.611 | 1701.561 | 1283.633 |
| reshape | 64 | 0.000252 | 0.000085 | 0.000275 |
| reshape | 256 | 0.000287 | 0.000084 | 0.000295 |
| reshape | 1024 | 0.000343 | 0.000082 | 0.000310 |
| reshape | 4096 | 0.000303 | 0.000084 | 0.000301 |
| solve | 64 | 0.031986 | 0.028095 | 0.033441 |
| solve | 256 | 0.547211 | 2.1357 | 1.7143 |
| solve | 1024 | 13.369 | 33.050 | 32.267 |
| solve | 4096 | 463.860 | 634.112 | 606.536 |
| sum | 64 | 0.002041 | 0.000635 | 0.000839 |
| sum | 256 | 0.017760 | 0.008773 | 0.008912 |
| sum | 1024 | 0.376473 | 0.322775 | 0.339889 |
| sum | 4096 | 11.970 | 1.9966 | 2.3238 |
| svd | 64 | 0.282254 | 0.565706 | 0.878348 |
| svd | 256 | 8.6197 | 12.158 | 18.701 |
| svd | 1024 | 274.237 | 376.331 | 634.889 |
| svd | 4096 | 9075.689 | 12457.223 | 13640.806 |
| transpose | 64 | 0.002430 | 0.003292 | 0.003500 |
| transpose | 256 | 0.078045 | 0.051588 | 0.052866 |
| transpose | 1024 | 5.0530 | 2.0392 | 2.1235 |
| transpose | 4096 | 272.944 | 71.295 | 56.902 |
| zeros | 64 | 0.000747 | 0.000407 | 0.000691 |
| zeros | 256 | 0.013558 | 0.002531 | 0.570760 |
| zeros | 1024 | 0.378460 | 0.004032 | 0.594923 |
| zeros | 4096 | 0.010040 | 0.021763 | 0.004437 |

</details>

<details>
<summary>Table B — f64 relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.27x | 0.52x | 1.91x |
| arange | 256 | 1.00x | 0.28x | 0.46x | 1.66x |
| arange | 1024 | 1.00x | 0.42x | 0.59x | 1.42x |
| arange | 4096 | 1.00x | 0.55x | 0.67x | 1.23x |
| cholesky | 64 | 1.00x | 0.93x | 2.79x | 2.99x |
| cholesky | 256 | 1.00x | 1.02x | 0.92x | 0.90x |
| cholesky | 1024 | 1.00x | 0.89x | 0.92x | 1.04x |
| cholesky | 4096 | 1.00x | 0.74x | 0.80x | 1.08x |
| copy | 64 | 1.00x | 0.73x | 0.83x | 1.13x |
| copy | 256 | 1.00x | 0.97x | 1.09x | 1.13x |
| copy | 1024 | 1.00x | 0.98x | 1.03x | 1.05x |
| copy | 4096 | 1.00x | 0.28x | 0.28x | 1.02x |
| dot | 64 | 1.00x | 0.08x | 0.46x | 5.48x |
| dot | 256 | 1.00x | 0.15x | 0.57x | 3.80x |
| dot | 1024 | 1.00x | 0.28x | 0.48x | 1.74x |
| dot | 4096 | 1.00x | 0.93x | 1.19x | 1.28x |
| elem_add | 64 | 1.00x | 0.82x | 0.94x | 1.14x |
| elem_add | 256 | 1.00x | 0.62x | 0.64x | 1.02x |
| elem_add | 1024 | 1.00x | 0.83x | 0.86x | 1.04x |
| elem_add | 4096 | 1.00x | 0.47x | 0.47x | 0.99x |
| elem_add_scalar | 64 | 1.00x | 0.70x | 0.74x | 1.06x |
| elem_add_scalar | 256 | 1.00x | 1.10x | 1.27x | 1.16x |
| elem_add_scalar | 1024 | 1.00x | 1.06x | 1.14x | 1.08x |
| elem_add_scalar | 4096 | 1.00x | 0.45x | 0.41x | 0.90x |
| elem_div | 64 | 1.00x | 0.92x | 0.90x | 0.98x |
| elem_div | 256 | 1.00x | 0.98x | 0.99x | 1.01x |
| elem_div | 1024 | 1.00x | 0.76x | 0.79x | 1.03x |
| elem_div | 4096 | 1.00x | 0.48x | 0.48x | 1.01x |
| elem_mul | 64 | 1.00x | 0.63x | 0.69x | 1.09x |
| elem_mul | 256 | 1.00x | 0.59x | 0.60x | 1.02x |
| elem_mul | 1024 | 1.00x | 0.72x | 0.75x | 1.04x |
| elem_mul | 4096 | 1.00x | 0.48x | 0.48x | 1.01x |
| elem_sub | 64 | 1.00x | 0.82x | 0.90x | 1.09x |
| elem_sub | 256 | 1.00x | 0.64x | 0.68x | 1.07x |
| elem_sub | 1024 | 1.00x | 0.65x | 0.67x | 1.03x |
| elem_sub | 4096 | 1.00x | 0.47x | 0.49x | 1.03x |
| eye | 64 | 1.00x | 0.14x | 0.29x | 2.04x |
| eye | 256 | 1.00x | 0.78x | 0.74x | 0.95x |
| eye | 1024 | 1.00x | 0.92x | 1.04x | 1.13x |
| eye | 4096 | 1.00x | 1.08x | 1.15x | 1.07x |
| fill | 64 | 1.00x | 0.28x | 0.63x | 2.23x |
| fill | 256 | 1.00x | 0.65x | 0.65x | 1.00x |
| fill | 1024 | 1.00x | 0.98x | 1.00x | 1.02x |
| fill | 4096 | 1.00x | 1.01x | 1.09x | 1.08x |
| full | 64 | 1.00x | 0.40x | 0.40x | 1.00x |
| full | 256 | 1.00x | 0.62x | 0.65x | 1.04x |
| full | 1024 | 1.00x | 0.96x | 0.92x | 0.95x |
| full | 4096 | 1.00x | 0.33x | 0.36x | 1.08x |
| matmul | 64 | 1.00x | 0.96x | 1.17x | 1.21x |
| matmul | 256 | 1.00x | 0.43x | 0.56x | 1.31x |
| matmul | 1024 | 1.00x | 1.00x | 1.10x | 1.09x |
| matmul | 4096 | 1.00x | 1.25x | 1.28x | 1.03x |
| max | 64 | 1.00x | 0.87x | 0.88x | 1.01x |
| max | 256 | 1.00x | 2.50x | 2.22x | 0.89x |
| max | 1024 | 1.00x | 1.09x | 1.15x | 1.06x |
| max | 4096 | 1.00x | 0.35x | 0.61x | 1.74x |
| mean | 64 | 1.00x | 0.18x | 0.23x | 1.33x |
| mean | 256 | 1.00x | 0.49x | 0.49x | 1.01x |
| mean | 1024 | 1.00x | 0.93x | 0.90x | 0.97x |
| mean | 4096 | 1.00x | 0.32x | 0.35x | 1.10x |
| min | 64 | 1.00x | 0.82x | 0.95x | 1.16x |
| min | 256 | 1.00x | 2.50x | 2.51x | 1.00x |
| min | 1024 | 1.00x | 1.13x | 1.14x | 1.01x |
| min | 4096 | 1.00x | 0.36x | 0.55x | 1.52x |
| norm | 64 | 1.00x | 0.38x | 0.44x | 1.13x |
| norm | 256 | 1.00x | 1.96x | 1.98x | 1.01x |
| norm | 1024 | 1.00x | 4.21x | 4.80x | 1.14x |
| norm | 4096 | 1.00x | 1.44x | 1.50x | 1.04x |
| ones | 64 | 1.00x | 0.39x | 0.44x | 1.12x |
| ones | 256 | 1.00x | 0.59x | 0.66x | 1.11x |
| ones | 1024 | 1.00x | 0.89x | 1.12x | 1.26x |
| ones | 4096 | 1.00x | 0.34x | 0.36x | 1.05x |
| qr | 64 | 1.00x | 5.49x | 2.38x | 0.43x |
| qr | 256 | 1.00x | 0.80x | 0.46x | 0.58x |
| qr | 1024 | 1.00x | 0.34x | 0.32x | 0.95x |
| qr | 4096 | 1.00x | 0.50x | 0.38x | 0.75x |
| reshape | 64 | 1.00x | 0.34x | 1.09x | 3.24x |
| reshape | 256 | 1.00x | 0.29x | 1.03x | 3.51x |
| reshape | 1024 | 1.00x | 0.24x | 0.90x | 3.78x |
| reshape | 4096 | 1.00x | 0.28x | 0.99x | 3.58x |
| solve | 64 | 1.00x | 0.88x | 1.05x | 1.19x |
| solve | 256 | 1.00x | 3.90x | 3.13x | 0.80x |
| solve | 1024 | 1.00x | 2.47x | 2.41x | 0.98x |
| solve | 4096 | 1.00x | 1.37x | 1.31x | 0.96x |
| sum | 64 | 1.00x | 0.31x | 0.41x | 1.32x |
| sum | 256 | 1.00x | 0.49x | 0.50x | 1.02x |
| sum | 1024 | 1.00x | 0.86x | 0.90x | 1.05x |
| sum | 4096 | 1.00x | 0.17x | 0.19x | 1.16x |
| svd | 64 | 1.00x | 2.00x | 3.11x | 1.55x |
| svd | 256 | 1.00x | 1.41x | 2.17x | 1.54x |
| svd | 1024 | 1.00x | 1.37x | 2.32x | 1.69x |
| svd | 4096 | 1.00x | 1.37x | 1.50x | 1.10x |
| transpose | 64 | 1.00x | 1.35x | 1.44x | 1.06x |
| transpose | 256 | 1.00x | 0.66x | 0.68x | 1.02x |
| transpose | 1024 | 1.00x | 0.40x | 0.42x | 1.04x |
| transpose | 4096 | 1.00x | 0.26x | 0.21x | 0.80x |
| zeros | 64 | 1.00x | 0.54x | 0.93x | 1.70x |
| zeros | 256 | 1.00x | 0.19x | 42.10x | 225.51x |
| zeros | 1024 | 1.00x | 0.01x | 1.57x | 147.55x |
| zeros | 4096 | 1.00x | 2.17x | 0.44x | 0.20x |

</details>

<details>
<summary>Table C — i64 absolute (ms) — matmul* NumPy column is f64 BLAS on integer-valued data</summary>

| op | n | NumPy int64 (ms) | MatLua Rust i64 (ms) | MatLua Lua i64 (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000456 | 0.000142 | 0.000271 |
| arange | 256 | 0.000609 | 0.000196 | 0.000340 |
| arange | 1024 | 0.000942 | 0.000486 | 0.000623 |
| arange | 4096 | 0.002057 | 0.001326 | 0.001517 |
| copy | 64 | 0.001305 | 0.001004 | 0.001155 |
| copy | 256 | 0.014747 | 0.013621 | 0.015286 |
| copy | 1024 | 0.678919 | 0.672548 | 0.720507 |
| copy | 4096 | 83.201 | 77.062 | 76.344 |
| dot | 64 | 0.000946 | 0.000067 | 0.000287 |
| dot | 256 | 0.001004 | 0.000123 | 0.000358 |
| dot | 1024 | 0.001386 | 0.000394 | 0.000670 |
| dot | 4096 | 0.003062 | 0.001546 | 0.001760 |
| elem_add | 64 | 0.001846 | 0.001252 | 0.001501 |
| elem_add | 256 | 0.022283 | 0.018735 | 0.027184 |
| elem_add | 1024 | 1.0076 | 0.965047 | 1.0596 |
| elem_add | 4096 | 72.381 | 67.202 | 65.389 |
| elem_div | 64 | 0.016143 | 0.008629 | 0.008843 |
| elem_div | 256 | 0.247777 | 0.136167 | 0.136882 |
| elem_div | 1024 | 3.8958 | 2.3386 | 2.5476 |
| elem_div | 4096 | 116.070 | 93.903 | 92.580 |
| elem_mul | 64 | 0.002181 | 0.001351 | 0.001547 |
| elem_mul | 256 | 0.030052 | 0.019463 | 0.025740 |
| elem_mul | 1024 | 0.996359 | 1.0071 | 1.0579 |
| elem_mul | 4096 | 75.974 | 68.593 | 67.305 |
| elem_sub | 64 | 0.001820 | 0.001253 | 0.001544 |
| elem_sub | 256 | 0.022229 | 0.018548 | 0.027297 |
| elem_sub | 1024 | 1.0077 | 0.980878 | 1.0496 |
| elem_sub | 4096 | 71.416 | 68.096 | 75.201 |
| eye | 64 | 0.002126 | 0.000315 | 0.000679 |
| eye | 256 | 0.014958 | 0.011702 | 0.011690 |
| eye | 1024 | 0.422676 | 0.380675 | 0.396229 |
| eye | 4096 | 5.2306 | 4.9566 | 4.8259 |
| fill | 64 | 0.001383 | 0.000394 | 0.000884 |
| fill | 256 | 0.017577 | 0.012310 | 0.012266 |
| fill | 1024 | 0.362981 | 0.379297 | 0.443983 |
| fill | 4096 | 19.249 | 13.898 | 18.763 |
| full | 64 | 0.002227 | 0.000883 | 0.001017 |
| full | 256 | 0.018467 | 0.012372 | 0.012625 |
| full | 1024 | 0.420000 | 0.392149 | 0.426589 |
| full | 4096 | 60.505 | 55.717 | 58.125 |
| isin | 64 | 0.020326 | 0.004563 | 0.005643 |
| isin | 256 | 0.082118 | 0.068793 | 0.126083 |
| isin | 1024 | 1.2892 | 1.1962 | 1.4936 |
| isin | 4096 | 53.286 | 85.000 | 32.050 |
| matmul | 64 | 0.008347 | 0.075517 | 0.076642 |
| matmul | 256 | 0.324496 | 2.1196 | 1.9291 |
| matmul | 1024 | 8.7940 | 54.132 | 56.544 |
| matmul | 4096 | 521.965 | 3448.514 | 3591.910 |
| matmul_at | 64 | 0.008372 | 0.075377 | 0.076756 |
| matmul_at | 256 | 0.354620 | 2.0933 | 2.0268 |
| matmul_at | 1024 | 8.7522 | 55.000 | 67.978 |
| matmul_at | 4096 | 506.037 | 3594.732 | 3554.662 |
| matmul_bt | 64 | 0.011751 | 0.072712 | 0.073638 |
| matmul_bt | 256 | 0.334205 | 2.2660 | 2.2617 |
| matmul_bt | 1024 | 8.6460 | 56.827 | 64.014 |
| matmul_bt | 4096 | 486.950 | 3657.345 | 3489.273 |
| max | 64 | 0.001763 | 0.000716 | 0.000889 |
| max | 256 | 0.008287 | 0.011329 | 0.011561 |
| max | 1024 | 0.337353 | 0.400409 | 0.379356 |
| max | 4096 | 5.5552 | 2.0600 | 2.9360 |
| min | 64 | 0.001758 | 0.000715 | 0.000890 |
| min | 256 | 0.008276 | 0.011317 | 0.011581 |
| min | 1024 | 0.328599 | 0.377357 | 0.404166 |
| min | 4096 | 5.5873 | 2.5387 | 3.1797 |
| ones | 64 | 0.002248 | 0.000902 | 0.000907 |
| ones | 256 | 0.020439 | 0.012431 | 0.012809 |
| ones | 1024 | 0.413656 | 0.383410 | 0.421691 |
| ones | 4096 | 62.256 | 56.053 | 56.433 |
| reshape | 64 | 0.000265 | 0.000084 | 0.000285 |
| reshape | 256 | 0.000300 | 0.000083 | 0.000289 |
| reshape | 1024 | 0.000321 | 0.000083 | 0.000286 |
| reshape | 4096 | 0.000317 | 0.000085 | 0.000300 |
| sum | 64 | 0.001897 | 0.000706 | 0.000886 |
| sum | 256 | 0.010124 | 0.011379 | 0.011576 |
| sum | 1024 | 0.334631 | 0.323767 | 0.393046 |
| sum | 4096 | 11.280 | 3.4764 | 4.0137 |
| transpose | 64 | 0.002563 | 0.003289 | 0.003508 |
| transpose | 256 | 0.078605 | 0.051029 | 0.052840 |
| transpose | 1024 | 5.0930 | 2.0857 | 2.1346 |
| transpose | 4096 | 264.398 | 99.476 | 92.284 |
| unique | 64 | 0.006663 | 0.000166 | 0.000314 |
| unique | 256 | 0.021598 | 0.000380 | 0.000614 |
| unique | 1024 | 0.108671 | 0.000960 | 0.001460 |
| unique | 4096 | 0.488477 | 0.003993 | 0.005539 |
| zeros | 64 | 0.000683 | 0.000308 | 0.000738 |
| zeros | 256 | 0.011588 | 0.011289 | 0.011329 |
| zeros | 1024 | 0.407514 | 0.363820 | 0.404799 |
| zeros | 4096 | 0.011178 | 0.010062 | 0.004606 |

</details>

<details>
<summary>Table D — i64 relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.31x | 0.59x | 1.91x |
| arange | 256 | 1.00x | 0.32x | 0.56x | 1.73x |
| arange | 1024 | 1.00x | 0.52x | 0.66x | 1.28x |
| arange | 4096 | 1.00x | 0.64x | 0.74x | 1.14x |
| copy | 64 | 1.00x | 0.77x | 0.89x | 1.15x |
| copy | 256 | 1.00x | 0.92x | 1.04x | 1.12x |
| copy | 1024 | 1.00x | 0.99x | 1.06x | 1.07x |
| copy | 4096 | 1.00x | 0.93x | 0.92x | 0.99x |
| dot | 64 | 1.00x | 0.07x | 0.30x | 4.28x |
| dot | 256 | 1.00x | 0.12x | 0.36x | 2.91x |
| dot | 1024 | 1.00x | 0.28x | 0.48x | 1.70x |
| dot | 4096 | 1.00x | 0.50x | 0.57x | 1.14x |
| elem_add | 64 | 1.00x | 0.68x | 0.81x | 1.20x |
| elem_add | 256 | 1.00x | 0.84x | 1.22x | 1.45x |
| elem_add | 1024 | 1.00x | 0.96x | 1.05x | 1.10x |
| elem_add | 4096 | 1.00x | 0.93x | 0.90x | 0.97x |
| elem_div | 64 | 1.00x | 0.53x | 0.55x | 1.02x |
| elem_div | 256 | 1.00x | 0.55x | 0.55x | 1.01x |
| elem_div | 1024 | 1.00x | 0.60x | 0.65x | 1.09x |
| elem_div | 4096 | 1.00x | 0.81x | 0.80x | 0.99x |
| elem_mul | 64 | 1.00x | 0.62x | 0.71x | 1.15x |
| elem_mul | 256 | 1.00x | 0.65x | 0.86x | 1.32x |
| elem_mul | 1024 | 1.00x | 1.01x | 1.06x | 1.05x |
| elem_mul | 4096 | 1.00x | 0.90x | 0.89x | 0.98x |
| elem_sub | 64 | 1.00x | 0.69x | 0.85x | 1.23x |
| elem_sub | 256 | 1.00x | 0.83x | 1.23x | 1.47x |
| elem_sub | 1024 | 1.00x | 0.97x | 1.04x | 1.07x |
| elem_sub | 4096 | 1.00x | 0.95x | 1.05x | 1.10x |
| eye | 64 | 1.00x | 0.15x | 0.32x | 2.16x |
| eye | 256 | 1.00x | 0.78x | 0.78x | 1.00x |
| eye | 1024 | 1.00x | 0.90x | 0.94x | 1.04x |
| eye | 4096 | 1.00x | 0.95x | 0.92x | 0.97x |
| fill | 64 | 1.00x | 0.28x | 0.64x | 2.24x |
| fill | 256 | 1.00x | 0.70x | 0.70x | 1.00x |
| fill | 1024 | 1.00x | 1.04x | 1.22x | 1.17x |
| fill | 4096 | 1.00x | 0.72x | 0.97x | 1.35x |
| full | 64 | 1.00x | 0.40x | 0.46x | 1.15x |
| full | 256 | 1.00x | 0.67x | 0.68x | 1.02x |
| full | 1024 | 1.00x | 0.93x | 1.02x | 1.09x |
| full | 4096 | 1.00x | 0.92x | 0.96x | 1.04x |
| isin | 64 | 1.00x | 0.22x | 0.28x | 1.24x |
| isin | 256 | 1.00x | 0.84x | 1.54x | 1.83x |
| isin | 1024 | 1.00x | 0.93x | 1.16x | 1.25x |
| isin | 4096 | 1.00x | 1.60x | 0.60x | 0.38x |
| matmul | 64 | 1.00x | 9.05x | 9.18x | 1.01x |
| matmul | 256 | 1.00x | 6.53x | 5.95x | 0.91x |
| matmul | 1024 | 1.00x | 6.16x | 6.43x | 1.04x |
| matmul | 4096 | 1.00x | 6.61x | 6.88x | 1.04x |
| matmul_at | 64 | 1.00x | 9.00x | 9.17x | 1.02x |
| matmul_at | 256 | 1.00x | 5.90x | 5.72x | 0.97x |
| matmul_at | 1024 | 1.00x | 6.28x | 7.77x | 1.24x |
| matmul_at | 4096 | 1.00x | 7.10x | 7.02x | 0.99x |
| matmul_bt | 64 | 1.00x | 6.19x | 6.27x | 1.01x |
| matmul_bt | 256 | 1.00x | 6.78x | 6.77x | 1.00x |
| matmul_bt | 1024 | 1.00x | 6.57x | 7.40x | 1.13x |
| matmul_bt | 4096 | 1.00x | 7.51x | 7.17x | 0.95x |
| max | 64 | 1.00x | 0.41x | 0.50x | 1.24x |
| max | 256 | 1.00x | 1.37x | 1.40x | 1.02x |
| max | 1024 | 1.00x | 1.19x | 1.12x | 0.95x |
| max | 4096 | 1.00x | 0.37x | 0.53x | 1.43x |
| min | 64 | 1.00x | 0.41x | 0.51x | 1.24x |
| min | 256 | 1.00x | 1.37x | 1.40x | 1.02x |
| min | 1024 | 1.00x | 1.15x | 1.23x | 1.07x |
| min | 4096 | 1.00x | 0.45x | 0.57x | 1.25x |
| ones | 64 | 1.00x | 0.40x | 0.40x | 1.01x |
| ones | 256 | 1.00x | 0.61x | 0.63x | 1.03x |
| ones | 1024 | 1.00x | 0.93x | 1.02x | 1.10x |
| ones | 4096 | 1.00x | 0.90x | 0.91x | 1.01x |
| reshape | 64 | 1.00x | 0.32x | 1.08x | 3.39x |
| reshape | 256 | 1.00x | 0.28x | 0.96x | 3.48x |
| reshape | 1024 | 1.00x | 0.26x | 0.89x | 3.45x |
| reshape | 4096 | 1.00x | 0.27x | 0.95x | 3.53x |
| sum | 64 | 1.00x | 0.37x | 0.47x | 1.25x |
| sum | 256 | 1.00x | 1.12x | 1.14x | 1.02x |
| sum | 1024 | 1.00x | 0.97x | 1.17x | 1.21x |
| sum | 4096 | 1.00x | 0.31x | 0.36x | 1.15x |
| transpose | 64 | 1.00x | 1.28x | 1.37x | 1.07x |
| transpose | 256 | 1.00x | 0.65x | 0.67x | 1.04x |
| transpose | 1024 | 1.00x | 0.41x | 0.42x | 1.02x |
| transpose | 4096 | 1.00x | 0.38x | 0.35x | 0.93x |
| unique | 64 | 1.00x | 0.02x | 0.05x | 1.89x |
| unique | 256 | 1.00x | 0.02x | 0.03x | 1.62x |
| unique | 1024 | 1.00x | 0.01x | 0.01x | 1.52x |
| unique | 4096 | 1.00x | 0.01x | 0.01x | 1.39x |
| zeros | 64 | 1.00x | 0.45x | 1.08x | 2.40x |
| zeros | 256 | 1.00x | 0.97x | 0.98x | 1.00x |
| zeros | 1024 | 1.00x | 0.89x | 0.99x | 1.11x |
| zeros | 4096 | 1.00x | 0.90x | 0.41x | 0.46x |

</details>

<details>
<summary>Table E — i64→f64 promote-out absolute (ms)</summary>

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 64 | 0.032377 | 0.036592 | 0.042560 |
| cholesky | 256 | 0.873824 | 0.965718 | 0.913872 |
| cholesky | 1024 | 22.581 | 20.793 | 19.874 |
| cholesky | 4096 | 883.006 | 671.939 | 809.168 |
| mean | 64 | 0.006421 | 0.000715 | 0.000923 |
| mean | 256 | 0.053738 | 0.011435 | 0.011460 |
| mean | 1024 | 0.852445 | 0.328353 | 0.340068 |
| mean | 4096 | 23.687 | 2.3876 | 3.1615 |
| median | 64 | 0.026296 | 0.003470 | 0.003742 |
| median | 256 | 0.105152 | 0.156874 | 0.054076 |
| median | 1024 | 1.8698 | 1.2506 | 1.3377 |
| median | 4096 | 103.015 | 90.015 | 88.240 |
| norm | 64 | 0.001928 | 0.020543 | 0.014241 |
| norm | 256 | 0.005424 | 0.031437 | 0.032429 |
| norm | 1024 | 0.089503 | 0.390496 | 0.412475 |
| norm | 4096 | 1.4172 | 2.9516 | 2.9528 |
| qr | 64 | 0.115112 | 1.3915 | 0.196651 |
| qr | 256 | 4.9964 | 7.0295 | 3.1597 |
| qr | 1024 | 97.984 | 71.459 | 63.341 |
| qr | 4096 | 3642.083 | 3486.847 | 3308.234 |
| quantile | 64 | 0.054927 | 0.004153 | 0.004425 |
| quantile | 256 | 0.223523 | 0.170447 | 0.064755 |
| quantile | 1024 | 2.8073 | 1.4201 | 1.4563 |
| quantile | 4096 | 118.958 | 91.973 | 89.009 |
| solve | 64 | 0.034812 | 0.033127 | 0.029068 |
| solve | 256 | 0.555228 | 2.2569 | 2.1334 |
| solve | 1024 | 14.318 | 29.472 | 34.644 |
| solve | 4096 | 572.642 | 1398.760 | 1396.753 |
| std | 64 | 0.016755 | 0.003659 | 0.003800 |
| std | 256 | 0.139272 | 0.056699 | 0.058085 |
| std | 1024 | 2.6168 | 1.1059 | 1.1610 |
| std | 4096 | 151.886 | 17.326 | 23.343 |

</details>

<details>
<summary>Table F — i64→f64 promote-out relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky | 64 | 1.00x | 1.13x | 1.31x | 1.16x |
| cholesky | 256 | 1.00x | 1.11x | 1.05x | 0.95x |
| cholesky | 1024 | 1.00x | 0.92x | 0.88x | 0.96x |
| cholesky | 4096 | 1.00x | 0.76x | 0.92x | 1.20x |
| mean | 64 | 1.00x | 0.11x | 0.14x | 1.29x |
| mean | 256 | 1.00x | 0.21x | 0.21x | 1.00x |
| mean | 1024 | 1.00x | 0.39x | 0.40x | 1.04x |
| mean | 4096 | 1.00x | 0.10x | 0.13x | 1.32x |
| median | 64 | 1.00x | 0.13x | 0.14x | 1.08x |
| median | 256 | 1.00x | 1.49x | 0.51x | 0.34x |
| median | 1024 | 1.00x | 0.67x | 0.72x | 1.07x |
| median | 4096 | 1.00x | 0.87x | 0.86x | 0.98x |
| norm | 64 | 1.00x | 10.66x | 7.39x | 0.69x |
| norm | 256 | 1.00x | 5.80x | 5.98x | 1.03x |
| norm | 1024 | 1.00x | 4.36x | 4.61x | 1.06x |
| norm | 4096 | 1.00x | 2.08x | 2.08x | 1.00x |
| qr | 64 | 1.00x | 12.09x | 1.71x | 0.14x |
| qr | 256 | 1.00x | 1.41x | 0.63x | 0.45x |
| qr | 1024 | 1.00x | 0.73x | 0.65x | 0.89x |
| qr | 4096 | 1.00x | 0.96x | 0.91x | 0.95x |
| quantile | 64 | 1.00x | 0.08x | 0.08x | 1.07x |
| quantile | 256 | 1.00x | 0.76x | 0.29x | 0.38x |
| quantile | 1024 | 1.00x | 0.51x | 0.52x | 1.03x |
| quantile | 4096 | 1.00x | 0.77x | 0.75x | 0.97x |
| solve | 64 | 1.00x | 0.95x | 0.83x | 0.88x |
| solve | 256 | 1.00x | 4.06x | 3.84x | 0.95x |
| solve | 1024 | 1.00x | 2.06x | 2.42x | 1.18x |
| solve | 4096 | 1.00x | 2.44x | 2.44x | 1.00x |
| std | 64 | 1.00x | 0.22x | 0.23x | 1.04x |
| std | 256 | 1.00x | 0.41x | 0.42x | 1.02x |
| std | 1024 | 1.00x | 0.42x | 0.44x | 1.05x |
| std | 4096 | 1.00x | 0.11x | 0.15x | 1.35x |

</details>

<!-- PERF_TABLES_END -->

## Roofline (i64 engineering yardstick)

Host: 4 vCPU Intel Xeon @ 2.10 GHz (shared cloud container, AVX-512DQ
available), rustc 1.94.1, 2026-08-03 (same session as the Results tables —
these are the calibration-gate baselines). Gops = 2 × MACs / s; median of 5
samples; ±10–20% shared-host noise. All rows are **default codegen** builds;
the shipped GEMM selects its ISA path at runtime via micro-calibration
(`array::isa::avx512_fast` — CPUID plus a ~1 ms kernel race, because CPUID
cannot see dynamic 512-bit downclocking; a ~6 h co-tenant throttle window on
this container cut the ISA tile to ~11 Gops while scalar stayed normal, and
the calibrated dispatch correctly fell back to the portable profile).

| kernel | Gops | note |
| --- | ---: | --- |
| scalar_chain | 2.00 | dependent MAC latency floor (context) |
| scalar_ilp8 | 3.36 | 8 independent accumulators |
| vec_mac_i64 | 3.36 | flat `c[j]+=a[j]*b[j]`, baseline codegen |
| vec_mac_f64 | 6.93 | ISA-physics context vs vec_mac_i64 |
| tile_4x8_i64 | 5.44 | GEBP register tile, baseline codegen, 1 thread |
| tile_4x8_i64_par | 21.1 | aggregate, 4 threads |
| tile_4x8_i64_isa | 12.6 | same tile, AVX-512DQ codegen (`vpmullq`), 1 thread |
| tile_4x8_i64_isa_par | 49.2 | aggregate, 4 threads |
| gemm_1024 (shipped, calibrated dispatch) | 36.4 (59.0 ms) | ≈74% of isa_tile_par |

History on this container (for scale): pre-rework named-scalar kernel
17.3 Gops; Goto-order + flat 4×8 portable tile 21.0; static-CPUID AVX-512
dispatch 27–34; micro-calibrated dispatch 35–36 on a healthy session.
Rejected tile shapes under `#[target_feature]` (recorded in
`linalg/i64_ops.rs`): 6×16 = 7.3 Gops, 8×8 = 13.8, 4×16 = 9.8 — only the
32-lane 4×8 tile stays register-clean. The residual i64-vs-BLAS matmul gap
in Table D is ISA physics (register-blocked f64 FMA vs 64-bit integer
multiply throughput), with the kernel at ~74–88% of the measured ceiling of
whichever path calibration selects.

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
