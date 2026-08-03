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
| arange | 4096 | 0.003249 | 0.001751 | 0.51x–0.62x |
| cholesky | 4096 | 632.894 | 514.529 | 0.81x–2.47x |
| copy | 4096 | 77.579 | 21.916 | 0.28x–1.24x |
| dot | 4096 | 0.001445 | 0.001742 | 0.48x–1.21x |
| elem_add | 4096 | 69.209 | 33.816 | 0.49x–0.94x |
| elem_add_scalar | 4096 | 63.065 | 25.902 | 0.41x–1.14x |
| elem_div | 4096 | 70.139 | 34.561 | 0.49x–0.99x |
| elem_mul | 4096 | 69.174 | 34.018 | 0.49x–0.94x |
| elem_sub | 4096 | 68.967 | 34.096 | 0.49x–0.91x |
| eye | 4096 | 5.1719 | 6.8946 | 0.31x–1.33x |
| fill | 4096 | 16.938 | 19.195 | 0.67x–1.13x |
| full | 4096 | 58.340 | 19.910 | 0.34x–1.07x |
| matmul | 4096 | 492.283 | 632.257 | 0.61x–1.28x |
| max | 4096 | 5.2721 | 23.767 | 1.19x–4.51x |
| mean | 4096 | 6.6606 | 14.285 | 0.28x–2.14x |
| min | 4096 | 5.1840 | 23.706 | 1.26x–4.57x |
| norm | 4096 | 1.4093 | 2.5073 | 1.78x–6.09x |
| ones | 4096 | 59.358 | 20.109 | 0.34x–1.13x |
| qr | 4096 | 3382.037 | 2039.319 | 0.31x–1.91x |
| reshape | 4096 | 0.000272 | 0.000540 | 0.72x–1.99x |
| solve | 4096 | 459.024 | 612.740 | 0.88x–3.00x |
| sum | 4096 | 13.598 | 12.311 | 0.50x–0.95x |
| svd | 4096 | 9280.692 | 13228.578 | 1.43x–3.08x |
| transpose | 4096 | 257.495 | 53.758 | 0.21x–1.45x |
| zeros | 4096 | 0.013153 | 0.005110 | 0.39x–17.78x |

#### i64

`matmul` / `matmul_at` / `matmul_bt` reference is NumPy **f64 BLAS** on
integer-valued data (see Yardsticks); MatLua times are exact wrapping i64.

| op | largest n | NumPy (ms) | MatLua Lua (ms) | Lua/NumPy, all n |
| --- | ---: | ---: | ---: | ---: |
| arange | 4096 | 0.002054 | 0.001572 | 0.55x–0.77x |
| copy | 4096 | 79.876 | 74.436 | 0.88x–1.04x |
| dot | 4096 | 0.003410 | 0.001736 | 0.30x–0.51x |
| elem_add | 4096 | 70.111 | 66.954 | 0.82x–1.02x |
| elem_div | 4096 | 116.326 | 94.776 | 0.54x–0.81x |
| elem_mul | 4096 | 75.372 | 67.051 | 0.66x–1.01x |
| elem_sub | 4096 | 69.687 | 67.605 | 0.87x–0.97x |
| eye | 4096 | 5.2134 | 5.2244 | 0.31x–1.00x |
| fill | 4096 | 18.280 | 18.181 | 0.69x–0.99x |
| full | 4096 | 58.558 | 55.854 | 0.43x–0.95x |
| isin | 4096 | 53.461 | 31.865 | 0.28x–1.07x |
| matmul | 4096 | 495.045 | 3455.757 | 6.73x–10.34x |
| matmul_at | 4096 | 489.889 | 3487.396 | 6.70x–10.26x |
| matmul_bt | 4096 | 490.219 | 3499.905 | 7.15x |
| max | 4096 | 5.1775 | 22.703 | 0.64x–4.38x |
| min | 4096 | 5.2345 | 21.489 | 0.66x–4.11x |
| ones | 4096 | 58.760 | 59.028 | 0.43x–1.00x |
| reshape | 4096 | 0.000554 | 0.000374 | 0.68x–1.15x |
| sum | 4096 | 9.3114 | 12.307 | 0.35x–1.32x |
| transpose | 4096 | 263.229 | 94.117 | 0.36x–1.34x |
| unique | 4096 | 0.496356 | 0.005504 | 0.01x–0.04x |
| zeros | 4096 | 0.013864 | 0.004142 | 0.30x–1.15x |

#### i64→f64 promote-out

| op | largest n | NumPy (ms) | MatLua Lua (ms) | Lua/NumPy, all n |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 4096 | 719.452 | 598.449 | 0.83x–2.15x |
| mean | 4096 | 24.156 | 6.2096 | 0.12x–0.46x |
| median | 4096 | 116.299 | 88.443 | 0.34x–0.86x |
| norm | 4096 | 1.4186 | 2.6547 | 1.87x–5.61x |
| qr | 4096 | 3296.766 | 2467.666 | 0.54x–2.16x |
| quantile | 4096 | 150.432 | 89.412 | 0.10x–0.59x |
| solve | 4096 | 552.132 | 2502.025 | 0.84x–4.53x |
| std | 4096 | 141.965 | 23.414 | 0.16x–0.46x |

### Appendix — full three-face tables

<details>
<summary>Table A — f64 absolute (ms)</summary>

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000545 | 0.000146 | 0.000287 |
| arange | 256 | 0.000717 | 0.000212 | 0.000366 |
| arange | 1024 | 0.001131 | 0.000492 | 0.000702 |
| arange | 4096 | 0.003249 | 0.001516 | 0.001751 |
| cholesky | 64 | 0.021525 | 0.030329 | 0.053270 |
| cholesky | 256 | 0.856383 | 1.0424 | 0.962927 |
| cholesky | 1024 | 17.705 | 16.872 | 16.128 |
| cholesky | 4096 | 632.894 | 622.775 | 514.529 |
| copy | 64 | 0.001339 | 0.000992 | 0.001137 |
| copy | 256 | 0.014234 | 0.014735 | 0.017677 |
| copy | 1024 | 0.671173 | 0.651097 | 0.677148 |
| copy | 4096 | 77.579 | 21.563 | 21.916 |
| dot | 64 | 0.000672 | 0.000084 | 0.000323 |
| dot | 256 | 0.000754 | 0.000122 | 0.000385 |
| dot | 1024 | 0.000801 | 0.000395 | 0.000670 |
| dot | 4096 | 0.001445 | 0.001489 | 0.001742 |
| elem_add | 64 | 0.001613 | 0.001347 | 0.001517 |
| elem_add | 256 | 0.042301 | 0.032825 | 0.027623 |
| elem_add | 1024 | 1.2774 | 0.970619 | 1.0041 |
| elem_add | 4096 | 69.209 | 33.127 | 33.816 |
| elem_add_scalar | 64 | 0.001651 | 0.001136 | 0.001211 |
| elem_add_scalar | 256 | 0.017854 | 0.016327 | 0.020388 |
| elem_add_scalar | 1024 | 0.672020 | 0.676536 | 0.688465 |
| elem_add_scalar | 4096 | 63.065 | 24.520 | 25.902 |
| elem_div | 64 | 0.003445 | 0.003184 | 0.003135 |
| elem_div | 256 | 0.046502 | 0.045494 | 0.046034 |
| elem_div | 1024 | 1.3597 | 1.0054 | 0.992856 |
| elem_div | 4096 | 70.139 | 32.181 | 34.561 |
| elem_mul | 64 | 0.001613 | 0.001521 | 0.001512 |
| elem_mul | 256 | 0.042090 | 0.026492 | 0.028666 |
| elem_mul | 1024 | 1.3614 | 0.967126 | 1.0098 |
| elem_mul | 4096 | 69.174 | 31.568 | 34.018 |
| elem_sub | 64 | 0.001645 | 0.001543 | 0.001500 |
| elem_sub | 256 | 0.041234 | 0.033902 | 0.031331 |
| elem_sub | 1024 | 1.3538 | 0.960759 | 0.983036 |
| elem_sub | 4096 | 68.967 | 31.164 | 34.096 |
| eye | 64 | 0.002244 | 0.000318 | 0.000687 |
| eye | 256 | 0.014906 | 0.011697 | 0.011658 |
| eye | 1024 | 0.368798 | 0.349846 | 0.395196 |
| eye | 4096 | 5.1719 | 5.3696 | 6.8946 |
| fill | 64 | 0.001387 | 0.000739 | 0.001001 |
| fill | 256 | 0.017571 | 0.012234 | 0.011767 |
| fill | 1024 | 0.358535 | 0.362907 | 0.383366 |
| fill | 4096 | 16.938 | 11.466 | 19.195 |
| full | 64 | 0.002188 | 0.000916 | 0.000941 |
| full | 256 | 0.018307 | 0.012324 | 0.013966 |
| full | 1024 | 0.367064 | 0.361521 | 0.392622 |
| full | 4096 | 58.340 | 19.110 | 19.910 |
| matmul | 64 | 0.008455 | 0.010428 | 0.009732 |
| matmul | 256 | 0.404162 | 0.225272 | 0.247579 |
| matmul | 1024 | 8.4589 | 10.733 | 9.8048 |
| matmul | 4096 | 492.283 | 644.249 | 632.257 |
| max | 64 | 0.001689 | 0.001912 | 0.002011 |
| max | 256 | 0.011664 | 0.028955 | 0.029183 |
| max | 1024 | 0.330711 | 0.553074 | 0.558945 |
| max | 4096 | 5.2721 | 10.662 | 23.767 |
| mean | 64 | 0.003649 | 0.000759 | 0.001017 |
| mean | 256 | 0.027865 | 0.011377 | 0.011539 |
| mean | 1024 | 0.335429 | 0.375463 | 0.329793 |
| mean | 4096 | 6.6606 | 5.3440 | 14.285 |
| min | 64 | 0.001679 | 0.001901 | 0.002122 |
| min | 256 | 0.011691 | 0.028928 | 0.029120 |
| min | 1024 | 0.338688 | 0.545058 | 0.538468 |
| min | 4096 | 5.1840 | 10.329 | 23.706 |
| norm | 64 | 0.002033 | 0.020048 | 0.012384 |
| norm | 256 | 0.005565 | 0.022613 | 0.028415 |
| norm | 1024 | 0.077604 | 0.361959 | 0.368749 |
| norm | 4096 | 1.4093 | 2.1867 | 2.5073 |
| ones | 64 | 0.002290 | 0.000891 | 0.000828 |
| ones | 256 | 0.018457 | 0.012302 | 0.013525 |
| ones | 1024 | 0.372435 | 0.447913 | 0.420699 |
| ones | 4096 | 59.358 | 18.999 | 20.109 |
| qr | 64 | 0.123038 | 0.567248 | 0.235020 |
| qr | 256 | 4.6729 | 3.3158 | 2.2167 |
| qr | 1024 | 99.754 | 42.125 | 30.729 |
| qr | 4096 | 3382.037 | 1960.189 | 2039.319 |
| reshape | 64 | 0.000251 | 0.000082 | 0.000287 |
| reshape | 256 | 0.000262 | 0.000084 | 0.000298 |
| reshape | 1024 | 0.000484 | 0.000084 | 0.000348 |
| reshape | 4096 | 0.000272 | 0.000087 | 0.000540 |
| solve | 64 | 0.032229 | 0.031091 | 0.028498 |
| solve | 256 | 0.574141 | 2.2612 | 1.7235 |
| solve | 1024 | 13.048 | 31.580 | 29.910 |
| solve | 4096 | 459.024 | 602.417 | 612.740 |
| sum | 64 | 0.002056 | 0.000754 | 0.001025 |
| sum | 256 | 0.021732 | 0.011356 | 0.011541 |
| sum | 1024 | 0.380001 | 0.339415 | 0.361196 |
| sum | 4096 | 13.598 | 6.4198 | 12.311 |
| svd | 64 | 0.285096 | 0.587796 | 0.877986 |
| svd | 256 | 8.9552 | 13.696 | 18.931 |
| svd | 1024 | 270.712 | 374.720 | 618.865 |
| svd | 4096 | 9280.692 | 13039.417 | 13228.578 |
| transpose | 64 | 0.002426 | 0.005845 | 0.003510 |
| transpose | 256 | 0.079090 | 0.051892 | 0.053268 |
| transpose | 1024 | 4.8675 | 2.0538 | 1.9817 |
| transpose | 4096 | 257.495 | 62.106 | 53.758 |
| zeros | 64 | 0.000724 | 0.000331 | 0.000732 |
| zeros | 256 | 0.011592 | 0.002808 | 0.206088 |
| zeros | 1024 | 0.353644 | 0.003427 | 0.553319 |
| zeros | 4096 | 0.013153 | 0.019904 | 0.005110 |

</details>

<details>
<summary>Table B — f64 relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.27x | 0.53x | 1.97x |
| arange | 256 | 1.00x | 0.30x | 0.51x | 1.73x |
| arange | 1024 | 1.00x | 0.44x | 0.62x | 1.43x |
| arange | 4096 | 1.00x | 0.47x | 0.54x | 1.16x |
| cholesky | 64 | 1.00x | 1.41x | 2.47x | 1.76x |
| cholesky | 256 | 1.00x | 1.22x | 1.12x | 0.92x |
| cholesky | 1024 | 1.00x | 0.95x | 0.91x | 0.96x |
| cholesky | 4096 | 1.00x | 0.98x | 0.81x | 0.83x |
| copy | 64 | 1.00x | 0.74x | 0.85x | 1.15x |
| copy | 256 | 1.00x | 1.04x | 1.24x | 1.20x |
| copy | 1024 | 1.00x | 0.97x | 1.01x | 1.04x |
| copy | 4096 | 1.00x | 0.28x | 0.28x | 1.02x |
| dot | 64 | 1.00x | 0.12x | 0.48x | 3.85x |
| dot | 256 | 1.00x | 0.16x | 0.51x | 3.16x |
| dot | 1024 | 1.00x | 0.49x | 0.84x | 1.70x |
| dot | 4096 | 1.00x | 1.03x | 1.21x | 1.17x |
| elem_add | 64 | 1.00x | 0.84x | 0.94x | 1.13x |
| elem_add | 256 | 1.00x | 0.78x | 0.65x | 0.84x |
| elem_add | 1024 | 1.00x | 0.76x | 0.79x | 1.03x |
| elem_add | 4096 | 1.00x | 0.48x | 0.49x | 1.02x |
| elem_add_scalar | 64 | 1.00x | 0.69x | 0.73x | 1.07x |
| elem_add_scalar | 256 | 1.00x | 0.91x | 1.14x | 1.25x |
| elem_add_scalar | 1024 | 1.00x | 1.01x | 1.02x | 1.02x |
| elem_add_scalar | 4096 | 1.00x | 0.39x | 0.41x | 1.06x |
| elem_div | 64 | 1.00x | 0.92x | 0.91x | 0.98x |
| elem_div | 256 | 1.00x | 0.98x | 0.99x | 1.01x |
| elem_div | 1024 | 1.00x | 0.74x | 0.73x | 0.99x |
| elem_div | 4096 | 1.00x | 0.46x | 0.49x | 1.07x |
| elem_mul | 64 | 1.00x | 0.94x | 0.94x | 0.99x |
| elem_mul | 256 | 1.00x | 0.63x | 0.68x | 1.08x |
| elem_mul | 1024 | 1.00x | 0.71x | 0.74x | 1.04x |
| elem_mul | 4096 | 1.00x | 0.46x | 0.49x | 1.08x |
| elem_sub | 64 | 1.00x | 0.94x | 0.91x | 0.97x |
| elem_sub | 256 | 1.00x | 0.82x | 0.76x | 0.92x |
| elem_sub | 1024 | 1.00x | 0.71x | 0.73x | 1.02x |
| elem_sub | 4096 | 1.00x | 0.45x | 0.49x | 1.09x |
| eye | 64 | 1.00x | 0.14x | 0.31x | 2.16x |
| eye | 256 | 1.00x | 0.78x | 0.78x | 1.00x |
| eye | 1024 | 1.00x | 0.95x | 1.07x | 1.13x |
| eye | 4096 | 1.00x | 1.04x | 1.33x | 1.28x |
| fill | 64 | 1.00x | 0.53x | 0.72x | 1.35x |
| fill | 256 | 1.00x | 0.70x | 0.67x | 0.96x |
| fill | 1024 | 1.00x | 1.01x | 1.07x | 1.06x |
| fill | 4096 | 1.00x | 0.68x | 1.13x | 1.67x |
| full | 64 | 1.00x | 0.42x | 0.43x | 1.03x |
| full | 256 | 1.00x | 0.67x | 0.76x | 1.13x |
| full | 1024 | 1.00x | 0.98x | 1.07x | 1.09x |
| full | 4096 | 1.00x | 0.33x | 0.34x | 1.04x |
| matmul | 64 | 1.00x | 1.23x | 1.15x | 0.93x |
| matmul | 256 | 1.00x | 0.56x | 0.61x | 1.10x |
| matmul | 1024 | 1.00x | 1.27x | 1.16x | 0.91x |
| matmul | 4096 | 1.00x | 1.31x | 1.28x | 0.98x |
| max | 64 | 1.00x | 1.13x | 1.19x | 1.05x |
| max | 256 | 1.00x | 2.48x | 2.50x | 1.01x |
| max | 1024 | 1.00x | 1.67x | 1.69x | 1.01x |
| max | 4096 | 1.00x | 2.02x | 4.51x | 2.23x |
| mean | 64 | 1.00x | 0.21x | 0.28x | 1.34x |
| mean | 256 | 1.00x | 0.41x | 0.41x | 1.01x |
| mean | 1024 | 1.00x | 1.12x | 0.98x | 0.88x |
| mean | 4096 | 1.00x | 0.80x | 2.14x | 2.67x |
| min | 64 | 1.00x | 1.13x | 1.26x | 1.12x |
| min | 256 | 1.00x | 2.47x | 2.49x | 1.01x |
| min | 1024 | 1.00x | 1.61x | 1.59x | 0.99x |
| min | 4096 | 1.00x | 1.99x | 4.57x | 2.30x |
| norm | 64 | 1.00x | 9.86x | 6.09x | 0.62x |
| norm | 256 | 1.00x | 4.06x | 5.11x | 1.26x |
| norm | 1024 | 1.00x | 4.66x | 4.75x | 1.02x |
| norm | 4096 | 1.00x | 1.55x | 1.78x | 1.15x |
| ones | 64 | 1.00x | 0.39x | 0.36x | 0.93x |
| ones | 256 | 1.00x | 0.67x | 0.73x | 1.10x |
| ones | 1024 | 1.00x | 1.20x | 1.13x | 0.94x |
| ones | 4096 | 1.00x | 0.32x | 0.34x | 1.06x |
| qr | 64 | 1.00x | 4.61x | 1.91x | 0.41x |
| qr | 256 | 1.00x | 0.71x | 0.47x | 0.67x |
| qr | 1024 | 1.00x | 0.42x | 0.31x | 0.73x |
| qr | 4096 | 1.00x | 0.58x | 0.60x | 1.04x |
| reshape | 64 | 1.00x | 0.33x | 1.14x | 3.50x |
| reshape | 256 | 1.00x | 0.32x | 1.14x | 3.55x |
| reshape | 1024 | 1.00x | 0.17x | 0.72x | 4.14x |
| reshape | 4096 | 1.00x | 0.32x | 1.99x | 6.21x |
| solve | 64 | 1.00x | 0.96x | 0.88x | 0.92x |
| solve | 256 | 1.00x | 3.94x | 3.00x | 0.76x |
| solve | 1024 | 1.00x | 2.42x | 2.29x | 0.95x |
| solve | 4096 | 1.00x | 1.31x | 1.33x | 1.02x |
| sum | 64 | 1.00x | 0.37x | 0.50x | 1.36x |
| sum | 256 | 1.00x | 0.52x | 0.53x | 1.02x |
| sum | 1024 | 1.00x | 0.89x | 0.95x | 1.06x |
| sum | 4096 | 1.00x | 0.47x | 0.91x | 1.92x |
| svd | 64 | 1.00x | 2.06x | 3.08x | 1.49x |
| svd | 256 | 1.00x | 1.53x | 2.11x | 1.38x |
| svd | 1024 | 1.00x | 1.38x | 2.29x | 1.65x |
| svd | 4096 | 1.00x | 1.41x | 1.43x | 1.01x |
| transpose | 64 | 1.00x | 2.41x | 1.45x | 0.60x |
| transpose | 256 | 1.00x | 0.66x | 0.67x | 1.03x |
| transpose | 1024 | 1.00x | 0.42x | 0.41x | 0.96x |
| transpose | 4096 | 1.00x | 0.24x | 0.21x | 0.87x |
| zeros | 64 | 1.00x | 0.46x | 1.01x | 2.21x |
| zeros | 256 | 1.00x | 0.24x | 17.78x | 73.39x |
| zeros | 1024 | 1.00x | 0.01x | 1.56x | 161.46x |
| zeros | 4096 | 1.00x | 1.51x | 0.39x | 0.26x |

</details>

<details>
<summary>Table C — i64 absolute (ms) — matmul* NumPy column is f64 BLAS on integer-valued data</summary>

| op | n | NumPy int64 (ms) | MatLua Rust i64 (ms) | MatLua Lua i64 (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000461 | 0.000139 | 0.000273 |
| arange | 256 | 0.000600 | 0.000195 | 0.000328 |
| arange | 1024 | 0.000920 | 0.000604 | 0.000608 |
| arange | 4096 | 0.002054 | 0.001337 | 0.001572 |
| copy | 64 | 0.001306 | 0.001001 | 0.001143 |
| copy | 256 | 0.014779 | 0.013614 | 0.015335 |
| copy | 1024 | 0.652091 | 0.679759 | 0.658632 |
| copy | 4096 | 79.876 | 77.787 | 74.436 |
| dot | 64 | 0.000943 | 0.000056 | 0.000282 |
| dot | 256 | 0.001000 | 0.000122 | 0.000374 |
| dot | 1024 | 0.001420 | 0.000393 | 0.000710 |
| dot | 4096 | 0.003410 | 0.001468 | 0.001736 |
| elem_add | 64 | 0.001835 | 0.001253 | 0.001505 |
| elem_add | 256 | 0.024976 | 0.019898 | 0.021304 |
| elem_add | 1024 | 0.980067 | 0.985936 | 1.0009 |
| elem_add | 4096 | 70.111 | 70.946 | 66.954 |
| elem_div | 64 | 0.016448 | 0.008651 | 0.008855 |
| elem_div | 256 | 0.232796 | 0.137425 | 0.148192 |
| elem_div | 1024 | 3.7943 | 2.2893 | 2.3267 |
| elem_div | 4096 | 116.326 | 97.081 | 94.776 |
| elem_mul | 64 | 0.002129 | 0.001350 | 0.001543 |
| elem_mul | 256 | 0.030670 | 0.020597 | 0.020092 |
| elem_mul | 1024 | 0.957279 | 0.997317 | 0.967783 |
| elem_mul | 4096 | 75.372 | 70.911 | 67.051 |
| elem_sub | 64 | 0.001828 | 0.001254 | 0.001583 |
| elem_sub | 256 | 0.024525 | 0.019858 | 0.022057 |
| elem_sub | 1024 | 1.0026 | 0.995735 | 0.973996 |
| elem_sub | 4096 | 69.687 | 69.841 | 67.605 |
| eye | 64 | 0.002105 | 0.000312 | 0.000656 |
| eye | 256 | 0.014970 | 0.011663 | 0.012560 |
| eye | 1024 | 0.371778 | 0.394631 | 0.338091 |
| eye | 4096 | 5.2134 | 5.2386 | 5.2244 |
| fill | 64 | 0.001388 | 0.000394 | 0.000954 |
| fill | 256 | 0.017527 | 0.012255 | 0.012214 |
| fill | 1024 | 0.411269 | 0.365602 | 0.351505 |
| fill | 4096 | 18.280 | 11.939 | 18.181 |
| full | 64 | 0.002199 | 0.000893 | 0.000937 |
| full | 256 | 0.018484 | 0.012199 | 0.012569 |
| full | 1024 | 0.382611 | 0.399474 | 0.356280 |
| full | 4096 | 58.558 | 57.572 | 55.854 |
| isin | 64 | 0.020268 | 0.004547 | 0.005578 |
| isin | 256 | 0.083647 | 0.068918 | 0.083921 |
| isin | 1024 | 1.3171 | 1.1538 | 1.4033 |
| isin | 4096 | 53.461 | 85.282 | 31.865 |
| matmul | 64 | 0.008252 | 0.077845 | 0.085327 |
| matmul | 256 | 0.320464 | 2.1503 | 2.1581 |
| matmul | 1024 | 8.0091 | 59.946 | 58.061 |
| matmul | 4096 | 495.045 | 3510.889 | 3455.757 |
| matmul_at | 64 | 0.008386 | 0.077899 | 0.086062 |
| matmul_at | 256 | 0.341699 | 2.3282 | 2.5240 |
| matmul_at | 1024 | 8.2987 | 56.298 | 55.609 |
| matmul_at | 4096 | 489.889 | 3509.282 | 3487.396 |
| matmul_bt | 64 | 0.011870 | 0.072823 | 0.084551 |
| matmul_bt | 256 | 0.334560 | 2.2344 | 2.4176 |
| matmul_bt | 1024 | 7.9386 | 62.100 | 56.183 |
| matmul_bt | 4096 | 490.219 | 3608.530 | 3499.905 |
| max | 64 | 0.001773 | 0.000967 | 0.001127 |
| max | 256 | 0.008288 | 0.015421 | 0.015638 |
| max | 1024 | 0.313737 | 0.408237 | 0.389988 |
| max | 4096 | 5.1775 | 6.5892 | 22.703 |
| min | 64 | 0.001739 | 0.000966 | 0.001148 |
| min | 256 | 0.008248 | 0.015501 | 0.015598 |
| min | 1024 | 0.305924 | 0.390296 | 0.400855 |
| min | 4096 | 5.2345 | 8.3203 | 21.489 |
| ones | 64 | 0.002209 | 0.000895 | 0.000945 |
| ones | 256 | 0.018383 | 0.012483 | 0.012794 |
| ones | 1024 | 0.392591 | 0.360588 | 0.352568 |
| ones | 4096 | 58.760 | 58.422 | 59.028 |
| reshape | 64 | 0.000261 | 0.000083 | 0.000300 |
| reshape | 256 | 0.000299 | 0.000085 | 0.000317 |
| reshape | 1024 | 0.000314 | 0.000083 | 0.000330 |
| reshape | 4096 | 0.000554 | 0.000100 | 0.000374 |
| sum | 64 | 0.001917 | 0.000348 | 0.000672 |
| sum | 256 | 0.010190 | 0.007959 | 0.008201 |
| sum | 1024 | 0.313053 | 0.316721 | 0.303181 |
| sum | 4096 | 9.3114 | 8.1825 | 12.307 |
| transpose | 64 | 0.002613 | 0.003285 | 0.003498 |
| transpose | 256 | 0.079455 | 0.050829 | 0.050975 |
| transpose | 1024 | 4.8158 | 2.0912 | 2.0589 |
| transpose | 4096 | 263.229 | 98.502 | 94.117 |
| unique | 64 | 0.006903 | 0.000174 | 0.000300 |
| unique | 256 | 0.021452 | 0.000389 | 0.000606 |
| unique | 1024 | 0.092979 | 0.001289 | 0.001532 |
| unique | 4096 | 0.496356 | 0.003935 | 0.005504 |
| zeros | 64 | 0.000705 | 0.000304 | 0.000720 |
| zeros | 256 | 0.011649 | 0.011555 | 0.012230 |
| zeros | 1024 | 0.333528 | 0.360924 | 0.383528 |
| zeros | 4096 | 0.013864 | 0.007106 | 0.004142 |

</details>

<details>
<summary>Table D — i64 relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.30x | 0.59x | 1.96x |
| arange | 256 | 1.00x | 0.33x | 0.55x | 1.68x |
| arange | 1024 | 1.00x | 0.66x | 0.66x | 1.01x |
| arange | 4096 | 1.00x | 0.65x | 0.77x | 1.18x |
| copy | 64 | 1.00x | 0.77x | 0.88x | 1.14x |
| copy | 256 | 1.00x | 0.92x | 1.04x | 1.13x |
| copy | 1024 | 1.00x | 1.04x | 1.01x | 0.97x |
| copy | 4096 | 1.00x | 0.97x | 0.93x | 0.96x |
| dot | 64 | 1.00x | 0.06x | 0.30x | 5.04x |
| dot | 256 | 1.00x | 0.12x | 0.37x | 3.07x |
| dot | 1024 | 1.00x | 0.28x | 0.50x | 1.81x |
| dot | 4096 | 1.00x | 0.43x | 0.51x | 1.18x |
| elem_add | 64 | 1.00x | 0.68x | 0.82x | 1.20x |
| elem_add | 256 | 1.00x | 0.80x | 0.85x | 1.07x |
| elem_add | 1024 | 1.00x | 1.01x | 1.02x | 1.02x |
| elem_add | 4096 | 1.00x | 1.01x | 0.95x | 0.94x |
| elem_div | 64 | 1.00x | 0.53x | 0.54x | 1.02x |
| elem_div | 256 | 1.00x | 0.59x | 0.64x | 1.08x |
| elem_div | 1024 | 1.00x | 0.60x | 0.61x | 1.02x |
| elem_div | 4096 | 1.00x | 0.83x | 0.81x | 0.98x |
| elem_mul | 64 | 1.00x | 0.63x | 0.72x | 1.14x |
| elem_mul | 256 | 1.00x | 0.67x | 0.66x | 0.98x |
| elem_mul | 1024 | 1.00x | 1.04x | 1.01x | 0.97x |
| elem_mul | 4096 | 1.00x | 0.94x | 0.89x | 0.95x |
| elem_sub | 64 | 1.00x | 0.69x | 0.87x | 1.26x |
| elem_sub | 256 | 1.00x | 0.81x | 0.90x | 1.11x |
| elem_sub | 1024 | 1.00x | 0.99x | 0.97x | 0.98x |
| elem_sub | 4096 | 1.00x | 1.00x | 0.97x | 0.97x |
| eye | 64 | 1.00x | 0.15x | 0.31x | 2.10x |
| eye | 256 | 1.00x | 0.78x | 0.84x | 1.08x |
| eye | 1024 | 1.00x | 1.06x | 0.91x | 0.86x |
| eye | 4096 | 1.00x | 1.00x | 1.00x | 1.00x |
| fill | 64 | 1.00x | 0.28x | 0.69x | 2.42x |
| fill | 256 | 1.00x | 0.70x | 0.70x | 1.00x |
| fill | 1024 | 1.00x | 0.89x | 0.85x | 0.96x |
| fill | 4096 | 1.00x | 0.65x | 0.99x | 1.52x |
| full | 64 | 1.00x | 0.41x | 0.43x | 1.05x |
| full | 256 | 1.00x | 0.66x | 0.68x | 1.03x |
| full | 1024 | 1.00x | 1.04x | 0.93x | 0.89x |
| full | 4096 | 1.00x | 0.98x | 0.95x | 0.97x |
| isin | 64 | 1.00x | 0.22x | 0.28x | 1.23x |
| isin | 256 | 1.00x | 0.82x | 1.00x | 1.22x |
| isin | 1024 | 1.00x | 0.88x | 1.07x | 1.22x |
| isin | 4096 | 1.00x | 1.60x | 0.60x | 0.37x |
| matmul | 64 | 1.00x | 9.43x | 10.34x | 1.10x |
| matmul | 256 | 1.00x | 6.71x | 6.73x | 1.00x |
| matmul | 1024 | 1.00x | 7.48x | 7.25x | 0.97x |
| matmul | 4096 | 1.00x | 7.09x | 6.98x | 0.98x |
| matmul_at | 64 | 1.00x | 9.29x | 10.26x | 1.10x |
| matmul_at | 256 | 1.00x | 6.81x | 7.39x | 1.08x |
| matmul_at | 1024 | 1.00x | 6.78x | 6.70x | 0.99x |
| matmul_at | 4096 | 1.00x | 7.16x | 7.12x | 0.99x |
| matmul_bt | 64 | 1.00x | 6.14x | 7.12x | 1.16x |
| matmul_bt | 256 | 1.00x | 6.68x | 7.23x | 1.08x |
| matmul_bt | 1024 | 1.00x | 7.82x | 7.08x | 0.90x |
| matmul_bt | 4096 | 1.00x | 7.36x | 7.14x | 0.97x |
| max | 64 | 1.00x | 0.55x | 0.64x | 1.17x |
| max | 256 | 1.00x | 1.86x | 1.89x | 1.01x |
| max | 1024 | 1.00x | 1.30x | 1.24x | 0.96x |
| max | 4096 | 1.00x | 1.27x | 4.38x | 3.45x |
| min | 64 | 1.00x | 0.56x | 0.66x | 1.19x |
| min | 256 | 1.00x | 1.88x | 1.89x | 1.01x |
| min | 1024 | 1.00x | 1.28x | 1.31x | 1.03x |
| min | 4096 | 1.00x | 1.59x | 4.11x | 2.58x |
| ones | 64 | 1.00x | 0.41x | 0.43x | 1.06x |
| ones | 256 | 1.00x | 0.68x | 0.70x | 1.02x |
| ones | 1024 | 1.00x | 0.92x | 0.90x | 0.98x |
| ones | 4096 | 1.00x | 0.99x | 1.00x | 1.01x |
| reshape | 64 | 1.00x | 0.32x | 1.15x | 3.61x |
| reshape | 256 | 1.00x | 0.28x | 1.06x | 3.73x |
| reshape | 1024 | 1.00x | 0.26x | 1.05x | 3.98x |
| reshape | 4096 | 1.00x | 0.18x | 0.68x | 3.74x |
| sum | 64 | 1.00x | 0.18x | 0.35x | 1.93x |
| sum | 256 | 1.00x | 0.78x | 0.80x | 1.03x |
| sum | 1024 | 1.00x | 1.01x | 0.97x | 0.96x |
| sum | 4096 | 1.00x | 0.88x | 1.32x | 1.50x |
| transpose | 64 | 1.00x | 1.26x | 1.34x | 1.06x |
| transpose | 256 | 1.00x | 0.64x | 0.64x | 1.00x |
| transpose | 1024 | 1.00x | 0.43x | 0.43x | 0.98x |
| transpose | 4096 | 1.00x | 0.37x | 0.36x | 0.96x |
| unique | 64 | 1.00x | 0.03x | 0.04x | 1.72x |
| unique | 256 | 1.00x | 0.02x | 0.03x | 1.56x |
| unique | 1024 | 1.00x | 0.01x | 0.02x | 1.19x |
| unique | 4096 | 1.00x | 0.01x | 0.01x | 1.40x |
| zeros | 64 | 1.00x | 0.43x | 1.02x | 2.37x |
| zeros | 256 | 1.00x | 0.99x | 1.05x | 1.06x |
| zeros | 1024 | 1.00x | 1.08x | 1.15x | 1.06x |
| zeros | 4096 | 1.00x | 0.51x | 0.30x | 0.58x |

</details>

<details>
<summary>Table E — i64→f64 promote-out absolute (ms)</summary>

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 64 | 0.021068 | 0.019298 | 0.045334 |
| cholesky | 256 | 0.939571 | 1.0404 | 0.897274 |
| cholesky | 1024 | 22.049 | 22.097 | 18.570 |
| cholesky | 4096 | 719.452 | 521.215 | 598.449 |
| mean | 64 | 0.005758 | 0.000356 | 0.000663 |
| mean | 256 | 0.046973 | 0.007983 | 0.008173 |
| mean | 1024 | 0.758283 | 0.309958 | 0.346480 |
| mean | 4096 | 24.156 | 8.7339 | 6.2096 |
| median | 64 | 0.014395 | 0.003497 | 0.004914 |
| median | 256 | 0.091657 | 0.161145 | 0.055898 |
| median | 1024 | 1.6599 | 1.3993 | 1.4281 |
| median | 4096 | 116.299 | 90.967 | 88.443 |
| norm | 64 | 0.002948 | 0.015227 | 0.014299 |
| norm | 256 | 0.005351 | 0.032632 | 0.030044 |
| norm | 1024 | 0.090173 | 0.413774 | 0.418968 |
| norm | 4096 | 1.4186 | 2.9409 | 2.6547 |
| qr | 64 | 0.135458 | 0.677416 | 0.292645 |
| qr | 256 | 4.7488 | 5.2114 | 2.9900 |
| qr | 1024 | 101.638 | 72.777 | 55.092 |
| qr | 4096 | 3296.766 | 3413.153 | 2467.666 |
| quantile | 64 | 0.049557 | 0.003483 | 0.004805 |
| quantile | 256 | 0.210562 | 0.185513 | 0.055536 |
| quantile | 1024 | 2.7996 | 1.2675 | 1.2977 |
| quantile | 4096 | 150.432 | 90.138 | 89.412 |
| solve | 64 | 0.036154 | 0.028923 | 0.030329 |
| solve | 256 | 0.590052 | 2.1490 | 1.7553 |
| solve | 1024 | 16.254 | 29.104 | 34.644 |
| solve | 4096 | 552.132 | 2153.099 | 2502.025 |
| std | 64 | 0.021104 | 0.003198 | 0.003488 |
| std | 256 | 0.132537 | 0.053199 | 0.053420 |
| std | 1024 | 2.4073 | 1.0963 | 1.1031 |
| std | 4096 | 141.965 | 21.144 | 23.414 |

</details>

<details>
<summary>Table F — i64→f64 promote-out relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky | 64 | 1.00x | 0.92x | 2.15x | 2.35x |
| cholesky | 256 | 1.00x | 1.11x | 0.95x | 0.86x |
| cholesky | 1024 | 1.00x | 1.00x | 0.84x | 0.84x |
| cholesky | 4096 | 1.00x | 0.72x | 0.83x | 1.15x |
| mean | 64 | 1.00x | 0.06x | 0.12x | 1.86x |
| mean | 256 | 1.00x | 0.17x | 0.17x | 1.02x |
| mean | 1024 | 1.00x | 0.41x | 0.46x | 1.12x |
| mean | 4096 | 1.00x | 0.36x | 0.26x | 0.71x |
| median | 64 | 1.00x | 0.24x | 0.34x | 1.41x |
| median | 256 | 1.00x | 1.76x | 0.61x | 0.35x |
| median | 1024 | 1.00x | 0.84x | 0.86x | 1.02x |
| median | 4096 | 1.00x | 0.78x | 0.76x | 0.97x |
| norm | 64 | 1.00x | 5.17x | 4.85x | 0.94x |
| norm | 256 | 1.00x | 6.10x | 5.61x | 0.92x |
| norm | 1024 | 1.00x | 4.59x | 4.65x | 1.01x |
| norm | 4096 | 1.00x | 2.07x | 1.87x | 0.90x |
| qr | 64 | 1.00x | 5.00x | 2.16x | 0.43x |
| qr | 256 | 1.00x | 1.10x | 0.63x | 0.57x |
| qr | 1024 | 1.00x | 0.72x | 0.54x | 0.76x |
| qr | 4096 | 1.00x | 1.04x | 0.75x | 0.72x |
| quantile | 64 | 1.00x | 0.07x | 0.10x | 1.38x |
| quantile | 256 | 1.00x | 0.88x | 0.26x | 0.30x |
| quantile | 1024 | 1.00x | 0.45x | 0.46x | 1.02x |
| quantile | 4096 | 1.00x | 0.60x | 0.59x | 0.99x |
| solve | 64 | 1.00x | 0.80x | 0.84x | 1.05x |
| solve | 256 | 1.00x | 3.64x | 2.97x | 0.82x |
| solve | 1024 | 1.00x | 1.79x | 2.13x | 1.19x |
| solve | 4096 | 1.00x | 3.90x | 4.53x | 1.16x |
| std | 64 | 1.00x | 0.15x | 0.17x | 1.09x |
| std | 256 | 1.00x | 0.40x | 0.40x | 1.00x |
| std | 1024 | 1.00x | 0.46x | 0.46x | 1.01x |
| std | 4096 | 1.00x | 0.15x | 0.16x | 1.11x |

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
