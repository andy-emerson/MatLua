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

**M7.c plan (durable):** keep exact i64 matmul (plan A). M7.c work is
complete on the optimization branch; closure is a Human ruling (DESIGN
§7.1.2), not something publishing these tables does.

**What the clock includes:** both MatLua faces and NumPy time the same call
on identical inputs — the Lua face's globals are byte-copies of the
Rust-face arrays. Promote ops (i64 → f64 results) pay the conversion inside
the clock on every face, NumPy included. One asymmetry is inherent: Lua
cells include Lua's allocator/GC interaction for result userdata (that *is*
the user-POV cost of the Lua face), while Rust cells see only Rust's
allocator.

### Yardsticks

- **NumPy is the product bar** (= 1.00x): the product is the Lua face, so
  the summary tables read Lua vs NumPy; the Rust face is developer
  diagnostics in the appendix.
- **i64 matmul family** (`matmul` / `matmul_at` / `matmul_bt`): the NumPy
  reference is **float64 BLAS** on the same integer-valued inputs (not
  `int64@int64` — NumPy has no integer BLAS, so that fallback is not a
  product bar). MatLua results are **always exact**; the runtime picks among
  three bit-identical kernel tiers on a derived range scan (DESIGN §7.1.2,
  2026-08-04 rulings), and the tables show one row per tier:
  - **Headline rows** use range-safe data (`k·max|A|·max|B| ≤ 2⁵³`, here
    |v| ≤ 1000): the guarded **f64-promote** tier — and the NumPy f64
    reference is itself exact there.
  - **`*_wide` rows** use ramps beyond 2⁵³ intermediates but inside i32:
    the **i32-pack widening** tier (32×32→64 products, exact for any k —
    bake-off winner 2026-08-04). NumPy's f64 reference silently rounds
    here (timing bar only).
  - **`*_huge` rows** use values ~10¹² (beyond i32): the exact wrapping
    **i64 GEBP**. NumPy's f64 reference rounds here too (timing bar only).
- **Machine roofline** (engineering yardstick): `i64_roofline` measures the
  running host's achievable wrapping i64 multiply-add throughput, so i64
  GEMM is also judged as **% of machine ceiling** — the BLAS ratio alone
  mixes kernel quality with ISA physics (no 64-bit vector multiply below
  AVX-512DQ). See the Roofline section.

### Provenance

Every table names the host that produced it; all faces of one table come
from one host and one session. Run-to-run noise on shared cloud hosts is
real (±10–20% observed); treat small deltas accordingly. (An earlier
revision blamed Rust/Lua spreads on "contention"; the real cause was the
harness feeding the faces different data — fixed, both faces now measure
identical inputs, and the attribution is retracted.)

**Calibration gate:** before publishing a refresh, run `i64_roofline` and
compare the register-resident tile ceilings against the Roofline section's
recorded values for the host. If they deviate beyond noise (±20%), the
session is unfit: cells would be honestly measured but unrepresentative —
the observed degradation is not a uniform slowdown (512-bit throughput has
dropped ~4× while scalar stayed normal), so even within-run ratios skew.
**Discarded runs are documented, not hidden** (Human ruling 2026-08-04):
every discard is logged below with its gate readings and suspected cause,
so the measurement problem stays visible and fixable instead of silently
forgotten. Do not publish the cells themselves — they describe the wrong
machine.

Discard log:

| Date (UTC) | Gate reading vs baseline (isa_tile_par 49.2 Gops / gemm 59.0 ms) | Suspected cause |
|---|---|---|
| 2026-08-03 (evening) | register-resident compute ~3× degraded across all kernels | hypervisor co-tenancy; no in-container load |
| 2026-08-03 (later) | AVX-512 tile ~4× degraded, scalar normal | dynamic 512-bit downclocking (CPUID still reports the features) |
| 2026-08-04 02:19 | gemm 76.5 ms (+30%), isa_tile_par 40.3 (−18%) — single failed gate attempt; passed on retry at 02:21 (46.0 / 67.1) | transient co-tenant burst |
| 2026-08-04 ~02:50 | full refresh ran on the 02:21 pass but the **post-gate failed**: isa_tile_par 37.2 (−24%), gemm 71.0 ms (+20%) — host degraded mid-run; all three suites' cells discarded | co-tenant load ramping during the run; the pre/post gate pair caught it |
| 2026-08-04 ~03:40 | **gate overruled by Human** — refresh run at isa_tile_par 11.8 (−76%) / gemm 213 ms (+260%) and **published with a degraded-window label** (see Results) after ~3 h of continuous throttling blocked every gated attempt | sustained co-tenant 512-bit throttling; publish-on-order ruling |

## Results

**Host:** 4 vCPU Intel Xeon @ 2.10 GHz (shared cloud container), rustc
1.94.1 at default codegen, NumPy 2.4.6 (bundled OpenBLAS), 2026-08-04.

**Publish-on-order (all tables, 2026-08-04):** every table below was
measured in one session inside a **documented degraded window** (gate
readings: start isa_tile_par 11.8 Gops vs 49.2 baseline / gemm_1024 213 ms
vs 59; end 10.3 / 213 — ~4–5× sustained 512-bit throttling, stable across
the run) and published on **Human order**, overriding the calibration gate.
All faces of every table ran in the same window, so face-vs-face ratios are
fair; ISA-heavy MatLua kernels (matmul tiers, parallel reductions) are hit
harder than NumPy's mix, so their ratios read **worse than a healthy host
would show**; absolute times are not comparable to other sessions' tables
or the roofline baselines. Re-measure under a passing gate for
representative numbers.

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
| arange | 4096 | 0.003359 | 0.003106 | 0.53x–0.92x |
| cholesky | 4096 | 860.850 | 852.772 | 0.78x–1.29x |
| cholesky_solve | 4096 | 624.168 | 710.900 | 0.51x–1.75x |
| copy | 4096 | 126.983 | 28.524 | 0.22x–2.54x |
| dot | 4096 | 0.001922 | 0.002807 | 0.31x–1.46x |
| elem_add | 4096 | 110.788 | 50.539 | 0.46x–1.05x |
| elem_add_scalar | 4096 | 105.821 | 40.176 | 0.38x–2.01x |
| elem_div | 4096 | 103.297 | 41.714 | 0.40x–1.15x |
| elem_mul | 4096 | 115.814 | 50.650 | 0.44x–1.31x |
| elem_sub | 4096 | 107.564 | 46.068 | 0.43x–1.14x |
| eye | 4096 | 9.3032 | 8.2632 | 0.52x–1.65x |
| fill | 4096 | 16.646 | 14.529 | 0.73x–0.99x |
| full | 4096 | 88.602 | 12.963 | 0.15x–1.45x |
| matmul | 4096 | 630.977 | 961.321 | 0.55x–1.52x |
| max | 4096 | 12.990 | 5.5361 | 0.43x–1.92x |
| mean | 4096 | 13.801 | 5.1485 | 0.26x–0.96x |
| min | 4096 | 13.232 | 5.1207 | 0.39x–1.93x |
| norm | 4096 | 3.5845 | 5.8986 | 0.41x–2.93x |
| ones | 4096 | 87.028 | 16.802 | 0.19x–2.31x |
| qr | 4096 | 5835.001 | 2330.377 | 0.40x–2.32x |
| reshape | 4096 | 0.000347 | 0.000400 | 1.01x–1.15x |
| solve | 4096 | 630.690 | 929.714 | 1.47x–3.30x |
| sum | 4096 | 13.821 | 5.2302 | 0.38x–1.00x |
| svd | 4096 | 20129.479 | 23415.668 | 1.16x–2.02x |
| transpose | 4096 | 457.967 | 95.478 | 0.21x–1.78x |
| zeros | 4096 | 0.013416 | 0.031973 | 0.96x–2.38x |

#### i64

`matmul` / `matmul_at` / `matmul_bt` reference is NumPy **f64 BLAS** on
integer-valued data (see Yardsticks); MatLua times are exact wrapping i64.

| op | largest n | NumPy (ms) | MatLua Lua (ms) | Lua/NumPy, all n |
| --- | ---: | ---: | ---: | ---: |
| arange | 4096 | 0.003984 | 0.002511 | 0.55x–0.84x |
| copy | 4096 | 113.056 | 88.672 | 0.75x–1.30x |
| dot | 4096 | 0.003804 | 0.003378 | 0.28x–0.89x |
| elem_add | 4096 | 94.348 | 90.778 | 0.77x–1.02x |
| elem_div | 4096 | 242.678 | 130.298 | 0.39x–0.54x |
| elem_mul | 4096 | 108.406 | 97.971 | 0.66x–0.99x |
| elem_sub | 4096 | 90.761 | 91.397 | 0.77x–1.05x |
| eye | 4096 | 8.9857 | 7.3948 | 0.50x–0.87x |
| fill | 4096 | 14.799 | 15.435 | 0.74x–1.04x |
| full | 4096 | 82.944 | 65.475 | 0.72x–0.90x |
| isin | 4096 | 72.063 | 99.667 | 0.22x–1.38x |
| matmul | 4096 | 602.092 | 1149.153 | 1.33x–2.84x |
| matmul_at | 4096 | 639.312 | 1749.507 | 1.48x–2.94x |
| matmul_at_wide | 4096 | 629.800 | 3080.613 | 1.46x–4.91x |
| matmul_bt | 4096 | 599.895 | 1123.516 | 1.42x–2.26x |
| matmul_bt_wide | 4096 | 619.747 | 3004.680 | 1.36x–4.98x |
| matmul_huge | 4096 | 616.919 | 12354.491 | 12.24x–20.03x |
| matmul_wide | 4096 | 631.572 | 2937.678 | 1.36x–5.32x |
| max | 4096 | 12.368 | 7.1951 | 0.58x–8.23x |
| min | 4096 | 12.210 | 7.9883 | 0.65x–8.16x |
| ones | 4096 | 82.833 | 68.717 | 0.42x–0.94x |
| reshape | 4096 | 0.000389 | 0.000386 | 0.98x |
| sum | 4096 | 12.802 | 7.1109 | 0.56x–8.71x |
| transpose | 4096 | 494.963 | 139.676 | 0.28x–1.92x |
| unique | 4096 | 0.758624 | 0.006117 | 0.01x–0.04x |
| zeros | 4096 | 0.014187 | 0.030885 | 0.89x–2.18x |

#### i64→f64 promote-out

| op | largest n | NumPy (ms) | MatLua Lua (ms) | Lua/NumPy, all n |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 4096 | 925.209 | 708.786 | 0.50x–0.90x |
| cholesky_solve | 4096 | 696.112 | 651.966 | 0.33x–1.10x |
| mean | 4096 | 23.902 | 24.748 | 0.25x–1.04x |
| median | 4096 | 146.453 | 100.331 | 0.15x–0.69x |
| norm | 4096 | 108.196 | 9.9148 | 0.09x–3.82x |
| qr | 4096 | 4280.981 | 2383.974 | 0.56x–2.10x |
| quantile | 4096 | 186.262 | 98.684 | 0.04x–0.53x |
| solve | 4096 | 740.373 | 971.076 | 0.55x–1.97x |
| std | 4096 | 168.863 | 45.775 | 0.22x–0.91x |

### Appendix — full three-face tables

<details>
<summary>Table A — f64 absolute (ms)</summary>

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000692 | 0.000167 | 0.000368 |
| arange | 256 | 0.000917 | 0.000402 | 0.000493 |
| arange | 1024 | 0.001338 | 0.001226 | 0.001017 |
| arange | 4096 | 0.003359 | 0.002871 | 0.003106 |
| cholesky | 64 | 0.038127 | 0.029246 | 0.029758 |
| cholesky | 256 | 1.1231 | 1.9039 | 1.4506 |
| cholesky | 1024 | 20.444 | 36.016 | 25.129 |
| cholesky | 4096 | 860.850 | 819.766 | 852.772 |
| cholesky_solve | 64 | 0.050442 | 0.024962 | 0.025531 |
| cholesky_solve | 256 | 0.751364 | 1.1837 | 1.3152 |
| cholesky_solve | 1024 | 16.347 | 23.946 | 19.800 |
| cholesky_solve | 4096 | 624.168 | 729.470 | 710.900 |
| copy | 64 | 0.001792 | 0.001115 | 0.001308 |
| copy | 256 | 0.018490 | 0.020028 | 0.023784 |
| copy | 1024 | 0.785521 | 0.835298 | 1.9951 |
| copy | 4096 | 126.983 | 93.454 | 28.524 |
| dot | 64 | 0.001238 | 0.000065 | 0.000389 |
| dot | 256 | 0.000953 | 0.000178 | 0.000504 |
| dot | 1024 | 0.001205 | 0.000632 | 0.000960 |
| dot | 4096 | 0.001922 | 0.002481 | 0.002807 |
| elem_add | 64 | 0.002625 | 0.001418 | 0.002050 |
| elem_add | 256 | 0.066753 | 0.058891 | 0.061115 |
| elem_add | 1024 | 2.0257 | 1.4629 | 2.1329 |
| elem_add | 4096 | 110.788 | 104.028 | 50.539 |
| elem_add_scalar | 64 | 0.002367 | 0.001231 | 0.001654 |
| elem_add_scalar | 256 | 0.023983 | 0.024138 | 0.033666 |
| elem_add_scalar | 1024 | 0.795992 | 1.0585 | 1.5996 |
| elem_add_scalar | 4096 | 105.821 | 92.856 | 40.176 |
| elem_div | 64 | 0.003931 | 0.003463 | 0.003798 |
| elem_div | 256 | 0.061314 | 0.059245 | 0.061515 |
| elem_div | 1024 | 1.6581 | 1.3921 | 1.9056 |
| elem_div | 4096 | 103.297 | 91.417 | 41.714 |
| elem_mul | 64 | 0.002581 | 0.001633 | 0.001973 |
| elem_mul | 256 | 0.066683 | 0.059317 | 0.061252 |
| elem_mul | 1024 | 1.7516 | 1.5823 | 2.2935 |
| elem_mul | 4096 | 115.814 | 106.676 | 50.650 |
| elem_sub | 64 | 0.002573 | 0.001420 | 0.001936 |
| elem_sub | 256 | 0.064574 | 0.059155 | 0.061278 |
| elem_sub | 1024 | 1.7597 | 1.4428 | 1.9981 |
| elem_sub | 4096 | 107.564 | 104.860 | 46.068 |
| eye | 64 | 0.002752 | 0.000918 | 0.001424 |
| eye | 256 | 0.012097 | 0.011529 | 0.019933 |
| eye | 1024 | 0.367504 | 0.336690 | 0.341907 |
| eye | 4096 | 9.3032 | 8.8304 | 8.2632 |
| fill | 64 | 0.001365 | 0.000887 | 0.000998 |
| fill | 256 | 0.012585 | 0.016512 | 0.012498 |
| fill | 1024 | 0.379557 | 0.381823 | 0.354916 |
| fill | 4096 | 16.646 | 14.433 | 14.529 |
| full | 64 | 0.002289 | 0.001328 | 0.001148 |
| full | 256 | 0.013654 | 0.018443 | 0.015963 |
| full | 1024 | 0.400536 | 0.380144 | 0.580421 |
| full | 4096 | 88.602 | 69.862 | 12.963 |
| matmul | 64 | 0.013762 | 0.009370 | 0.010400 |
| matmul | 256 | 0.573038 | 0.276818 | 0.316165 |
| matmul | 1024 | 10.754 | 12.431 | 12.640 |
| matmul | 4096 | 630.977 | 1015.196 | 961.321 |
| max | 64 | 0.002644 | 0.001555 | 0.001553 |
| max | 256 | 0.010787 | 0.020479 | 0.020761 |
| max | 1024 | 0.379854 | 0.388872 | 0.363350 |
| max | 4096 | 12.990 | 4.6871 | 5.5361 |
| mean | 64 | 0.004352 | 0.000880 | 0.001143 |
| mean | 256 | 0.019706 | 0.013178 | 0.013460 |
| mean | 1024 | 0.353083 | 0.355514 | 0.340326 |
| mean | 4096 | 13.801 | 5.5044 | 5.1485 |
| min | 64 | 0.002300 | 0.001513 | 0.001538 |
| min | 256 | 0.010733 | 0.019996 | 0.020730 |
| min | 1024 | 0.386664 | 0.373947 | 0.345002 |
| min | 4096 | 13.232 | 4.3476 | 5.1207 |
| norm | 64 | 0.002838 | 0.000743 | 0.001168 |
| norm | 256 | 0.005839 | 0.013191 | 0.013456 |
| norm | 1024 | 0.124847 | 0.383796 | 0.365603 |
| norm | 4096 | 3.5845 | 4.7097 | 5.8986 |
| ones | 64 | 0.002486 | 0.001350 | 0.001135 |
| ones | 256 | 0.020198 | 0.017027 | 0.022327 |
| ones | 1024 | 0.398947 | 0.390195 | 0.920149 |
| ones | 4096 | 87.028 | 69.072 | 16.802 |
| qr | 64 | 0.227729 | 0.648680 | 0.529299 |
| qr | 256 | 5.5438 | 6.0041 | 3.6486 |
| qr | 1024 | 104.893 | 117.167 | 77.297 |
| qr | 4096 | 5835.001 | 3557.654 | 2330.377 |
| reshape | 64 | 0.000348 | 0.000070 | 0.000369 |
| reshape | 256 | 0.000362 | 0.000098 | 0.000365 |
| reshape | 1024 | 0.000359 | 0.000073 | 0.000385 |
| reshape | 4096 | 0.000347 | 0.000071 | 0.000400 |
| solve | 64 | 0.061550 | 0.109278 | 0.094497 |
| solve | 256 | 0.715685 | 3.2908 | 2.3616 |
| solve | 1024 | 16.793 | 51.621 | 44.001 |
| solve | 4096 | 630.690 | 1192.530 | 929.714 |
| sum | 64 | 0.002795 | 0.000880 | 0.001200 |
| sum | 256 | 0.020286 | 0.013174 | 0.013436 |
| sum | 1024 | 0.348858 | 0.379101 | 0.348770 |
| sum | 4096 | 13.821 | 6.0723 | 5.2302 |
| svd | 64 | 0.472850 | 1.0003 | 0.954078 |
| svd | 256 | 10.895 | 17.554 | 17.096 |
| svd | 1024 | 332.306 | 459.710 | 506.522 |
| svd | 4096 | 20129.479 | 24346.531 | 23415.668 |
| transpose | 64 | 0.003850 | 0.007620 | 0.006855 |
| transpose | 256 | 0.061169 | 0.091426 | 0.094287 |
| transpose | 1024 | 4.3776 | 2.9329 | 4.5523 |
| transpose | 4096 | 457.967 | 140.662 | 95.478 |
| zeros | 64 | 0.001654 | 0.001496 | 0.001584 |
| zeros | 256 | 0.018076 | 0.018564 | 0.022075 |
| zeros | 1024 | 0.357937 | 0.351170 | 0.358664 |
| zeros | 4096 | 0.013416 | 0.071078 | 0.031973 |

</details>

<details>
<summary>Table B — f64 relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.24x | 0.53x | 2.20x |
| arange | 256 | 1.00x | 0.44x | 0.54x | 1.23x |
| arange | 1024 | 1.00x | 0.92x | 0.76x | 0.83x |
| arange | 4096 | 1.00x | 0.85x | 0.92x | 1.08x |
| cholesky | 64 | 1.00x | 0.77x | 0.78x | 1.02x |
| cholesky | 256 | 1.00x | 1.70x | 1.29x | 0.76x |
| cholesky | 1024 | 1.00x | 1.76x | 1.23x | 0.70x |
| cholesky | 4096 | 1.00x | 0.95x | 0.99x | 1.04x |
| cholesky_solve | 64 | 1.00x | 0.49x | 0.51x | 1.02x |
| cholesky_solve | 256 | 1.00x | 1.58x | 1.75x | 1.11x |
| cholesky_solve | 1024 | 1.00x | 1.46x | 1.21x | 0.83x |
| cholesky_solve | 4096 | 1.00x | 1.17x | 1.14x | 0.97x |
| copy | 64 | 1.00x | 0.62x | 0.73x | 1.17x |
| copy | 256 | 1.00x | 1.08x | 1.29x | 1.19x |
| copy | 1024 | 1.00x | 1.06x | 2.54x | 2.39x |
| copy | 4096 | 1.00x | 0.74x | 0.22x | 0.31x |
| dot | 64 | 1.00x | 0.05x | 0.31x | 5.98x |
| dot | 256 | 1.00x | 0.19x | 0.53x | 2.83x |
| dot | 1024 | 1.00x | 0.52x | 0.80x | 1.52x |
| dot | 4096 | 1.00x | 1.29x | 1.46x | 1.13x |
| elem_add | 64 | 1.00x | 0.54x | 0.78x | 1.45x |
| elem_add | 256 | 1.00x | 0.88x | 0.92x | 1.04x |
| elem_add | 1024 | 1.00x | 0.72x | 1.05x | 1.46x |
| elem_add | 4096 | 1.00x | 0.94x | 0.46x | 0.49x |
| elem_add_scalar | 64 | 1.00x | 0.52x | 0.70x | 1.34x |
| elem_add_scalar | 256 | 1.00x | 1.01x | 1.40x | 1.39x |
| elem_add_scalar | 1024 | 1.00x | 1.33x | 2.01x | 1.51x |
| elem_add_scalar | 4096 | 1.00x | 0.88x | 0.38x | 0.43x |
| elem_div | 64 | 1.00x | 0.88x | 0.97x | 1.10x |
| elem_div | 256 | 1.00x | 0.97x | 1.00x | 1.04x |
| elem_div | 1024 | 1.00x | 0.84x | 1.15x | 1.37x |
| elem_div | 4096 | 1.00x | 0.88x | 0.40x | 0.46x |
| elem_mul | 64 | 1.00x | 0.63x | 0.76x | 1.21x |
| elem_mul | 256 | 1.00x | 0.89x | 0.92x | 1.03x |
| elem_mul | 1024 | 1.00x | 0.90x | 1.31x | 1.45x |
| elem_mul | 4096 | 1.00x | 0.92x | 0.44x | 0.47x |
| elem_sub | 64 | 1.00x | 0.55x | 0.75x | 1.36x |
| elem_sub | 256 | 1.00x | 0.92x | 0.95x | 1.04x |
| elem_sub | 1024 | 1.00x | 0.82x | 1.14x | 1.38x |
| elem_sub | 4096 | 1.00x | 0.97x | 0.43x | 0.44x |
| eye | 64 | 1.00x | 0.33x | 0.52x | 1.55x |
| eye | 256 | 1.00x | 0.95x | 1.65x | 1.73x |
| eye | 1024 | 1.00x | 0.92x | 0.93x | 1.02x |
| eye | 4096 | 1.00x | 0.95x | 0.89x | 0.94x |
| fill | 64 | 1.00x | 0.65x | 0.73x | 1.13x |
| fill | 256 | 1.00x | 1.31x | 0.99x | 0.76x |
| fill | 1024 | 1.00x | 1.01x | 0.94x | 0.93x |
| fill | 4096 | 1.00x | 0.87x | 0.87x | 1.01x |
| full | 64 | 1.00x | 0.58x | 0.50x | 0.86x |
| full | 256 | 1.00x | 1.35x | 1.17x | 0.87x |
| full | 1024 | 1.00x | 0.95x | 1.45x | 1.53x |
| full | 4096 | 1.00x | 0.79x | 0.15x | 0.19x |
| matmul | 64 | 1.00x | 0.68x | 0.76x | 1.11x |
| matmul | 256 | 1.00x | 0.48x | 0.55x | 1.14x |
| matmul | 1024 | 1.00x | 1.16x | 1.18x | 1.02x |
| matmul | 4096 | 1.00x | 1.61x | 1.52x | 0.95x |
| max | 64 | 1.00x | 0.59x | 0.59x | 1.00x |
| max | 256 | 1.00x | 1.90x | 1.92x | 1.01x |
| max | 1024 | 1.00x | 1.02x | 0.96x | 0.93x |
| max | 4096 | 1.00x | 0.36x | 0.43x | 1.18x |
| mean | 64 | 1.00x | 0.20x | 0.26x | 1.30x |
| mean | 256 | 1.00x | 0.67x | 0.68x | 1.02x |
| mean | 1024 | 1.00x | 1.01x | 0.96x | 0.96x |
| mean | 4096 | 1.00x | 0.40x | 0.37x | 0.94x |
| min | 64 | 1.00x | 0.66x | 0.67x | 1.02x |
| min | 256 | 1.00x | 1.86x | 1.93x | 1.04x |
| min | 1024 | 1.00x | 0.97x | 0.89x | 0.92x |
| min | 4096 | 1.00x | 0.33x | 0.39x | 1.18x |
| norm | 64 | 1.00x | 0.26x | 0.41x | 1.57x |
| norm | 256 | 1.00x | 2.26x | 2.30x | 1.02x |
| norm | 1024 | 1.00x | 3.07x | 2.93x | 0.95x |
| norm | 4096 | 1.00x | 1.31x | 1.65x | 1.25x |
| ones | 64 | 1.00x | 0.54x | 0.46x | 0.84x |
| ones | 256 | 1.00x | 0.84x | 1.11x | 1.31x |
| ones | 1024 | 1.00x | 0.98x | 2.31x | 2.36x |
| ones | 4096 | 1.00x | 0.79x | 0.19x | 0.24x |
| qr | 64 | 1.00x | 2.85x | 2.32x | 0.82x |
| qr | 256 | 1.00x | 1.08x | 0.66x | 0.61x |
| qr | 1024 | 1.00x | 1.12x | 0.74x | 0.66x |
| qr | 4096 | 1.00x | 0.61x | 0.40x | 0.66x |
| reshape | 64 | 1.00x | 0.20x | 1.06x | 5.27x |
| reshape | 256 | 1.00x | 0.27x | 1.01x | 3.72x |
| reshape | 1024 | 1.00x | 0.20x | 1.07x | 5.27x |
| reshape | 4096 | 1.00x | 0.20x | 1.15x | 5.63x |
| solve | 64 | 1.00x | 1.78x | 1.54x | 0.86x |
| solve | 256 | 1.00x | 4.60x | 3.30x | 0.72x |
| solve | 1024 | 1.00x | 3.07x | 2.62x | 0.85x |
| solve | 4096 | 1.00x | 1.89x | 1.47x | 0.78x |
| sum | 64 | 1.00x | 0.31x | 0.43x | 1.36x |
| sum | 256 | 1.00x | 0.65x | 0.66x | 1.02x |
| sum | 1024 | 1.00x | 1.09x | 1.00x | 0.92x |
| sum | 4096 | 1.00x | 0.44x | 0.38x | 0.86x |
| svd | 64 | 1.00x | 2.12x | 2.02x | 0.95x |
| svd | 256 | 1.00x | 1.61x | 1.57x | 0.97x |
| svd | 1024 | 1.00x | 1.38x | 1.52x | 1.10x |
| svd | 4096 | 1.00x | 1.21x | 1.16x | 0.96x |
| transpose | 64 | 1.00x | 1.98x | 1.78x | 0.90x |
| transpose | 256 | 1.00x | 1.49x | 1.54x | 1.03x |
| transpose | 1024 | 1.00x | 0.67x | 1.04x | 1.55x |
| transpose | 4096 | 1.00x | 0.31x | 0.21x | 0.68x |
| zeros | 64 | 1.00x | 0.90x | 0.96x | 1.06x |
| zeros | 256 | 1.00x | 1.03x | 1.22x | 1.19x |
| zeros | 1024 | 1.00x | 0.98x | 1.00x | 1.02x |
| zeros | 4096 | 1.00x | 5.30x | 2.38x | 0.45x |

</details>

<details>
<summary>Table C — i64 absolute (ms) — matmul* NumPy column is f64 BLAS on integer-valued data</summary>

| op | n | NumPy int64 (ms) | MatLua Rust i64 (ms) | MatLua Lua i64 (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000645 | 0.000166 | 0.000357 |
| arange | 256 | 0.000742 | 0.000265 | 0.000456 |
| arange | 1024 | 0.001016 | 0.001128 | 0.000854 |
| arange | 4096 | 0.003984 | 0.002266 | 0.002511 |
| copy | 64 | 0.001346 | 0.001409 | 0.001753 |
| copy | 256 | 0.018677 | 0.016759 | 0.020484 |
| copy | 1024 | 1.0012 | 0.994519 | 0.753422 |
| copy | 4096 | 113.056 | 112.209 | 88.672 |
| dot | 64 | 0.001321 | 0.000087 | 0.000370 |
| dot | 256 | 0.001303 | 0.000223 | 0.000522 |
| dot | 1024 | 0.001846 | 0.000781 | 0.001091 |
| dot | 4096 | 0.003804 | 0.003052 | 0.003378 |
| elem_add | 64 | 0.002139 | 0.001718 | 0.001648 |
| elem_add | 256 | 0.057648 | 0.059073 | 0.058998 |
| elem_add | 1024 | 1.7971 | 1.9019 | 1.6580 |
| elem_add | 4096 | 94.348 | 105.251 | 90.778 |
| elem_div | 64 | 0.038740 | 0.016278 | 0.016413 |
| elem_div | 256 | 0.670078 | 0.274457 | 0.263919 |
| elem_div | 1024 | 10.521 | 4.6616 | 4.5171 |
| elem_div | 4096 | 242.678 | 144.216 | 130.298 |
| elem_mul | 64 | 0.002953 | 0.001834 | 0.001948 |
| elem_mul | 256 | 0.061997 | 0.059849 | 0.061148 |
| elem_mul | 1024 | 1.7785 | 2.2013 | 1.5407 |
| elem_mul | 4096 | 108.406 | 110.522 | 97.971 |
| elem_sub | 64 | 0.002145 | 0.001708 | 0.001662 |
| elem_sub | 256 | 0.055870 | 0.063851 | 0.058859 |
| elem_sub | 1024 | 1.4809 | 1.8479 | 1.3693 |
| elem_sub | 4096 | 90.761 | 104.470 | 91.397 |
| eye | 64 | 0.002697 | 0.000922 | 0.001339 |
| eye | 256 | 0.012127 | 0.008867 | 0.008782 |
| eye | 1024 | 0.400549 | 0.384103 | 0.346977 |
| eye | 4096 | 8.9857 | 9.4456 | 7.3948 |
| fill | 64 | 0.001357 | 0.000833 | 0.001002 |
| fill | 256 | 0.012763 | 0.012118 | 0.012838 |
| fill | 1024 | 0.455052 | 0.388648 | 0.387073 |
| fill | 4096 | 14.799 | 15.010 | 15.435 |
| full | 64 | 0.002716 | 0.000955 | 0.001955 |
| full | 256 | 0.014255 | 0.014296 | 0.012892 |
| full | 1024 | 0.475346 | 0.391378 | 0.371292 |
| full | 4096 | 82.944 | 81.101 | 65.475 |
| isin | 64 | 0.030662 | 0.006375 | 0.006679 |
| isin | 256 | 0.148760 | 0.098372 | 0.104945 |
| isin | 1024 | 1.7173 | 1.9024 | 1.7779 |
| isin | 4096 | 72.063 | 102.878 | 99.667 |
| matmul | 64 | 0.013552 | 0.037354 | 0.038496 |
| matmul | 256 | 0.561693 | 0.932612 | 0.745750 |
| matmul | 1024 | 11.288 | 31.027 | 22.760 |
| matmul | 4096 | 602.092 | 1256.449 | 1149.153 |
| matmul_at | 64 | 0.013332 | 0.038010 | 0.039257 |
| matmul_at | 256 | 0.562953 | 0.830606 | 0.831586 |
| matmul_at | 1024 | 13.062 | 37.733 | 36.933 |
| matmul_at | 4096 | 639.312 | 2037.288 | 1749.507 |
| matmul_at_wide | 64 | 0.013160 | 0.051457 | 0.039234 |
| matmul_at_wide | 256 | 0.554242 | 0.961345 | 0.808951 |
| matmul_at_wide | 1024 | 11.631 | 59.043 | 57.088 |
| matmul_at_wide | 4096 | 629.800 | 3145.812 | 3080.613 |
| matmul_bt | 64 | 0.017325 | 0.037988 | 0.039184 |
| matmul_bt | 256 | 0.523586 | 0.832825 | 0.745291 |
| matmul_bt | 1024 | 10.475 | 26.091 | 23.594 |
| matmul_bt | 4096 | 599.895 | 1190.928 | 1123.516 |
| matmul_bt_wide | 64 | 0.017822 | 0.051221 | 0.038909 |
| matmul_bt_wide | 256 | 0.551970 | 0.962301 | 0.749691 |
| matmul_bt_wide | 1024 | 10.692 | 52.503 | 53.257 |
| matmul_bt_wide | 4096 | 619.747 | 3016.696 | 3004.680 |
| matmul_huge | 64 | 0.013562 | 0.242128 | 0.224817 |
| matmul_huge | 256 | 0.532653 | 8.2466 | 6.5186 |
| matmul_huge | 1024 | 10.711 | 209.182 | 199.018 |
| matmul_huge | 4096 | 616.919 | 12595.292 | 12354.491 |
| matmul_wide | 64 | 0.013903 | 0.038391 | 0.038257 |
| matmul_wide | 256 | 0.538114 | 0.959759 | 0.734219 |
| matmul_wide | 1024 | 10.349 | 57.396 | 55.082 |
| matmul_wide | 4096 | 631.572 | 3085.114 | 2937.678 |
| max | 64 | 0.002289 | 0.005647 | 0.005059 |
| max | 256 | 0.009283 | 0.088457 | 0.076356 |
| max | 1024 | 0.386725 | 1.3975 | 1.3467 |
| max | 4096 | 12.368 | 7.9471 | 7.1951 |
| min | 64 | 0.002255 | 0.004792 | 0.005063 |
| min | 256 | 0.009360 | 0.084288 | 0.076381 |
| min | 1024 | 0.388766 | 1.3788 | 1.3435 |
| min | 4096 | 12.210 | 9.3236 | 7.9883 |
| ones | 64 | 0.002781 | 0.001014 | 0.001180 |
| ones | 256 | 0.013760 | 0.012391 | 0.012945 |
| ones | 1024 | 0.409886 | 0.405696 | 0.366494 |
| ones | 4096 | 82.833 | 78.904 | 68.717 |
| reshape | 64 | 0.000354 | 0.000070 | 0.000359 |
| reshape | 256 | 0.000369 | 0.000070 | 0.000358 |
| reshape | 1024 | 0.000385 | 0.000087 | 0.000365 |
| reshape | 4096 | 0.000389 | 0.000072 | 0.000386 |
| sum | 64 | 0.001942 | 0.004843 | 0.005098 |
| sum | 256 | 0.008883 | 0.079357 | 0.077341 |
| sum | 1024 | 0.351868 | 1.4057 | 1.3717 |
| sum | 4096 | 12.802 | 9.1207 | 7.1109 |
| transpose | 64 | 0.003382 | 0.006256 | 0.006507 |
| transpose | 256 | 0.064866 | 0.088516 | 0.090389 |
| transpose | 1024 | 4.3309 | 5.9080 | 2.7322 |
| transpose | 4096 | 494.963 | 151.676 | 139.676 |
| unique | 64 | 0.010248 | 0.000359 | 0.000377 |
| unique | 256 | 0.038242 | 0.001209 | 0.000735 |
| unique | 1024 | 0.134594 | 0.001723 | 0.001762 |
| unique | 4096 | 0.758624 | 0.005660 | 0.006117 |
| zeros | 64 | 0.001271 | 0.000661 | 0.001537 |
| zeros | 256 | 0.021268 | 0.018808 | 0.019889 |
| zeros | 1024 | 0.407885 | 0.430003 | 0.363714 |
| zeros | 4096 | 0.014187 | 0.008856 | 0.030885 |

</details>

<details>
<summary>Table D — i64 relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.26x | 0.55x | 2.15x |
| arange | 256 | 1.00x | 0.36x | 0.61x | 1.72x |
| arange | 1024 | 1.00x | 1.11x | 0.84x | 0.76x |
| arange | 4096 | 1.00x | 0.57x | 0.63x | 1.11x |
| copy | 64 | 1.00x | 1.05x | 1.30x | 1.24x |
| copy | 256 | 1.00x | 0.90x | 1.10x | 1.22x |
| copy | 1024 | 1.00x | 0.99x | 0.75x | 0.76x |
| copy | 4096 | 1.00x | 0.99x | 0.78x | 0.79x |
| dot | 64 | 1.00x | 0.07x | 0.28x | 4.25x |
| dot | 256 | 1.00x | 0.17x | 0.40x | 2.34x |
| dot | 1024 | 1.00x | 0.42x | 0.59x | 1.40x |
| dot | 4096 | 1.00x | 0.80x | 0.89x | 1.11x |
| elem_add | 64 | 1.00x | 0.80x | 0.77x | 0.96x |
| elem_add | 256 | 1.00x | 1.02x | 1.02x | 1.00x |
| elem_add | 1024 | 1.00x | 1.06x | 0.92x | 0.87x |
| elem_add | 4096 | 1.00x | 1.12x | 0.96x | 0.86x |
| elem_div | 64 | 1.00x | 0.42x | 0.42x | 1.01x |
| elem_div | 256 | 1.00x | 0.41x | 0.39x | 0.96x |
| elem_div | 1024 | 1.00x | 0.44x | 0.43x | 0.97x |
| elem_div | 4096 | 1.00x | 0.59x | 0.54x | 0.90x |
| elem_mul | 64 | 1.00x | 0.62x | 0.66x | 1.06x |
| elem_mul | 256 | 1.00x | 0.97x | 0.99x | 1.02x |
| elem_mul | 1024 | 1.00x | 1.24x | 0.87x | 0.70x |
| elem_mul | 4096 | 1.00x | 1.02x | 0.90x | 0.89x |
| elem_sub | 64 | 1.00x | 0.80x | 0.77x | 0.97x |
| elem_sub | 256 | 1.00x | 1.14x | 1.05x | 0.92x |
| elem_sub | 1024 | 1.00x | 1.25x | 0.92x | 0.74x |
| elem_sub | 4096 | 1.00x | 1.15x | 1.01x | 0.87x |
| eye | 64 | 1.00x | 0.34x | 0.50x | 1.45x |
| eye | 256 | 1.00x | 0.73x | 0.72x | 0.99x |
| eye | 1024 | 1.00x | 0.96x | 0.87x | 0.90x |
| eye | 4096 | 1.00x | 1.05x | 0.82x | 0.78x |
| fill | 64 | 1.00x | 0.61x | 0.74x | 1.20x |
| fill | 256 | 1.00x | 0.95x | 1.01x | 1.06x |
| fill | 1024 | 1.00x | 0.85x | 0.85x | 1.00x |
| fill | 4096 | 1.00x | 1.01x | 1.04x | 1.03x |
| full | 64 | 1.00x | 0.35x | 0.72x | 2.05x |
| full | 256 | 1.00x | 1.00x | 0.90x | 0.90x |
| full | 1024 | 1.00x | 0.82x | 0.78x | 0.95x |
| full | 4096 | 1.00x | 0.98x | 0.79x | 0.81x |
| isin | 64 | 1.00x | 0.21x | 0.22x | 1.05x |
| isin | 256 | 1.00x | 0.66x | 0.71x | 1.07x |
| isin | 1024 | 1.00x | 1.11x | 1.04x | 0.93x |
| isin | 4096 | 1.00x | 1.43x | 1.38x | 0.97x |
| matmul | 64 | 1.00x | 2.76x | 2.84x | 1.03x |
| matmul | 256 | 1.00x | 1.66x | 1.33x | 0.80x |
| matmul | 1024 | 1.00x | 2.75x | 2.02x | 0.73x |
| matmul | 4096 | 1.00x | 2.09x | 1.91x | 0.91x |
| matmul_at | 64 | 1.00x | 2.85x | 2.94x | 1.03x |
| matmul_at | 256 | 1.00x | 1.48x | 1.48x | 1.00x |
| matmul_at | 1024 | 1.00x | 2.89x | 2.83x | 0.98x |
| matmul_at | 4096 | 1.00x | 3.19x | 2.74x | 0.86x |
| matmul_at_wide | 64 | 1.00x | 3.91x | 2.98x | 0.76x |
| matmul_at_wide | 256 | 1.00x | 1.73x | 1.46x | 0.84x |
| matmul_at_wide | 1024 | 1.00x | 5.08x | 4.91x | 0.97x |
| matmul_at_wide | 4096 | 1.00x | 4.99x | 4.89x | 0.98x |
| matmul_bt | 64 | 1.00x | 2.19x | 2.26x | 1.03x |
| matmul_bt | 256 | 1.00x | 1.59x | 1.42x | 0.89x |
| matmul_bt | 1024 | 1.00x | 2.49x | 2.25x | 0.90x |
| matmul_bt | 4096 | 1.00x | 1.99x | 1.87x | 0.94x |
| matmul_bt_wide | 64 | 1.00x | 2.87x | 2.18x | 0.76x |
| matmul_bt_wide | 256 | 1.00x | 1.74x | 1.36x | 0.78x |
| matmul_bt_wide | 1024 | 1.00x | 4.91x | 4.98x | 1.01x |
| matmul_bt_wide | 4096 | 1.00x | 4.87x | 4.85x | 1.00x |
| matmul_huge | 64 | 1.00x | 17.85x | 16.58x | 0.93x |
| matmul_huge | 256 | 1.00x | 15.48x | 12.24x | 0.79x |
| matmul_huge | 1024 | 1.00x | 19.53x | 18.58x | 0.95x |
| matmul_huge | 4096 | 1.00x | 20.42x | 20.03x | 0.98x |
| matmul_wide | 64 | 1.00x | 2.76x | 2.75x | 1.00x |
| matmul_wide | 256 | 1.00x | 1.78x | 1.36x | 0.77x |
| matmul_wide | 1024 | 1.00x | 5.55x | 5.32x | 0.96x |
| matmul_wide | 4096 | 1.00x | 4.88x | 4.65x | 0.95x |
| max | 64 | 1.00x | 2.47x | 2.21x | 0.90x |
| max | 256 | 1.00x | 9.53x | 8.23x | 0.86x |
| max | 1024 | 1.00x | 3.61x | 3.48x | 0.96x |
| max | 4096 | 1.00x | 0.64x | 0.58x | 0.91x |
| min | 64 | 1.00x | 2.13x | 2.25x | 1.06x |
| min | 256 | 1.00x | 9.01x | 8.16x | 0.91x |
| min | 1024 | 1.00x | 3.55x | 3.46x | 0.97x |
| min | 4096 | 1.00x | 0.76x | 0.65x | 0.86x |
| ones | 64 | 1.00x | 0.36x | 0.42x | 1.16x |
| ones | 256 | 1.00x | 0.90x | 0.94x | 1.04x |
| ones | 1024 | 1.00x | 0.99x | 0.89x | 0.90x |
| ones | 4096 | 1.00x | 0.95x | 0.83x | 0.87x |
| reshape | 64 | 1.00x | 0.20x | 1.01x | 5.13x |
| reshape | 256 | 1.00x | 0.19x | 0.97x | 5.11x |
| reshape | 1024 | 1.00x | 0.23x | 0.95x | 4.20x |
| reshape | 4096 | 1.00x | 0.19x | 0.99x | 5.36x |
| sum | 64 | 1.00x | 2.49x | 2.63x | 1.05x |
| sum | 256 | 1.00x | 8.93x | 8.71x | 0.97x |
| sum | 1024 | 1.00x | 4.00x | 3.90x | 0.98x |
| sum | 4096 | 1.00x | 0.71x | 0.56x | 0.78x |
| transpose | 64 | 1.00x | 1.85x | 1.92x | 1.04x |
| transpose | 256 | 1.00x | 1.36x | 1.39x | 1.02x |
| transpose | 1024 | 1.00x | 1.36x | 0.63x | 0.46x |
| transpose | 4096 | 1.00x | 0.31x | 0.28x | 0.92x |
| unique | 64 | 1.00x | 0.04x | 0.04x | 1.05x |
| unique | 256 | 1.00x | 0.03x | 0.02x | 0.61x |
| unique | 1024 | 1.00x | 0.01x | 0.01x | 1.02x |
| unique | 4096 | 1.00x | 0.01x | 0.01x | 1.08x |
| zeros | 64 | 1.00x | 0.52x | 1.21x | 2.33x |
| zeros | 256 | 1.00x | 0.88x | 0.94x | 1.06x |
| zeros | 1024 | 1.00x | 1.05x | 0.89x | 0.85x |
| zeros | 4096 | 1.00x | 0.62x | 2.18x | 3.49x |

</details>

<details>
<summary>Table E — i64→f64 promote-out absolute (ms)</summary>

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 64 | 0.055327 | 0.027087 | 0.027903 |
| cholesky | 256 | 2.0156 | 1.3772 | 1.8059 |
| cholesky | 1024 | 43.865 | 25.992 | 32.196 |
| cholesky | 4096 | 925.209 | 818.384 | 708.786 |
| cholesky_solve | 64 | 0.083350 | 0.026671 | 0.027337 |
| cholesky_solve | 256 | 1.7241 | 1.2122 | 1.8882 |
| cholesky_solve | 1024 | 33.939 | 21.678 | 28.137 |
| cholesky_solve | 4096 | 696.112 | 689.166 | 651.966 |
| mean | 64 | 0.009591 | 0.001705 | 0.002359 |
| mean | 256 | 0.089887 | 0.031201 | 0.040101 |
| mean | 1024 | 0.891097 | 0.774286 | 0.655351 |
| mean | 4096 | 23.902 | 16.635 | 24.748 |
| median | 64 | 0.029835 | 0.004078 | 0.004445 |
| median | 256 | 0.189313 | 0.274213 | 0.129291 |
| median | 1024 | 2.7790 | 1.9588 | 1.8084 |
| median | 4096 | 146.453 | 112.873 | 100.331 |
| norm | 64 | 0.007682 | 0.032097 | 0.029360 |
| norm | 256 | 0.094272 | 0.122195 | 0.131243 |
| norm | 1024 | 2.5781 | 1.7730 | 1.7918 |
| norm | 4096 | 108.196 | 7.7956 | 9.9148 |
| qr | 64 | 0.284241 | 1.1081 | 0.596976 |
| qr | 256 | 6.2293 | 11.245 | 8.0370 |
| qr | 1024 | 151.832 | 93.202 | 114.443 |
| qr | 4096 | 4280.981 | 2263.615 | 2383.974 |
| quantile | 64 | 0.116736 | 0.003848 | 0.004202 |
| quantile | 256 | 0.451861 | 0.275103 | 0.119965 |
| quantile | 1024 | 5.5776 | 1.5677 | 1.7483 |
| quantile | 4096 | 186.262 | 103.989 | 98.684 |
| solve | 64 | 0.091631 | 0.057445 | 0.050123 |
| solve | 256 | 1.8480 | 2.9592 | 3.6339 |
| solve | 1024 | 36.096 | 49.941 | 45.547 |
| solve | 4096 | 740.373 | 898.574 | 971.076 |
| std | 64 | 0.034196 | 0.006534 | 0.007479 |
| std | 256 | 0.274084 | 0.225048 | 0.249348 |
| std | 1024 | 3.9113 | 1.8316 | 1.9402 |
| std | 4096 | 168.863 | 42.347 | 45.775 |

</details>

<details>
<summary>Table F — i64→f64 promote-out relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky | 64 | 1.00x | 0.49x | 0.50x | 1.03x |
| cholesky | 256 | 1.00x | 0.68x | 0.90x | 1.31x |
| cholesky | 1024 | 1.00x | 0.59x | 0.73x | 1.24x |
| cholesky | 4096 | 1.00x | 0.88x | 0.77x | 0.87x |
| cholesky_solve | 64 | 1.00x | 0.32x | 0.33x | 1.02x |
| cholesky_solve | 256 | 1.00x | 0.70x | 1.10x | 1.56x |
| cholesky_solve | 1024 | 1.00x | 0.64x | 0.83x | 1.30x |
| cholesky_solve | 4096 | 1.00x | 0.99x | 0.94x | 0.95x |
| mean | 64 | 1.00x | 0.18x | 0.25x | 1.38x |
| mean | 256 | 1.00x | 0.35x | 0.45x | 1.29x |
| mean | 1024 | 1.00x | 0.87x | 0.74x | 0.85x |
| mean | 4096 | 1.00x | 0.70x | 1.04x | 1.49x |
| median | 64 | 1.00x | 0.14x | 0.15x | 1.09x |
| median | 256 | 1.00x | 1.45x | 0.68x | 0.47x |
| median | 1024 | 1.00x | 0.70x | 0.65x | 0.92x |
| median | 4096 | 1.00x | 0.77x | 0.69x | 0.89x |
| norm | 64 | 1.00x | 4.18x | 3.82x | 0.91x |
| norm | 256 | 1.00x | 1.30x | 1.39x | 1.07x |
| norm | 1024 | 1.00x | 0.69x | 0.70x | 1.01x |
| norm | 4096 | 1.00x | 0.07x | 0.09x | 1.27x |
| qr | 64 | 1.00x | 3.90x | 2.10x | 0.54x |
| qr | 256 | 1.00x | 1.81x | 1.29x | 0.71x |
| qr | 1024 | 1.00x | 0.61x | 0.75x | 1.23x |
| qr | 4096 | 1.00x | 0.53x | 0.56x | 1.05x |
| quantile | 64 | 1.00x | 0.03x | 0.04x | 1.09x |
| quantile | 256 | 1.00x | 0.61x | 0.27x | 0.44x |
| quantile | 1024 | 1.00x | 0.28x | 0.31x | 1.12x |
| quantile | 4096 | 1.00x | 0.56x | 0.53x | 0.95x |
| solve | 64 | 1.00x | 0.63x | 0.55x | 0.87x |
| solve | 256 | 1.00x | 1.60x | 1.97x | 1.23x |
| solve | 1024 | 1.00x | 1.38x | 1.26x | 0.91x |
| solve | 4096 | 1.00x | 1.21x | 1.31x | 1.08x |
| std | 64 | 1.00x | 0.19x | 0.22x | 1.14x |
| std | 256 | 1.00x | 0.82x | 0.91x | 1.11x |
| std | 1024 | 1.00x | 0.47x | 0.50x | 1.06x |
| std | 4096 | 1.00x | 0.25x | 0.27x | 1.08x |

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

Published-run gate readings (2026-08-03, the session that produced the
Results tables): session start isa_tile_par 45.9, gemm_1024 30.3 (70.8 ms);
session end 46.9 and 34.1 (63.0 ms, 73% of that reading's ceiling). Both
within ±20% of the baselines above → run published.

History on this container (for scale): pre-rework named-scalar kernel
17.3 Gops; Goto-order + flat 4×8 portable tile 21.0; static-CPUID AVX-512
dispatch 27–34; micro-calibrated dispatch 35–36 on a healthy session.
Rejected tile shapes under `#[target_feature]` (recorded in
`linalg/i64_ops.rs`): 6×16 = 7.3 Gops, 8×8 = 13.8, 4×16 = 9.8 — only the
32-lane 4×8 tile stays register-clean. The residual i64-vs-BLAS matmul gap
in Table D is ISA physics (register-blocked f64 FMA vs 64-bit integer
multiply throughput), with the kernel at ~73–88% of the measured ceiling of
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
