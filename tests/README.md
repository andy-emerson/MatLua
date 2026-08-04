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
1.94.1 at default codegen, NumPy 2.4.6 (bundled OpenBLAS), 2026-08-03.

**2026-08-04 publish-on-order (f64 table):** the f64 table below was
re-measured in a **documented degraded window** (session gate reading:
isa_tile_par 11.8 Gops vs 49.2 baseline, gemm_1024 213 ms vs 59 — ~4×
512-bit throttling) and published on **Human order**, overriding the
calibration gate. Within this table the three faces ran in the same window,
so face-vs-face ratios are fair; absolute times are understated vs a
healthy host, and cells must not be compared against other sessions'
tables or the roofline baselines. The i64 / promote tables refresh from
the same session when their suites complete.

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
| arange | 4096 | 0.002115 | 0.001502 | 0.59x–0.71x |
| copy | 4096 | 84.961 | 81.273 | 0.88x–1.03x |
| dot | 4096 | 0.003086 | 0.001819 | 0.30x–0.59x |
| elem_add | 4096 | 72.710 | 71.869 | 0.86x–1.05x |
| elem_div | 4096 | 117.717 | 97.046 | 0.54x–0.82x |
| elem_mul | 4096 | 81.040 | 69.072 | 0.74x–1.04x |
| elem_sub | 4096 | 71.099 | 70.688 | 1.02x |
| eye | 4096 | 6.2001 | 5.0839 | 0.30x–0.91x |
| fill | 4096 | 20.155 | 21.111 | 0.65x–1.09x |
| full | 4096 | 69.441 | 59.023 | 0.40x–1.02x |
| isin | 4096 | 65.439 | 80.102 | 0.23x–1.22x |
| matmul | 4096 | 557.300 | 3762.527 | 5.78x–10.13x |
| matmul_at | 4096 | 519.028 | 3913.520 | 6.40x–8.63x |
| matmul_bt | 4096 | 519.396 | 3944.109 | 5.00x–7.59x |
| max | 4096 | 5.4210 | 3.1291 | 0.51x–1.41x |
| min | 4096 | 5.9981 | 3.5485 | 0.50x–1.41x |
| ones | 4096 | 66.563 | 57.091 | 0.42x–0.95x |
| reshape | 4096 | 0.000369 | 0.000514 | 0.80x–1.39x |
| sum | 4096 | 12.654 | 4.1735 | 0.33x–1.34x |
| transpose | 4096 | 263.553 | 103.081 | 0.39x–1.36x |
| unique | 4096 | 0.530427 | 0.004106 | 0.01x–0.04x |
| zeros | 4096 | 0.009568 | 0.004782 | 0.50x–1.08x |

#### i64→f64 promote-out

| op | largest n | NumPy (ms) | MatLua Lua (ms) | Lua/NumPy, all n |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 4096 | 714.413 | 492.859 | 0.56x–0.87x |
| mean | 4096 | 26.431 | 24.745 | 0.27x–0.94x |
| median | 4096 | 122.480 | 92.214 | 0.20x–0.77x |
| norm | 4096 | 77.208 | 3.9016 | 0.05x–3.30x |
| qr | 4096 | 3053.131 | 1919.713 | 0.63x–2.13x |
| quantile | 4096 | 158.545 | 89.863 | 0.08x–0.57x |
| solve | 4096 | 596.921 | 746.530 | 0.77x–1.69x |
| std | 4096 | 143.563 | 50.327 | 0.28x–0.53x |

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
| arange | 64 | 0.000468 | 0.000142 | 0.000274 |
| arange | 256 | 0.000598 | 0.000200 | 0.000354 |
| arange | 1024 | 0.000922 | 0.000443 | 0.000599 |
| arange | 4096 | 0.002115 | 0.002021 | 0.001502 |
| copy | 64 | 0.001318 | 0.001001 | 0.001156 |
| copy | 256 | 0.015500 | 0.013780 | 0.015460 |
| copy | 1024 | 0.706958 | 0.685692 | 0.725278 |
| copy | 4096 | 84.961 | 80.771 | 81.273 |
| dot | 64 | 0.000940 | 0.000057 | 0.000280 |
| dot | 256 | 0.000980 | 0.000123 | 0.000494 |
| dot | 1024 | 0.001395 | 0.000750 | 0.000753 |
| dot | 4096 | 0.003086 | 0.001484 | 0.001819 |
| elem_add | 64 | 0.001592 | 0.001254 | 0.001548 |
| elem_add | 256 | 0.027803 | 0.020500 | 0.023859 |
| elem_add | 1024 | 0.955324 | 0.985937 | 1.0042 |
| elem_add | 4096 | 72.710 | 72.892 | 71.869 |
| elem_div | 64 | 0.016264 | 0.008626 | 0.008840 |
| elem_div | 256 | 0.238233 | 0.141168 | 0.145128 |
| elem_div | 1024 | 3.9548 | 2.3324 | 2.5919 |
| elem_div | 4096 | 117.717 | 99.810 | 97.046 |
| elem_mul | 64 | 0.002079 | 0.001347 | 0.001545 |
| elem_mul | 256 | 0.029254 | 0.021317 | 0.024769 |
| elem_mul | 1024 | 0.991020 | 1.0148 | 1.0336 |
| elem_mul | 4096 | 81.040 | 72.825 | 69.072 |
| elem_sub | 64 | 0.001574 | 0.001259 | 0.001549 |
| elem_sub | 256 | 0.026115 | 0.020375 | 0.025913 |
| elem_sub | 1024 | 0.971423 | 0.990996 | 1.0301 |
| elem_sub | 4096 | 71.099 | 72.421 | 70.688 |
| eye | 64 | 0.002254 | 0.000316 | 0.000684 |
| eye | 256 | 0.016991 | 0.011681 | 0.011687 |
| eye | 1024 | 0.400245 | 0.350254 | 0.365400 |
| eye | 4096 | 6.2001 | 7.7677 | 5.0839 |
| fill | 64 | 0.001401 | 0.000393 | 0.000904 |
| fill | 256 | 0.017652 | 0.012251 | 0.012240 |
| fill | 1024 | 0.341021 | 0.378982 | 0.373042 |
| fill | 4096 | 20.155 | 19.649 | 21.111 |
| full | 64 | 0.002569 | 0.000892 | 0.001040 |
| full | 256 | 0.018351 | 0.012460 | 0.011938 |
| full | 1024 | 0.393361 | 0.353754 | 0.399707 |
| full | 4096 | 69.441 | 57.645 | 59.023 |
| isin | 64 | 0.020638 | 0.004519 | 0.004814 |
| isin | 256 | 0.108938 | 0.071709 | 0.073645 |
| isin | 1024 | 1.4825 | 1.1347 | 1.2417 |
| isin | 4096 | 65.439 | 89.302 | 80.102 |
| matmul | 64 | 0.008641 | 0.074929 | 0.087548 |
| matmul | 256 | 0.369111 | 2.3055 | 2.1325 |
| matmul | 1024 | 8.5158 | 92.208 | 75.875 |
| matmul | 4096 | 557.300 | 3738.249 | 3762.527 |
| matmul_at | 64 | 0.009708 | 0.089490 | 0.083822 |
| matmul_at | 256 | 0.378298 | 2.3234 | 2.7674 |
| matmul_at | 1024 | 9.8849 | 70.515 | 63.242 |
| matmul_at | 4096 | 519.028 | 3876.565 | 3913.520 |
| matmul_bt | 64 | 0.015927 | 0.078169 | 0.079674 |
| matmul_bt | 256 | 0.373954 | 2.8310 | 2.7036 |
| matmul_bt | 1024 | 9.9496 | 64.682 | 69.838 |
| matmul_bt | 4096 | 519.396 | 3916.443 | 3944.109 |
| max | 64 | 0.001774 | 0.000718 | 0.000913 |
| max | 256 | 0.008253 | 0.011244 | 0.011634 |
| max | 1024 | 0.316083 | 0.318544 | 0.358912 |
| max | 4096 | 5.4210 | 2.8977 | 3.1291 |
| min | 64 | 0.001775 | 0.000719 | 0.000885 |
| min | 256 | 0.008241 | 0.011342 | 0.011609 |
| min | 1024 | 0.337241 | 0.343419 | 0.378131 |
| min | 4096 | 5.9981 | 3.3892 | 3.5485 |
| ones | 64 | 0.002195 | 0.000894 | 0.000915 |
| ones | 256 | 0.018438 | 0.012498 | 0.013200 |
| ones | 1024 | 0.401860 | 0.406950 | 0.382351 |
| ones | 4096 | 66.563 | 58.411 | 57.091 |
| reshape | 64 | 0.000258 | 0.000085 | 0.000310 |
| reshape | 256 | 0.000512 | 0.000083 | 0.000412 |
| reshape | 1024 | 0.000301 | 0.000083 | 0.000315 |
| reshape | 4096 | 0.000369 | 0.000105 | 0.000514 |
| sum | 64 | 0.001923 | 0.000710 | 0.000878 |
| sum | 256 | 0.009979 | 0.011397 | 0.011595 |
| sum | 1024 | 0.330762 | 0.332085 | 0.441736 |
| sum | 4096 | 12.654 | 4.6726 | 4.1735 |
| transpose | 64 | 0.002573 | 0.003273 | 0.003500 |
| transpose | 256 | 0.082726 | 0.052388 | 0.058210 |
| transpose | 1024 | 4.8379 | 2.1517 | 2.2072 |
| transpose | 4096 | 263.553 | 129.244 | 103.081 |
| unique | 64 | 0.006902 | 0.000152 | 0.000287 |
| unique | 256 | 0.029633 | 0.000363 | 0.000572 |
| unique | 1024 | 0.094853 | 0.000959 | 0.001201 |
| unique | 4096 | 0.530427 | 0.003912 | 0.004106 |
| zeros | 64 | 0.000708 | 0.000316 | 0.000763 |
| zeros | 256 | 0.013116 | 0.011288 | 0.011534 |
| zeros | 1024 | 0.381016 | 0.423131 | 0.355512 |
| zeros | 4096 | 0.009568 | 0.008578 | 0.004782 |

</details>

<details>
<summary>Table D — i64 relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.30x | 0.59x | 1.93x |
| arange | 256 | 1.00x | 0.33x | 0.59x | 1.77x |
| arange | 1024 | 1.00x | 0.48x | 0.65x | 1.35x |
| arange | 4096 | 1.00x | 0.96x | 0.71x | 0.74x |
| copy | 64 | 1.00x | 0.76x | 0.88x | 1.15x |
| copy | 256 | 1.00x | 0.89x | 1.00x | 1.12x |
| copy | 1024 | 1.00x | 0.97x | 1.03x | 1.06x |
| copy | 4096 | 1.00x | 0.95x | 0.96x | 1.01x |
| dot | 64 | 1.00x | 0.06x | 0.30x | 4.91x |
| dot | 256 | 1.00x | 0.13x | 0.50x | 4.02x |
| dot | 1024 | 1.00x | 0.54x | 0.54x | 1.00x |
| dot | 4096 | 1.00x | 0.48x | 0.59x | 1.23x |
| elem_add | 64 | 1.00x | 0.79x | 0.97x | 1.23x |
| elem_add | 256 | 1.00x | 0.74x | 0.86x | 1.16x |
| elem_add | 1024 | 1.00x | 1.03x | 1.05x | 1.02x |
| elem_add | 4096 | 1.00x | 1.00x | 0.99x | 0.99x |
| elem_div | 64 | 1.00x | 0.53x | 0.54x | 1.02x |
| elem_div | 256 | 1.00x | 0.59x | 0.61x | 1.03x |
| elem_div | 1024 | 1.00x | 0.59x | 0.66x | 1.11x |
| elem_div | 4096 | 1.00x | 0.85x | 0.82x | 0.97x |
| elem_mul | 64 | 1.00x | 0.65x | 0.74x | 1.15x |
| elem_mul | 256 | 1.00x | 0.73x | 0.85x | 1.16x |
| elem_mul | 1024 | 1.00x | 1.02x | 1.04x | 1.02x |
| elem_mul | 4096 | 1.00x | 0.90x | 0.85x | 0.95x |
| elem_sub | 64 | 1.00x | 0.80x | 0.98x | 1.23x |
| elem_sub | 256 | 1.00x | 0.78x | 0.99x | 1.27x |
| elem_sub | 1024 | 1.00x | 1.02x | 1.06x | 1.04x |
| elem_sub | 4096 | 1.00x | 1.02x | 0.99x | 0.98x |
| eye | 64 | 1.00x | 0.14x | 0.30x | 2.16x |
| eye | 256 | 1.00x | 0.69x | 0.69x | 1.00x |
| eye | 1024 | 1.00x | 0.88x | 0.91x | 1.04x |
| eye | 4096 | 1.00x | 1.25x | 0.82x | 0.65x |
| fill | 64 | 1.00x | 0.28x | 0.65x | 2.30x |
| fill | 256 | 1.00x | 0.69x | 0.69x | 1.00x |
| fill | 1024 | 1.00x | 1.11x | 1.09x | 0.98x |
| fill | 4096 | 1.00x | 0.97x | 1.05x | 1.07x |
| full | 64 | 1.00x | 0.35x | 0.40x | 1.17x |
| full | 256 | 1.00x | 0.68x | 0.65x | 0.96x |
| full | 1024 | 1.00x | 0.90x | 1.02x | 1.13x |
| full | 4096 | 1.00x | 0.83x | 0.85x | 1.02x |
| isin | 64 | 1.00x | 0.22x | 0.23x | 1.07x |
| isin | 256 | 1.00x | 0.66x | 0.68x | 1.03x |
| isin | 1024 | 1.00x | 0.77x | 0.84x | 1.09x |
| isin | 4096 | 1.00x | 1.36x | 1.22x | 0.90x |
| matmul | 64 | 1.00x | 8.67x | 10.13x | 1.17x |
| matmul | 256 | 1.00x | 6.25x | 5.78x | 0.92x |
| matmul | 1024 | 1.00x | 10.83x | 8.91x | 0.82x |
| matmul | 4096 | 1.00x | 6.71x | 6.75x | 1.01x |
| matmul_at | 64 | 1.00x | 9.22x | 8.63x | 0.94x |
| matmul_at | 256 | 1.00x | 6.14x | 7.32x | 1.19x |
| matmul_at | 1024 | 1.00x | 7.13x | 6.40x | 0.90x |
| matmul_at | 4096 | 1.00x | 7.47x | 7.54x | 1.01x |
| matmul_bt | 64 | 1.00x | 4.91x | 5.00x | 1.02x |
| matmul_bt | 256 | 1.00x | 7.57x | 7.23x | 0.95x |
| matmul_bt | 1024 | 1.00x | 6.50x | 7.02x | 1.08x |
| matmul_bt | 4096 | 1.00x | 7.54x | 7.59x | 1.01x |
| max | 64 | 1.00x | 0.40x | 0.51x | 1.27x |
| max | 256 | 1.00x | 1.36x | 1.41x | 1.03x |
| max | 1024 | 1.00x | 1.01x | 1.14x | 1.13x |
| max | 4096 | 1.00x | 0.53x | 0.58x | 1.08x |
| min | 64 | 1.00x | 0.41x | 0.50x | 1.23x |
| min | 256 | 1.00x | 1.38x | 1.41x | 1.02x |
| min | 1024 | 1.00x | 1.02x | 1.12x | 1.10x |
| min | 4096 | 1.00x | 0.57x | 0.59x | 1.05x |
| ones | 64 | 1.00x | 0.41x | 0.42x | 1.02x |
| ones | 256 | 1.00x | 0.68x | 0.72x | 1.06x |
| ones | 1024 | 1.00x | 1.01x | 0.95x | 0.94x |
| ones | 4096 | 1.00x | 0.88x | 0.86x | 0.98x |
| reshape | 64 | 1.00x | 0.33x | 1.20x | 3.65x |
| reshape | 256 | 1.00x | 0.16x | 0.80x | 4.96x |
| reshape | 1024 | 1.00x | 0.28x | 1.05x | 3.80x |
| reshape | 4096 | 1.00x | 0.28x | 1.39x | 4.90x |
| sum | 64 | 1.00x | 0.37x | 0.46x | 1.24x |
| sum | 256 | 1.00x | 1.14x | 1.16x | 1.02x |
| sum | 1024 | 1.00x | 1.00x | 1.34x | 1.33x |
| sum | 4096 | 1.00x | 0.37x | 0.33x | 0.89x |
| transpose | 64 | 1.00x | 1.27x | 1.36x | 1.07x |
| transpose | 256 | 1.00x | 0.63x | 0.70x | 1.11x |
| transpose | 1024 | 1.00x | 0.44x | 0.46x | 1.03x |
| transpose | 4096 | 1.00x | 0.49x | 0.39x | 0.80x |
| unique | 64 | 1.00x | 0.02x | 0.04x | 1.89x |
| unique | 256 | 1.00x | 0.01x | 0.02x | 1.58x |
| unique | 1024 | 1.00x | 0.01x | 0.01x | 1.25x |
| unique | 4096 | 1.00x | 0.01x | 0.01x | 1.05x |
| zeros | 64 | 1.00x | 0.45x | 1.08x | 2.41x |
| zeros | 256 | 1.00x | 0.86x | 0.88x | 1.02x |
| zeros | 1024 | 1.00x | 1.11x | 0.93x | 0.84x |
| zeros | 4096 | 1.00x | 0.90x | 0.50x | 0.56x |

</details>

<details>
<summary>Table E — i64→f64 promote-out absolute (ms)</summary>

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 64 | 0.022591 | 0.019272 | 0.019625 |
| cholesky | 256 | 1.2024 | 0.916057 | 0.936306 |
| cholesky | 1024 | 26.420 | 14.299 | 14.853 |
| cholesky | 4096 | 714.413 | 456.424 | 492.859 |
| mean | 64 | 0.006657 | 0.001644 | 0.001782 |
| mean | 256 | 0.061210 | 0.025534 | 0.025673 |
| mean | 1024 | 0.829003 | 0.512956 | 0.496312 |
| mean | 4096 | 26.431 | 23.726 | 24.745 |
| median | 64 | 0.018710 | 0.003480 | 0.003739 |
| median | 256 | 0.083350 | 0.163724 | 0.058745 |
| median | 1024 | 1.6351 | 1.6173 | 1.2584 |
| median | 4096 | 122.480 | 95.207 | 92.214 |
| norm | 64 | 0.004244 | 0.020255 | 0.013987 |
| norm | 256 | 0.054854 | 0.038806 | 0.029085 |
| norm | 1024 | 0.984959 | 0.459550 | 0.407572 |
| norm | 4096 | 77.208 | 4.4199 | 3.9016 |
| qr | 64 | 0.117623 | 0.414549 | 0.250229 |
| qr | 256 | 5.1411 | 7.8515 | 5.5903 |
| qr | 1024 | 101.041 | 58.715 | 73.209 |
| qr | 4096 | 3053.131 | 2616.128 | 1919.713 |
| quantile | 64 | 0.049181 | 0.003500 | 0.003762 |
| quantile | 256 | 0.208905 | 0.160741 | 0.055427 |
| quantile | 1024 | 2.9736 | 1.3067 | 1.3333 |
| quantile | 4096 | 158.545 | 96.181 | 89.863 |
| solve | 64 | 0.037972 | 0.032507 | 0.029302 |
| solve | 256 | 1.0554 | 2.4484 | 1.7800 |
| solve | 1024 | 18.297 | 32.013 | 22.667 |
| solve | 4096 | 596.921 | 604.396 | 746.530 |
| std | 64 | 0.016444 | 0.005129 | 0.004606 |
| std | 256 | 0.133008 | 0.073773 | 0.070918 |
| std | 1024 | 2.4397 | 1.3668 | 1.2714 |
| std | 4096 | 143.563 | 46.765 | 50.327 |

</details>

<details>
<summary>Table F — i64→f64 promote-out relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky | 64 | 1.00x | 0.85x | 0.87x | 1.02x |
| cholesky | 256 | 1.00x | 0.76x | 0.78x | 1.02x |
| cholesky | 1024 | 1.00x | 0.54x | 0.56x | 1.04x |
| cholesky | 4096 | 1.00x | 0.64x | 0.69x | 1.08x |
| mean | 64 | 1.00x | 0.25x | 0.27x | 1.08x |
| mean | 256 | 1.00x | 0.42x | 0.42x | 1.01x |
| mean | 1024 | 1.00x | 0.62x | 0.60x | 0.97x |
| mean | 4096 | 1.00x | 0.90x | 0.94x | 1.04x |
| median | 64 | 1.00x | 0.19x | 0.20x | 1.07x |
| median | 256 | 1.00x | 1.96x | 0.70x | 0.36x |
| median | 1024 | 1.00x | 0.99x | 0.77x | 0.78x |
| median | 4096 | 1.00x | 0.78x | 0.75x | 0.97x |
| norm | 64 | 1.00x | 4.77x | 3.30x | 0.69x |
| norm | 256 | 1.00x | 0.71x | 0.53x | 0.75x |
| norm | 1024 | 1.00x | 0.47x | 0.41x | 0.89x |
| norm | 4096 | 1.00x | 0.06x | 0.05x | 0.88x |
| qr | 64 | 1.00x | 3.52x | 2.13x | 0.60x |
| qr | 256 | 1.00x | 1.53x | 1.09x | 0.71x |
| qr | 1024 | 1.00x | 0.58x | 0.72x | 1.25x |
| qr | 4096 | 1.00x | 0.86x | 0.63x | 0.73x |
| quantile | 64 | 1.00x | 0.07x | 0.08x | 1.07x |
| quantile | 256 | 1.00x | 0.77x | 0.27x | 0.34x |
| quantile | 1024 | 1.00x | 0.44x | 0.45x | 1.02x |
| quantile | 4096 | 1.00x | 0.61x | 0.57x | 0.93x |
| solve | 64 | 1.00x | 0.86x | 0.77x | 0.90x |
| solve | 256 | 1.00x | 2.32x | 1.69x | 0.73x |
| solve | 1024 | 1.00x | 1.75x | 1.24x | 0.71x |
| solve | 4096 | 1.00x | 1.01x | 1.25x | 1.24x |
| std | 64 | 1.00x | 0.31x | 0.28x | 0.90x |
| std | 256 | 1.00x | 0.55x | 0.53x | 0.96x |
| std | 1024 | 1.00x | 0.56x | 0.52x | 0.93x |
| std | 4096 | 1.00x | 0.33x | 0.35x | 1.08x |

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
