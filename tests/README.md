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
  product bar). MatLua times are **exact wrapping i64**. See DESIGN §7.1.2.
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
| arange | 4096 | 0.002807 | 0.001764 | 0.29x–0.76x |
| cholesky | 4096 | 707.329 | 482.999 | 0.68x–1.01x |
| copy | 4096 | 79.387 | 21.547 | 0.27x–1.34x |
| dot | 4096 | 0.001502 | 0.001810 | 0.27x–1.21x |
| elem_add | 4096 | 71.389 | 33.796 | 0.46x–1.94x |
| elem_add_scalar | 4096 | 67.390 | 26.981 | 0.40x–1.21x |
| elem_div | 4096 | 72.601 | 33.531 | 0.46x–1.01x |
| elem_mul | 4096 | 71.141 | 33.748 | 0.42x–1.23x |
| elem_sub | 4096 | 69.914 | 34.198 | 0.42x–1.89x |
| eye | 4096 | 5.3566 | 4.9009 | 0.19x–0.95x |
| fill | 4096 | 18.987 | 18.973 | 0.48x–1.00x |
| full | 4096 | 57.292 | 19.736 | 0.26x–0.96x |
| matmul | 4096 | 538.087 | 639.778 | 0.78x–1.23x |
| max | 4096 | 5.8312 | 1.8190 | 0.31x–2.46x |
| mean | 4096 | 17.397 | 2.0098 | 0.12x–0.99x |
| min | 4096 | 8.7667 | 1.8740 | 0.21x–2.21x |
| norm | 4096 | 1.6948 | 2.2965 | 0.30x–5.25x |
| ones | 4096 | 57.650 | 20.375 | 0.28x–1.21x |
| qr | 4096 | 3063.772 | 2124.509 | 0.53x–2.09x |
| reshape | 4096 | 0.000332 | 0.000302 | 0.59x–1.12x |
| solve | 4096 | 494.056 | 700.088 | 0.71x–2.98x |
| sum | 4096 | 17.929 | 2.4702 | 0.14x–1.02x |
| svd | 4096 | 9974.817 | 13677.859 | 1.36x–1.66x |
| transpose | 4096 | 282.269 | 56.962 | 0.20x–1.00x |
| zeros | 4096 | 0.009415 | 0.005663 | 0.60x–0.93x |

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
| arange | 64 | 0.000978 | 0.000146 | 0.000282 |
| arange | 256 | 0.000710 | 0.000207 | 0.000358 |
| arange | 1024 | 0.001167 | 0.000508 | 0.000886 |
| arange | 4096 | 0.002807 | 0.002087 | 0.001764 |
| cholesky | 64 | 0.032875 | 0.019013 | 0.024540 |
| cholesky | 256 | 0.887034 | 1.1294 | 0.896047 |
| cholesky | 1024 | 17.411 | 18.938 | 14.738 |
| cholesky | 4096 | 707.329 | 694.845 | 482.999 |
| copy | 64 | 0.001704 | 0.001002 | 0.001200 |
| copy | 256 | 0.014121 | 0.013591 | 0.018916 |
| copy | 1024 | 0.692069 | 0.695256 | 0.659991 |
| copy | 4096 | 79.387 | 79.865 | 21.547 |
| dot | 64 | 0.001209 | 0.000084 | 0.000323 |
| dot | 256 | 0.000737 | 0.000123 | 0.000381 |
| dot | 1024 | 0.000767 | 0.000396 | 0.000722 |
| dot | 4096 | 0.001502 | 0.001464 | 0.001810 |
| elem_add | 64 | 0.003502 | 0.001351 | 0.001605 |
| elem_add | 256 | 0.021738 | 0.027967 | 0.042213 |
| elem_add | 1024 | 1.2604 | 0.964599 | 1.0395 |
| elem_add | 4096 | 71.389 | 68.988 | 33.796 |
| elem_add_scalar | 64 | 0.002611 | 0.001121 | 0.001331 |
| elem_add_scalar | 256 | 0.014494 | 0.013613 | 0.017591 |
| elem_add_scalar | 1024 | 0.664558 | 0.705251 | 0.747823 |
| elem_add_scalar | 4096 | 67.390 | 64.395 | 26.981 |
| elem_div | 64 | 0.004082 | 0.003183 | 0.003139 |
| elem_div | 256 | 0.046315 | 0.045926 | 0.046589 |
| elem_div | 1024 | 1.1370 | 1.0010 | 1.0882 |
| elem_div | 4096 | 72.601 | 70.280 | 33.531 |
| elem_mul | 64 | 0.003824 | 0.001340 | 0.001593 |
| elem_mul | 256 | 0.021773 | 0.025678 | 0.026832 |
| elem_mul | 1024 | 1.2578 | 1.0016 | 1.0593 |
| elem_mul | 4096 | 71.141 | 69.255 | 33.748 |
| elem_sub | 64 | 0.003774 | 0.001342 | 0.001581 |
| elem_sub | 256 | 0.022091 | 0.026282 | 0.041734 |
| elem_sub | 1024 | 1.1555 | 0.969639 | 1.0124 |
| elem_sub | 4096 | 69.914 | 69.640 | 34.198 |
| eye | 64 | 0.003611 | 0.000318 | 0.000669 |
| eye | 256 | 0.014911 | 0.011650 | 0.011713 |
| eye | 1024 | 0.392741 | 0.359807 | 0.374786 |
| eye | 4096 | 5.3566 | 4.8644 | 4.9009 |
| fill | 64 | 0.001903 | 0.000400 | 0.000922 |
| fill | 256 | 0.017567 | 0.012306 | 0.012366 |
| fill | 1024 | 0.410451 | 0.373007 | 0.368652 |
| fill | 4096 | 18.987 | 16.109 | 18.973 |
| full | 64 | 0.003238 | 0.000871 | 0.000833 |
| full | 256 | 0.019662 | 0.012434 | 0.015478 |
| full | 1024 | 0.388961 | 0.382597 | 0.373918 |
| full | 4096 | 57.292 | 55.950 | 19.736 |
| matmul | 64 | 0.012583 | 0.008264 | 0.009838 |
| matmul | 256 | 0.331902 | 0.262957 | 0.263749 |
| matmul | 1024 | 8.5669 | 10.887 | 10.505 |
| matmul | 4096 | 538.087 | 675.751 | 639.778 |
| max | 64 | 0.003040 | 0.001397 | 0.001551 |
| max | 256 | 0.009143 | 0.020520 | 0.022447 |
| max | 1024 | 0.349165 | 0.354395 | 0.391399 |
| max | 4096 | 5.8312 | 2.1100 | 1.8190 |
| mean | 64 | 0.005753 | 0.000641 | 0.000735 |
| mean | 256 | 0.017771 | 0.008789 | 0.008920 |
| mean | 1024 | 0.381269 | 0.312194 | 0.375625 |
| mean | 4096 | 17.397 | 2.0846 | 2.0098 |
| min | 64 | 0.002965 | 0.001407 | 0.001459 |
| min | 256 | 0.009135 | 0.019908 | 0.020145 |
| min | 1024 | 0.366945 | 0.351309 | 0.418614 |
| min | 4096 | 8.7667 | 2.1302 | 1.8740 |
| norm | 64 | 0.002911 | 0.000768 | 0.000881 |
| norm | 256 | 0.005648 | 0.010790 | 0.010605 |
| norm | 1024 | 0.070224 | 0.353304 | 0.368340 |
| norm | 4096 | 1.6948 | 2.2736 | 2.2965 |
| ones | 64 | 0.003044 | 0.000886 | 0.000849 |
| ones | 256 | 0.019829 | 0.012368 | 0.012524 |
| ones | 1024 | 0.401621 | 0.384564 | 0.485057 |
| ones | 4096 | 57.650 | 57.636 | 20.375 |
| qr | 64 | 0.132586 | 0.592591 | 0.277226 |
| qr | 256 | 5.1361 | 3.5392 | 3.2460 |
| qr | 1024 | 109.543 | 57.052 | 57.886 |
| qr | 4096 | 3063.772 | 2131.071 | 2124.509 |
| reshape | 64 | 0.000469 | 0.000083 | 0.000278 |
| reshape | 256 | 0.000263 | 0.000085 | 0.000294 |
| reshape | 1024 | 0.000402 | 0.000091 | 0.000302 |
| reshape | 4096 | 0.000332 | 0.000085 | 0.000302 |
| solve | 64 | 0.048344 | 0.032239 | 0.034253 |
| solve | 256 | 0.586502 | 2.1891 | 1.7450 |
| solve | 1024 | 13.385 | 28.831 | 34.717 |
| solve | 4096 | 494.056 | 554.631 | 700.088 |
| sum | 64 | 0.003291 | 0.000634 | 0.000728 |
| sum | 256 | 0.016151 | 0.008788 | 0.008884 |
| sum | 1024 | 0.380388 | 0.314204 | 0.389062 |
| sum | 4096 | 17.929 | 3.8456 | 2.4702 |
| svd | 64 | 0.317135 | 0.559389 | 0.526482 |
| svd | 256 | 9.2683 | 13.199 | 12.624 |
| svd | 1024 | 267.280 | 389.062 | 393.254 |
| svd | 4096 | 9974.817 | 13837.868 | 13677.859 |
| transpose | 64 | 0.003531 | 0.003309 | 0.003514 |
| transpose | 256 | 0.078178 | 0.055931 | 0.054362 |
| transpose | 1024 | 5.1331 | 2.0421 | 2.1656 |
| transpose | 4096 | 282.269 | 93.720 | 56.962 |
| zeros | 64 | 0.001310 | 0.000306 | 0.000826 |
| zeros | 256 | 0.013563 | 0.011371 | 0.012605 |
| zeros | 1024 | 0.408545 | 0.383439 | 0.375100 |
| zeros | 4096 | 0.009415 | 0.012432 | 0.005663 |

</details>

<details>
<summary>Table B — f64 relative (NumPy = 1.00x)</summary>

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.15x | 0.29x | 1.93x |
| arange | 256 | 1.00x | 0.29x | 0.50x | 1.73x |
| arange | 1024 | 1.00x | 0.44x | 0.76x | 1.74x |
| arange | 4096 | 1.00x | 0.74x | 0.63x | 0.85x |
| cholesky | 64 | 1.00x | 0.58x | 0.75x | 1.29x |
| cholesky | 256 | 1.00x | 1.27x | 1.01x | 0.79x |
| cholesky | 1024 | 1.00x | 1.09x | 0.85x | 0.78x |
| cholesky | 4096 | 1.00x | 0.98x | 0.68x | 0.70x |
| copy | 64 | 1.00x | 0.59x | 0.70x | 1.20x |
| copy | 256 | 1.00x | 0.96x | 1.34x | 1.39x |
| copy | 1024 | 1.00x | 1.00x | 0.95x | 0.95x |
| copy | 4096 | 1.00x | 1.01x | 0.27x | 0.27x |
| dot | 64 | 1.00x | 0.07x | 0.27x | 3.85x |
| dot | 256 | 1.00x | 0.17x | 0.52x | 3.10x |
| dot | 1024 | 1.00x | 0.52x | 0.94x | 1.82x |
| dot | 4096 | 1.00x | 0.97x | 1.21x | 1.24x |
| elem_add | 64 | 1.00x | 0.39x | 0.46x | 1.19x |
| elem_add | 256 | 1.00x | 1.29x | 1.94x | 1.51x |
| elem_add | 1024 | 1.00x | 0.77x | 0.82x | 1.08x |
| elem_add | 4096 | 1.00x | 0.97x | 0.47x | 0.49x |
| elem_add_scalar | 64 | 1.00x | 0.43x | 0.51x | 1.19x |
| elem_add_scalar | 256 | 1.00x | 0.94x | 1.21x | 1.29x |
| elem_add_scalar | 1024 | 1.00x | 1.06x | 1.13x | 1.06x |
| elem_add_scalar | 4096 | 1.00x | 0.96x | 0.40x | 0.42x |
| elem_div | 64 | 1.00x | 0.78x | 0.77x | 0.99x |
| elem_div | 256 | 1.00x | 0.99x | 1.01x | 1.01x |
| elem_div | 1024 | 1.00x | 0.88x | 0.96x | 1.09x |
| elem_div | 4096 | 1.00x | 0.97x | 0.46x | 0.48x |
| elem_mul | 64 | 1.00x | 0.35x | 0.42x | 1.19x |
| elem_mul | 256 | 1.00x | 1.18x | 1.23x | 1.04x |
| elem_mul | 1024 | 1.00x | 0.80x | 0.84x | 1.06x |
| elem_mul | 4096 | 1.00x | 0.97x | 0.47x | 0.49x |
| elem_sub | 64 | 1.00x | 0.36x | 0.42x | 1.18x |
| elem_sub | 256 | 1.00x | 1.19x | 1.89x | 1.59x |
| elem_sub | 1024 | 1.00x | 0.84x | 0.88x | 1.04x |
| elem_sub | 4096 | 1.00x | 1.00x | 0.49x | 0.49x |
| eye | 64 | 1.00x | 0.09x | 0.19x | 2.10x |
| eye | 256 | 1.00x | 0.78x | 0.79x | 1.01x |
| eye | 1024 | 1.00x | 0.92x | 0.95x | 1.04x |
| eye | 4096 | 1.00x | 0.91x | 0.91x | 1.01x |
| fill | 64 | 1.00x | 0.21x | 0.48x | 2.30x |
| fill | 256 | 1.00x | 0.70x | 0.70x | 1.00x |
| fill | 1024 | 1.00x | 0.91x | 0.90x | 0.99x |
| fill | 4096 | 1.00x | 0.85x | 1.00x | 1.18x |
| full | 64 | 1.00x | 0.27x | 0.26x | 0.96x |
| full | 256 | 1.00x | 0.63x | 0.79x | 1.24x |
| full | 1024 | 1.00x | 0.98x | 0.96x | 0.98x |
| full | 4096 | 1.00x | 0.98x | 0.34x | 0.35x |
| matmul | 64 | 1.00x | 0.66x | 0.78x | 1.19x |
| matmul | 256 | 1.00x | 0.79x | 0.79x | 1.00x |
| matmul | 1024 | 1.00x | 1.27x | 1.23x | 0.96x |
| matmul | 4096 | 1.00x | 1.26x | 1.19x | 0.95x |
| max | 64 | 1.00x | 0.46x | 0.51x | 1.11x |
| max | 256 | 1.00x | 2.24x | 2.46x | 1.09x |
| max | 1024 | 1.00x | 1.01x | 1.12x | 1.10x |
| max | 4096 | 1.00x | 0.36x | 0.31x | 0.86x |
| mean | 64 | 1.00x | 0.11x | 0.13x | 1.15x |
| mean | 256 | 1.00x | 0.49x | 0.50x | 1.01x |
| mean | 1024 | 1.00x | 0.82x | 0.99x | 1.20x |
| mean | 4096 | 1.00x | 0.12x | 0.12x | 0.96x |
| min | 64 | 1.00x | 0.47x | 0.49x | 1.04x |
| min | 256 | 1.00x | 2.18x | 2.21x | 1.01x |
| min | 1024 | 1.00x | 0.96x | 1.14x | 1.19x |
| min | 4096 | 1.00x | 0.24x | 0.21x | 0.88x |
| norm | 64 | 1.00x | 0.26x | 0.30x | 1.15x |
| norm | 256 | 1.00x | 1.91x | 1.88x | 0.98x |
| norm | 1024 | 1.00x | 5.03x | 5.25x | 1.04x |
| norm | 4096 | 1.00x | 1.34x | 1.35x | 1.01x |
| ones | 64 | 1.00x | 0.29x | 0.28x | 0.96x |
| ones | 256 | 1.00x | 0.62x | 0.63x | 1.01x |
| ones | 1024 | 1.00x | 0.96x | 1.21x | 1.26x |
| ones | 4096 | 1.00x | 1.00x | 0.35x | 0.35x |
| qr | 64 | 1.00x | 4.47x | 2.09x | 0.47x |
| qr | 256 | 1.00x | 0.69x | 0.63x | 0.92x |
| qr | 1024 | 1.00x | 0.52x | 0.53x | 1.01x |
| qr | 4096 | 1.00x | 0.70x | 0.69x | 1.00x |
| reshape | 64 | 1.00x | 0.18x | 0.59x | 3.35x |
| reshape | 256 | 1.00x | 0.32x | 1.12x | 3.46x |
| reshape | 1024 | 1.00x | 0.23x | 0.75x | 3.32x |
| reshape | 4096 | 1.00x | 0.26x | 0.91x | 3.55x |
| solve | 64 | 1.00x | 0.67x | 0.71x | 1.06x |
| solve | 256 | 1.00x | 3.73x | 2.98x | 0.80x |
| solve | 1024 | 1.00x | 2.15x | 2.59x | 1.20x |
| solve | 4096 | 1.00x | 1.12x | 1.42x | 1.26x |
| sum | 64 | 1.00x | 0.19x | 0.22x | 1.15x |
| sum | 256 | 1.00x | 0.54x | 0.55x | 1.01x |
| sum | 1024 | 1.00x | 0.83x | 1.02x | 1.24x |
| sum | 4096 | 1.00x | 0.21x | 0.14x | 0.64x |
| svd | 64 | 1.00x | 1.76x | 1.66x | 0.94x |
| svd | 256 | 1.00x | 1.42x | 1.36x | 0.96x |
| svd | 1024 | 1.00x | 1.46x | 1.47x | 1.01x |
| svd | 4096 | 1.00x | 1.39x | 1.37x | 0.99x |
| transpose | 64 | 1.00x | 0.94x | 1.00x | 1.06x |
| transpose | 256 | 1.00x | 0.72x | 0.70x | 0.97x |
| transpose | 1024 | 1.00x | 0.40x | 0.42x | 1.06x |
| transpose | 4096 | 1.00x | 0.33x | 0.20x | 0.61x |
| zeros | 64 | 1.00x | 0.23x | 0.63x | 2.70x |
| zeros | 256 | 1.00x | 0.84x | 0.93x | 1.11x |
| zeros | 1024 | 1.00x | 0.94x | 0.92x | 0.98x |
| zeros | 4096 | 1.00x | 1.32x | 0.60x | 0.46x |

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
