# Tests

| Path | Job |
|------|-----|
| [`correctness/`](correctness/) | Public API + Lua face (`cargo test`, `cargo test --features lua`). |
| [`bench/`](bench/) | Performance microbenches vs NumPy (results below). |

## Measurement

Sizes: **64, 256, 1024, 4096**.

Each cell is the **median** of several timed single calls after a short warmup (not one-shot). Setup sits outside the clock. Relative tables use **NumPy = 1.00x** when that face was measured; otherwise **—** (never invent a baseline).

**i64 matmul reference:** NumPy **float64 BLAS** on the same integer-valued inputs (not `int64@int64` — no integer BLAS, not a product bar). MatLua times are **exact wrapping i64**. See DESIGN §7.1.2.

**Roofline (engineering yardstick):** `i64_roofline` measures the running
host's achievable wrapping i64 multiply-add throughput, so i64 GEMM can be
judged as **% of machine ceiling**, not only as a ratio to f64 BLAS (which
mixes kernel quality with ISA physics — no 64-bit vector multiply below
AVX-512DQ). See the Roofline section below.

**Provenance:** every table names the host that produced it; all faces of one
table come from one host. Run-to-run noise on shared cloud hosts is real
(±10–20% observed); treat small deltas accordingly.

**M7.c plan (durable):** keep exact i64 matmul (plan A); complete honest numbers before any competitiveness threshold. M7.c is **not closed** by publishing these tables.


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

## Roofline (i64 engineering yardstick)

Host: 4 vCPU Intel Xeon @ 2.10 GHz (shared cloud container, AVX-512DQ
available), rustc 1.94.1, 2026-08-03. Gops = 2 × MACs / s. Median of 5
samples; ±10–20% run-to-run noise observed on this shared host.

| kernel | default codegen (Gops) | `target-cpu=native` (Gops) | note |
| --- | ---: | ---: | --- |
| scalar_chain | 2.30 | 2.35 | dependent MAC latency floor (context) |
| scalar_ilp8 | 3.81 | 7.77 | 8 independent accumulators |
| vec_mac_i64 | 3.81 | 6.6–8.2 | flat `c[j]+=a[j]*b[j]`; native uses `vpmullq` |
| vec_mac_f64 | 8.57 | 8.3–10.3 | ISA-physics context vs vec_mac_i64 |
| tile_4x8_i64 | 6.13 | 6.79 | GEBP-shaped register tile, 1 thread |
| tile_4x8_i64_par | 23.1 | 26.2 | aggregate, 4 threads |
| gemm_1024, pre-rework kernel | 17.3 (≈75% of tile_par) | 22.1 (≈84%) | named-scalar 8×8, NC=64 structure |
| gemm_1024 (shipped) | 21.0 (≈88%) | 22.9 (≈88%) | Goto order + flat 4×8 tile (`i64_ops`) |

Readings: (1) at **default** codegen (baseline x86-64, SSE2) the shipped GEMM
sits near the measured tile ceiling — remaining kernel-shape headroom there
is bounded; (2) the wider ISA roughly doubles the flat-loop i64 ceiling
(`vpmullq`); the reworked kernel autovectorizes from portable source, and a
6×16 tile variant reached 26.2 Gops at native but was rejected for regressing
the default build to 11.3 (recorded in `linalg/i64_ops.rs`); (3) exact-i64
ceilings are the same order as a *streaming* f64 loop, while BLAS f64 GEMM
reaches far higher through register-blocked FMA — a gap ISA physics
guarantees for exact i64 below AVX-512DQ. Hosts that build with
`-C target-cpu=native` (or `x86-64-v4`) get the higher ceiling at zero source
cost.

## Results

**Provenance:** 4 vCPU Intel Xeon @ 2.10 GHz (shared cloud container),
rustc 1.94.1 at default codegen, NumPy 2.4.6 (bundled OpenBLAS), 2026-08-03.
All faces of every table below ran on this host in one session. Shared-host
noise is ±10–20%; occasional wider spreads between the Rust and Lua faces of
the same op (e.g. i64 matmul n=1024) are contention — both faces call the
same kernel.

<!-- PERF_TABLES_START -->

### Table A — f64 absolute (ms)

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000426 | 0.000131 | 0.000231 |
| arange | 256 | 0.000583 | 0.000182 | 0.000422 |
| arange | 1024 | 0.001472 | 0.000389 | 0.000522 |
| arange | 4096 | 0.003099 | 0.006381 | 0.006181 |
| cholesky | 64 | 0.029095 | 0.013681 | 0.011188 |
| cholesky | 256 | 0.982017 | 1.2159 | 1.0394 |
| cholesky | 1024 | 29.195 | 31.883 | 25.674 |
| cholesky | 4096 | 816.717 | 771.349 | 786.670 |
| copy | 64 | 0.001122 | 0.000906 | 0.000975 |
| copy | 256 | 0.013604 | 0.012106 | 0.018612 |
| copy | 1024 | 0.681191 | 0.619803 | 0.682283 |
| copy | 4096 | 80.067 | 77.786 | 21.399 |
| dot | 64 | 0.000594 | 0.000048 | 0.000268 |
| dot | 256 | 0.000632 | 0.000128 | 0.000500 |
| dot | 1024 | 0.001390 | 0.000313 | 0.000558 |
| dot | 4096 | 0.003256 | 0.001283 | 0.001794 |
| elem_add | 64 | 0.002386 | 0.001377 | 0.001517 |
| elem_add | 256 | 0.018000 | 0.032312 | 0.037915 |
| elem_add | 1024 | 1.1849 | 0.920405 | 0.982288 |
| elem_add | 4096 | 75.146 | 65.125 | 35.526 |
| elem_add_scalar | 64 | 0.001939 | 0.000990 | 0.001028 |
| elem_add_scalar | 256 | 0.012351 | 0.015524 | 0.029976 |
| elem_add_scalar | 1024 | 0.682880 | 0.621931 | 0.752425 |
| elem_add_scalar | 4096 | 69.347 | 61.828 | 22.887 |
| elem_div | 64 | 0.002808 | 0.002899 | 0.002720 |
| elem_div | 256 | 0.039565 | 0.040452 | 0.073062 |
| elem_div | 1024 | 1.1879 | 0.903728 | 1.0181 |
| elem_div | 4096 | 70.387 | 71.713 | 32.757 |
| elem_mul | 64 | 0.002399 | 0.001598 | 0.001525 |
| elem_mul | 256 | 0.018179 | 0.031875 | 0.034485 |
| elem_mul | 1024 | 1.1125 | 0.907759 | 1.0152 |
| elem_mul | 4096 | 69.592 | 69.857 | 33.328 |
| elem_sub | 64 | 0.002391 | 0.001240 | 0.001547 |
| elem_sub | 256 | 0.017977 | 0.031913 | 0.049025 |
| elem_sub | 1024 | 1.1220 | 0.920389 | 1.0421 |
| elem_sub | 4096 | 77.450 | 72.925 | 33.923 |
| eye | 64 | 0.001807 | 0.000285 | 0.000591 |
| eye | 256 | 0.012442 | 0.010277 | 0.011525 |
| eye | 1024 | 0.435788 | 0.353338 | 0.435253 |
| eye | 4096 | 5.2877 | 4.9642 | 6.9176 |
| fill | 64 | 0.001141 | 0.000348 | 0.000655 |
| fill | 256 | 0.014814 | 0.009860 | 0.012607 |
| fill | 1024 | 0.409159 | 0.417829 | 0.474589 |
| fill | 4096 | 19.322 | 14.986 | 16.845 |
| full | 64 | 0.001780 | 0.000701 | 0.001043 |
| full | 256 | 0.015007 | 0.012332 | 0.014175 |
| full | 1024 | 0.429098 | 0.420078 | 0.527773 |
| full | 4096 | 59.652 | 57.165 | 11.941 |
| matmul | 64 | 0.009340 | 0.008600 | 0.012471 |
| matmul | 256 | 0.443209 | 0.292858 | 0.378600 |
| matmul | 1024 | 11.560 | 11.971 | 13.496 |
| matmul | 4096 | 719.954 | 862.842 | 921.282 |
| max | 64 | 0.002402 | 0.001628 | 0.001748 |
| max | 256 | 0.012381 | 0.025219 | 0.028016 |
| max | 1024 | 0.411953 | 0.534864 | 0.640212 |
| max | 4096 | 6.2137 | 10.126 | 13.071 |
| mean | 64 | 0.004137 | 0.000656 | 0.000775 |
| mean | 256 | 0.015056 | 0.010344 | 0.011618 |
| mean | 1024 | 0.499009 | 0.320549 | 0.365587 |
| mean | 4096 | 8.6900 | 6.1000 | 7.1854 |
| min | 64 | 0.001392 | 0.001621 | 0.001754 |
| min | 256 | 0.007361 | 0.025763 | 0.028103 |
| min | 1024 | 0.413801 | 0.525096 | 0.621478 |
| min | 4096 | 6.5632 | 9.9203 | 11.467 |
| norm | 64 | 0.001605 | 0.000659 | 0.000815 |
| norm | 256 | 0.006859 | 0.010486 | 0.010660 |
| norm | 1024 | 0.139443 | 0.356481 | 0.459128 |
| norm | 4096 | 3.2159 | 13.763 | 12.201 |
| ones | 64 | 0.001834 | 0.000699 | 0.000979 |
| ones | 256 | 0.015291 | 0.011998 | 0.013296 |
| ones | 1024 | 0.438212 | 0.434434 | 0.556235 |
| ones | 4096 | 59.595 | 59.801 | 21.062 |
| qr | 64 | 0.147334 | 0.691251 | 0.307918 |
| qr | 256 | 5.3383 | 5.9410 | 4.3382 |
| qr | 1024 | 142.142 | 85.382 | 67.679 |
| qr | 4096 | 5567.463 | 4190.250 | 4060.001 |
| reshape | 64 | 0.000196 | 0.000071 | 0.000227 |
| reshape | 256 | 0.000206 | 0.000072 | 0.000317 |
| reshape | 1024 | 0.000288 | 0.000078 | 0.000282 |
| reshape | 4096 | 0.001181 | 0.000620 | 0.000495 |
| solve | 64 | 0.033380 | 0.128478 | 0.182633 |
| solve | 256 | 0.607549 | 2.6083 | 2.0775 |
| solve | 1024 | 18.585 | 37.363 | 47.135 |
| solve | 4096 | 551.200 | 1369.587 | 1469.838 |
| sum | 64 | 0.002496 | 0.000652 | 0.000751 |
| sum | 256 | 0.013156 | 0.010542 | 0.010310 |
| sum | 1024 | 0.538030 | 0.320177 | 0.421743 |
| sum | 4096 | 15.956 | 12.113 | 12.239 |
| svd | 64 | 0.441365 | 0.768178 | 0.904257 |
| svd | 256 | 10.230 | 14.698 | 22.953 |
| svd | 1024 | 329.209 | 498.252 | 763.256 |
| svd | 4096 | 12894.197 | 15288.165 | 16882.935 |
| transpose | 64 | 0.002012 | 0.002702 | 0.002910 |
| transpose | 256 | 0.070919 | 0.065750 | 0.084264 |
| transpose | 1024 | 9.6577 | 2.1283 | 2.6404 |
| transpose | 4096 | 274.818 | 104.519 | 76.531 |
| zeros | 64 | 0.001265 | 0.000297 | 0.000640 |
| zeros | 256 | 0.010084 | 0.013543 | 0.010601 |
| zeros | 1024 | 0.481643 | 0.368432 | 0.379261 |
| zeros | 4096 | 0.055947 | 0.034590 | 0.005124 |

### Table B — f64 relative (NumPy = 1.00x)

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.31x | 0.54x | 1.76x |
| arange | 256 | 1.00x | 0.31x | 0.72x | 2.32x |
| arange | 1024 | 1.00x | 0.26x | 0.35x | 1.34x |
| arange | 4096 | 1.00x | 2.06x | 1.99x | 0.97x |
| cholesky | 64 | 1.00x | 0.47x | 0.38x | 0.82x |
| cholesky | 256 | 1.00x | 1.24x | 1.06x | 0.85x |
| cholesky | 1024 | 1.00x | 1.09x | 0.88x | 0.81x |
| cholesky | 4096 | 1.00x | 0.94x | 0.96x | 1.02x |
| copy | 64 | 1.00x | 0.81x | 0.87x | 1.08x |
| copy | 256 | 1.00x | 0.89x | 1.37x | 1.54x |
| copy | 1024 | 1.00x | 0.91x | 1.00x | 1.10x |
| copy | 4096 | 1.00x | 0.97x | 0.27x | 0.28x |
| dot | 64 | 1.00x | 0.08x | 0.45x | 5.58x |
| dot | 256 | 1.00x | 0.20x | 0.79x | 3.91x |
| dot | 1024 | 1.00x | 0.23x | 0.40x | 1.78x |
| dot | 4096 | 1.00x | 0.39x | 0.55x | 1.40x |
| elem_add | 64 | 1.00x | 0.58x | 0.64x | 1.10x |
| elem_add | 256 | 1.00x | 1.80x | 2.11x | 1.17x |
| elem_add | 1024 | 1.00x | 0.78x | 0.83x | 1.07x |
| elem_add | 4096 | 1.00x | 0.87x | 0.47x | 0.55x |
| elem_add_scalar | 64 | 1.00x | 0.51x | 0.53x | 1.04x |
| elem_add_scalar | 256 | 1.00x | 1.26x | 2.43x | 1.93x |
| elem_add_scalar | 1024 | 1.00x | 0.91x | 1.10x | 1.21x |
| elem_add_scalar | 4096 | 1.00x | 0.89x | 0.33x | 0.37x |
| elem_div | 64 | 1.00x | 1.03x | 0.97x | 0.94x |
| elem_div | 256 | 1.00x | 1.02x | 1.85x | 1.81x |
| elem_div | 1024 | 1.00x | 0.76x | 0.86x | 1.13x |
| elem_div | 4096 | 1.00x | 1.02x | 0.47x | 0.46x |
| elem_mul | 64 | 1.00x | 0.67x | 0.64x | 0.95x |
| elem_mul | 256 | 1.00x | 1.75x | 1.90x | 1.08x |
| elem_mul | 1024 | 1.00x | 0.82x | 0.91x | 1.12x |
| elem_mul | 4096 | 1.00x | 1.00x | 0.48x | 0.48x |
| elem_sub | 64 | 1.00x | 0.52x | 0.65x | 1.25x |
| elem_sub | 256 | 1.00x | 1.78x | 2.73x | 1.54x |
| elem_sub | 1024 | 1.00x | 0.82x | 0.93x | 1.13x |
| elem_sub | 4096 | 1.00x | 0.94x | 0.44x | 0.47x |
| eye | 64 | 1.00x | 0.16x | 0.33x | 2.07x |
| eye | 256 | 1.00x | 0.83x | 0.93x | 1.12x |
| eye | 1024 | 1.00x | 0.81x | 1.00x | 1.23x |
| eye | 4096 | 1.00x | 0.94x | 1.31x | 1.39x |
| fill | 64 | 1.00x | 0.30x | 0.57x | 1.88x |
| fill | 256 | 1.00x | 0.67x | 0.85x | 1.28x |
| fill | 1024 | 1.00x | 1.02x | 1.16x | 1.14x |
| fill | 4096 | 1.00x | 0.78x | 0.87x | 1.12x |
| full | 64 | 1.00x | 0.39x | 0.59x | 1.49x |
| full | 256 | 1.00x | 0.82x | 0.94x | 1.15x |
| full | 1024 | 1.00x | 0.98x | 1.23x | 1.26x |
| full | 4096 | 1.00x | 0.96x | 0.20x | 0.21x |
| matmul | 64 | 1.00x | 0.92x | 1.34x | 1.45x |
| matmul | 256 | 1.00x | 0.66x | 0.85x | 1.29x |
| matmul | 1024 | 1.00x | 1.04x | 1.17x | 1.13x |
| matmul | 4096 | 1.00x | 1.20x | 1.28x | 1.07x |
| max | 64 | 1.00x | 0.68x | 0.73x | 1.07x |
| max | 256 | 1.00x | 2.04x | 2.26x | 1.11x |
| max | 1024 | 1.00x | 1.30x | 1.55x | 1.20x |
| max | 4096 | 1.00x | 1.63x | 2.10x | 1.29x |
| mean | 64 | 1.00x | 0.16x | 0.19x | 1.18x |
| mean | 256 | 1.00x | 0.69x | 0.77x | 1.12x |
| mean | 1024 | 1.00x | 0.64x | 0.73x | 1.14x |
| mean | 4096 | 1.00x | 0.70x | 0.83x | 1.18x |
| min | 64 | 1.00x | 1.16x | 1.26x | 1.08x |
| min | 256 | 1.00x | 3.50x | 3.82x | 1.09x |
| min | 1024 | 1.00x | 1.27x | 1.50x | 1.18x |
| min | 4096 | 1.00x | 1.51x | 1.75x | 1.16x |
| norm | 64 | 1.00x | 0.41x | 0.51x | 1.24x |
| norm | 256 | 1.00x | 1.53x | 1.55x | 1.02x |
| norm | 1024 | 1.00x | 2.56x | 3.29x | 1.29x |
| norm | 4096 | 1.00x | 4.28x | 3.79x | 0.89x |
| ones | 64 | 1.00x | 0.38x | 0.53x | 1.40x |
| ones | 256 | 1.00x | 0.78x | 0.87x | 1.11x |
| ones | 1024 | 1.00x | 0.99x | 1.27x | 1.28x |
| ones | 4096 | 1.00x | 1.00x | 0.35x | 0.35x |
| qr | 64 | 1.00x | 4.69x | 2.09x | 0.45x |
| qr | 256 | 1.00x | 1.11x | 0.81x | 0.73x |
| qr | 1024 | 1.00x | 0.60x | 0.48x | 0.79x |
| qr | 4096 | 1.00x | 0.75x | 0.73x | 0.97x |
| reshape | 64 | 1.00x | 0.36x | 1.16x | 3.20x |
| reshape | 256 | 1.00x | 0.35x | 1.54x | 4.40x |
| reshape | 1024 | 1.00x | 0.27x | 0.98x | 3.62x |
| reshape | 4096 | 1.00x | 0.52x | 0.42x | 0.80x |
| solve | 64 | 1.00x | 3.85x | 5.47x | 1.42x |
| solve | 256 | 1.00x | 4.29x | 3.42x | 0.80x |
| solve | 1024 | 1.00x | 2.01x | 2.54x | 1.26x |
| solve | 4096 | 1.00x | 2.48x | 2.67x | 1.07x |
| sum | 64 | 1.00x | 0.26x | 0.30x | 1.15x |
| sum | 256 | 1.00x | 0.80x | 0.78x | 0.98x |
| sum | 1024 | 1.00x | 0.60x | 0.78x | 1.32x |
| sum | 4096 | 1.00x | 0.76x | 0.77x | 1.01x |
| svd | 64 | 1.00x | 1.74x | 2.05x | 1.18x |
| svd | 256 | 1.00x | 1.44x | 2.24x | 1.56x |
| svd | 1024 | 1.00x | 1.51x | 2.32x | 1.53x |
| svd | 4096 | 1.00x | 1.19x | 1.31x | 1.10x |
| transpose | 64 | 1.00x | 1.34x | 1.45x | 1.08x |
| transpose | 256 | 1.00x | 0.93x | 1.19x | 1.28x |
| transpose | 1024 | 1.00x | 0.22x | 0.27x | 1.24x |
| transpose | 4096 | 1.00x | 0.38x | 0.28x | 0.73x |
| zeros | 64 | 1.00x | 0.23x | 0.51x | 2.15x |
| zeros | 256 | 1.00x | 1.34x | 1.05x | 0.78x |
| zeros | 1024 | 1.00x | 0.76x | 0.79x | 1.03x |
| zeros | 4096 | 1.00x | 0.62x | 0.09x | 0.15x |

### Table C — i64 absolute (ms)

Matmul NumPy column is **f64 BLAS** on integer-valued data (see Measurement).

| op | n | NumPy int64 (ms) | MatLua Rust i64 (ms) | MatLua Lua i64 (ms) |
| --- | ---: | ---: | ---: | ---: |
| arange | 64 | 0.000532 | 0.000130 | 0.000225 |
| arange | 256 | 0.000484 | 0.000193 | 0.000310 |
| arange | 1024 | 0.000742 | 0.000402 | 0.000599 |
| arange | 4096 | 0.002620 | 0.006940 | 0.006549 |
| copy | 64 | 0.001158 | 0.000879 | 0.000986 |
| copy | 256 | 0.012731 | 0.011911 | 0.012912 |
| copy | 1024 | 0.642526 | 0.819116 | 0.717954 |
| copy | 4096 | 78.787 | 78.088 | 71.619 |
| dot | 64 | 0.000777 | 0.000051 | 0.000241 |
| dot | 256 | 0.000840 | 0.000145 | 0.000483 |
| dot | 1024 | 0.001231 | 0.000377 | 0.000764 |
| dot | 4096 | 0.004141 | 0.001436 | 0.001973 |
| elem_add | 64 | 0.001904 | 0.002161 | 0.002308 |
| elem_add | 256 | 0.022045 | 0.043309 | 0.038542 |
| elem_add | 1024 | 0.930652 | 1.2170 | 1.0718 |
| elem_add | 4096 | 70.976 | 89.561 | 77.706 |
| elem_div | 64 | 0.015710 | 0.007571 | 0.007717 |
| elem_div | 256 | 0.220867 | 0.125644 | 0.124111 |
| elem_div | 1024 | 3.3460 | 2.3421 | 2.1628 |
| elem_div | 4096 | 148.622 | 112.755 | 109.631 |
| elem_mul | 64 | 0.002310 | 0.002163 | 0.002321 |
| elem_mul | 256 | 0.024849 | 0.042862 | 0.037951 |
| elem_mul | 1024 | 0.936219 | 1.1622 | 1.0468 |
| elem_mul | 4096 | 74.924 | 82.468 | 74.042 |
| elem_sub | 64 | 0.001995 | 0.002152 | 0.002315 |
| elem_sub | 256 | 0.021751 | 0.042612 | 0.038143 |
| elem_sub | 1024 | 0.961197 | 1.1971 | 1.0481 |
| elem_sub | 4096 | 69.433 | 78.969 | 74.294 |
| eye | 64 | 0.001995 | 0.000283 | 0.000571 |
| eye | 256 | 0.012737 | 0.010267 | 0.010191 |
| eye | 1024 | 0.387034 | 0.460971 | 0.413666 |
| eye | 4096 | 6.0650 | 5.2287 | 4.7546 |
| fill | 64 | 0.001389 | 0.000346 | 0.000637 |
| fill | 256 | 0.015007 | 0.009871 | 0.009843 |
| fill | 1024 | 0.428273 | 0.591668 | 0.557905 |
| fill | 4096 | 21.914 | 20.148 | 15.940 |
| full | 64 | 0.001864 | 0.000693 | 0.000779 |
| full | 256 | 0.015834 | 0.009971 | 0.010118 |
| full | 1024 | 0.436365 | 0.558234 | 0.525365 |
| full | 4096 | 58.296 | 64.065 | 49.523 |
| isin | 64 | 0.017879 | 0.002760 | 0.004858 |
| isin | 256 | 0.084094 | 0.061115 | 0.072028 |
| isin | 1024 | 1.3765 | 1.0152 | 1.3900 |
| isin | 4096 | 53.324 | 78.988 | 36.689 |
| matmul | 64 | 0.008294 | 0.099707 | 0.099006 |
| matmul | 256 | 0.414520 | 4.0391 | 3.1700 |
| matmul | 1024 | 12.201 | 164.123 | 91.460 |
| matmul | 4096 | 662.838 | 6101.252 | 6098.063 |
| max | 64 | 0.001421 | 0.000850 | 0.000983 |
| max | 256 | 0.006707 | 0.013499 | 0.013675 |
| max | 1024 | 0.349972 | 0.599713 | 0.556145 |
| max | 4096 | 5.5566 | 9.9644 | 10.179 |
| min | 64 | 0.001435 | 0.000835 | 0.001028 |
| min | 256 | 0.006616 | 0.013512 | 0.013678 |
| min | 1024 | 0.308097 | 0.635083 | 0.576488 |
| min | 4096 | 5.9935 | 9.9931 | 11.118 |
| ones | 64 | 0.001777 | 0.000703 | 0.000768 |
| ones | 256 | 0.015643 | 0.010007 | 0.010133 |
| ones | 1024 | 0.447854 | 0.554368 | 0.536707 |
| ones | 4096 | 59.473 | 60.019 | 52.461 |
| reshape | 64 | 0.000330 | 0.000071 | 0.000232 |
| reshape | 256 | 0.000248 | 0.000072 | 0.000233 |
| reshape | 1024 | 0.000311 | 0.000080 | 0.000423 |
| reshape | 4096 | 0.000694 | 0.000149 | 0.000635 |
| sum | 64 | 0.001657 | 0.000307 | 0.000625 |
| sum | 256 | 0.008340 | 0.007800 | 0.007194 |
| sum | 1024 | 0.307494 | 0.424457 | 0.368268 |
| sum | 4096 | 13.631 | 14.539 | 13.684 |
| transpose | 64 | 0.002110 | 0.003160 | 0.002949 |
| transpose | 256 | 0.069850 | 0.043849 | 0.044282 |
| transpose | 1024 | 9.6875 | 2.5587 | 2.4108 |
| transpose | 4096 | 261.455 | 100.978 | 93.257 |
| unique | 64 | 0.005805 | 0.000150 | 0.000259 |
| unique | 256 | 0.018892 | 0.000312 | 0.000465 |
| unique | 1024 | 0.079945 | 0.000846 | 0.001015 |
| unique | 4096 | 0.477116 | 0.003585 | 0.007214 |
| zeros | 64 | 0.000549 | 0.000270 | 0.000626 |
| zeros | 256 | 0.009980 | 0.009929 | 0.009895 |
| zeros | 1024 | 0.373772 | 0.454945 | 0.421362 |
| zeros | 4096 | 0.012572 | 0.008434 | 0.016765 |

### Table D — i64 relative (NumPy = 1.00x)

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| arange | 64 | 1.00x | 0.24x | 0.42x | 1.73x |
| arange | 256 | 1.00x | 0.40x | 0.64x | 1.61x |
| arange | 1024 | 1.00x | 0.54x | 0.81x | 1.49x |
| arange | 4096 | 1.00x | 2.65x | 2.50x | 0.94x |
| copy | 64 | 1.00x | 0.76x | 0.85x | 1.12x |
| copy | 256 | 1.00x | 0.94x | 1.01x | 1.08x |
| copy | 1024 | 1.00x | 1.27x | 1.12x | 0.88x |
| copy | 4096 | 1.00x | 0.99x | 0.91x | 0.92x |
| dot | 64 | 1.00x | 0.07x | 0.31x | 4.73x |
| dot | 256 | 1.00x | 0.17x | 0.57x | 3.33x |
| dot | 1024 | 1.00x | 0.31x | 0.62x | 2.03x |
| dot | 4096 | 1.00x | 0.35x | 0.48x | 1.37x |
| elem_add | 64 | 1.00x | 1.13x | 1.21x | 1.07x |
| elem_add | 256 | 1.00x | 1.96x | 1.75x | 0.89x |
| elem_add | 1024 | 1.00x | 1.31x | 1.15x | 0.88x |
| elem_add | 4096 | 1.00x | 1.26x | 1.09x | 0.87x |
| elem_div | 64 | 1.00x | 0.48x | 0.49x | 1.02x |
| elem_div | 256 | 1.00x | 0.57x | 0.56x | 0.99x |
| elem_div | 1024 | 1.00x | 0.70x | 0.65x | 0.92x |
| elem_div | 4096 | 1.00x | 0.76x | 0.74x | 0.97x |
| elem_mul | 64 | 1.00x | 0.94x | 1.00x | 1.07x |
| elem_mul | 256 | 1.00x | 1.72x | 1.53x | 0.89x |
| elem_mul | 1024 | 1.00x | 1.24x | 1.12x | 0.90x |
| elem_mul | 4096 | 1.00x | 1.10x | 0.99x | 0.90x |
| elem_sub | 64 | 1.00x | 1.08x | 1.16x | 1.08x |
| elem_sub | 256 | 1.00x | 1.96x | 1.75x | 0.90x |
| elem_sub | 1024 | 1.00x | 1.25x | 1.09x | 0.88x |
| elem_sub | 4096 | 1.00x | 1.14x | 1.07x | 0.94x |
| eye | 64 | 1.00x | 0.14x | 0.29x | 2.02x |
| eye | 256 | 1.00x | 0.81x | 0.80x | 0.99x |
| eye | 1024 | 1.00x | 1.19x | 1.07x | 0.90x |
| eye | 4096 | 1.00x | 0.86x | 0.78x | 0.91x |
| fill | 64 | 1.00x | 0.25x | 0.46x | 1.84x |
| fill | 256 | 1.00x | 0.66x | 0.66x | 1.00x |
| fill | 1024 | 1.00x | 1.38x | 1.30x | 0.94x |
| fill | 4096 | 1.00x | 0.92x | 0.73x | 0.79x |
| full | 64 | 1.00x | 0.37x | 0.42x | 1.12x |
| full | 256 | 1.00x | 0.63x | 0.64x | 1.01x |
| full | 1024 | 1.00x | 1.28x | 1.20x | 0.94x |
| full | 4096 | 1.00x | 1.10x | 0.85x | 0.77x |
| isin | 64 | 1.00x | 0.15x | 0.27x | 1.76x |
| isin | 256 | 1.00x | 0.73x | 0.86x | 1.18x |
| isin | 1024 | 1.00x | 0.74x | 1.01x | 1.37x |
| isin | 4096 | 1.00x | 1.48x | 0.69x | 0.46x |
| matmul | 64 | 1.00x | 12.02x | 11.94x | 0.99x |
| matmul | 256 | 1.00x | 9.74x | 7.65x | 0.78x |
| matmul | 1024 | 1.00x | 13.45x | 7.50x | 0.56x |
| matmul | 4096 | 1.00x | 9.20x | 9.20x | 1.00x |
| max | 64 | 1.00x | 0.60x | 0.69x | 1.16x |
| max | 256 | 1.00x | 2.01x | 2.04x | 1.01x |
| max | 1024 | 1.00x | 1.71x | 1.59x | 0.93x |
| max | 4096 | 1.00x | 1.79x | 1.83x | 1.02x |
| min | 64 | 1.00x | 0.58x | 0.72x | 1.23x |
| min | 256 | 1.00x | 2.04x | 2.07x | 1.01x |
| min | 1024 | 1.00x | 2.06x | 1.87x | 0.91x |
| min | 4096 | 1.00x | 1.67x | 1.86x | 1.11x |
| ones | 64 | 1.00x | 0.40x | 0.43x | 1.09x |
| ones | 256 | 1.00x | 0.64x | 0.65x | 1.01x |
| ones | 1024 | 1.00x | 1.24x | 1.20x | 0.97x |
| ones | 4096 | 1.00x | 1.01x | 0.88x | 0.87x |
| reshape | 64 | 1.00x | 0.22x | 0.70x | 3.27x |
| reshape | 256 | 1.00x | 0.29x | 0.94x | 3.24x |
| reshape | 1024 | 1.00x | 0.26x | 1.36x | 5.29x |
| reshape | 4096 | 1.00x | 0.21x | 0.91x | 4.26x |
| sum | 64 | 1.00x | 0.19x | 0.38x | 2.04x |
| sum | 256 | 1.00x | 0.94x | 0.86x | 0.92x |
| sum | 1024 | 1.00x | 1.38x | 1.20x | 0.87x |
| sum | 4096 | 1.00x | 1.07x | 1.00x | 0.94x |
| transpose | 64 | 1.00x | 1.50x | 1.40x | 0.93x |
| transpose | 256 | 1.00x | 0.63x | 0.63x | 1.01x |
| transpose | 1024 | 1.00x | 0.26x | 0.25x | 0.94x |
| transpose | 4096 | 1.00x | 0.39x | 0.36x | 0.92x |
| unique | 64 | 1.00x | 0.03x | 0.04x | 1.73x |
| unique | 256 | 1.00x | 0.02x | 0.02x | 1.49x |
| unique | 1024 | 1.00x | 0.01x | 0.01x | 1.20x |
| unique | 4096 | 1.00x | 0.01x | 0.02x | 2.01x |
| zeros | 64 | 1.00x | 0.49x | 1.14x | 2.32x |
| zeros | 256 | 1.00x | 0.99x | 0.99x | 1.00x |
| zeros | 1024 | 1.00x | 1.22x | 1.13x | 0.93x |
| zeros | 4096 | 1.00x | 0.67x | 1.33x | 1.99x |

### Table E — i64→f64 promote-out absolute (ms)

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) |
| --- | ---: | ---: | ---: | ---: |
| cholesky | 64 | 0.020272 | 0.012422 | 0.011784 |
| cholesky | 256 | 0.967911 | 1.3772 | 1.0327 |
| cholesky | 1024 | 50.419 | 24.697 | 24.112 |
| cholesky | 4096 | 1215.196 | 807.134 | 752.159 |
| mean | 64 | 0.004931 | 0.000314 | 0.000563 |
| mean | 256 | 0.041636 | 0.006981 | 0.007181 |
| mean | 1024 | 0.858150 | 0.388974 | 0.394092 |
| mean | 4096 | 17.892 | 11.488 | 11.245 |
| median | 64 | 0.010710 | 0.003032 | 0.003101 |
| median | 256 | 0.059938 | 0.193823 | 0.064735 |
| median | 1024 | 1.5430 | 1.3678 | 1.3680 |
| median | 4096 | 86.327 | 78.902 | 80.751 |
| norm | 64 | 0.001874 | 0.001305 | 0.001363 |
| norm | 256 | 0.006824 | 0.023377 | 0.022160 |
| norm | 1024 | 0.064243 | 0.584984 | 0.588584 |
| norm | 4096 | 3.4534 | 16.843 | 17.850 |
| qr | 64 | 0.123029 | 0.794337 | 0.372822 |
| qr | 256 | 5.3356 | 8.3102 | 3.4426 |
| qr | 1024 | 132.614 | 89.780 | 68.016 |
| qr | 4096 | 4545.119 | 4518.199 | 2865.385 |
| quantile | 64 | 0.038537 | 0.003092 | 0.003116 |
| quantile | 256 | 0.144425 | 0.198296 | 0.058296 |
| quantile | 1024 | 2.7817 | 1.3608 | 1.4322 |
| quantile | 4096 | 110.731 | 78.202 | 85.178 |
| solve | 64 | 0.035014 | 0.148853 | 0.155792 |
| solve | 256 | 0.708992 | 2.5540 | 2.0510 |
| solve | 1024 | 20.935 | 37.495 | 34.581 |
| solve | 4096 | 1033.847 | 1104.739 | 1292.341 |
| std | 64 | 0.014182 | 0.002884 | 0.002980 |
| std | 256 | 0.137415 | 0.046704 | 0.050563 |
| std | 1024 | 2.8741 | 1.1476 | 1.1869 |
| std | 4096 | 112.306 | 18.949 | 26.642 |

### Table F — i64→f64 promote-out relative (NumPy = 1.00x)

| op | n | NumPy | Rust/NumPy | Lua/NumPy | Lua/Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky | 64 | 1.00x | 0.61x | 0.58x | 0.95x |
| cholesky | 256 | 1.00x | 1.42x | 1.07x | 0.75x |
| cholesky | 1024 | 1.00x | 0.49x | 0.48x | 0.98x |
| cholesky | 4096 | 1.00x | 0.66x | 0.62x | 0.93x |
| mean | 64 | 1.00x | 0.06x | 0.11x | 1.79x |
| mean | 256 | 1.00x | 0.17x | 0.17x | 1.03x |
| mean | 1024 | 1.00x | 0.45x | 0.46x | 1.01x |
| mean | 4096 | 1.00x | 0.64x | 0.63x | 0.98x |
| median | 64 | 1.00x | 0.28x | 0.29x | 1.02x |
| median | 256 | 1.00x | 3.23x | 1.08x | 0.33x |
| median | 1024 | 1.00x | 0.89x | 0.89x | 1.00x |
| median | 4096 | 1.00x | 0.91x | 0.94x | 1.02x |
| norm | 64 | 1.00x | 0.70x | 0.73x | 1.04x |
| norm | 256 | 1.00x | 3.43x | 3.25x | 0.95x |
| norm | 1024 | 1.00x | 9.11x | 9.16x | 1.01x |
| norm | 4096 | 1.00x | 4.88x | 5.17x | 1.06x |
| qr | 64 | 1.00x | 6.46x | 3.03x | 0.47x |
| qr | 256 | 1.00x | 1.56x | 0.65x | 0.41x |
| qr | 1024 | 1.00x | 0.68x | 0.51x | 0.76x |
| qr | 4096 | 1.00x | 0.99x | 0.63x | 0.63x |
| quantile | 64 | 1.00x | 0.08x | 0.08x | 1.01x |
| quantile | 256 | 1.00x | 1.37x | 0.40x | 0.29x |
| quantile | 1024 | 1.00x | 0.49x | 0.51x | 1.05x |
| quantile | 4096 | 1.00x | 0.71x | 0.77x | 1.09x |
| solve | 64 | 1.00x | 4.25x | 4.45x | 1.05x |
| solve | 256 | 1.00x | 3.60x | 2.89x | 0.80x |
| solve | 1024 | 1.00x | 1.79x | 1.65x | 0.92x |
| solve | 4096 | 1.00x | 1.07x | 1.25x | 1.17x |
| std | 64 | 1.00x | 0.20x | 0.21x | 1.03x |
| std | 256 | 1.00x | 0.34x | 0.37x | 1.08x |
| std | 1024 | 1.00x | 0.40x | 0.41x | 1.03x |
| std | 4096 | 1.00x | 0.17x | 0.24x | 1.41x |

<!-- PERF_TABLES_END -->
