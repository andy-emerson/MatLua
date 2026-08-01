# Tests

Two kinds of checks live here:

| Path | Job |
|------|-----|
| [`correctness/`](correctness/) | Integration tests against the **public** crate API (and Lua face with `--features lua`). Module unit tests remain next to code under `src/`. |
| [`bench/`](bench/) | **Fair** three-way microbenches: NumPy, MatLua Rust (no Lua), MatLua Lua (interpreter + same Rust). |

## How measurement works

For each op and size \(n \in \{64, 256, 1024\}\):

1. Build inputs with a **shared generation rule** (same shapes and values).
2. Time **one call** of the op (setup outside the clock).
3. Report **median wall time** in milliseconds after warmup.
4. Faces: **NumPy** · **MatLua Rust** · **MatLua Lua**.

```bash
# Correctness
cargo test
cargo test --features lua

# Fair microbench (release) + table
python3 tests/bench/compare_fair.py
# or pieces:
cargo test --release --features lua --test fair_all -- --run --sizes 64,256,1024
python3 tests/bench/numpy_fair.py --sizes 64,256,1024
```

Open performance work is tracked as GitHub Issues (one function per issue). Close the issue and update DESIGN when the Human is satisfied with that function’s performance.

| Function | Issue |
|----------|-------|
| `reshape` | [#12](https://github.com/andy-emerson/MatLua/issues/12) |
| `min` | [#13](https://github.com/andy-emerson/MatLua/issues/13) |
| `max` | [#14](https://github.com/andy-emerson/MatLua/issues/14) |
| `norm` | [#15](https://github.com/andy-emerson/MatLua/issues/15) |

## Latest fair results (snapshot)

Host: Linux x86_64, 2 CPUs, MatLua **release**, NumPy + OpenBLAS.  
Run date: 2026-08-01 (post #12–#15: Arc reshape, chunked min/max, sum-of-squares norm; `Array::Clone` deep).  
Re-run: `python3 tests/bench/compare_fair.py`.

**Caveats:** Micro-ops under ~0.01 ms are noisy. Heavy ops at n=1024 use few samples. `reshape` is metadata + `Arc` share (not a full NumPy ndarray view type). `fill` is timed in-place on a dedicated buffer.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) | Rust/NumPy | Lua/NumPy | Lua/Rust |
|----|---:|-----------:|-----------------:|----------------:|-----------:|----------:|---------:|
| arange | 64 | 0.0006 | 0.0003 | 0.0004 | 0.45× | 0.69× | 1.52× |
| cholesky | 64 | 0.0236 | 0.0163 | 0.0157 | 0.69× | 0.67× | 0.97× |
| copy | 64 | 0.0014 | 0.0013 | 0.0034 | 0.98× | 2.48× | 2.54× |
| dot | 64 | 0.0007 | 0.0001 | 0.0002 | 0.07× | 0.27× | 3.69× |
| elem_add | 64 | 0.0026 | 0.0019 | 0.0033 | 0.74× | 1.24× | 1.68× |
| elem_add_scalar | 64 | 0.0017 | 0.0015 | 0.0030 | 0.87× | 1.79× | 2.06× |
| elem_div | 64 | 0.0034 | 0.0035 | 0.0049 | 1.04× | 1.45× | 1.39× |
| elem_mul | 64 | 0.0026 | 0.0019 | 0.0034 | 0.74× | 1.28× | 1.74× |
| elem_sub | 64 | 0.0027 | 0.0019 | 0.0033 | 0.74× | 1.25× | 1.70× |
| eye | 64 | 0.0024 | 0.0003 | 0.0016 | 0.13× | 0.66× | 5.24× |
| fill | 64 | 0.0017 | 0.0004 | 0.0006 | 0.22× | 0.34× | 1.55× |
| full | 64 | 0.0027 | 0.0005 | 0.0024 | 0.17× | 0.88× | 5.10× |
| matmul | 64 | 0.0139 | 0.0119 | 0.0134 | 0.86× | 0.97× | 1.13× |
| max | 64 | 0.0019 | 0.0014 | 0.0015 | 0.73× | 0.78× | 1.08× |
| mean | 64 | 0.0044 | 0.0007 | 0.0009 | 0.16× | 0.20× | 1.25× |
| min | 64 | 0.0019 | 0.0014 | 0.0017 | 0.73× | 0.87× | 1.19× |
| norm | 64 | 0.0027 | 0.0007 | 0.0009 | 0.27× | 0.34× | 1.25× |
| ones | 64 | 0.0028 | 0.0005 | 0.0040 | 0.16× | 1.42× | 8.67× |
| qr | 64 | 0.1218 | 0.2632 | 0.2589 | 2.16× | 2.13× | 0.98× |
| reshape | 64 | 0.0003 | 0.0001 | 0.0003 | 0.23× | 0.95× | 4.10× |
| solve | 64 | 0.0366 | 0.0721 | 0.0727 | 1.97× | 1.99× | 1.01× |
| sum | 64 | 0.0023 | 0.0007 | 0.0009 | 0.32× | 0.40× | 1.27× |
| svd | 64 | 0.2889 | 0.4891 | 0.4378 | 1.69× | 1.52× | 0.90× |
| transpose | 64 | 0.0026 | 0.0019 | 0.0039 | 0.76× | 1.50× | 1.98× |
| zeros | 64 | 0.0008 | 0.0003 | 0.0032 | 0.34× | 3.77× | 10.97× |
| arange | 256 | 0.0008 | 0.0009 | 0.0012 | 1.04× | 1.43× | 1.37× |
| cholesky | 256 | 0.8101 | 0.6236 | 0.6633 | 0.77× | 0.82× | 1.06× |
| copy | 256 | 0.0139 | 0.0132 | 0.2193 | 0.95× | 15.74× | 16.66× |
| dot | 256 | 0.0009 | 0.0001 | 0.0003 | 0.13× | 0.28× | 2.16× |
| elem_add | 256 | 0.0394 | 0.0325 | 0.2296 | 0.82× | 5.82× | 7.06× |
| elem_add_scalar | 256 | 0.0139 | 0.0236 | 0.0624 | 1.70× | 4.50× | 2.65× |
| elem_div | 256 | 0.0449 | 0.0548 | 0.0859 | 1.22× | 1.91× | 1.57× |
| elem_mul | 256 | 0.0399 | 0.0299 | 0.0710 | 0.75× | 1.78× | 2.38× |
| elem_sub | 256 | 0.0405 | 0.0306 | 0.2404 | 0.76× | 5.94× | 7.86× |
| eye | 256 | 0.0145 | 0.0134 | 0.2080 | 0.92× | 14.35× | 15.57× |
| fill | 256 | 0.0224 | 0.0116 | 0.0118 | 0.52× | 0.53× | 1.02× |
| full | 256 | 0.0233 | 0.0118 | 0.2041 | 0.50× | 8.77× | 17.36× |
| matmul | 256 | 0.5333 | 0.3789 | 0.4201 | 0.71× | 0.79× | 1.11× |
| max | 256 | 0.0109 | 0.0219 | 0.0221 | 2.02× | 2.03× | 1.01× |
| mean | 256 | 0.0168 | 0.0110 | 0.0112 | 0.65× | 0.67× | 1.02× |
| min | 256 | 0.0147 | 0.0219 | 0.0221 | 1.49× | 1.51× | 1.01× |
| norm | 256 | 0.0096 | 0.0110 | 0.0112 | 1.14× | 1.17× | 1.02× |
| ones | 256 | 0.0234 | 0.0117 | 0.2007 | 0.50× | 8.56× | 17.14× |
| qr | 256 | 4.8970 | 2.7426 | 2.4649 | 0.56× | 0.50× | 0.90× |
| reshape | 256 | 0.0003 | 0.0001 | 0.0003 | 0.23× | 0.90× | 3.97× |
| solve | 256 | 0.6492 | 1.2143 | 1.1076 | 1.87× | 1.71× | 0.91× |
| sum | 256 | 0.0146 | 0.0110 | 0.0113 | 0.75× | 0.77× | 1.03× |
| svd | 256 | 9.3906 | 13.5032 | 11.4897 | 1.44× | 1.22× | 0.85× |
| transpose | 256 | 0.0524 | 0.0614 | 0.0996 | 1.17× | 1.90× | 1.62× |
| zeros | 256 | 0.0113 | 0.0125 | 0.0374 | 1.11× | 3.31× | 2.99× |
| arange | 1024 | 0.0012 | 0.0031 | 0.0039 | 2.48× | 3.18× | 1.28× |
| cholesky | 1024 | 23.6187 | 16.9584 | 16.7286 | 0.72× | 0.71× | 0.99× |
| copy | 1024 | 0.6169 | 0.5990 | 3.8448 | 0.97× | 6.23× | 6.42× |
| dot | 1024 | 0.0009 | 0.0004 | 0.0005 | 0.40× | 0.55× | 1.37× |
| elem_add | 1024 | 1.1547 | 1.2036 | 3.5451 | 1.04× | 3.07× | 2.95× |
| elem_add_scalar | 1024 | 0.6553 | 0.9090 | 2.0371 | 1.39× | 3.11× | 2.24× |
| elem_div | 1024 | 0.9978 | 1.2065 | 2.3744 | 1.21× | 2.38× | 1.97× |
| elem_mul | 1024 | 1.1360 | 1.2061 | 3.4827 | 1.06× | 3.07× | 2.89× |
| elem_sub | 1024 | 1.1492 | 1.2007 | 3.4850 | 1.04× | 3.03× | 2.90× |
| eye | 1024 | 0.3559 | 0.3017 | 1.6350 | 0.85× | 4.59× | 5.42× |
| fill | 1024 | 0.3563 | 0.3260 | 0.3391 | 0.91× | 0.95× | 1.04× |
| full | 1024 | 0.3654 | 0.3297 | 3.2220 | 0.90× | 8.82× | 9.77× |
| matmul | 1024 | 16.9346 | 20.2720 | 21.1744 | 1.20× | 1.25× | 1.04× |
| max | 1024 | 0.3133 | 0.3782 | 0.3837 | 1.21× | 1.22× | 1.01× |
| mean | 1024 | 0.3352 | 0.2737 | 0.2947 | 0.82× | 0.88× | 1.08× |
| min | 1024 | 0.2855 | 0.3835 | 0.3896 | 1.34× | 1.37× | 1.02× |
| norm | 1024 | 0.1565 | 0.2872 | 0.3185 | 1.84× | 2.04× | 1.11× |
| ones | 1024 | 0.3965 | 0.3298 | 3.3911 | 0.83× | 8.55× | 10.28× |
| qr | 1024 | 112.4217 | 69.5297 | 58.3509 | 0.62× | 0.52× | 0.84× |
| reshape | 1024 | 0.0003 | 0.0001 | 0.0006 | 0.23× | 1.74× | 7.65× |
| solve | 1024 | 26.2932 | 30.2587 | 25.0201 | 1.15× | 0.95× | 0.83× |
| sum | 1024 | 0.3361 | 0.2766 | 0.2780 | 0.82× | 0.83× | 1.00× |
| svd | 1024 | 342.0571 | 417.6127 | 375.8791 | 1.22× | 1.10× | 0.90× |
| transpose | 1024 | 4.0206 | 5.0443 | 5.3882 | 1.25× | 1.34× | 1.07× |
| zeros | 1024 | 0.3349 | 0.2984 | 0.0640 | 0.89× | 0.19× | 0.21× |
