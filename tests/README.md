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
Run date: 2026-08-01. Re-run with `python3 tests/bench/compare_fair.py`; numbers move by machine.

**Caveats:** NumPy `reshape` is usually a view (not a full copy). Lua `fill` in an early spike was measured as copy+fill; production `a:fill` is in-place. Heavy ops at n=1024 use few samples — treat tiny ratio differences as noise.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) | Rust/NumPy | Lua/NumPy | Lua/Rust |
|----|---:|-----------:|-----------------:|----------------:|-----------:|----------:|---------:|
| zeros | 64 | 0.0008 | 0.0003 | 0.0036 | 0.32× | 4.43× | 14.05× |
| ones | 64 | 0.0028 | 0.0004 | 0.0131 | 0.15× | 4.63× | 30.52× |
| full | 64 | 0.0027 | 0.0004 | 0.0134 | 0.16× | 4.99× | 31.20× |
| eye | 64 | 0.0024 | 0.0003 | 0.0143 | 0.11× | 5.93× | 54.25× |
| arange | 64 | 0.0006 | 0.0002 | 0.0005 | 0.40× | 0.76× | 1.89× |
| copy | 64 | 0.0014 | 0.0011 | 0.0037 | 0.82× | 2.74× | 3.36× |
| reshape | 64 | 0.0003 | 0.0009 | 0.0034 | 3.14× | 11.25× | 3.58× |
| fill | 64 | 0.0031 | 0.0004 | 0.0039 | 0.13× | 1.25× | 9.84× |
| elem_add | 64 | 0.0017 | 0.0019 | 0.0176 | 1.13× | 10.30× | 9.11× |
| elem_sub | 64 | 0.0017 | 0.0017 | 0.0034 | 1.02× | 2.01× | 1.97× |
| elem_mul | 64 | 0.0017 | 0.0017 | 0.0034 | 1.01× | 2.00× | 1.98× |
| elem_div | 64 | 0.0034 | 0.0035 | 0.0051 | 1.02× | 1.49× | 1.46× |
| elem_add_scalar | 64 | 0.0017 | 0.0016 | 0.0039 | 0.93× | 2.34× | 2.52× |
| sum | 64 | 0.0023 | 0.0007 | 0.0008 | 0.31× | 0.36× | 1.14× |
| mean | 64 | 0.0043 | 0.0007 | 0.0008 | 0.17× | 0.19× | 1.14× |
| min | 64 | 0.0019 | 0.0042 | 0.0043 | 2.20× | 2.25× | 1.02× |
| max | 64 | 0.0019 | 0.0041 | 0.0043 | 2.13× | 2.25× | 1.05× |
| transpose | 64 | 0.0025 | 0.0020 | 0.0036 | 0.78× | 1.42× | 1.82× |
| dot | 64 | 0.0007 | 0.0001 | 0.0002 | 0.09× | 0.32× | 3.52× |
| norm | 64 | 0.0027 | 0.0023 | 0.0022 | 0.85× | 0.83× | 0.98× |
| matmul | 64 | 0.0110 | 0.0122 | 0.0120 | 1.10× | 1.09× | 0.98× |
| solve | 64 | 0.0357 | 0.0778 | 0.0810 | 2.18× | 2.27× | 1.04× |
| cholesky | 64 | 0.0225 | 0.0161 | 0.0174 | 0.72× | 0.77× | 1.08× |
| qr | 64 | 0.1121 | 0.2841 | 0.2744 | 2.53× | 2.45× | 0.97× |
| svd | 64 | 0.3165 | 0.4854 | 0.4272 | 1.53× | 1.35× | 0.88× |
| zeros | 256 | 0.0113 | 0.0121 | 0.0585 | 1.07× | 5.19× | 4.85× |
| ones | 256 | 0.0235 | 0.0117 | 0.2005 | 0.50× | 8.54× | 17.10× |
| full | 256 | 0.0233 | 0.0117 | 0.1495 | 0.50× | 6.42× | 12.78× |
| eye | 256 | 0.0159 | 0.0133 | 0.1751 | 0.84× | 11.04× | 13.14× |
| arange | 256 | 0.0008 | 0.0008 | 0.0011 | 1.04× | 1.38× | 1.32× |
| copy | 256 | 0.0135 | 0.0132 | 0.2318 | 0.97× | 17.15× | 17.62× |
| reshape | 256 | 0.0003 | 0.0131 | 0.2465 | 43.05× | 808× | 18.77× |
| fill | 256 | 0.0359 | 0.0117 | 0.2685 | 0.32× | 7.48× | 23.04× |
| elem_add | 256 | 0.0409 | 0.0332 | 0.2466 | 0.81× | 6.03× | 7.44× |
| elem_sub | 256 | 0.0406 | 0.0330 | 0.0847 | 0.81× | 2.08× | 2.57× |
| elem_mul | 256 | 0.0406 | 0.0311 | 0.2195 | 0.77× | 5.41× | 7.05× |
| elem_div | 256 | 0.0450 | 0.0545 | 0.0932 | 1.21× | 2.07× | 1.71× |
| elem_add_scalar | 256 | 0.0139 | 0.0235 | 0.0689 | 1.70× | 4.97× | 2.93× |
| sum | 256 | 0.0150 | 0.0110 | 0.0112 | 0.73× | 0.74× | 1.02× |
| mean | 256 | 0.0165 | 0.0110 | 0.0112 | 0.67× | 0.68× | 1.02× |
| min | 256 | 0.0127 | 0.0659 | 0.0664 | 5.19× | 5.22× | 1.01× |
| max | 256 | 0.0124 | 0.0660 | 0.0662 | 5.31× | 5.33× | 1.00× |
| transpose | 256 | 0.0521 | 0.0615 | 0.1035 | 1.18× | 1.99× | 1.68× |
| dot | 256 | 0.0009 | 0.0002 | 0.0003 | 0.18× | 0.35× | 1.93× |
| norm | 256 | 0.0094 | 0.0328 | 0.0331 | 3.48× | 3.51× | 1.01× |
| matmul | 256 | 0.5626 | 0.3856 | 0.4538 | 0.69× | 0.81× | 1.18× |
| solve | 256 | 0.6649 | 1.2826 | 1.1664 | 1.93× | 1.75× | 0.91× |
| cholesky | 256 | 0.8827 | 0.6129 | 0.7475 | 0.69× | 0.85× | 1.22× |
| qr | 256 | 5.5965 | 3.0674 | 2.9375 | 0.55× | 0.52× | 0.96× |
| svd | 256 | 10.6156 | 10.6182 | 11.2834 | 1.00× | 1.06× | 1.06× |
| zeros | 1024 | 0.3676 | 0.3451 | 0.0690 | 0.94× | 0.19× | 0.20× |
| ones | 1024 | 0.3817 | 0.4012 | 3.4374 | 1.05× | 9.01× | 8.57× |
| full | 1024 | 0.3785 | 0.3712 | 3.3359 | 0.98× | 8.81× | 8.99× |
| eye | 1024 | 0.3657 | 0.3354 | 2.2997 | 0.92× | 6.29× | 6.86× |
| arange | 1024 | 0.0012 | 0.0038 | 0.0042 | 3.16× | 3.44× | 1.09× |
| copy | 1024 | 0.6576 | 0.6493 | 1.0406 | 0.99× | 1.58× | 1.60× |
| reshape | 1024 | 0.0003 | 0.6511 | 3.9825 | (view vs copy) | (view vs copy) | 6.12× |
| fill | 1024 | 1.0330 | 0.3726 | 5.1485 | 0.36× | 4.98× | 13.82× |
| elem_add | 1024 | 1.4310 | 1.3644 | 4.1183 | 0.95× | 2.88× | 3.02× |
| elem_sub | 1024 | 1.4303 | 1.3634 | 4.1192 | 0.95× | 2.88× | 3.02× |
| elem_mul | 1024 | 1.4331 | 1.3638 | 32.3520 | 0.95× | 22.58× | 23.72× |
| elem_div | 1024 | 1.3665 | 1.3733 | 2.4404 | 1.00× | 1.79× | 1.78× |
| elem_add_scalar | 1024 | 0.7466 | 1.0803 | 2.0914 | 1.45× | 2.80× | 1.94× |
| sum | 1024 | 0.3934 | 0.3421 | 0.3230 | 0.87× | 0.82× | 0.94× |
| mean | 1024 | 0.4194 | 0.3299 | 0.3239 | 0.79× | 0.77× | 0.98× |
| min | 1024 | 0.3356 | 1.1065 | 1.0876 | 3.30× | 3.24× | 0.98× |
| max | 1024 | 0.3350 | 1.0928 | 1.1149 | 3.26× | 3.33× | 1.02× |
| transpose | 1024 | 6.7787 | 7.0202 | 7.6748 | 1.04× | 1.13× | 1.09× |
| dot | 1024 | 0.0008 | 0.0006 | 0.0007 | 0.67× | 0.90× | 1.34× |
| norm | 1024 | 0.1779 | 0.6382 | 0.6072 | 3.59× | 3.41× | 0.95× |
| matmul | 1024 | 17.5607 | 22.2391 | 22.8179 | 1.27× | 1.30× | 1.03× |
| solve | 1024 | 27.1718 | 35.1404 | 30.4433 | 1.29× | 1.12× | 0.87× |
| cholesky | 1024 | 29.2298 | 25.6419 | 21.4684 | 0.88× | 0.73× | 0.84× |
| qr | 1024 | 135.5692 | 92.5847 | 59.7139 | 0.68× | 0.44× | 0.64× |
| svd | 1024 | 377.9728 | 445.8364 | 402.7613 | 1.18× | 1.07× | 0.90× |
