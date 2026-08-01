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
Run date: 2026-08-01 (session closeout: GCSTEP policy, single-touch buffers, blocked min/transpose).  
Re-run: `python3 tests/bench/compare_fair.py`.

**Caveats:** Micro-ops under ~0.01 ms are noisy (ratios can look large). Blocked kernels are O(n)/O(mn) and meant to **help** large n via cache, not only n≤1024. Hosts with their own `lua_State` should call `matlua::lua::enable_generational_gc`.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) | Rust/NumPy | Lua/NumPy | Lua/Rust |
|----|---:|-----------:|-----------------:|----------------:|-----------:|----------:|---------:|
| arange | 64 | 0.0006 | 0.0002 | 0.0003 | 0.27× | 0.56× | 2.11× |
| cholesky | 64 | 0.0226 | 0.0159 | 0.0157 | 0.70× | 0.69× | 0.98× |
| copy | 64 | 0.0014 | 0.0010 | 0.0022 | 0.74× | 1.60× | 2.16× |
| dot | 64 | 0.0007 | 0.0001 | 0.0002 | 0.08× | 0.29× | 3.79× |
| elem_add | 64 | 0.0027 | 0.0015 | 0.0021 | 0.55× | 0.80× | 1.44× |
| elem_add_scalar | 64 | 0.0017 | 0.0010 | 0.0049 | 0.60× | 2.93× | 4.84× |
| elem_div | 64 | 0.0034 | 0.0032 | 0.0038 | 0.94× | 1.13× | 1.21× |
| elem_mul | 64 | 0.0027 | 0.0013 | 0.0025 | 0.49× | 0.92× | 1.87× |
| elem_sub | 64 | 0.0026 | 0.0015 | 0.0031 | 0.56× | 1.17× | 2.08× |
| eye | 64 | 0.0024 | 0.0007 | 0.0012 | 0.31× | 0.50× | 1.64× |
| fill | 64 | 0.0017 | 0.0007 | 0.0009 | 0.43× | 0.53× | 1.25× |
| full | 64 | 0.0027 | 0.0009 | 0.0034 | 0.31× | 1.23× | 3.97× |
| matmul | 64 | 0.0107 | 0.0277 | 0.0121 | 2.57× | 1.13× | 0.44× |
| max | 64 | 0.0019 | 0.0011 | 0.0012 | 0.55× | 0.61× | 1.11× |
| mean | 64 | 0.0043 | 0.0007 | 0.0008 | 0.17× | 0.19× | 1.13× |
| min | 64 | 0.0019 | 0.0011 | 0.0012 | 0.58× | 0.60× | 1.04× |
| norm | 64 | 0.0026 | 0.0007 | 0.0009 | 0.28× | 0.35× | 1.26× |
| ones | 64 | 0.0029 | 0.0009 | 0.0015 | 0.30× | 0.53× | 1.78× |
| qr | 64 | 0.1216 | 0.2748 | 0.2875 | 2.26× | 2.36× | 1.05× |
| reshape | 64 | 0.0003 | 0.0001 | 0.0004 | 0.27× | 1.23× | 4.48× |
| solve | 64 | 0.0361 | 0.0763 | 0.0743 | 2.11× | 2.06× | 0.97× |
| sum | 64 | 0.0023 | 0.0007 | 0.0009 | 0.32× | 0.40× | 1.25× |
| svd | 64 | 0.3005 | 0.4758 | 0.4308 | 1.58× | 1.43× | 0.91× |
| transpose | 64 | 0.0026 | 0.0030 | 0.0041 | 1.14× | 1.56× | 1.37× |
| zeros | 64 | 0.0008 | 0.0008 | 0.0040 | 0.91× | 4.82× | 5.30× |
| arange | 256 | 0.0008 | 0.0004 | 0.0005 | 0.50× | 0.64× | 1.29× |
| cholesky | 256 | 0.8351 | 0.6812 | 0.6574 | 0.82× | 0.79× | 0.97× |
| copy | 256 | 0.0140 | 0.0133 | 0.0145 | 0.95× | 1.04× | 1.09× |
| dot | 256 | 0.0008 | 0.0001 | 0.0002 | 0.15× | 0.31× | 2.03× |
| elem_add | 256 | 0.0403 | 0.0228 | 0.0240 | 0.57× | 0.60× | 1.05× |
| elem_add_scalar | 256 | 0.0164 | 0.0128 | 0.0200 | 0.78× | 1.22× | 1.57× |
| elem_div | 256 | 0.0451 | 0.0444 | 0.0458 | 0.98× | 1.02× | 1.03× |
| elem_mul | 256 | 0.0401 | 0.0207 | 0.0253 | 0.52× | 0.63× | 1.22× |
| elem_sub | 256 | 0.0401 | 0.0220 | 0.0269 | 0.55× | 0.67× | 1.22× |
| eye | 256 | 0.0145 | 0.0125 | 0.0117 | 0.86× | 0.80× | 0.93× |
| fill | 256 | 0.0224 | 0.0116 | 0.0117 | 0.52× | 0.52× | 1.01× |
| full | 256 | 0.0234 | 0.0112 | 0.0123 | 0.48× | 0.53× | 1.10× |
| matmul | 256 | 0.5306 | 0.4047 | 0.5951 | 0.76× | 1.12× | 1.47× |
| max | 256 | 0.0096 | 0.0155 | 0.0159 | 1.61× | 1.65× | 1.03× |
| mean | 256 | 0.0167 | 0.0110 | 0.0112 | 0.66× | 0.67× | 1.02× |
| min | 256 | 0.0096 | 0.0155 | 0.0159 | 1.62× | 1.66× | 1.03× |
| norm | 256 | 0.0087 | 0.0110 | 0.0112 | 1.26× | 1.28× | 1.02× |
| ones | 256 | 0.0235 | 0.0117 | 0.0123 | 0.50× | 0.52× | 1.05× |
| qr | 256 | 5.1507 | 2.7753 | 2.3551 | 0.54× | 0.46× | 0.85× |
| reshape | 256 | 0.0003 | 0.0001 | 0.0004 | 0.24× | 1.11× | 4.53× |
| solve | 256 | 0.6511 | 1.3246 | 1.2835 | 2.03× | 1.97× | 0.97× |
| sum | 256 | 0.0145 | 0.0110 | 0.0112 | 0.76× | 0.77× | 1.02× |
| svd | 256 | 9.7012 | 10.2616 | 10.9188 | 1.06× | 1.13× | 1.06× |
| transpose | 256 | 0.0535 | 0.1928 | 0.1877 | 3.60× | 3.51× | 0.97× |
| zeros | 256 | 0.0113 | 0.0121 | 0.0113 | 1.07× | 1.00× | 0.93× |
| arange | 1024 | 0.0012 | 0.0011 | 0.0019 | 0.87× | 1.53× | 1.76× |
| cholesky | 1024 | 22.9667 | 24.1626 | 19.5284 | 1.05× | 0.85× | 0.81× |
| copy | 1024 | 0.6089 | 0.5949 | 0.6757 | 0.98× | 1.11× | 1.14× |
| dot | 1024 | 0.0009 | 0.0004 | 0.0005 | 0.42× | 0.57× | 1.37× |
| elem_add | 1024 | 1.0565 | 0.8721 | 0.9710 | 0.83× | 0.92× | 1.11× |
| elem_add_scalar | 1024 | 0.5992 | 0.6176 | 1.2004 | 1.03× | 2.00× | 1.94× |
| elem_div | 1024 | 0.9198 | 0.8771 | 1.4360 | 0.95× | 1.56× | 1.64× |
| elem_mul | 1024 | 1.0199 | 0.8713 | 0.9844 | 0.85× | 0.97× | 1.13× |
| elem_sub | 1024 | 1.0191 | 0.8707 | 0.9659 | 0.85× | 0.95× | 1.11× |
| eye | 1024 | 0.3466 | 0.3298 | 0.3489 | 0.95× | 1.01× | 1.06× |
| fill | 1024 | 0.3810 | 0.3254 | 0.3509 | 0.85× | 0.92× | 1.08× |
| full | 1024 | 0.3923 | 0.3558 | 0.4062 | 0.91× | 1.04× | 1.14× |
| matmul | 1024 | 17.1713 | 20.4939 | 22.5765 | 1.19× | 1.31× | 1.10× |
| max | 1024 | 0.2902 | 0.3251 | 0.3091 | 1.12× | 1.07× | 0.95× |
| mean | 1024 | 0.3272 | 0.2791 | 0.2868 | 0.85× | 0.88× | 1.03× |
| min | 1024 | 0.2774 | 0.3479 | 0.2950 | 1.25× | 1.06× | 0.85× |
| norm | 1024 | 0.1676 | 0.3167 | 0.3153 | 1.89× | 1.88× | 1.00× |
| ones | 1024 | 0.3785 | 0.3576 | 0.4062 | 0.94× | 1.07× | 1.14× |
| qr | 1024 | 112.6534 | 80.3390 | 60.3200 | 0.71× | 0.54× | 0.75× |
| reshape | 1024 | 0.0003 | 0.0001 | 0.0005 | 0.26× | 1.69× | 6.59× |
| solve | 1024 | 25.8654 | 32.2379 | 28.6589 | 1.25× | 1.11× | 0.89× |
| sum | 1024 | 0.3121 | 0.2796 | 0.2574 | 0.90× | 0.82× | 0.92× |
| svd | 1024 | 340.7707 | 444.5843 | 390.2499 | 1.30× | 1.15× | 0.88× |
| transpose | 1024 | 4.3359 | 4.5253 | 4.4600 | 1.04× | 1.03× | 0.99× |
| zeros | 1024 | 0.3259 | 0.2989 | 0.3692 | 0.92× | 1.13× | 1.24× |
