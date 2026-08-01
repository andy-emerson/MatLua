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
Run date: 2026-08-01 (post #12–#15 + buffer pool + blocked max + Lua GC debt on `push_array` / generational GC).  
Re-run: `python3 tests/bench/compare_fair.py`.

**Caveats:** Micro-ops under ~0.01 ms are noisy. Heavy ops at n=1024 use few samples. `reshape` is metadata + `Arc` share. `fill` is in-place on a dedicated buffer. Hosts using their own `lua_State` should call `matlua::lua::enable_generational_gc`.

| op | n | NumPy (ms) | MatLua Rust (ms) | MatLua Lua (ms) | Rust/NumPy | Lua/NumPy | Lua/Rust |
|----|---:|-----------:|-----------------:|----------------:|-----------:|----------:|---------:|
| arange | 64 | 0.0006 | 0.0002 | 0.0005 | 0.32× | 0.88× | 2.77× |
| cholesky | 64 | 0.0226 | 0.0161 | 0.0153 | 0.71× | 0.68× | 0.95× |
| copy | 64 | 0.0014 | 0.0015 | 0.0023 | 1.08× | 1.70× | 1.57× |
| dot | 64 | 0.0007 | 0.0001 | 0.0002 | 0.07× | 0.27× | 3.60× |
| elem_add | 64 | 0.0017 | 0.0022 | 0.0030 | 1.27× | 1.72× | 1.35× |
| elem_add_scalar | 64 | 0.0017 | 0.0016 | 0.0026 | 0.93× | 1.56× | 1.67× |
| elem_div | 64 | 0.0034 | 0.0037 | 0.0048 | 1.11× | 1.42× | 1.28× |
| elem_mul | 64 | 0.0017 | 0.0022 | 0.0030 | 1.27× | 1.75× | 1.38× |
| elem_sub | 64 | 0.0017 | 0.0022 | 0.0030 | 1.29× | 1.81× | 1.40× |
| eye | 64 | 0.0024 | 0.0009 | 0.0014 | 0.38× | 0.60× | 1.55× |
| fill | 64 | 0.0017 | 0.0007 | 0.0005 | 0.42× | 0.29× | 0.69× |
| full | 64 | 0.0026 | 0.0015 | 0.0020 | 0.59× | 0.77× | 1.31× |
| matmul | 64 | 0.0115 | 0.0121 | 0.0148 | 1.05× | 1.30× | 1.23× |
| max | 64 | 0.0019 | 0.0011 | 0.0013 | 0.55× | 0.67× | 1.21× |
| mean | 64 | 0.0043 | 0.0007 | 0.0009 | 0.17× | 0.21× | 1.25× |
| min | 64 | 0.0019 | 0.0014 | 0.0017 | 0.73× | 0.87× | 1.19× |
| norm | 64 | 0.0026 | 0.0007 | 0.0009 | 0.27× | 0.34× | 1.24× |
| ones | 64 | 0.0030 | 0.0012 | 0.0017 | 0.41× | 0.58× | 1.40× |
| qr | 64 | 0.1205 | 0.2587 | 0.2731 | 2.15× | 2.27× | 1.06× |
| reshape | 64 | 0.0003 | 0.0001 | 0.0006 | 0.26× | 1.91× | 7.28× |
| solve | 64 | 0.0366 | 0.0734 | 0.0751 | 2.01× | 2.05× | 1.02× |
| sum | 64 | 0.0023 | 0.0007 | 0.0009 | 0.32× | 0.40× | 1.26× |
| svd | 64 | 0.2837 | 0.4710 | 0.4693 | 1.66× | 1.65× | 1.00× |
| transpose | 64 | 0.0026 | 0.0021 | 0.0044 | 0.80× | 1.68× | 2.11× |
| zeros | 64 | 0.0008 | 0.0009 | 0.0015 | 1.14× | 1.76× | 1.54× |
| arange | 256 | 0.0008 | 0.0004 | 0.0008 | 0.56× | 0.98× | 1.76× |
| cholesky | 256 | 0.8145 | 0.5767 | 0.6214 | 0.71× | 0.76× | 1.08× |
| copy | 256 | 0.0135 | 0.0352 | 0.0379 | 2.60× | 2.80× | 1.08× |
| dot | 256 | 0.0008 | 0.0001 | 0.0002 | 0.15× | 0.31× | 2.09× |
| elem_add | 256 | 0.0224 | 0.0406 | 0.0542 | 1.82× | 2.43× | 1.33× |
| elem_add_scalar | 256 | 0.0133 | 0.0352 | 0.0387 | 2.64× | 2.90× | 1.10× |
| elem_div | 256 | 0.0447 | 0.0655 | 0.0699 | 1.47× | 1.56× | 1.07× |
| elem_mul | 256 | 0.0216 | 0.0414 | 0.0508 | 1.92× | 2.35× | 1.23× |
| elem_sub | 256 | 0.0208 | 0.0407 | 0.0459 | 1.96× | 2.21× | 1.13× |
| eye | 256 | 0.0144 | 0.0220 | 0.0225 | 1.53× | 1.56× | 1.02× |
| fill | 256 | 0.0224 | 0.0116 | 0.0118 | 0.52× | 0.53× | 1.02× |
| full | 256 | 0.0233 | 0.0232 | 0.0243 | 1.00× | 1.04× | 1.05× |
| matmul | 256 | 0.5105 | 0.3662 | 0.5627 | 0.72× | 1.10× | 1.54× |
| max | 256 | 0.0093 | 0.0156 | 0.0157 | 1.68× | 1.69× | 1.01× |
| mean | 256 | 0.0166 | 0.0110 | 0.0112 | 0.66× | 0.67× | 1.02× |
| min | 256 | 0.0093 | 0.0219 | 0.0221 | 2.34× | 2.36× | 1.01× |
| norm | 256 | 0.0097 | 0.0110 | 0.0112 | 1.13× | 1.15× | 1.01× |
| ones | 256 | 0.0235 | 0.0234 | 0.0240 | 1.00× | 1.02× | 1.03× |
| qr | 256 | 4.9151 | 3.0070 | 2.5614 | 0.61× | 0.52× | 0.85× |
| reshape | 256 | 0.0003 | 0.0001 | 0.0006 | 0.26× | 1.98× | 7.60× |
| solve | 256 | 0.6446 | 1.1304 | 1.4678 | 1.75× | 2.28× | 1.30× |
| sum | 256 | 0.0145 | 0.0110 | 0.0112 | 0.76× | 0.77× | 1.02× |
| svd | 256 | 9.5980 | 11.1637 | 13.7239 | 1.16× | 1.43× | 1.23× |
| transpose | 256 | 0.0521 | 0.0616 | 0.2507 | 1.18× | 4.81× | 4.07× |
| zeros | 256 | 0.0113 | 0.0217 | 0.0225 | 1.92× | 1.99× | 1.04× |
| arange | 1024 | 0.0012 | 0.0014 | 0.0028 | 1.18× | 2.37× | 2.00× |
| cholesky | 1024 | 22.5556 | 21.9958 | 16.9982 | 0.98× | 0.75× | 0.77× |
| copy | 1024 | 0.6161 | 1.2565 | 1.2173 | 2.04× | 1.98× | 0.97× |
| dot | 1024 | 0.0008 | 0.0004 | 0.0005 | 0.45× | 0.62× | 1.36× |
| elem_add | 1024 | 1.0905 | 1.4989 | 1.5016 | 1.37× | 1.38× | 1.00× |
| elem_add_scalar | 1024 | 0.6071 | 1.2511 | 1.2203 | 2.06× | 2.01× | 0.98× |
| elem_div | 1024 | 1.0012 | 1.5002 | 1.8901 | 1.50× | 1.89× | 1.26× |
| elem_mul | 1024 | 1.0780 | 1.5118 | 1.5259 | 1.40× | 1.42× | 1.01× |
| elem_sub | 1024 | 1.0783 | 1.4999 | 1.5288 | 1.39× | 1.42× | 1.02× |
| eye | 1024 | 0.3300 | 0.6246 | 0.6416 | 1.89× | 1.94× | 1.03× |
| fill | 1024 | 0.3855 | 0.3243 | 0.3230 | 0.84× | 0.84× | 1.00× |
| full | 1024 | 0.3651 | 0.6799 | 0.7087 | 1.86× | 1.94× | 1.04× |
| matmul | 1024 | 17.4750 | 19.6925 | 21.9253 | 1.13× | 1.25× | 1.11× |
| max | 1024 | 0.2780 | 0.3135 | 0.2775 | 1.13× | 1.00× | 0.89× |
| mean | 1024 | 0.3332 | 0.2754 | 0.2635 | 0.83× | 0.79× | 0.96× |
| min | 1024 | 0.2780 | 0.3843 | 0.3817 | 1.38× | 1.37× | 0.99× |
| norm | 1024 | 0.1685 | 0.2849 | 0.2583 | 1.69× | 1.53× | 0.91× |
| ones | 1024 | 0.3919 | 0.6834 | 0.7114 | 1.74× | 1.82× | 1.04× |
| qr | 1024 | 113.3407 | 79.5826 | 55.4013 | 0.70× | 0.49× | 0.70× |
| reshape | 1024 | 0.0003 | 0.0001 | 0.0006 | 0.25× | 1.94× | 7.71× |
| solve | 1024 | 25.5601 | 28.7311 | 31.2658 | 1.12× | 1.22× | 1.09× |
| sum | 1024 | 0.3188 | 0.2792 | 0.2669 | 0.88× | 0.84× | 0.96× |
| svd | 1024 | 335.6471 | 405.4092 | 364.7228 | 1.21× | 1.09× | 0.90× |
| transpose | 1024 | 4.7769 | 4.9378 | 7.6889 | 1.03× | 1.61× | 1.56× |
| zeros | 1024 | 0.3130 | 0.6218 | 0.6376 | 1.99× | 2.04× | 1.03× |
