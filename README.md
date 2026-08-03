# MatLua

Dense numeric arrays and linear algebra for **Lua 5.4**, implemented as a
**Rust crate**. Scripts see a Lua library; hosts embed the crate and register
it into a PUC Lua 5.4 state.

Architectural counterpart to NumPy for the Lua world: high-level language in
front, systems language in the core. Python → Lua 5.4, C → Rust, `ndarray` →
[Apache Arrow](https://arrow.apache.org/), BLAS/LAPACK →
[faer](https://github.com/sarah-quinones/faer-rs).

| You are… | You get… |
|----------|----------|
| **Lua author** (inside a host) | `require "matlua"`, 1-based arrays, operators, `matmul` / `matmul_at` / `matmul_bt` / `normal_eq` / `solve` / factorizations |
| **Host / embedder** | Rust crate: owned `f64` n-D arrays, faer LA, optional `lua` feature + `matlua::lua::register` |
| **Rust-only consumer** | Same arrays + LA without linking Lua |

Implementer rulings, scope, and architecture: [DESIGN.md](DESIGN.md).  
Process: [AGENTS.md](AGENTS.md).

## Quick feel (Lua)

After the host registers MatLua:

```lua
local ml = require "matlua"

local A = ml.array({{3, 1}, {1, 2}})
local b = ml.array({9, 8})
local x = ml.solve(A, b)          -- ≈ {2, 3}

-- one expression, one result (no forced temps for subexpressions)
local X = ml.array({{1, 0}, {1, 1}, {1, 2}, {1, 3}})
local y = ml.array({1, 3, 5, 7})
local beta = ml.normal_eq(X, y)  -- short path: matmul_at + solve (see DESIGN)
-- or: ml.solve(ml.matmul_at(X, X), ml.matmul_at(X, y))
```

**Indexing is 1-based on the Lua face.** Elementwise `+ - * /` and unary `-`
work on arrays (and array ↔ number). Matrix product is always explicit:
`ml.matmul(a, b)` — Lua has a single `*`, so it stays elementwise.

## What works today

| Area | Surface |
|------|---------|
| **Masks / compare** | `where`; `eq`/`ne`/`lt`/`le`/`gt`/`ge` (0/1); `isnan` / `isfinite` |
| **Indexing** | `nonzero`, `compress`, `put`, `put_mask`, `take` (i64 or f64 indices) |
| **Host embed (Rust)** | `lua::push_view_f64` / `push_view_i64` (zero-copy read-only), `push_array_copy_*` |
| **Broadcast / views** | elementwise broadcast; `broadcast_to`; `slice`/`rows`/`row`/`col` (1-based half-open on face) |
| **NaN reductions** | `nansum`, `nanmean`, `nanmin`, `nanmax`, `nanvar`, `nanstd` |
| **Tier-2** | `cov`/`corrcoef`, `outer`/`diag`/`trace`, `argsort`/`take`, axis on `sum`/`mean`/…, `any`/`all` |
| **Random** | `seed`, `random`, `randn`, `uniform`, `normal`, `integers` (i64), `choice` |
| **Constructors** | `zeros`, `ones`, `full`, `arange` (`start, stop[, step]`, half-open), `array` (nested tables → dense `f64`), `eye` |
| **Array methods** | `shape`, `rank`, `get` / `set`, `sum` / `mean` / `min` / `max`, `var` / `std`, **`median`/`quantile`**, `argmin` / `argmax`, ufuncs (`abs`/`sqrt`/`exp`/`log`/`log1p`/`sign`/`power`/`clip`/`isnan`/`isfinite`/`cumsum`), `copy`, `reshape` (may share; write COWs), `transpose`, `fill`, `#a` |
| **Elementwise** | `+`, `-`, `*`, `/`, unary `-`; **`add_out`/`mul_out`/…** and **`matmul_out`** (preallocated) |
| **Linear algebra (`f64`)** | `matmul`, …, `solve`, `lstsq`, `eigh`, `pinv`, `cholesky`, `qr`, `svd`, **`det`/`slogdet`/`matrix_rank`/`cond`/`eig`/`eigvals`** (M7.b) |
| **`i64`-unique** | bitwise / rem / shift, `unique` / `isin` / `bincount` / `searchsorted` / `sort`, `divmod` / `gcd` / `lcm`, bit counts |
| **Linear algebra (`i64`)** | Integer path: `matmul_i64` / … (wrapping). **Same** `solve`/`eigh`/… accept `ArrayI64` and return **`f64`** (NumPy-style). Also `linalg::from_i64` / `i64_ops` in Rust |
| **Rust core** | `Array` (`f64`), `ArrayI64` (`i64`), views over host `f64`/`i64` buffers, Arrow `Float64`/`Int64`, LA `matlua::linalg` + `linalg::i64_ops` |

**`f64`** remains the primary continuous/LA dtype; **`i64`** (M7 **Done**) is first-class for
keys, indices, integer arithmetic/LA, and i64-unique ops. Solvers on integer inputs
promote to **`f64` results** (NumPy-style). Lua: `zeros_i64`, `array_i64`, `where_i64`, …
Storage is a dense buffer, not nested Lua tables (tables are constructor sugar only).

### Crate features

| Feature | Purpose |
|---------|---------|
| *(default)* | Arrays + faer LA (Rust API only) |
| `lua` | Hand-rolled Lua 5.4 bindings; vendors PUC 5.4.7 for tests/tools |

Hosts keep their own `lua_State` and call `unsafe { matlua::lua::register(L) }`.
The vendored interpreter is for MatLua’s own tests and simple tools.

```text
cargo test
cargo test --features lua
python3 tests/bench/compare_fair.py   # fair perf table (release)
```

## Host sketch (Rust)

```rust
// L: *mut lua_State owned by the host (PUC Lua 5.4)
unsafe {
    matlua::lua::register(L);
    // Recommended: large f64 buffers live on the Rust heap; help Lua GC.
    matlua::lua::enable_generational_gc(L);
}
// scripts: local ml = require "matlua"
```

Rust-side desk math without Lua:

```rust
use matlua::{Array, linalg};

let a = Array::from_shape_slice(vec![2, 2], &[3., 1., 1., 2.])?;
let b = Array::from_shape_slice(vec![2], &[9., 8.])?;
let x = linalg::solve(&a, &b)?;
```

## Design in one breath

- **User product:** Lua library. **Ship form:** Rust crate.
- **Hosts fit to MatLua** — engines adapt buffers and embed to MatLua’s contracts.
- **Portable dense LA** — faer in-tree; no system BLAS/LAPACK default.
- **Arrow** for buffer model and interchange; faer for dense LA (views in, owned results out).
- Curated dense surface first — not full NumPy/SciPy parity.

Closed decisions and milestones: [DESIGN.md](DESIGN.md).

## Status

**M0–M6 are on `main`** (see [DESIGN.md](DESIGN.md) §7.1). The tree is a **v0.1
candidate**: a host can embed MatLua and scripts can do ordinary dense array and
linear-algebra work end-to-end. Crate version remains **`0.0.1`** until a formal
`0.1.0` cut. **M7 Done. M7.b Done.** **M7.c in progress** (not closed): optimize f64+i64; exact
wrapping i64 matmul is plan A; get complete honest numbers before setting any
performance target — [DESIGN.md](DESIGN.md) §7.1.2. Embed/TallyDB letter work is
**M8–M12** (§7.1.1).

**Performance:** three-way microbenches (NumPy · MatLua Rust · MatLua Lua) under
[`tests/README.md`](tests/README.md) — summary tables first (Lua vs NumPy, ranges
across sizes), full per-n three-face tables in the appendix, plus an i64
machine-roofline yardstick. i64 **matmul** is timed against **NumPy f64 BLAS** on
integer-valued inputs (not `int64@int64`).

**Known limits (tracked M7.c–M12 / issues):** exact i64 GEMM runs ~5–7× NumPy f64
BLAS (integer-valued) at ~80–88% of the measured machine ceiling — the rest is
integer-multiply ISA physics (see tests/README Roofline); small-buffer pool min size; embed error boundary; dtypes
beyond f64/i64; `arrow-lite` cutover. Full in-place `out=` is
[#21](https://github.com/andy-emerson/MatLua/issues/21) (partial `*_out` exists).
Host→Lua views: `push_view_f64` / `push_view_i64` (M7.b).

## License

[MIT](LICENSE) © 2026 Andy Emerson
