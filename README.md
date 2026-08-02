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
| **Lua author** (inside a host) | `require "matlua"`, 1-based arrays, operators, `matmul` / `matmul_at` / `normal_eq` / `solve` / factorizations |
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
| **Broadcast / views** | elementwise broadcast; `broadcast_to`; `slice`/`rows`/`row`/`col` (1-based half-open on face) |
| **NaN reductions** | `nansum`, `nanmean`, `nanmin`, `nanmax`, `nanvar`, `nanstd` |
| **Tier-2** | `cov`/`corrcoef`, `outer`/`diag`/`trace`, `argsort`/`take`, axis on `sum`/`mean`/…, `any`/`all` |
| **Constructors** | `zeros`, `ones`, `full`, `arange` (`start, stop[, step]`, half-open), `array` (nested tables → dense `f64`), `eye` |
| **Array methods** | `shape`, `rank`, `get` / `set`, `sum` / `mean` / `min` / `max`, `var` / `std`, `argmin` / `argmax`, ufuncs (`abs`/`sqrt`/`exp`/`log`/`log1p`/`sign`/`power`/`clip`/`isnan`/`isfinite`/`cumsum`), `copy`, `reshape` (may share; write COWs), `transpose`, `fill`, `#a` |
| **Elementwise** | `+`, `-`, `*`, `/`, unary `-` (array–array or array–number) |
| **Linear algebra** | `matmul`, `matmul_at`, `normal_eq`, `solve`, `lstsq`, `eigh`, `pinv`, `transpose`, `dot`, `norm`, `cholesky`, `qr`, `svd` |
| **Rust core** | `Array` (row-major n-D `f64`), views over host buffers, Arrow `Float64Array` interchange, same LA under `matlua::linalg` |

Quality bar is **`f64`**. Storage is a dense buffer, not nested Lua tables
(tables are constructor sugar only).

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

**M0–M3 are on `main`.** The tree is a **v0.1 candidate**: a host can embed
MatLua and scripts can do ordinary dense array and linear-algebra work
end-to-end. Crate version remains **`0.0.1`** until a formal `0.1.0` cut.

**Performance:** fair three-way microbench (NumPy · MatLua Rust · MatLua Lua) lives
under [`tests/`](tests/README.md). Run `python3 tests/bench/compare_fair.py`.
Open function-level work is in GitHub Issues; DESIGN holds closed rulings.

Known thin spots vs a full “leave late” desk (column views, richer slicing,
broadcasting, host zero-copy *from* Lua) are intentional feature follow-ups,
not blockers for basic embed + bulk math.

## License

[MIT](LICENSE) © 2026 Andy Emerson
