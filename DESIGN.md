# MatLua design

Durable record of **what MatLua is and why**, for implementers and agents.
Present-tense design intent only. Open debate lives in GitHub Discussions;
**closed** rulings are written here.

| Document | Audience / job |
|----------|----------------|
| **[README.md](README.md)** | Visitors and users — what it is, how it feels, what works |
| **This file** | Implementers — rulings, scope, architecture, milestones |
| **[AGENTS.md](AGENTS.md)** | Human ↔ agent working agreement (do not edit ad hoc) |

Do not restate the README’s product pitch or tutorial samples here beyond what
rulings and frozen API need. Prefer linking to the README for “how it feels.”

---

## 1. Product

MatLua is a **general-purpose dense array and linear-algebra library** with a
**Lua 5.4 face** and a **Rust implementation**. Deep enough that analytic engines
and quant workflows stay in-process for ordinary matrix work; portable enough
that **no single host owns the API**.

| Layer | Choice |
|-------|--------|
| Scripting face | **Lua 5.4** (PUC reference family), 1-based |
| Core | **Rust** crate |
| Bindings | **Hand-rolled Lua C API** (no mlua / rlua / LuaJIT in-tree) |
| Array / buffer model | **Apache Arrow** (lean `arrow-array` / `buffer` / `schema`) |
| Dense LA | **faer** |
| Deliverable | **Rust crate** (hosts embed; optional native module packaging later) |

**User POV:** a Lua library. Users never need Rust.  
**Dev / host POV:** a Rust crate. Hosts adapt buffers, own `lua_State`, register MatLua.

### 1.1 Downstream contract

- **Hosts fit to MatLua.** Public buffer and API contracts are defined here; engines adapt at the boundary.
- **TallyDB is design pressure, not the product boundary.** Easy adoption for TallyDB-class hosts without TallyDB types, SQL, or storage in the public API.
- **Portable dense LA.** No system BLAS/LAPACK default so ports (including future `wasm32`) need not rewrite numerics.

### 1.2 Success criterion (“leave late”)

A script author doing ordinary dense array and dense LA work stays on **Lua +
MatLua** until the problem is genuinely outside that domain (sparse, ML stacks,
plotting, full SciPy-class ecosystems, …). Leaving should be clean (Arrow
interchange) but unnecessary for day-to-day matrix work.

### 1.3 Dual audience (principle)

Build what **both** a **Lua** desk author and a **NumPy**-literate user would
expect for dense arrays and dense LA.

| Clause | Rule |
|--------|------|
| Dual expectation | Behavior and vocabulary should not surprise either audience on ordinary desk work. |
| Conflict | **Lua face wins** (1-based indices, `*` elementwise, explicit `matmul`, Lua-shaped methods). |
| Performance floor | Lua-first must not force designs that cannot stay **competitive with NumPy** on bulk `f64` work (fair three-way: product is Lua; wall-time bar is NumPy). |

**Scope:** dense arrays, dense LA, host contracts — not full SciPy, not TallyDB windows/SQL.  
**Reopen when:** a Lua-idiomatic API systematically exceeds ~2× NumPy on medium+ bulk ops with no shorter path, or a NumPy-shaped API is unusable on the Lua face.

---

## 2. Scope

### 2.1 In scope

- Dense numeric arrays with an **n-D** data model (shape + rank) from the start
- **NumPy-shaped** capabilities: constructors, elementwise ops, reductions, indexing helpers, dense LA
- **Lua-shaped** syntax (see §5)
- Primary quality bar on **`f64`**, dtype story free to grow later
- **faer** for dense matmul, factorizations, solvers
- **Arrow** as primary in-memory layout and interchange
- Explicit **ownership / view / copy** rules compatible with zero-copy host buffers
- Optional Cargo feature **`lua`** for the Lua face; product incomplete for the README promise without real bulk work on that face

### 2.2 Non-goals (for a long time)

| Non-goal | Reason |
|----------|--------|
| Full NumPy / SciPy module parity | Unbounded; not required for “leave late” on dense work |
| Nested Lua tables as runtime storage | Slow; the problem MatLua exists to avoid |
| System BLAS/LAPACK as default backend | Portability, embed story, pure-Rust build |
| mlua / rlua / LuaJIT as in-tree binding | Host-owned PUC 5.4; hand-rolled C API |
| Sparse matrices, graphs, DataFrames | Different data models |
| FFT, random suites, special functions | Separate product surfaces |
| Autodiff / ML frameworks | Wrong layer |
| IO ecosystem (`.npy`, CSV, HDF5, …) as core mission | Host / Arrow concern |
| TallyDB-specific types or SQL in the public API | Hosts fit to MatLua |
| Research-grade novel LA algorithms | faer owns serious dense LA |

### 2.3 Relationship to TallyDB (informative)

If MatLua succeeds at general dense LA:

- TallyDB can depend on **MatLua** for matrix-class work and need not link **faer** directly.
- TallyDB keeps **engine-native** specialty compute (e.g. closed-form window statistics).
- TallyDB adapts engine buffers and its Lua embed to MatLua’s contracts.

MatLua does not subsume TallyDB.

---

## 3. Closed design rulings

### 3.1 Indexing

| Face | Convention |
|------|------------|
| **Lua (user)** | **1-based** |
| **Rust (implementation)** | **0-based** |

The binding translates. Users never need Rust indices.

### 3.2 Rank and shape

- Public model is **n-D from the start**: rank + shape, not permanently separate vector/matrix types that cannot grow.
- **Storage** is a dense typed buffer plus shape metadata (Arrow-shaped), not a tree of Lua tables.
- Implementations **may specialize** rank-1 and rank-2 kernels; specialization is invisible except where math requires shape compatibility.
- **v0.1 depth** prioritizes real desk math on rank 1–2 LA while keeping the type model n-D.

**LA rank conventions (implemented):**

| Input | Interpretation |
|-------|----------------|
| rank 2, shape `(m, n)` | `m × n` matrix |
| rank 1, shape `(n,)` | `n × 1` column vector |

Matrix×vector `matmul` returns **rank-1**. `solve` preserves the rank style of `b`.

### 3.3 Dtypes

- **v0.1 quality bar: `f64`.** Dense LA and the performance contract stay `f64`-first.
- **Next deliberate addition: `i64`** (milestone **M7**) for ordering keys and exact integer columns; not full dtype parity.
- Further types (`f32`, complex, …) by deliberate addition after M7–M12 unless a new need appears.

### 3.4 Ownership, views, and copies

1. **Owned arrays** — MatLua-managed results own their buffer.
2. **Borrowed views (Rust)** — `ArrayView` / `ArrayViewMut` over contiguous host or parent memory; **caller guarantees lifetime**.
3. **LA results** — always **owned** row-major arrays. **Inputs** are zero-copy faer `MatRef` when contiguous. **Outputs:** `matmul` / `matmul_at` / `matmul_bt` / blocked `transpose` write dest buffers; `solve` factors then solves in place on a row-major RHS copy. **Factorizations** (`cholesky`, `qr`, `svd`) still pack out from faer views into owned buffers.
4. **Rule of thumb:** views share memory until the user **copies**; bulk math results are new arrays unless documented otherwise. `reshape` may share an owned buffer until a write (copy-on-write).

Lua userdata today holds **owned** arrays. Host zero-copy *into* scripts remains a host/Rust-side concern until a view face is exposed to Lua.

### 3.5 Construction

**Implemented:**

- NumPy-like: `zeros`, `ones`, `full`, `arange` / `arange_step` (half-open `[start, stop)`), shape-first APIs
- From flat buffers: `Array::from_shape_vec` / `from_shape_slice`
- Arrow: `to_arrow` / `from_arrow` (`Float64Array`, non-null, with explicit shape)
- Identity: `eye`
- **Nested Lua tables** as constructor sugar only (`ml.array({...})` copies into dense `f64`)

**Not yet:** `empty` / uninit constructors, column/row view helpers, advanced slicing.

Arrays are **not** plain Lua tables at runtime.

### 3.6 Elementwise vs matmul operators

**Frozen for the Lua face:**

- Metamethods: elementwise `+`, `-`, `*`, `/`, unary `-` (array–array or array–number).
- **Matrix multiplication:** module function `ml.matmul(a, b)` only (not `*`, no `@`).
- **Transpose:** `ml.transpose(a)` and method `a:transpose()`.
- Other LA: `matmul_at`, `matmul_bt`, `normal_eq`, `solve`, `lstsq`, `eigh`, `pinv`, `dot`, `norm`, `cholesky`, `qr`, `svd`.

Rationale: Lua has a single `*`; naming matmul avoids silent confusion.

### 3.7 Expression style

- One mathematical formula → **one expression** assigning **one** result.
- Intermediate `local`s are optional for human clarity, never required for subexpressions of a single equation.

### 3.8 Deliverable and Lua face

- Repository product: **Rust crate** first.
- User product: **Lua library** (`require "matlua"` after host registration).
- Feature **`lua`**: hand-rolled bindings + vendored PUC 5.4.7 for tests/tools. Hosts may link their own 5.4.
- Public host entry: `unsafe fn matlua::lua::register(*mut lua_State)`.
- Hosts should call `enable_generational_gc` on allocate-heavy workloads (see README host sketch).
- Product-complete for the README promise means real **bulk** numeric work on the Lua face, not a single demo op.

### 3.9 Binding stack

- PUC **Lua 5.4** only in-tree.
- **Hand-rolled** thin C API (host-owned `lua_State`).
- No mlua, no rlua, no LuaJIT as the supported in-tree path.

### 3.10 Dependencies

| Dependency | Role |
|------------|------|
| **faer** | Dense linear algebra |
| **arrow-array / arrow-buffer / arrow-schema** (lean defaults) | Arrays, buffers, types |
| **thiserror** | Public error types |
| **cc** (build) | Compile vendored Lua when `lua` is enabled |
| Lua C sources | Vendored under `vendor/lua` for the `lua` feature only |

Explicitly not defaults: nalgebra, system BLAS/LAPACK, mlua, ndarray as data model, Parquet/Flight/DataFusion.

### 3.11 Project nature

**Library / systems engineering**, not research. Novel dense algorithms live in
**faer**. MatLua owns contracts, API, glue, Lua face, and evidence.

### 3.12 Layout

- Owned `Array` storage is contiguous **row-major (C-order)** `f64`.
- Dense LA **inputs** are faer `MatRef` views over that storage when the buffer is contiguous row-major (**zero-copy in**). The public `Array` model stays row-major; we do not re-store matrices column-major.
- Dense LA **results** are **owned** row-major `Array`s (see §3.4 for dest-write vs factorization pack-out). Out-parameter APIs are not frozen yet.
- faer’s native owned `Mat` remains column-major internally where faer allocates; that is an implementation detail of kernels, not MatLua’s user-facing layout.

### 3.13 Reshape buffer sharing

Value storage is `Arc<Vec<f64>>`. `reshape` (Rust and Lua) shares the buffer
(metadata + `Arc` clone). In-place mutation copy-on-writes when the buffer is
still shared (`Arc::make_mut`). `Clone`, `to_owned_array`, and Lua `copy` are
**deep** unique copies.

### 3.14 Composed dense paths

Prefer `matmul_at(A, B)` for `AᵀB`, `matmul_bt(A, B)` for `ABᵀ` (used by `cov`), and `normal_eq(X, y)` for `solve(XᵀX, Xᵀy)`
over materializing `transpose` then `matmul`. Same numerics as the long
composition. Large same-buffer `AᵀA` (`k ≥ 512`) may materialize `Aᵀ` once
internally. Measurement: `tests/bench/compare_compose.py`.

### 3.15 Broadcast, compares, views, NaN policy (M5)

- **Broadcast:** NumPy right-align rules for elementwise ops and `broadcast_to`. Practical focus rank ≤ 2; higher ranks supported by the same algorithm.
- **Compares:** `eq`/`ne`/`lt`/`le`/`gt`/`ge` return **0/1** dense `f64` masks (no separate bool dtype). IEEE: NaN comparisons are false.
- **Ufuncs:** IEEE **propagate** NaN. Skipping NaN uses explicit `nan*` reductions.
- **Views:** rank-1 `slice`, rank-2 `rows`/`row` are zero-copy when contiguous; `col` **copies** (row-major). Lua face uses **1-based half-open** ranges for `slice`/`rows` (stop exclusive).

### 3.16 Tier-2 quant helpers (M6)

- **`cov` / `corrcoef`:** variables in **rows** (NumPy `rowvar=True`). `cov` default `ddof=1`.
- **Axis:** reductions take **0-based axis** (NumPy-shaped) even on the Lua face; element indices remain 1-based.
- **`argsort` / `take`:** Rust 0-based; Lua face converts to/from 1-based indices.
- **`diag`:** vector→matrix or matrix→diagonal; `diagonal` / `trace` on matrices; `outer` for rank-1×rank-1.
- **`any` / `all`:** nonzero non-NaN is true (same as `where` cond); optional axis → 0/1 mask.

---

## 4. Lua face (frozen names)

Names match the `lua` feature on `main`. Tutorial samples live in
[README.md](README.md); this section freezes the surface for implementers.

### 3.17 `i64` arrays (M7)

- **`ArrayI64`**: owned row-major `i64`, same shape/rank model as `f64` [`Array`].
- **Introduction order:** `f64` first, then `i64` (not a permanent “LA is only f64” hierarchy).
- **Integer LA path:** `matmul` / `matmul_at` / `matmul_bt` / `dot` / `transpose` / `eye` on `ArrayI64` via `linalg::i64_ops` (wrapping `i64` accumulators; not faer). Integer×integer→integer in \(\mathbb{Z}\); fixed-width may wrap.
- **Real LA on integer inputs (NumPy-style):** `linalg::from_i64::{solve,lstsq,normal_eq,pinv,eigh,cholesky,qr,svd}` promote with `to_f64` and return **`f64` arrays**. Lua `ml.solve` / `eigh` / … accept `ArrayI64` the same way. Not exact rational solve; values \(>2^{53}\) lose integer exactness.
- **Still not pure-`i64` codomain:** those ops never return `ArrayI64` (math is real-valued).
- **Stats that are real-valued:** `mean` / `var` / `std` (+ axis) take `i64` and return `f64`.
- **Arithmetic:** wrapping add/sub/mul/neg/abs; truncating `/`; division by zero → `0` (no panic).
- **Mean** (scalar or axis) returns **`f64`** (or `f64` array).
- **Casts:** `ArrayI64::to_f64` / `Array::to_i64` (truncate toward zero).
- **Arrow:** `Int64Array` interchange (non-null).
- **Lua face:** `*_i64` constructors above; methods include shared grammar + i64-unique (`unique`, `isin`, `bincount`, `searchsorted`, `sort`, bitwise, `rem`, `divmod`, `gcd`/`lcm`, …); `get`/`set` integers; `to_f64` / `dtype`.
- **Also on i64 (Rust+Lua):** `where_cond`, `concatenate`/`stack`, `sign`/`clip`, `var`/`std` (as `f64`), `any`/`all` (+ axis), `slice`/`rows`/`row`/`col`, `broadcast_to`, compares (array and scalar).
- **i64-unique (M7):** bitwise/rem/shift, `unique`/`isin`/`bincount`/`searchsorted`/`sort`, `divmod`/`gcd`/`lcm`, bit counts; **`ArrayViewI64` / `ArrayViewMutI64`** (Rust host buffers).
- **Not M7 / later:** float-only ufuncs (`exp`/`log`/…), `cov`/`corrcoef`, nan* (as needed); performance (**M7.c**). Lua host views: entry in **M7.b**; richer face / linalg-on-views in **M8**.


### 3.18 M7.b LA diagnostics (f64-first)

- **`det` / `slogdet`**: square real matrices via partial-pivoted LU; `slogdet` → `(sign, log|det|)`.
- **`matrix_rank`**: numerical rank from SVD; default tol `max(m,n)·ε·σ_max` (optional override).
- **`cond`**: 2-norm condition `σ_max/σ_min` (∞ if singular).
- **`eigvals` / `eig`**: general (possibly non-symmetric) eigen via faer; complex results as **real/imag split** arrays (no `c64` yet). Prefer **`eigh`** for symmetric.
- **i64 inputs:** `linalg::from_i64` and Lua dual face promote to `f64` results (same as `solve`).

### 3.19 M7.b order statistics

- **`median` / `quantile` / `quantiles`**: linear interpolation (NumPy-style `q∈[0,1]`); empty errors.
- **Axis:** `median_axis` / `quantile_axis` on rank-2 (`f64`).
- **`i64`:** scalar `median`/`quantile` return **`f64`** (even-length median averages).

### 3.20 M7.b random

- **PRNG:** process-global xoshiro256**; `seed(u64)` resets stream (not crypto).
- **`random` / `randn` / `uniform` / `normal`:** `f64` arrays.
- **`integers`:** `i64` arrays, half-open `[low, high)`.
- **`choice`:** sample with replacement from rank-1 (`f64` or `i64`).

### 3.21 M7.b indexing

- **`nonzero`**: flat 0-based indices as `ArrayI64` (Lua: 1-based).
- **`compress(mask)`**: rank-1 boolean-style select (nonzero mask entries).
- **`put` / `put_mask`**: scatter by indices or mask (in-place).
- **`take` / `take_i64`**: gather; Lua `take` accepts f64 or i64 index arrays (1-based).

### 3.22 M7.b `out=` (partial, #21)

- **Rust:** `add_out`/`sub_out`/`mul_out`/`div_out`/`neg_out`/`abs_out` on `Array`; same-shape `*_out` on `ArrayI64`; `linalg::matmul_out`.
- **Lua:** `a:add_out(b, out)`, …, `ml.matmul_out(A,B,out)` — returns `out`.
- **Not yet:** full surface `out=` on reductions/LA/ufuncs (tracked #21).

### 3.23 M7.b host buffer → Lua

Embedders (TallyDB) push engine columns without a second interpreter:

| API | Behavior |
|-----|----------|
| `lua::push_view_f64` / `push_view_i64` | **Zero-copy** read-only userdata (`matlua.ArrayView` / `ArrayViewI64`). Host owns memory; MatLua does **not** free. Methods: `shape`, `rank`, `get`, `dtype`, `to_array` (copy to owned). |
| `lua::push_array_copy_f64` / `push_array_copy_i64` | Safe **copy** into owned `ml.array` / `ml.array_i64`. |

Views are not yet accepted by `linalg` (owned `&Array` only) — TallyDB letter §5 notes copies are fine for current windows; view-aware LA is a later optimization.

### 3.24 TallyDB requirements letter (alignment)

External letter *“TallyDB → MatLua: what we need”* (not a MatLua file). **Authoritative milestone mapping: §7.1.1.** Summary:

| § | Ask | MatLua stance / milestone |
|---|-----|---------------------------|
| **1.1** | Face without vendored second Lua | `register(L)` already host-state; **split `lua` feature** (face vs interpreter) → **M10** |
| **1.2** | No `Drop` live across longjmp | Documented risk; systematic fix → **M10** |
| **1.3** | No panic across C | `catch_unwind` boundary → **M10** |
| **2.1** | `i64` exact end-to-end | **M7 done**; keep no silent i64→f64 at face for values |
| **2.2** | Documented absence contract | Prefer **explicit refuse / mask at boundary** (aligns with their validity bytes); document in M8/Arrow work — **default: non-null in, refuse nulls** (today’s Arrow path) until mask lands |
| **3.1–3.4** | Arrow C Data Interface, release callback, no null buffer reads | **arrow-lite / C ABI** track when lite v0.1; drop unused arrow-rs deps opportunistically |
| **4** | Design freedom | Ours (1-based Lua, faer, etc.) |
| **5** | View-aware linalg later | Acknowledged; not required now |
| **6** | One array type in Lua tier | Prefer **MatLua as the math array**; host pushes via view/copy APIs; retiring `tallydb.vector` for math is their call — we make adoption cheap |
| **7** | APICHECK / differential tests | Welcome; our CI should grow APICHECK (M10) |

### 4.1 Module functions

**`f64`:** `zeros`, `ones`, `full`, `arange` (`start, stop[, step]`), `array`, `eye`, `where`,
`matmul`, …, `svd`, **`det`/`slogdet`/`matrix_rank`/`cond`/`eig`/`eigvals`**, **`seed`/`random`/`randn`/`uniform`/`normal`/`integers`/`choice`**.

**`i64` constructors / helpers:** `zeros_i64`, `ones_i64`, `full_i64`, `arange_i64`, `array_i64`, `eye_i64`, `diag_i64`, `outer_i64`, `where_i64`, `concatenate_i64`, `stack_i64`, `broadcast_to_i64`, plus `matmul_i64` / `dot_i64` / …  
**Dual:** `matmul` / `solve` / `eigh` / … accept `ArrayI64` where applicable (integer matmul stays i64; solvers return `f64` arrays). See §3.17.

### 4.2 Array methods and metamethods

Methods: `shape`, `rank`, `get`, `set`, `sum`, `mean`, `min`, `max`, `nansum`,
`nanmean`, `nanmin`, `nanmax`, `nanvar`, `nanstd`, `copy`, `reshape`, `transpose`,
`fill`, ufuncs, compares (`eq`/`ne`/`lt`/`le`/`gt`/`ge`), `slice`/`rows`/`row`/`col`,
`cumsum`, `argmin`, `argmax`, `var`, `std`  
Module also: `where`, `concatenate`, `stack`, `broadcast_to`  
Metamethods: `__add`, `__sub`, `__mul`, `__div`, `__unm`, `__len`, `__tostring`, `__gc`

### 4.3 NumPy contrast (capability, not syntax parity)

```python
# NumPy
beta = np.linalg.solve(X.T @ X, X.T @ y)
```

```lua
-- MatLua: explicit matmul; 1-based; no @
-- Preferred short path (no materializing Xᵀ):
local beta = ml.normal_eq(X, y)
-- Equivalent primitive: ml.solve(ml.matmul_at(X, X), ml.matmul_at(X, y))
-- Long path (still correct): ml.solve(ml.matmul(X:transpose(), X), ml.matmul(X:transpose(), y))
```

Rank-1 vectors are valid `matmul` / `solve` operands (column convention). Column-fill helpers (`:col`, `:assign`) are **not** implemented; use `get`/`set` loops or build from tables until view sugar lands.

---

## 5. Lua-shaped vs NumPy-shaped

| Concern | Choice |
|---------|--------|
| Ideas / capability set | **NumPy-shaped** (dense arrays, bulk ops, linalg role) |
| Syntax / indexing / load | **Lua-shaped** (`require`, 1-based, metamethods, colon methods, no `@`) |
| Storage | Dense buffers, not Python lists / Lua tables |
| Everyday arithmetic | Short operators + sharp methods |
| Linear algebra vocabulary | Explicit names (`transpose`, `matmul`, `solve`, …) |

---

## 6. Architecture

```text
Lua 5.4 scripts
      │  require "matlua" after host registration (1-based API)
      ▼
Hand-rolled Lua C API  ← feature = "lua"
      ▼
Rust crate (MatLua)
      ├── Array / views / Arrow Float64 interchange (row-major f64)
      └── faer  (MatRef over Array for inputs; owned Array results)
```

Arrow owns the **data model**. faer owns **dense linear algebra**. They meet at
explicit boundaries (zero-copy views in, owned results out).

**Modules (crate):**

| Path | Role |
|------|------|
| `matlua::array` | `Array`, `Shape`, views, elementwise ops |
| `matlua::linalg` | matmul, solve, decompositions, norm, eye; `i64_ops` integer LA; `from_i64` promote solvers |
| `matlua::error` | `Error` / `Result` |
| `matlua::lua` | register, userdata, optional test `Lua` helper (`lua` feature) |

---

## 7. Milestones

### 7.1 Product surface (M0–M3 / v0.1)

| Milestone | Intent | Status |
|-----------|--------|--------|
| **M0** | Crate skeleton, module layout, deps, error type | **Done** |
| **M1** | `f64` n-D arrays: construct, shape, view/copy, elementwise | **Done** |
| **M2** | Dense LA via faer: matmul, solve, decompositions | **Done** |
| **M3** | Lua face: register into host state; bulk ops; 1-based API | **Done** |
| **v0.1** | M1–M3 good enough that a host embeds MatLua and scripts do ordinary dense work without leaving for NumPy | **Candidate** (not version-tagged; crate still `0.0.1`) |
| **M4a** | Job-A LA pack: `lstsq`, `eigh`, `pinv` | **Done** |
| **M5** | Tier-1 leave-late array ops | **Done** |
| **M6** | Tier-2 quant sugar: `cov`/`corrcoef`, `outer`/`diag`/`trace`, `argsort`/`take`, axis reductions (rank-2), `any`/`all` | **Done** |
| **v0.1** tag | Explicit release cut | **Deferred** |
| **M7** | **`i64` surface (correctness):** shared array grammar + integer-path LA (wrapping) + **i64-unique** + views + gcd/lcm/divmod/bitcount + **`from_i64` solvers** (i64 in → f64 out). | **Done** |
| **M7.b** | **Quant leave-late pack:** LA diagnostics, median/quantile, random, indexing, partial `out=` (#21), host view entry (`push_view_*` / `push_array_copy_*`). | **Done** |
| **M7.c** | **Optimize entire surface** (f64 + i64): structural and kernel performance once M7/M7.b correctness holds | **In progress** |
| **M8** | **Host integration depth** (TallyDB letter §5–§6 + view face): see **§7.1.1** | **Planned** |
| **M9** | **Small-window pool** — freelist for *n* ≪ 256 (TallyDB hot path; letter pressure) | **Planned** |
| **M10** | **Embed-safe Lua boundary** — letter **§1.1–§1.3** (feature-split face, longjmp/`Drop`, no panic across C) | **Planned** |
| **M11** | **CI + embed hygiene** — letter **§7** (APICHECK, ASan) + no `DLOPEN` embed profile, Miri-clean `take_uninit` | **Planned** |
| **M12** | **Arrow C Data Interface + `arrow-lite`** — letter **§3**; cutover when shared lite v0.1 ships | **Gated** |

**Priority:** **M7.c** (in progress on `feat-m7c-optimize`) → embed **M8–M11** → **M12**. Further dtypes (`f32`, complex, …) after this arc unless a new need appears.

**Also tracked:** GitHub **#21** full `out=` surface (beyond M7.b partial); TallyDB engine cutover (other repo).

**TallyDB readiness bar:** M7 `i64` + M7.b host entry + **M8–M11** embed safety/pool/CI + **M12** C ABI / lite layout. Copies into owned arrays remain acceptable for current window sizes (letter §5).

### 7.1.1 TallyDB letter → milestone agreements

Source: external letter *“TallyDB → MatLua: what we need, and what is yours”* (Human-supplied). These are **agreed planning constraints**, not “we implement whatever TallyDB codes.”

#### Required (their §1–§3) — owned by MatLua milestones

| Letter | Agreement | Milestone |
|--------|-----------|-----------|
| **§1.1** Host-owned interpreter; **no second vendored Lua** at link time | Keep `register(L)`. **Split Cargo features**: face/bindings vs vendored PUC (tests/tools only). Hosts compile ANSI Lua as C (`longjmp`); no `package.loadlib` required. | **M10** |
| **§1.2** No Rust value needing `Drop` live across a Lua call that can `longjmp` | Systematic audit; error paths must drop owned values before `lua_error`. Reserve stack before multi-push. | **M10** |
| **§1.3** No panic may cross the C boundary | `catch_unwind` (or equivalent) on `extern "C"`; map to Lua error / `Result`. | **M10** |
| **§2.1** `i64` exact end-to-end; no silent widen past 2⁵³ at the boundary | **M7 done.** Maintain: keys/timestamps stay `i64`; f64 only on explicit promote / real LA. | **M7** (maintain) |
| **§2.2** Documented absence contract | **Default now:** non-null in, non-null out; **refuse** Arrow/host buffers with nulls (do not read null slots). Longer term: optional **validity mask** path so crossings align with TallyDB’s per-element validity bytes (cheaper than NaN conversion). Document in public face when M8/M12 touch interchange. | **M8** (doc + host seam) / **M12** (Arrow) |
| **§3.1** Contiguous `f64`/`i64` buffer + length + validity (or refuse nulls) | Already refuse-null on Arrow import; keep for every dtype. | **M12** + maintain |
| **§3.2** Reachable **without linking arrow-rs on the host** | Prefer **Arrow C Data Interface** (`ArrowArray` / `ArrowSchema` + release callback) for zero-copy interchange. | **M12** |
| **§3.3** Release-callback discipline | Producer sets `release`; consumer calls once; moved-from nulls `release`. | **M12** |
| **§3.4** Never interpret null slot buffer contents as data | Keep refuse-or-proper-nulls policy. | maintain / **M12** |
| **§3 preference** Drop unused direct arrow-rs crates | Manifest honesty; do when touching Arrow deps. | opportunistic / **M12** |

#### Host product choices (their §4 — our design wins)

1-based Lua face, row-major dense core, faer-backed LA, singular/rank-deficient behavior, error taxonomy, dtype order — **unchanged MatLua decisions**.

#### Their §5–§7 — agreed responses

| Letter | Agreement | Milestone |
|--------|-----------|-----------|
| **§5** `linalg` takes owned `&Array`; views not accepted yet | **Accepted for now** (copy at boundary OK for current windows). Shape APIs so view-aware LA can land later without API break if cheap. | **M8** (optional view-accepting linalg) / not blocking |
| **§6** Two array types (`tallydb.vector` vs `ml.array`) is a poor surface | **Prefer one math type = MatLua.** Make host→MatLua cheap via `push_view_*` / copy. TallyDB may retire overlapping vocabulary; we do not require them to keep `tallydb.vector` for matrix work. | **M7.b** entry / **M8** deepen |
| **§7** APICHECK + ASan/UBSan; differential tests vs NumPy/DuckDB | Welcome; wire **APICHECK** profile and encourage shared diffs for `solve`/`lstsq`/`qr`. | **M11** (+ their contrib) |

#### What M8 specifically means (post–M7.b)

M7.b delivered **host entry** (`push_view_f64`/`i64`, `push_array_copy_*`) as read-only views + copies. **M8** is the rest of the integration depth:

1. Document host contracts (lifetime, absence §2.2, 1-based vs 0-based) as a stable embed chapter.
2. Optional: richer view face (more methods; clear mutation → `to_array` only).
3. Optional: `linalg` accepting views / `ArrayView` to cut allocs on small windows when measured.
4. Alignment helpers so TallyDB can present **one** array type to scripts (conversions or direct userdata accept — design at M8 kickoff, MatLua-led).
5. Does **not** replace M10 safety or M12 Arrow C ABI.

### 7.1.2 M7.c optimization program (in progress)

**Goal:** whole-surface performance (f64 + i64) without breaking correctness contracts.

**Wave 1 (landed on this branch):**
- `i64` matmul / `matmul_at` / `matmul_bt` / `dot`: `i–k–j` / unrolled accumulation (wrapping preserved).
- `i64` elementwise add/mul/sum: 4-wide unroll.
- `ArrayI64::transpose`: blocked out-of-place (same idea as f64).
- `median`: order-statistic path (`select_nth`) instead of full sort for scalar median.

**Next waves (planned):** re-run fair table (`compare_fair.py`); attack remaining Rust/NumPy ≫ 1 and Lua/Rust ≫ 2; extend `out=` paths where they remove allocs; avoid scale-hostile fixed chunking (prefer algorithms that scale past n=1024).

#### What remains explicitly *not* TallyDB-owned

MatLua public API stays free of TallyDB types, SQL, and storage. TallyDB is a first consumer and design pressure source (§2.3).

### 7.2 Performance program (P0–P6)

Goal: make the **current** surface competitive with NumPy for ordinary dense
`f64` desk work, then **prove** it. Optimize structural costs first; formal
NumPy comparison is the **gate**, not the driver of every change.

#### Performance contract (“on par”)

| Axis | Contract |
|------|----------|
| **Scope** | Dense `f64`, rank 1–2: elementwise bulk ops, `matmul`, `solve`, and the shipped decompositions |
| **Sizes** | Report at least \(n \in \{64, 256, 1024\}\) (matmul may also use 2048) |
| **Faces** | Always **three-way**: **NumPy** (baseline **1.00×**), **Rust** (critical path), **Lua** (product) |
| **Reporting** | Relative wall time to NumPy (NumPy column is always 1.00×); absolute ms secondary |
| **Bar** | On medium+ matmul/solve (release, same shapes), MatLua wall time within about **1–2×** of NumPy on the same machine |
| **Method** | Fixed harness + NumPy scripts under `tests/bench/`; publish tables in `tests/README.md` |
| **Non-goals** | Beat MKL/OpenBLAS on every micro-op; research kernels; replace faer with system BLAS; full broadcasting engine |

**Residuals vs bar (present tense):** medium+ matmul and many bulk kernels sit near the band; large pure `XᵀX` and some micro-ops can exceed 1–2× (OpenBLAS residual / noise). Lua bulk paths track Rust when hosts use generational GC. Prefer `tests/README.md` for measured numbers, not success language here.

| Milestone | Intent | Status |
|-----------|--------|--------|
| **P0** | Write this contract and the P1–P5 wall into durable docs | **Done** |
| **P1** | Kill the LA tax: zero-copy faer `MatRef` over row-major `Array` inputs; copy out only for owned results | **Done** |
| **P2** | Elementwise / reductions: contiguous bulk loops, less alloc, SIMD-friendly code | **Done** |
| **P3** | Hot-path hygiene: fewer intermediate arrays, cheap constructors, avoid needless clones | **Done** |
| **P4** | Lua face cost: reduce per-op overhead on bulk paths | **Done** |
| **P5** | Comparison harness vs NumPy; meet §7.2 bar or document gaps | **Done** |
| **P6** | Matmul dest-GEMM into row-major + parallel large products; remeasure 3-way | **Done** |

Order: **P0 → P1 → P2 → P3 → P4 → P5 → P6** (P6 = close matmul residual).

Harness and latest table: [`tests/README.md`](tests/README.md) and
`python3 tests/bench/compare_fair.py`. Open perf work: GitHub Issues (one
function per issue); close the issue and update this file when the Human is
satisfied with that function’s performance.

---

## 8. Records and process (repo-specific)

| Record | Job |
|--------|-----|
| **README.md** | User / visitor surface — accurate to what ships |
| **DESIGN.md** (this file) | Rulings, architecture, scope — accurate to why and how |
| **AGENTS.md** | Human/agent process (replace whole from upstream only on explicit decision) |
| **tests/** | Correctness integration tests + fair three-way microbenches; results prose in `tests/README.md` |
| **GitHub Discussions** | Open design while unsettled |
| **GitHub Issues** | Todos, bugs, and **per-function performance work** (proposed fix; close when Human is satisfied) |

When a Discussion closes: write the ruling here in present tense; note rejected
alternatives when useful; optional reopen triggers.

### 8.1 Attribution

Human is always **author** of record; agent may be **co-author** when allowed for that session.

---

## 9. Rejected alternatives (summary)

| Rejected | Why |
|----------|-----|
| Pure Lua implementation | Too slow for the NumPy role |
| mlua/rlua as primary binding | Host-owned state, zero-copy, PUC 5.4 discipline |
| nalgebra as LA backend | Wrong performance sweet spot for medium/large dense LA |
| System BLAS/LAPACK default | Portability and embed constraints |
| Nested tables as the array model | Defeats the purpose |
| API shaped only for TallyDB | Hosts fit to MatLua; TallyDB is one consumer |
| Full NumPy parity as v1 goal | Scope unbounded; curated dense surface first |
| Forcing multi-line temps for one formula | Poor Lua ergonomics |
| `*` for matrix multiply | Collides with elementwise; explicit `matmul` |

---

## 10. Status

**Shipped surface** matches §3–§4: row-major `f64` arrays, **`ArrayI64` (M7)**, views
(`f64` and `i64`), elementwise, Arrow `Float64`/`Int64`, faer LA + `i64_ops` / `from_i64`,
Lua face (1-based) including `*_i64` and dual-dtype `solve`/`matmul`, M4a–M6, reshape COW (§3.13).

Package version is **`0.0.1`**. Call **v0.1** when the human tags a release;
until then treat the tree as a **v0.1 candidate** per §7.1. **M7** and **M7.b** are **Done**.
**Next:** **M7.c** (optimize, new branch after merge), then embed track **M8–M11** and **M12**
(Arrow C Data / arrow-lite) per §7.1 / §7.1.1 (TallyDB letter agreements).

Open work: §7.1 M7.c–M12, GitHub **#21** (`out=` full surface), measured tables in
[`tests/README.md`](tests/README.md) — not as a living log in this file.

Update this document when rulings or the frozen public face change — not on
every internal refactor.
