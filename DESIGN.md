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
- **Product bar for LA remains `f64`.** Integer arrays do not enter `matmul`/`solve`/factorizations.
- **Arithmetic:** wrapping add/sub/mul/neg/abs; truncating `/`; division by zero → `0` (no panic).
- **Mean** (scalar or axis) returns **`f64`** (or `f64` array).
- **Casts:** `ArrayI64::to_f64` / `Array::to_i64` (truncate toward zero).
- **Arrow:** `Int64Array` interchange (non-null).
- **Lua face:** `zeros_i64` / `ones_i64` / `full_i64` / `arange_i64` / `array_i64` / `eye_i64` / `diag_i64` / `outer_i64`; userdata methods mirror a subset of the `f64` face (`get`/`set` use integers; `to_f64`).
- **Not yet on i64 (follow-ups inside M7 or later):** full ufunc set, `cov`/`corrcoef`, nan*, host views, every axis helper, performance tuning.


### 4.1 Module functions

`zeros`, `ones`, `full`, `arange` (`start, stop[, step]`), `array`, `eye`, `where`,
`matmul`, `matmul_at` (AᵀB; large same-buffer AᵀA may materialize Aᵀ once), `matmul_bt` (ABᵀ; large AAᵀ same rule), `normal_eq`, `solve`, `lstsq`, `eigh`, `pinv`, `transpose`, `dot`, `norm`, `cholesky`, `qr`, `svd`

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
| `matlua::linalg` | matmul, solve, decompositions, norm, eye |
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
| **M7** | **`i64` arrays** (first multi-dtype step): owned construct/index/elementwise needed for ordering keys and exact integer columns; LA remains `f64`-first | **In progress** (correctness) |
| **M8** | **Lua host-buffer / view face** — scripts can use engine (or other host) memory without only owned copies (`from_host` / view userdata; lifetime contract) | Planned |
| **M9** | **Small-window pool** — freelist / recycle policy that covers *n* ≪ 256 (TallyDB-style 64-row windows), not only bulk desk sizes | Planned |
| **M10** | **Embed-safe Lua boundary** — no `lua_error` longjmp over live Rust drops; `catch_unwind` on every `extern "C"` entry | Planned |
| **M11** | **CI + embed hygiene** — `.github` CI (tests, `MATLUA_LUA_APICHECK`); no `LUA_USE_DLOPEN` on embed/vendored profile; `take_uninit` Miri-clean (or init-only public paths) | Planned |
| **M12** | **`arrow-lite` cutover** — runtime off `arrow-array`/`arrow-buffer`/`arrow-schema` once shared **`arrow-lite` v0.1** is released; refactor sooner rather than later after that gate | **Gated** on arrow-lite v0.1 |

**Priority (Human, 2026-08-02):** **M7 (`i64`) first** among open work; then **M8–M11** (TallyDB fusion / embed bar from host review); **M12** as soon as **arrow-lite v0.1** exists (layout refactor is cheaper early). Further dtypes (`f32`, complex, …) wait until after M7–M12 unless a new need appears.

**Also tracked (not renumbered milestones):** in-place `out=` ([#21](https://github.com/andy-emerson/MatLua/issues/21)); TallyDB engine cutover (other repo); optional later dtypes beyond `i64`.

**TallyDB readiness framing:** M0–M6 solve bulk desk math and vocabulary. Fusion needs **M8–M9** (host views + small allocs) and embed bar **M10–M11**. Shared layout **M12** and keys **M7** complete the long joint stack; **M7 is urgent for integer columns even before full fusion polish.**

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

**Shipped surface** matches §3–§4: row-major `f64` arrays, views, elementwise,
Arrow interchange, faer LA, Lua face with 1-based indexing, composed paths
(`matmul_at`, `matmul_bt`, `normal_eq`), M4a–M6 surface, reshape buffer sharing (§3.13).

Package version is **`0.0.1`**. Call **v0.1** when the human tags a release;
until then treat the tree as a **v0.1 candidate** per §7.1. **Next open milestone: M7 (`i64`)** (§7.1).

Open work: §7.1 M7–M12, GitHub Issues (e.g. `out=` #21), and measured tables in
[`tests/README.md`](tests/README.md) — not as a living log in this file.

Update this document when rulings or the frozen public face change — not on
every internal refactor.
