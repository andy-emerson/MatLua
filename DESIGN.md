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

- **v0.1 quality bar: `f64`.**
- Extensible later (`f32`, integers, complex) by deliberate addition, not day-one parity.

### 3.4 Ownership, views, and copies

1. **Owned arrays** — MatLua-managed results own their buffer.
2. **Borrowed views (Rust)** — `ArrayView` / `ArrayViewMut` over contiguous host or parent memory; **caller guarantees lifetime**.
3. **LA results** — matmul / factorizations / solvers return **owned** arrays (faer path currently copies at the Mat boundary).
4. **Rule of thumb:** views share memory until the user **copies**; bulk math results are new arrays unless documented otherwise.

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
- Other LA: module functions `solve`, `dot`, `norm`, `cholesky`, `qr`, `svd`.

Rationale: Lua has a single `*`; naming matmul avoids silent confusion.

### 3.7 Expression style

- One mathematical formula → **one expression** assigning **one** result.
- Intermediate `local`s are optional for human clarity, never required for subexpressions of a single equation.

### 3.8 Deliverable and Lua face

- Repository product: **Rust crate** first.
- User product: **Lua library** (`require "matlua"` after host registration).
- Feature **`lua`**: hand-rolled bindings + vendored PUC 5.4.7 for tests/tools. Hosts may link their own 5.4.
- Public host entry: `unsafe fn matlua::lua::register(*mut lua_State)`.
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
- faer `Mat` is column-major; the crate currently copies at the boundary. Zero-copy faer views may come later without changing the public Array model.

---

## 4. Lua face (frozen names)

Names match the `lua` feature on `main`. Tutorial samples live in
[README.md](README.md); this section freezes the surface for implementers.

### 4.1 Module functions

`zeros`, `ones`, `full`, `arange` (`start, stop[, step]`), `array`, `eye`,
`matmul`, `solve`, `transpose`, `dot`, `norm`, `cholesky`, `qr`, `svd`

### 4.2 Array methods and metamethods

Methods: `shape`, `rank`, `get`, `set`, `sum`, `mean`, `min`, `max`, `copy`,
`reshape`, `transpose`, `fill`  
Metamethods: `__add`, `__sub`, `__mul`, `__div`, `__unm`, `__len`, `__tostring`, `__gc`

### 4.3 NumPy contrast (capability, not syntax parity)

```python
# NumPy
beta = np.linalg.solve(X.T @ X, X.T @ y)
```

```lua
-- MatLua: explicit matmul; 1-based; no @
local beta = ml.solve(
  ml.matmul(X:transpose(), X),
  ml.matmul(X:transpose(), y)
)
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
      ├── Array / views / Arrow Float64 interchange
      └── faer  (copy in/out of column-major Mat today)
```

Arrow owns the **data model**. faer owns **dense linear algebra**. They meet at
explicit boundaries (views, gathers, copies).

**Modules (crate):**

| Path | Role |
|------|------|
| `matlua::array` | `Array`, `Shape`, views, elementwise ops |
| `matlua::linalg` | matmul, solve, decompositions, norm, eye |
| `matlua::error` | `Error` / `Result` |
| `matlua::lua` | register, userdata, optional test `Lua` helper (`lua` feature) |

---

## 7. Milestones

| Milestone | Intent | Status |
|-----------|--------|--------|
| **M0** | Crate skeleton, module layout, deps, error type | **Done** |
| **M1** | `f64` n-D arrays: construct, shape, view/copy, elementwise | **Done** |
| **M2** | Dense LA via faer: matmul, solve, decompositions | **Done** |
| **M3** | Lua face: register into host state; bulk ops; 1-based API | **Done** |
| **v0.1** | M1–M3 good enough that a host embeds MatLua and scripts do ordinary dense work without leaving for NumPy | **Candidate** (not version-tagged; crate still `0.0.1`) |

Natural follow-ups after a v0.1 cut (not blocking the candidate bar): richer
indexing/views from Lua, broadcasting policy, host buffer handles in Lua,
docs.rs polish, more reductions.

---

## 8. Records and process (repo-specific)

| Record | Job |
|--------|-----|
| **README.md** | User / visitor surface — accurate to what ships |
| **DESIGN.md** (this file) | Rulings, architecture, scope — accurate to why and how |
| **AGENTS.md** | Human/agent process (replace whole from upstream only on explicit decision) |
| **GitHub Discussions** | Open design while unsettled |
| **GitHub Issues** | Todos and bugs after rulings |

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

**Implementation through M3 is on `main`.** Closed rulings above match the code:

- Rust: row-major owned `f64` n-D arrays, views, elementwise, Arrow interchange, faer LA.
- Lua (`lua` feature): 1-based userdata, constructors, metamethods, linalg module functions.

Package version is **`0.0.1`**. Call **v0.1** when the human tags a release;
until then treat the tree as a **v0.1 candidate** per §7.

Update this document when rulings or the frozen public face change — not on
every internal refactor.
