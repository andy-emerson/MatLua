# MatLua design

This document is the durable record of **what MatLua is and why**, for implementers and agents. It describes the present design intent. Open work and debate while a choice is unsettled live in GitHub Discussions; **closed** rulings are written here. The user-facing summary remains in [README.md](README.md). Process rules live in [AGENTS.md](AGENTS.md).

---

## 1. Product

MatLua is a **general-purpose dense array and linear-algebra library** with a **Lua 5.4 face** and a **Rust implementation**. It is deep enough that analytic engines and quant workflows can stay in-process for ordinary matrix work, and portable enough that **no single host owns the API**.

It is the architectural counterpart to NumPy for the Lua ecosystem: high-level language in front, systems language in the core.

| Layer | NumPy world | MatLua world |
|-------|-------------|--------------|
| Scripting language | Python | **Lua 5.4** (PUC reference family) |
| Core implementation | C (C++/Fortran) | **Rust** |
| Binding API | Python C API, Cython | **Hand-rolled Lua C API** (no mlua, no LuaJIT in-tree) |
| Array / buffer model | `ndarray` | **Apache Arrow** |
| Dense linear algebra | BLAS / LAPACK | **faer** |
| What ships | wheels / `pip` package | **Rust crate** (hosts embed; optional native module packaging later) |

**User POV:** MatLua looks and feels like a **Lua library**. Users never need to touch Rust.

**Dev / host POV:** MatLua is implemented and integrated as a **Rust crate**. Hosts depend on it, adapt their buffers to its contracts, and register it into a host-owned Lua 5.4 state (or load a module built from the same code).

### 1.1 Downstream contract

- **Hosts fit to MatLua.** The crate defines buffer and API contracts. Downstream engines adapt at the boundary rather than the reverse.
- **TallyDB is a design pressure, not the product boundary.** MatLua must be easy for TallyDB (and similar engines) to adopt, without baking TallyDB types, SQL, or storage into the public API.
- **MatLua stays portable.** The core avoids system BLAS/LAPACK and other non-portable native LA dependencies so ports (including future `wasm32`) do not require a rewrite of the numerics stack.

### 1.2 Success criterion (“leave late”)

A script author doing ordinary dense array and dense linear-algebra work should stay on **Lua + MatLua** (inside whatever host embeds them) until the problem is *genuinely* outside that domain (sparse, ML frameworks, plotting, full scientific ecosystem packages, etc.). Leaving should be possible **cleanly** (Arrow interchange), but not necessary for day-to-day matrix work.

---

## 2. Scope

### 2.1 In scope

- Dense numeric arrays with an **n-D data model** (shape + rank) from the start
- **NumPy-shaped** capabilities: constructors, elementwise ops, reductions, indexing/slicing helpers, dense linear algebra
- **Lua-shaped** syntax and ergonomics (see §5)
- Primary quality bar on **`f64`**, with a dtype story that can grow
- **faer** for dense matmul, factorizations, and solvers
- **Arrow** as the primary in-memory layout and interchange format
- Explicit **ownership / view / copy** rules compatible with zero-copy host buffers
- Optional Cargo feature for the **Lua face** (packaging); the product is incomplete for its stated purpose until that face can do real bulk work

### 2.2 Non-goals (for a long time)

| Non-goal | Reason |
|----------|--------|
| Full NumPy / SciPy module parity | Unbounded surface; not required for “leave late” on dense work |
| Nested Lua tables as runtime storage | Slow; the problem MatLua exists to avoid |
| System BLAS/LAPACK as the default backend | Portability, embed story, pure-Rust build |
| mlua / rlua / LuaJIT as the in-tree binding | Host-owned PUC 5.4; hand-rolled C API |
| Sparse matrices, graphs, DataFrames | Different data models |
| FFT, random suites, special-function libraries | Separate product surfaces |
| Autodiff / ML frameworks | Wrong layer |
| IO ecosystem (`.npy`, CSV, HDF5, …) as a core mission | Host / Arrow concern |
| TallyDB-specific types or SQL in the public API | Hosts fit to MatLua |
| Research-grade novel LA algorithms | faer owns serious dense LA; MatLua is integration and product boundary |

### 2.3 Relationship to TallyDB (informative)

If MatLua succeeds at general dense LA:

- TallyDB can depend on **MatLua** for matrix-class work and need not link **faer** directly.
- TallyDB keeps **engine-native** specialty compute (e.g. closed-form window statistics) that MatLua is not meant to replace.
- TallyDB adapts **arrow-lite** / engine buffers and its Lua embed to MatLua’s contracts.

MatLua does not subsume TallyDB.

---

## 3. Closed design rulings

### 3.1 Indexing

| Face | Convention |
|------|------------|
| **Lua (user)** | **1-based** |
| **Rust (implementation)** | **0-based** |

The binding translates. Users never need to know Rust indices.

### 3.2 Rank and shape

- Public data model is **n-D from the start**: an array has rank and shape, not a permanently separate “vector type” and “matrix type” that cannot grow.
- **Storage** is a dense typed buffer plus shape metadata (Arrow-shaped), not a tree of Lua tables.
- **Implementations may specialize** rank-1 and rank-2 kernels (and other fast paths) for performance. Specialization is **invisible** to users except where mathematics requires it (e.g. `matmul` needs compatible 2-D shapes).
- v0.1 **depth** of operations and tests prioritizes the paths needed for real desk math (especially rank 1–2 linear algebra) while keeping the type model n-D.

### 3.3 Dtypes

- **v0.1 quality bar: `f64`.**
- Architecture remains **extensible** toward other widths and kinds that matter later (e.g. `f32`, integers, complex). Those are deliberate later additions, not day-one parity.

### 3.4 Ownership, views, and copies

Direction (API details may refine without changing the spirit):

1. **Owned arrays** — a MatLua array owns (or jointly owns) its buffer when it is a MatLua-managed result or transferred ownership.
2. **Borrowed views** — allowed over **host-owned** memory so engines can offer zero-copy access. The **host guarantees lifetime** for the duration of the operation or documented handle scope.
3. **Results** of matmul / factorizations / solvers are **owned** arrays unless an explicit out-parameter API is added later.
4. **User-facing rule of thumb:** slices/views share memory unless the user **copies**; bulk math results are new arrays unless documented otherwise.

This matches the needs of zero-copy analytic hosts without coupling MatLua to any one engine.

### 3.5 Construction

- **Primary:** NumPy-like constructors — `zeros`, `ones`, `empty`/`uninit` as appropriate, `arange`, `full`, shape-first APIs, and construction from host/Arrow buffers.
- **Secondary sugar:** build from **nested Lua tables** for small literals (copy into dense storage). Tables are an input format, not the array implementation.
- MatLua arrays are **not** plain Lua tables at runtime.

### 3.6 Elementwise vs matmul operators

**Lean (illustrative API; names can freeze at implementation):**

- Metamethods for **elementwise** arithmetic: `+`, `-`, `*`, `/` (and unary minus).
- **Matrix multiplication** uses an explicit name: `matmul` (method and/or function).
- Linear-algebra entry points use clear names: `transpose`, `solve`, factorizations as designed.

Rationale: Lua has a single `*`; naming matmul avoids silent confusion. Elementwise operators keep everyday scripts short.

### 3.7 Expression style

- One mathematical formula should be expressible as **one expression** assigning **one** result.
- Intermediate `local`s are optional for human clarity, never required by the API for subexpressions of a single equation.
- `local` itself is normal Lua scoping, not a MatLua invention.

### 3.8 Deliverable and Lua face

- Repository product: **Rust crate** first.
- User product: **Lua library experience** (`require` and/or host registration).
- Cargo feature (intended name: `lua`) compiles registration helpers / bindings; default builds can exercise the Rust core without linking Lua.
- **Product-complete for the README promise** means the Lua face can perform **real bulk numeric work**, not only a demo op. Pure Lua reimplementations of kernels are non-goals.

### 3.9 Binding stack

- PUC **Lua 5.4** only in-tree.
- **Hand-rolled** thin C API bindings (host-owned `lua_State`).
- No mlua, no rlua, no LuaJIT as the supported in-tree path.
- Pattern aligned with serious embeds (vendoring/linking discipline, API-check builds where applicable): safety and zero-copy over high-level binding frameworks.

### 3.10 Dependencies (crate start)

| Dependency | Role |
|------------|------|
| **faer** | Dense linear algebra |
| **arrow-array / arrow-buffer / arrow-schema** (lean Arrow; avoid fat defaults) | Arrays, buffers, types |
| **thiserror** (or equivalent) | Public error types |
| Lua | Only behind the Lua feature; host or documented link strategy |

Explicitly not defaults: nalgebra (wrong sweet spot), system BLAS/LAPACK, mlua, ndarray as the data model, Parquet/Flight/DataFusion as core deps.

### 3.11 Project nature

MatLua is **library / systems engineering**, not a research project. Novel high-performance dense algorithms live in **faer**. MatLua owns contracts, API, glue, Lua face, and evidence.

---

## 4. Illustrative Lua face

Names below are **illustrative** until implementation freezes them. Semantics match §3.

### 4.1 Side-by-side sample

**NumPy**

```python
import numpy as np

X = np.zeros((100, 3))
y = np.arange(100, dtype=np.float64)

X[:, 0] = 1.0
X[:, 1] = y
X[:, 2] = y ** 2

y_centered = y - y.mean()
beta = np.linalg.solve(X.T @ X, X.T @ y)

print(beta, X[0, 1], np.linalg.norm(beta))
```

**MatLua (target feel)**

```lua
local ml = require "matlua"

local X = ml.zeros(100, 3)
local y = ml.arange(0, 99)

X:col(1):fill(1)
X:col(2):assign(y)
X:col(3):assign(y * y)

local y_centered = y - y:mean()
local beta = ml.solve(X:transpose():matmul(X), X:transpose():matmul(y))

print(beta, X:get(1, 2), beta:norm())
```

### 4.2 Single-equation matmul / transpose

```lua
local XtX = X:transpose():matmul(X)
local z = X:matmul(v)
```

### 4.3 Small literal sugar

```lua
local A = ml.array({ {1, 2}, {3, 4} })  -- copy into dense f64 storage
```

---

## 5. Lua-shaped vs NumPy-shaped

| Concern | Choice |
|---------|--------|
| Ideas / capability set | **NumPy-shaped** (dense arrays, bulk ops, linalg role) |
| Syntax / indexing / module load | **Lua-shaped** (`require`, 1-based, metamethods, colon methods, no `@`, no `1:3` token slices) |
| Storage | Dense buffers (NumPy-like), not Python lists / Lua tables |
| Everyday arithmetic | Short operators + a few sharp methods |
| Linear algebra vocabulary | Explicit names (`transpose`, `matmul`, `solve`, …) |

---

## 6. Architecture sketch

```text
Lua 5.4 scripts
      │  require / host registration (1-based API)
      ▼
Hand-rolled Lua C API  ← feature-gated
      ▼
Rust crate (MatLua)
      ├── Arrow arrays / buffers     ← data model & interchange
      └── faer                       ← dense matmul, factorizations, solvers
```

Arrow owns the **data model**. faer owns **dense linear algebra**. They meet at explicit boundaries (views, gathers, copies).

---

## 7. Milestones (intent)

Exact issue tracking is living status; this is the intended wall.

| Milestone | Intent |
|-----------|--------|
| **M0** | Crate skeleton, module layout, deps, error type, design docs in sync |
| **M1** | `f64` n-D arrays: construct, shape, basic index/view/copy rules, core elementwise |
| **M2** | Dense LA via faer: matmul, solve, initial decompositions as warranted |
| **M3** | Lua face: register into host state; real bulk ops from scripts; 1-based API |
| **v0.1** | M1–M3 good enough that a host can embed MatLua and scripts can do ordinary dense work without leaving for NumPy |

Build order may implement Rust-testable core before Lua; **v0.1** still includes the Lua face for the product promise.

---

## 8. Records and process (repo-specific)

| Record | Job |
|--------|-----|
| **README.md** | What it is, for a visitor |
| **DESIGN.md** (this file) | What we build and why; closed rulings |
| **AGENTS.md** | How human and agent work (do not refresh from upstream without an explicit experiment decision; this repo may pin a version under test) |
| **GitHub Discussions** | Open design conversation while a decision is unsettled |
| **GitHub Issues** | Todos, bugs, and trackable follow-ups after rulings |

When a Discussion decision closes: write the ruling here (present tense), note rejected alternatives when worth keeping, and optional reopen triggers.

### 8.1 Attribution

On commits produced with an agent: the Human is always **author**; the agent may be listed as **co-author** when the Human has allowed it for that session.

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

---

## 10. Status

Design-phase implementation has not begun. This document freezes the agreements above so scaffolding and API work do not rely on chat history alone. Illustrative Lua names freeze when the corresponding code lands and tests/docs are updated to match.
