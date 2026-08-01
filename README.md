# MatLua

MatLua is a general-purpose dense array and linear-algebra library for Rust and Lua 5.4 — deep enough that analytic engines and quant workflows stay in-process for ordinary matrix work, portable enough that no single host owns the API.

MatLua is the architectural counterpart to NumPy for the Lua ecosystem: a high-level language in front, a systems language in the core. Python is replaced by Lua 5.4, C by Rust, `ndarray` by Apache Arrow, and BLAS/LAPACK by [faer](https://github.com/sarah-quinones/faer-rs).

This repository ships **a Rust crate**. Hosts embed it; ports compile it.

## High-level stack

| Layer | NumPy world | MatLua world |
|-------|-------------|--------------|
| Scripting language | Python | **Lua 5.4** |
| Core implementation | C (C++/Fortran) | **Rust** |
| Binding / extension API | Python C API, Cython | **Hand-rolled Lua C API** (no mlua) |
| Primary array / buffer model | `ndarray` | **Apache Arrow** |
| Dense linear algebra | BLAS / LAPACK | **faer** |
| **Deliverable** | wheels / `pip` package | **Rust crate** |

```text
Lua 5.4  ← scripts / embedders
   ↓
Hand-rolled Lua C API  (optional feature)
   ↓
Rust crate (MatLua)
   ├── Arrow arrays / batches   ← data model & interchange
   └── faer                     ← dense matmul, factorizations, solvers
```

Arrow owns the **data model**. faer owns **dense linear algebra**. They meet at explicit boundaries (views, gathers, copies)—the same separation NumPy has between `ndarray` and BLAS.

## Goals

- Provide a **NumPy-shaped** numeric layer for Lua: arrays/matrices, elementwise ops, and linear algebra
- Ship as an **embeddable Rust library** with an optional Lua face
- Use **pure-Rust LA** via faer (no system BLAS/LAPACK, no Fortran toolchain)
- Use **Arrow** as the primary in-memory layout and interchange format
- Keep the surface **curated**: dense numerics and linear algebra first; grow by deliberate addition

## Who it is for

| Audience | How they use MatLua |
|----------|---------------------|
| **Embedded analytic / database hosts** | Depend on the crate; adapt host buffers and Lua to MatLua’s API |
| **WASM and other constrained targets** | Compile the crate for the target; supply their own Lua runtime and glue |
| **Rust developers** shipping Lua-scriptable tools | Embed MatLua + Lua; expose matrices and arrays to scripts |
| **Lua users** who need fast dense numeric work | Use a host that embeds MatLua (or a future module build) |

### Downstream contract

- **Hosts fit to MatLua.** The crate defines buffer and API contracts. Downstream engines adapt at the boundary rather than the reverse.
- **MatLua stays portable.** This is not a WASM project, but the core avoids non-portable native LA dependencies so a separate port can target `wasm32` (and similar) without a rewrite.

## Design principles

1. **Rust crate first** — the library is the product; system-Lua modules and finished WASM binaries are downstream packaging.
2. **Lua 5.4** (PUC reference family) — hand-rolled C API bindings; no mlua, no LuaJIT in-tree.
3. **Arrow for arrays** — primary in-memory model and interchange.
4. **faer for dense LA** — solve, QR, SVD, Cholesky, and related ops without BLAS/LAPACK.
5. **Thin Lua, hot path in Rust** — scripts orchestrate; kernels run in compiled code.
6. **Embeddable and explicit** — clear ownership, documented view/copy rules, feature-gated Lua bindings.

## Crate features (intended)

| Feature | Purpose |
|---------|---------|
| default | Core arrays + faer LA (Rust API) |
| `lua` | Hand-rolled Lua 5.4 bindings / registration helpers |
| lean / target-friendly flags | Trim unnecessary `std` extras where practical |

Exact feature names will be fixed as the crate is scaffolded.

## Status

Design-phase. API surface, indexing convention, and first milestones are still being locked.

## License

MIT (intended; finalize when the repo is published).
