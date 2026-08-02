//! MatLua — dense numeric arrays and linear algebra for Lua 5.4.
//!
//! Rust core: Arrow-shaped dense buffers (`f64` product bar, `i64` M7) and faer-backed dense LA. Optional
//! Lua face (`lua` feature) registers into a host-owned PUC Lua 5.4 state.
//!
//! - Visitors / users: repository `README.md`
//! - Implementer rulings: repository `DESIGN.md`
//!
//! Package version is `0.0.1` until a formal v0.1 cut. See repository README / DESIGN.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

pub mod array;
pub mod error;
pub mod linalg;

#[cfg(feature = "lua")]
#[cfg_attr(docsrs, doc(cfg(feature = "lua")))]
pub mod lua;

pub use array::{Array, ArrayI64, ArrayView, ArrayViewMut, ArrayViewI64, ArrayViewMutI64, DType, Shape};
pub use error::{Error, Result};
