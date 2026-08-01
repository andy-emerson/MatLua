//! MatLua — dense numeric arrays and linear algebra for Lua 5.4.
//!
//! Rust core: Arrow-shaped `f64` buffers and faer-backed dense LA. Optional
//! Lua face (`lua` feature) registers into a host-owned PUC Lua 5.4 state.
//!
//! - Visitors / users: repository `README.md`
//! - Implementer rulings: repository `DESIGN.md`
//!
//! # Status
//!
//! M0–M3 complete (arrays, dense LA, Lua face). Package version is `0.0.1`
//! until a formal v0.1 release cut.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

pub mod array;
pub mod error;
pub mod linalg;

#[cfg(feature = "lua")]
#[cfg_attr(docsrs, doc(cfg(feature = "lua")))]
pub mod lua;

pub use array::{Array, ArrayView, ArrayViewMut, Shape};
pub use error::{Error, Result};
