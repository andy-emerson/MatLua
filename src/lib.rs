//! MatLua — dense numeric arrays and linear algebra for Lua 5.4.
//!
//! This crate is the Rust core: Arrow-shaped buffers and faer-backed dense LA.
//! The Lua face is feature-gated (`lua`) and registers into a host-owned
//! PUC Lua 5.4 state. See `DESIGN.md` for product rules and closed decisions.
//!
//! M0 is scaffolding only: module layout, dependencies, and a public error
//! type. Array and linalg APIs are not implemented yet.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

pub mod array;
pub mod error;
pub mod linalg;

#[cfg(feature = "lua")]
#[cfg_attr(docsrs, doc(cfg(feature = "lua")))]
pub mod lua;

pub use error::{Error, Result};
