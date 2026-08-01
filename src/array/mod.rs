//! Dense n-D `f64` arrays: owned buffers, views, constructors, element-wise ops.
//!
//! # Layout
//!
//! Contiguous **row-major (C-order)** storage. Rust indices are **0-based**.
//! The Lua face (`lua` feature) presents **1-based** indexing.
//!
//! # Ownership
//!
//! - [`Array`] — owned values.
//! - [`ArrayView`] / [`ArrayViewMut`] — borrowed views over a contiguous buffer
//!   (parent array or host memory). Lifetime is the caller's responsibility.
//! - [`ArrayView::to_owned_array`] copies into an owned array.
//!
//! # Arrow
//!
//! [`Array::to_arrow`] / [`Array::from_arrow`] interchange flat non-null
//! [`arrow_array::Float64Array`] values with an explicit shape.

mod array;
mod ops;
mod shape;
mod view;

pub use array::Array;
pub use ops::TryElemwise;
pub use shape::Shape;
pub use view::{ArrayView, ArrayViewMut};
