//! Dense n-D `f64` arrays: owned buffers, views, constructors, element-wise ops.
//!
//! Elementwise arithmetic and reductions use contiguous slice kernels (`kernels`).
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
pub(crate) mod kernels;
mod pool;
mod ops;
mod shape;
mod view;

pub use array::Array;
pub use shape::Shape;
pub use view::{ArrayView, ArrayViewMut};

/// Pooled uninitialized buffer for sibling modules (linalg kernels).
#[inline]
pub(crate) fn pool_take_uninit(len: usize) -> Vec<f64> {
    pool::take_uninit(len)
}
