//! Dense n-D arrays: owned buffers, views, constructors, element-wise ops.
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
//! - [`Array`] — owned `f64` values (product bar / dense LA).
//! - [`ArrayI64`] — owned `i64` values (M7; keys / exact integers).
//! - [`ArrayView`] / [`ArrayViewMut`] — borrowed views over contiguous `f64` buffers.
//! - [`ArrayViewI64`] / [`ArrayViewMutI64`] — borrowed views over contiguous `i64` buffers.
//! - [`ArrayView::to_owned_array`] copies into an owned array.
//!
//! # Arrow
//!
//! [`Array::to_arrow`] / [`Array::from_arrow`] interchange flat non-null
//! [`arrow_array::Float64Array`] values with an explicit shape.
//! [`ArrayI64::to_arrow`] / [`ArrayI64::from_arrow`] use [`arrow_array::Int64Array`].

mod array;
mod array_i64;
pub mod dtype;
pub(crate) mod kernels;
pub(crate) mod kernels_i64;
mod pool;
pub(crate) mod pool_i64;
mod ops;
mod ops_i64;
mod shape;
mod view;
mod view_i64;

pub use array::Array;
pub use array_i64::ArrayI64;
pub use dtype::DType;
pub use shape::Shape;
pub use view::{ArrayView, ArrayViewMut};
pub use view_i64::{ArrayViewI64, ArrayViewMutI64};

/// Pooled uninitialized buffer for sibling modules (linalg kernels).
#[inline]
pub(crate) fn pool_take_uninit(len: usize) -> Vec<f64> {
    pool::take_uninit(len)
}
