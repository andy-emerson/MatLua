//! Borrowed views over dense `f64` buffers.

use super::array::Array;
use super::shape::Shape;
use crate::error::Result;

/// Shared view of a dense row-major `f64` buffer with an n-D shape.
///
/// The borrowed slice must outlive the view (host or parent array lifetime).
#[derive(Clone, Debug)]
pub struct ArrayView<'a> {
    shape: Shape,
    data: &'a [f64],
}

impl<'a> ArrayView<'a> {
    /// Construct a view; `data.len()` must equal `shape.numel()`.
    pub fn from_shape_slice(shape: Shape, data: &'a [f64]) -> Self {
        debug_assert_eq!(data.len(), shape.numel());
        Self { shape, data }
    }

    /// Construct from dims and slice.
    pub fn try_from_dims(dims: impl Into<Vec<usize>>, data: &'a [f64]) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if data.len() != shape.numel() {
            return Err(crate::error::Error::Shape(format!(
                "view data length {} does not match shape {} ({} elements)",
                data.len(),
                shape,
                shape.numel()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Shape of the view.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Axis lengths.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Rank.
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Underlying contiguous values.
    #[inline]
    pub fn as_slice(&self) -> &'a [f64] {
        self.data
    }

    /// Read one element (0-based multi-index).
    pub fn get(&self, indices: &[usize]) -> Result<f64> {
        Ok(self.data[self.shape.offset(indices)?])
    }

    /// Copy into an owned [`Array`].
    pub fn to_owned_array(&self) -> Array {
        Array::from_shape_vec(self.shape.dims().to_vec(), self.data.to_vec())
            .expect("view shape and data already validated")
    }
}

/// Mutable view of a dense row-major `f64` buffer with an n-D shape.
#[derive(Debug)]
pub struct ArrayViewMut<'a> {
    shape: Shape,
    data: &'a mut [f64],
}

impl<'a> ArrayViewMut<'a> {
    /// Construct a mutable view; `data.len()` must equal `shape.numel()`.
    pub fn from_shape_slice(shape: Shape, data: &'a mut [f64]) -> Self {
        debug_assert_eq!(data.len(), shape.numel());
        Self { shape, data }
    }

    /// Construct from dims and mutable slice.
    pub fn try_from_dims(dims: impl Into<Vec<usize>>, data: &'a mut [f64]) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if data.len() != shape.numel() {
            return Err(crate::error::Error::Shape(format!(
                "view data length {} does not match shape {} ({} elements)",
                data.len(),
                shape,
                shape.numel()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Shape.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Axis lengths.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Rank.
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Contiguous values.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        self.data
    }

    /// Contiguous values (mutable).
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        self.data
    }

    /// Shared reborrow.
    pub fn as_view(&self) -> ArrayView<'_> {
        ArrayView::from_shape_slice(self.shape.clone(), self.data)
    }

    /// Read one element.
    pub fn get(&self, indices: &[usize]) -> Result<f64> {
        Ok(self.data[self.shape.offset(indices)?])
    }

    /// Write one element.
    pub fn set(&mut self, indices: &[usize], value: f64) -> Result<()> {
        let off = self.shape.offset(indices)?;
        self.data[off] = value;
        Ok(())
    }

    /// Fill all elements.
    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// Copy into an owned [`Array`].
    pub fn to_owned_array(&self) -> Array {
        Array::from_shape_vec(self.shape.dims().to_vec(), self.data.to_vec())
            .expect("view shape and data already validated")
    }
}
