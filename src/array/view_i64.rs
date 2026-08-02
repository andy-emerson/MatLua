//! Borrowed views over dense `i64` buffers.

use super::array_i64::ArrayI64;
use super::shape::Shape;
use crate::error::Result;

/// Shared view of a dense row-major `i64` buffer with an n-D shape.
#[derive(Clone, Debug)]
pub struct ArrayViewI64<'a> {
    shape: Shape,
    data: &'a [i64],
}

impl<'a> ArrayViewI64<'a> {
    /// Construct a view; panics if lengths mismatch. Prefer [`try_from_dims`].
    pub fn from_shape_slice(shape: Shape, data: &'a [i64]) -> Self {
        assert_eq!(
            data.len(),
            shape.numel(),
            "view data length {} != shape numel {}",
            data.len(),
            shape.numel()
        );
        Self { shape, data }
    }

    /// Construct from dims and slice.
    pub fn try_from_dims(dims: impl Into<Vec<usize>>, data: &'a [i64]) -> Result<Self> {
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
    pub fn as_slice(&self) -> &'a [i64] {
        self.data
    }
    /// Read one element (0-based multi-index).
    pub fn get(&self, indices: &[usize]) -> Result<i64> {
        Ok(self.data[self.shape.offset(indices)?])
    }
    /// Copy into an owned [`ArrayI64`].
    pub fn to_owned_array(&self) -> ArrayI64 {
        let mut data = super::pool_i64::take_uninit(self.data.len());
        data.copy_from_slice(self.data);
        ArrayI64::from_parts(self.shape.clone(), data)
    }
}

/// Mutable view of a dense row-major `i64` buffer with an n-D shape.
#[derive(Debug)]
pub struct ArrayViewMutI64<'a> {
    shape: Shape,
    data: &'a mut [i64],
}

impl<'a> ArrayViewMutI64<'a> {
    /// Construct a mutable view; panics if lengths mismatch.
    pub fn from_shape_slice(shape: Shape, data: &'a mut [i64]) -> Self {
        assert_eq!(
            data.len(),
            shape.numel(),
            "view data length {} != shape numel {}",
            data.len(),
            shape.numel()
        );
        Self { shape, data }
    }

    /// Construct from dims and mutable slice.
    pub fn try_from_dims(dims: impl Into<Vec<usize>>, data: &'a mut [i64]) -> Result<Self> {
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
    pub fn as_slice(&self) -> &[i64] {
        self.data
    }
    /// Contiguous values (mutable).
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [i64] {
        self.data
    }
    /// Shared reborrow.
    pub fn as_view(&self) -> ArrayViewI64<'_> {
        ArrayViewI64::from_shape_slice(self.shape.clone(), self.data)
    }
    /// Read one element.
    pub fn get(&self, indices: &[usize]) -> Result<i64> {
        Ok(self.data[self.shape.offset(indices)?])
    }
    /// Write one element.
    pub fn set(&mut self, indices: &[usize], value: i64) -> Result<()> {
        let off = self.shape.offset(indices)?;
        self.data[off] = value;
        Ok(())
    }
    /// Fill all elements.
    pub fn fill(&mut self, value: i64) {
        self.data.fill(value);
    }
    /// Copy into an owned [`ArrayI64`].
    pub fn to_owned_array(&self) -> ArrayI64 {
        let mut data = super::pool_i64::take_uninit(self.data.len());
        data.copy_from_slice(self.data);
        ArrayI64::from_parts(self.shape.clone(), data)
    }
}
