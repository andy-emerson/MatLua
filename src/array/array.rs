//! Owned dense `f64` arrays (row-major, n-D).

use arrow_array::{Array as ArrowArray, Float64Array};
use arrow_buffer::Buffer;

use super::shape::{numel_checked, Shape};
use super::view::{ArrayView, ArrayViewMut};
use crate::error::{Error, Result};

/// Owned dense n-D array of `f64` values in row-major order.
///
/// This is the primary Rust-side array type for M1. Storage is a contiguous
/// buffer compatible with Arrow [`Float64Array`] interchange (no nulls).
#[derive(Clone, Debug)]
pub struct Array {
    shape: Shape,
    /// Contiguous row-major values; length always equals `shape.numel()`.
    data: Vec<f64>,
}

impl Array {
    /// Shape of the array.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Axis lengths.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Number of axes.
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if there are zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Contiguous row-major values.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Contiguous row-major values (mutable).
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Shared view of the whole array.
    #[inline]
    pub fn view(&self) -> ArrayView<'_> {
        ArrayView::from_shape_slice(self.shape.clone(), &self.data)
    }

    /// Mutable view of the whole array.
    #[inline]
    pub fn view_mut(&mut self) -> ArrayViewMut<'_> {
        ArrayViewMut::from_shape_slice(self.shape.clone(), &mut self.data)
    }

    /// Deep copy (same as [`Clone`] for owned data).
    #[inline]
    pub fn to_owned_array(&self) -> Array {
        self.clone()
    }

    /// Build from shape and a flat row-major buffer.
    pub fn from_shape_vec(shape: impl Into<Vec<usize>>, data: Vec<f64>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        if data.len() != shape.numel() {
            return Err(Error::Shape(format!(
                "data length {} does not match shape {} ({} elements)",
                data.len(),
                shape,
                shape.numel()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Build from shape and a flat row-major slice (copies).
    pub fn from_shape_slice(shape: impl Into<Vec<usize>>, data: &[f64]) -> Result<Self> {
        Self::from_shape_vec(shape, data.to_vec())
    }

    /// Zeros with the given shape.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        Ok(Self {
            data: vec![0.0; shape.numel()],
            shape,
        })
    }

    /// Ones with the given shape.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        Ok(Self {
            data: vec![1.0; shape.numel()],
            shape,
        })
    }

    /// Fill every element with `value`.
    pub fn full(shape: impl Into<Vec<usize>>, value: f64) -> Result<Self> {
        let shape = Shape::new(shape)?;
        Ok(Self {
            data: vec![value; shape.numel()],
            shape,
        })
    }

    /// Rank-1 range `[start, stop)` with step `1.0` (NumPy-style, exclusive stop).
    pub fn arange(start: f64, stop: f64) -> Result<Self> {
        Self::arange_step(start, stop, 1.0)
    }

    /// Rank-1 range `[start, stop)` with the given step.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if `step` is zero or the length overflows.
    pub fn arange_step(start: f64, stop: f64, step: f64) -> Result<Self> {
        if step == 0.0 {
            return Err(Error::Shape("arange step must be non-zero".into()));
        }
        if (step > 0.0 && start >= stop) || (step < 0.0 && start <= stop) {
            return Self::from_shape_vec(vec![0], Vec::new());
        }
        let n_f = ((stop - start) / step).ceil();
        if !n_f.is_finite() || n_f < 0.0 {
            return Err(Error::Shape("arange produced non-finite length".into()));
        }
        if n_f > usize::MAX as f64 {
            return Err(Error::Shape("arange length overflow".into()));
        }
        let n = n_f as usize;
        let mut data = Vec::with_capacity(n);
        let mut x = start;
        for _ in 0..n {
            if (step > 0.0 && x >= stop) || (step < 0.0 && x <= stop) {
                break;
            }
            data.push(x);
            x += step;
        }
        Self::from_shape_vec(vec![data.len()], data)
    }

    /// Read one element at a multi-index (0-based).
    pub fn get(&self, indices: &[usize]) -> Result<f64> {
        Ok(self.data[self.shape.offset(indices)?])
    }

    /// Write one element at a multi-index (0-based).
    pub fn set(&mut self, indices: &[usize], value: f64) -> Result<()> {
        let off = self.shape.offset(indices)?;
        self.data[off] = value;
        Ok(())
    }

    /// Set every element to `value`.
    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// Reshape without changing values if the element count matches (copies data).
    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        if shape.numel() != self.len() {
            return Err(Error::Shape(format!(
                "cannot reshape array of {} elements into {}",
                self.len(),
                shape
            )));
        }
        Ok(Self {
            shape,
            data: self.data.clone(),
        })
    }

    /// Reshape in place without copying if the element count matches.
    pub fn reshape_inplace(&mut self, shape: impl Into<Vec<usize>>) -> Result<()> {
        let shape = Shape::new(shape)?;
        if shape.numel() != self.len() {
            return Err(Error::Shape(format!(
                "cannot reshape array of {} elements into {}",
                self.len(),
                shape
            )));
        }
        self.shape = shape;
        Ok(())
    }

    /// Sum of all elements.
    pub fn sum(&self) -> f64 {
        self.data.iter().copied().sum()
    }

    /// Mean of all elements.
    ///
    /// # Errors
    ///
    /// Empty arrays return [`Error::Shape`].
    pub fn mean(&self) -> Result<f64> {
        if self.is_empty() {
            return Err(Error::Shape("mean of empty array".into()));
        }
        Ok(self.sum() / self.len() as f64)
    }

    /// Minimum element.
    pub fn min(&self) -> Result<f64> {
        self.data
            .iter()
            .copied()
            .reduce(f64::min)
            .ok_or_else(|| Error::Shape("min of empty array".into()))
    }

    /// Maximum element.
    pub fn max(&self) -> Result<f64> {
        self.data
            .iter()
            .copied()
            .reduce(f64::max)
            .ok_or_else(|| Error::Shape("max of empty array".into()))
    }

    /// Export as a non-null Arrow [`Float64Array`] (flat row-major values).
    pub fn to_arrow(&self) -> Float64Array {
        Float64Array::from(self.data.clone())
    }

    /// Import from a non-null Arrow [`Float64Array`] with an explicit shape.
    ///
    /// Nulls are rejected. Values are interpreted as row-major for `shape`.
    pub fn from_arrow(array: &Float64Array, shape: impl Into<Vec<usize>>) -> Result<Self> {
        if array.null_count() != 0 {
            return Err(Error::Shape(
                "from_arrow does not accept nulls in M1".into(),
            ));
        }
        let shape = Shape::new(shape)?;
        if array.len() != shape.numel() {
            return Err(Error::Shape(format!(
                "arrow length {} does not match shape {} ({} elements)",
                array.len(),
                shape,
                shape.numel()
            )));
        }
        let data = array.values().iter().copied().collect();
        Ok(Self { shape, data })
    }

    /// Copy values into a new Arrow [`Buffer`].
    pub fn to_buffer(&self) -> Buffer {
        Buffer::from_vec(self.data.clone())
    }

    /// Element-wise binary op with shape check; result is owned.
    pub(crate) fn binary_op<F>(&self, other: &Array, op: F) -> Result<Array>
    where
        F: Fn(f64, f64) -> f64,
    {
        if !self.shape.same_as(other.shape()) {
            return Err(Error::Shape(format!(
                "shape mismatch: {} vs {}",
                self.shape, other.shape
            )));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();
        Ok(Array {
            shape: self.shape.clone(),
            data,
        })
    }

    /// Element-wise op with a scalar; result is owned.
    pub(crate) fn scalar_op<F>(&self, scalar: f64, op: F) -> Array
    where
        F: Fn(f64, f64) -> f64,
    {
        let data = self.data.iter().map(|&a| op(a, scalar)).collect();
        Array {
            shape: self.shape.clone(),
            data,
        }
    }

    /// In-place element-wise binary op with shape check.
    pub(crate) fn binary_op_assign<F>(&mut self, other: &Array, op: F) -> Result<()>
    where
        F: Fn(f64, f64) -> f64,
    {
        if !self.shape.same_as(other.shape()) {
            return Err(Error::Shape(format!(
                "shape mismatch: {} vs {}",
                self.shape, other.shape
            )));
        }
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = op(*a, b);
        }
        Ok(())
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Array) -> Result<Array> {
        self.binary_op(other, |a, b| a + b)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Array) -> Result<Array> {
        self.binary_op(other, |a, b| a - b)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Array) -> Result<Array> {
        self.binary_op(other, |a, b| a * b)
    }

    /// Element-wise division.
    pub fn div(&self, other: &Array) -> Result<Array> {
        self.binary_op(other, |a, b| a / b)
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Array {
        let data = self.data.iter().map(|&a| -a).collect();
        Array {
            shape: self.shape.clone(),
            data,
        }
    }
}

impl PartialEq for Array {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
    }
}

/// Validate shape product without constructing (used by views).
#[allow(dead_code)]
pub(crate) fn check_shape(dims: &[usize]) -> Result<usize> {
    numel_checked(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{ArrayView, ArrayViewMut};

    #[test]
    fn zeros_and_get_set() {
        let mut a = Array::zeros(vec![2, 3]).unwrap();
        assert_eq!(a.rank(), 2);
        assert_eq!(a.len(), 6);
        assert_eq!(a.get(&[0, 0]).unwrap(), 0.0);
        a.set(&[1, 2], 3.5).unwrap();
        assert_eq!(a.get(&[1, 2]).unwrap(), 3.5);
        assert!(a.get(&[2, 0]).is_err());
    }

    #[test]
    fn arange_and_reshape() {
        let a = Array::arange(0.0, 6.0).unwrap();
        assert_eq!(a.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = a.reshape(vec![2, 3]).unwrap();
        assert_eq!(b.get(&[1, 0]).unwrap(), 3.0);
        assert_eq!(b.sum(), 15.0);
        assert!((b.mean().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn elementwise_and_arrow_roundtrip() {
        let a = Array::from_shape_slice(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array::full(vec![2, 2], 10.0).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.as_slice(), &[11.0, 12.0, 13.0, 14.0]);
        let arrow = c.to_arrow();
        let back = Array::from_arrow(&arrow, vec![2, 2]).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn view_shares_then_owned_copy() {
        let mut a = Array::arange(0.0, 4.0).unwrap();
        {
            let v = a.view();
            assert_eq!(v.get(&[2]).unwrap(), 2.0);
        }
        {
            let mut v = a.view_mut();
            v.set(&[1], 9.0).unwrap();
        }
        assert_eq!(a.get(&[1]).unwrap(), 9.0);
        let owned = a.view().to_owned_array();
        assert_eq!(owned.as_slice(), a.as_slice());
    }

    #[test]
    fn host_buffer_view() {
        let mut host = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        {
            let view = ArrayViewMut::try_from_dims(vec![2, 3], &mut host).unwrap();
            assert_eq!(view.get(&[1, 1]).unwrap(), 5.0);
        }
        let v = ArrayView::try_from_dims(vec![3, 2], &host).unwrap();
        assert_eq!(v.get(&[2, 1]).unwrap(), 6.0);
    }
}
