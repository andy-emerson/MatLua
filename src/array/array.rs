//! Owned dense `f64` arrays (row-major, n-D).

use std::sync::Arc;

use arrow_array::{Array as ArrowArray, Float64Array};
use arrow_buffer::Buffer;

use super::kernels;
use super::pool;
use super::shape::{numel_checked, Shape};
use super::view::{ArrayView, ArrayViewMut};
use crate::error::{Error, Result};

/// Owned dense n-D array of `f64` values in row-major order.
///
/// Primary Rust-side array type. Storage is a contiguous buffer compatible
/// with Arrow [`Float64Array`] interchange (no nulls).
///
/// The value buffer is reference-counted so [`Self::reshape`] can share storage
/// (metadata-only). In-place mutation uses copy-on-write when the buffer is shared.
#[derive(Debug)]
pub struct Array {
    shape: Shape,
    /// Contiguous row-major values; length always equals `shape.numel()`.
    data: Arc<Vec<f64>>,
}

impl Clone for Array {
    /// Deep copy of values (unique buffer). Prefer [`Self::reshape`] for
    /// zero-copy shape changes that intentionally share storage.
    fn clone(&self) -> Self {
        let mut data = pool::take_uninit(self.len());
        data.copy_from_slice(self.as_slice());
        Self::from_parts(self.shape.clone(), data)
    }
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
        self.data.as_ref()
    }

    /// `Arc` strong count for the value buffer (1 = uniquely owned payload).
    #[inline]
    pub(crate) fn buffer_strong_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    /// Contiguous row-major values (mutable).
    ///
    /// Clones the buffer if other arrays still share it (e.g. after [`Self::reshape`]).
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// Shared view of the whole array.
    #[inline]
    pub fn view(&self) -> ArrayView<'_> {
        ArrayView::from_shape_slice(self.shape.clone(), self.data.as_ref())
    }

    /// Mutable view of the whole array.
    #[inline]
    pub fn view_mut(&mut self) -> ArrayViewMut<'_> {
        let shape = self.shape.clone();
        ArrayViewMut::from_shape_slice(shape, self.as_mut_slice())
    }

    /// Deep copy of shape and values (unique buffer).
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
        Ok(Self {
            shape,
            data: Arc::new(data),
        })
    }

    /// Assemble an array when shape and length are already known to match.
    #[inline]
    pub(crate) fn from_parts(shape: Shape, data: Vec<f64>) -> Self {
        debug_assert_eq!(data.len(), shape.numel());
        Self {
            shape,
            data: Arc::new(data),
        }
    }

    /// Build from shape and a flat row-major slice (copies).
    pub fn from_shape_slice(shape: impl Into<Vec<usize>>, data: &[f64]) -> Result<Self> {
        Self::from_shape_vec(shape, data.to_vec())
    }

    /// Zeros with the given shape.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        let n = shape.numel();
        Ok(Self::from_parts(shape, pool::take_zeroed(n)))
    }

    /// Ones with the given shape.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        let n = shape.numel();
        Ok(Self::from_parts(shape, pool::take_filled(n, 1.0)))
    }

    /// Fill every element with `value`.
    pub fn full(shape: impl Into<Vec<usize>>, value: f64) -> Result<Self> {
        let shape = Shape::new(shape)?;
        let n = shape.numel();
        Ok(Self::from_parts(shape, pool::take_filled(n, value)))
    }

    /// Rank-1 range `[start, stop)` with step `1.0` (NumPy-style, exclusive stop).
    pub fn arange(start: f64, stop: f64) -> Result<Self> {
        Self::arange_step(start, stop, 1.0)
    }

    /// Rank-1 range `[start, stop)` with the given step.
    pub fn arange_step(start: f64, stop: f64, step: f64) -> Result<Self> {
        if step == 0.0 {
            return Err(Error::Shape("arange step must be non-zero".into()));
        }
        if (step > 0.0 && start >= stop) || (step < 0.0 && start <= stop) {
            return Ok(Self::from_parts(Shape::from_len(0), Vec::new()));
        }
        let n_f = ((stop - start) / step).ceil();
        if !n_f.is_finite() || n_f < 0.0 {
            return Err(Error::Shape("arange produced non-finite length".into()));
        }
        if n_f > usize::MAX as f64 {
            return Err(Error::Shape("arange length overflow".into()));
        }
        let n = n_f as usize;
        let mut data = pool::take_uninit(n);
        let mut x = start;
        let mut written = 0usize;
        for i in 0..n {
            if (step > 0.0 && x >= stop) || (step < 0.0 && x <= stop) {
                break;
            }
            data[i] = x;
            written = i + 1;
            x += step;
        }
        data.truncate(written);
        Ok(Self::from_parts(Shape::from_len(data.len()), data))
    }

    /// Read one element at a multi-index (0-based).
    pub fn get(&self, indices: &[usize]) -> Result<f64> {
        Ok(self.data[self.shape.offset(indices)?])
    }

    /// Write one element at a multi-index (0-based).
    pub fn set(&mut self, indices: &[usize], value: f64) -> Result<()> {
        let off = self.shape.offset(indices)?;
        self.as_mut_slice()[off] = value;
        Ok(())
    }

    /// Set every element to `value`.
    pub fn fill(&mut self, value: f64) {
        self.as_mut_slice().fill(value);
    }

    /// Reshape without changing values if the element count matches.
    ///
    /// **Zero-copy:** shares the underlying buffer (`Arc` clone). A later
    /// in-place mutation on either array copies the buffer on write
    /// (`Arc::make_mut`).
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
            data: Arc::clone(&self.data),
        })
    }

    /// Consume `self` and reshape without copying the value buffer.
    pub fn into_reshape(mut self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        if shape.numel() != self.len() {
            return Err(Error::Shape(format!(
                "cannot reshape array of {} elements into {}",
                self.len(),
                shape
            )));
        }
        // Move Arc out without running Drop recycle on the buffer.
        let data = std::mem::replace(&mut self.data, Arc::new(Vec::new()));
        Ok(Self { shape, data })
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

    /// Identity matrix of order `n` (row-major), diagonal written in one pass.
    pub fn eye(n: usize) -> Result<Self> {
        let shape = Shape::matrix(n, n)?;
        let mut data = pool::take_zeroed(shape.numel());
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Ok(Self::from_parts(shape, data))
    }

    /// Sum of all elements.
    pub fn sum(&self) -> f64 {
        kernels::sum_slice(self.as_slice())
    }

    /// Mean of all elements.
    pub fn mean(&self) -> Result<f64> {
        if self.is_empty() {
            return Err(Error::Shape("mean of empty array".into()));
        }
        Ok(self.sum() / self.len() as f64)
    }

    /// Minimum element.
    pub fn min(&self) -> Result<f64> {
        kernels::min_slice(self.as_slice())
            .ok_or_else(|| Error::Shape("min of empty array".into()))
    }

    /// Maximum element.
    pub fn max(&self) -> Result<f64> {
        kernels::max_slice(self.as_slice())
            .ok_or_else(|| Error::Shape("max of empty array".into()))
    }

    /// Export as a non-null Arrow [`Float64Array`] (flat row-major values).
    pub fn to_arrow(&self) -> Float64Array {
        Float64Array::from(self.as_slice().to_vec())
    }

    /// Import from a non-null Arrow [`Float64Array`] with an explicit shape.
    pub fn from_arrow(array: &Float64Array, shape: impl Into<Vec<usize>>) -> Result<Self> {
        if array.null_count() != 0 {
            return Err(Error::Shape("from_arrow does not accept nulls".into()));
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
        let data: Vec<f64> = array.values().iter().copied().collect();
        Ok(Self::from_parts(shape, data))
    }

    /// Copy values into a new Arrow [`Buffer`].
    pub fn to_buffer(&self) -> Buffer {
        Buffer::from_vec(self.as_slice().to_vec())
    }

    fn same_shape(&self, other: &Array) -> Result<()> {
        if !self.shape.same_as(other.shape()) {
            return Err(Error::Shape(format!(
                "shape mismatch: {} vs {}",
                self.shape, other.shape
            )));
        }
        Ok(())
    }

    fn owned_from_kernel<F>(&self, other: &Array, f: F) -> Result<Array>
    where
        F: FnOnce(&[f64], &[f64], &mut [f64]),
    {
        self.same_shape(other)?;
        let n = self.len();
        let mut data = pool::take_uninit(n);
        f(self.as_slice(), other.as_slice(), &mut data);
        Ok(Self::from_parts(self.shape.clone(), data))
    }

    fn owned_unary_kernel<F>(&self, f: F) -> Array
    where
        F: FnOnce(&[f64], &mut [f64]),
    {
        let n = self.len();
        let mut data = pool::take_uninit(n);
        f(self.as_slice(), &mut data);
        Self::from_parts(self.shape.clone(), data)
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::add_slices)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::sub_slices)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::mul_slices)
    }

    /// Element-wise division.
    pub fn div(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::div_slices)
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Array {
        self.owned_unary_kernel(kernels::neg_slice)
    }

    /// In-place element-wise addition.
    pub fn add_assign_arr(&mut self, other: &Array) -> Result<()> {
        self.same_shape(other)?;
        kernels::add_assign_slices(self.as_mut_slice(), other.as_slice());
        Ok(())
    }

    /// In-place element-wise subtraction.
    pub fn sub_assign_arr(&mut self, other: &Array) -> Result<()> {
        self.same_shape(other)?;
        kernels::sub_assign_slices(self.as_mut_slice(), other.as_slice());
        Ok(())
    }

    /// In-place element-wise multiplication.
    pub fn mul_assign_arr(&mut self, other: &Array) -> Result<()> {
        self.same_shape(other)?;
        kernels::mul_assign_slices(self.as_mut_slice(), other.as_slice());
        Ok(())
    }

    /// In-place element-wise division.
    pub fn div_assign_arr(&mut self, other: &Array) -> Result<()> {
        self.same_shape(other)?;
        kernels::div_assign_slices(self.as_mut_slice(), other.as_slice());
        Ok(())
    }

    pub(crate) fn add_scalar(&self, scalar: f64) -> Array {
        self.owned_unary_kernel(|a, out| kernels::add_scalar(a, scalar, out))
    }

    pub(crate) fn sub_scalar(&self, scalar: f64) -> Array {
        self.owned_unary_kernel(|a, out| kernels::sub_scalar(a, scalar, out))
    }

    pub(crate) fn scalar_sub(&self, scalar: f64) -> Array {
        self.owned_unary_kernel(|a, out| kernels::scalar_sub(a, scalar, out))
    }

    pub(crate) fn mul_scalar(&self, scalar: f64) -> Array {
        self.owned_unary_kernel(|a, out| kernels::mul_scalar(a, scalar, out))
    }

    pub(crate) fn div_scalar(&self, scalar: f64) -> Array {
        self.owned_unary_kernel(|a, out| kernels::div_scalar(a, scalar, out))
    }

    pub(crate) fn scalar_div(&self, scalar: f64) -> Array {
        self.owned_unary_kernel(|a, out| kernels::scalar_div(a, scalar, out))
    }
}

impl PartialEq for Array {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self
                .as_slice()
                .iter()
                .zip(other.as_slice().iter())
                .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        // Recycle only uniquely owned buffers (not reshape-shared Arcs).
        if let Some(inner) = Arc::get_mut(&mut self.data) {
            let buf = std::mem::take(inner);
            pool::recycle(buf);
        }
    }
}

#[allow(dead_code)]
pub(crate) fn check_shape(dims: &[usize]) -> Result<usize> {
    numel_checked(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{ArrayView, ArrayViewMut};

    #[test]
    fn eye_and_into_reshape() {
        let e = Array::eye(3).unwrap();
        assert_eq!(e.dims(), &[3, 3]);
        assert_eq!(e.as_slice(), &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
        let v = Array::arange(0.0, 6.0).unwrap();
        let m = v.into_reshape(vec![2, 3]).unwrap();
        assert_eq!(m.dims(), &[2, 3]);
        assert_eq!(m.as_slice(), &[0., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn reshape_shares_buffer_until_write() {
        let a = Array::arange(0.0, 6.0).unwrap();
        let b = a.reshape(vec![2, 3]).unwrap();
        assert_eq!(b.dims(), &[2, 3]);
        assert_eq!(a.as_slice().as_ptr(), b.as_slice().as_ptr());
        let mut c = b.reshape(vec![3, 2]).unwrap();
        assert_eq!(a.as_slice().as_ptr(), c.as_slice().as_ptr());
        c.set(&[0, 0], 99.0).unwrap();
        assert_eq!(a.get(&[0]).unwrap(), 0.0);
        assert_eq!(c.get(&[0, 0]).unwrap(), 99.0);
        assert_ne!(a.as_slice().as_ptr(), c.as_slice().as_ptr());
    }

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
