//! Owned dense `f64` arrays (row-major, n-D).

use std::sync::Arc;

use arrow_array::{Array as ArrowArray, Float64Array};

use super::kernels;
use super::pool;
use super::shape::{broadcast_shapes, Shape};
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
        assert_eq!(
            data.len(),
            shape.numel(),
            "from_parts: data length {} != shape numel {}",
            data.len(),
            shape.numel()
        );
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
        if self.shape.same_as(other.shape()) {
            let n = self.len();
            let mut data = pool::take_uninit(n);
            f(self.as_slice(), other.as_slice(), &mut data);
            return Ok(Self::from_parts(self.shape.clone(), data));
        }
        let out_dims = broadcast_shapes(self.dims(), other.dims())?;
        let out_shape = Shape::new(out_dims)?;
        let left = self.broadcast_to(out_shape.dims())?;
        let right = other.broadcast_to(out_shape.dims())?;
        let n = left.len();
        let mut data = pool::take_uninit(n);
        f(left.as_slice(), right.as_slice(), &mut data);
        Ok(Self::from_parts(out_shape, data))
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

    /// Element-wise absolute value (IEEE).
    pub fn abs(&self) -> Array {
        self.owned_unary_kernel(kernels::abs_slice)
    }

    /// Element-wise square root (IEEE; domain errors → NaN).
    pub fn sqrt(&self) -> Array {
        self.owned_unary_kernel(kernels::sqrt_slice)
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> Array {
        self.owned_unary_kernel(kernels::exp_slice)
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Array {
        self.owned_unary_kernel(kernels::log_slice)
    }

    /// Element-wise `ln(1 + x)`.
    pub fn log1p(&self) -> Array {
        self.owned_unary_kernel(kernels::log1p_slice)
    }

    /// Element-wise sign: −1, 0, +1 (NaN → NaN).
    pub fn sign(&self) -> Array {
        self.owned_unary_kernel(kernels::sign_slice)
    }

    /// Element-wise power `self ** other` (same shape).
    pub fn power(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::power_slices)
    }

    /// Element-wise power with a scalar exponent.
    pub fn power_scalar(&self, p: f64) -> Array {
        self.owned_unary_kernel(|a, out| kernels::power_scalar(a, p, out))
    }

    /// Clip values to `[lo, hi]` (NaNs unchanged if outside compare fails).
    pub fn clip(&self, lo: f64, hi: f64) -> Result<Array> {
        if lo > hi {
            return Err(Error::shape(format!("clip lo {lo} > hi {hi}")));
        }
        Ok(self.owned_unary_kernel(|a, out| kernels::clip_slice(a, lo, hi, out)))
    }

    /// Dense mask: 1.0 where NaN, else 0.0.
    pub fn isnan(&self) -> Array {
        self.owned_unary_kernel(kernels::isnan_slice)
    }

    /// Dense mask: 1.0 where finite, else 0.0.
    pub fn isfinite(&self) -> Array {
        self.owned_unary_kernel(kernels::isfinite_slice)
    }

    /// `where(cond, x, y)` — nonzero finite `cond` selects `x`, else `y` (same shape).
    pub fn where_cond(cond: &Array, x: &Array, y: &Array) -> Result<Array> {
        cond.same_shape(x)?;
        cond.same_shape(y)?;
        let n = cond.len();
        let mut data = pool::take_uninit(n);
        kernels::where_slices(cond.as_slice(), x.as_slice(), y.as_slice(), &mut data);
        Ok(Self::from_parts(cond.shape.clone(), data))
    }

    /// Inclusive cumulative sum along the flat row-major buffer.
    pub fn cumsum(&self) -> Array {
        self.owned_unary_kernel(kernels::cumsum_slice)
    }

    /// Index of minimum in flat order (0-based). Empty → error.
    pub fn argmin(&self) -> Result<usize> {
        kernels::argmin_slice(self.as_slice()).ok_or_else(|| Error::shape("argmin of empty array"))
    }

    /// Index of maximum in flat order (0-based). Empty → error.
    pub fn argmax(&self) -> Result<usize> {
        kernels::argmax_slice(self.as_slice()).ok_or_else(|| Error::shape("argmax of empty array"))
    }

    /// Variance; `ddof` matches NumPy (`0` population, `1` sample).
    pub fn var(&self, ddof: usize) -> Result<f64> {
        kernels::var_slice(self.as_slice(), ddof).ok_or_else(|| {
            Error::shape(format!(
                "var requires len > ddof (len={}, ddof={ddof})",
                self.len()
            ))
        })
    }

    /// Standard deviation; `ddof` matches NumPy.
    pub fn std(&self, ddof: usize) -> Result<f64> {
        Ok(self.var(ddof)?.sqrt())
    }

    /// Concatenate arrays along `axis` (0 or 1 for rank ≤ 2).
    ///
    /// All inputs must share the same rank and matching sizes on non-`axis` dims.
    pub fn concatenate(axis: usize, parts: &[&Array]) -> Result<Array> {
        if parts.is_empty() {
            return Err(Error::shape("concatenate needs at least one array"));
        }
        let rank = parts[0].rank();
        if rank == 0 || rank > 2 {
            return Err(Error::shape("concatenate supports rank 1 or 2"));
        }
        if axis >= rank {
            return Err(Error::shape(format!("axis {axis} out of range for rank {rank}")));
        }
        for p in parts {
            if p.rank() != rank {
                return Err(Error::shape("concatenate: rank mismatch"));
            }
            for d in 0..rank {
                if d != axis && p.dims()[d] != parts[0].dims()[d] {
                    return Err(Error::shape("concatenate: shape mismatch on non-axis"));
                }
            }
        }
        let mut out_dims = parts[0].dims().to_vec();
        out_dims[axis] = parts.iter().map(|p| p.dims()[axis]).sum();
        let shape = Shape::new(out_dims)?;
        let mut data = pool::take_uninit(shape.numel());
        if rank == 1 {
            let mut off = 0;
            for p in parts {
                let n = p.len();
                data[off..off + n].copy_from_slice(p.as_slice());
                off += n;
            }
        } else {
            // rank 2
            let cols_out = shape.dims()[1];
            if axis == 0 {
                let mut row = 0usize;
                for p in parts {
                    let pr = p.dims()[0];
                    let pc = p.dims()[1];
                    for i in 0..pr {
                        let src = &p.as_slice()[i * pc..(i + 1) * pc];
                        let dst = &mut data[(row + i) * cols_out..(row + i) * cols_out + pc];
                        dst.copy_from_slice(src);
                    }
                    row += pr;
                }
            } else {
                // axis 1
                let rows = shape.dims()[0];
                for i in 0..rows {
                    let mut col = 0usize;
                    for p in parts {
                        let pc = p.dims()[1];
                        let src = &p.as_slice()[i * pc..(i + 1) * pc];
                        data[i * cols_out + col..i * cols_out + col + pc].copy_from_slice(src);
                        col += pc;
                    }
                }
            }
        }
        Ok(Self::from_parts(shape, data))
    }

    /// Stack arrays along a new axis (0 or 1); inputs must be same shape, rank 1.
    ///
    /// Rank-1 inputs → rank-2. `axis=0` stacks as rows; `axis=1` as columns.
    pub fn stack(axis: usize, parts: &[&Array]) -> Result<Array> {
        if parts.is_empty() {
            return Err(Error::shape("stack needs at least one array"));
        }
        if !parts.iter().all(|p| p.rank() == 1) {
            return Err(Error::shape("stack currently supports rank-1 inputs only"));
        }
        let n = parts[0].len();
        if parts.iter().any(|p| p.len() != n) {
            return Err(Error::shape("stack: length mismatch"));
        }
        let k = parts.len();
        match axis {
            0 => {
                // k × n
                let mut data = pool::take_uninit(k * n);
                for (i, p) in parts.iter().enumerate() {
                    data[i * n..(i + 1) * n].copy_from_slice(p.as_slice());
                }
                Ok(Self::from_parts(Shape::matrix(k, n)?, data))
            }
            1 => {
                // n × k
                let mut data = pool::take_uninit(n * k);
                for i in 0..n {
                    for (j, p) in parts.iter().enumerate() {
                        data[i * k + j] = p.as_slice()[i];
                    }
                }
                Ok(Self::from_parts(Shape::matrix(n, k)?, data))
            }
            _ => Err(Error::shape("stack axis must be 0 or 1")),
        }
    }

    /// Broadcast this array to `dims` (NumPy right-align rules).
    ///
    /// Same shape returns a shared clone. Expanding size-1 axes allocates and tiles.
    pub fn broadcast_to(&self, dims: impl Into<Vec<usize>>) -> Result<Array> {
        let target = Shape::new(dims)?;
        if self.shape.same_as(&target) {
            return Ok(self.clone());
        }
        let tr = target.rank();
        let sr = self.rank();
        if sr > tr {
            return Err(Error::Shape(format!(
                "cannot broadcast rank {sr} down to rank {tr}"
            )));
        }
        // Pad source dims on the left with 1s to rank `tr`
        let mut sdims = vec![1usize; tr];
        for i in 0..sr {
            sdims[tr - sr + i] = self.dims()[i];
        }
        let tdims = target.dims();
        for ax in 0..tr {
            if sdims[ax] != tdims[ax] && sdims[ax] != 1 {
                return Err(Error::Shape(format!(
                    "cannot broadcast {} to {}",
                    self.shape, target
                )));
            }
        }

        let n = target.numel();
        let mut data = pool::take_uninit(n);
        let src = self.as_slice();
        let mut idx = vec![0usize; tr];
        for flat in 0..n {
            let mut soff = 0usize;
            let mut stride = 1usize;
            for ax in (0..tr).rev() {
                let sc = if sdims[ax] == 1 { 0 } else { idx[ax] };
                soff += sc * stride;
                stride *= sdims[ax];
            }
            data[flat] = src[soff];
            for ax in (0..tr).rev() {
                idx[ax] += 1;
                if idx[ax] < tdims[ax] {
                    break;
                }
                idx[ax] = 0;
            }
        }
        Ok(Self::from_parts(target, data))
    }

    /// Element-wise `==` as 0/1 mask (broadcasts).
    pub fn eq_elem(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::eq_slices)
    }
    /// Element-wise `!=` as 0/1 mask.
    pub fn ne_elem(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::ne_slices)
    }
    /// Element-wise `<` as 0/1 mask.
    pub fn lt(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::lt_slices)
    }
    /// Element-wise `<=` as 0/1 mask.
    pub fn le(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::le_slices)
    }
    /// Element-wise `>` as 0/1 mask.
    pub fn gt(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::gt_slices)
    }
    /// Element-wise `>=` as 0/1 mask.
    pub fn ge(&self, other: &Array) -> Result<Array> {
        self.owned_from_kernel(other, kernels::ge_slices)
    }

    /// Compare every element to a scalar; 0/1 mask.
    pub fn eq_scalar_elem(&self, s: f64) -> Array {
        self.owned_unary_kernel(|a, o| kernels::eq_scalar(a, s, o))
    }
    /// `!=` scalar mask.
    pub fn ne_scalar_elem(&self, s: f64) -> Array {
        self.owned_unary_kernel(|a, o| kernels::ne_scalar(a, s, o))
    }
    /// `<` scalar mask.
    pub fn lt_scalar(&self, s: f64) -> Array {
        self.owned_unary_kernel(|a, o| kernels::lt_scalar(a, s, o))
    }
    /// `<=` scalar mask.
    pub fn le_scalar(&self, s: f64) -> Array {
        self.owned_unary_kernel(|a, o| kernels::le_scalar(a, s, o))
    }
    /// `>` scalar mask.
    pub fn gt_scalar(&self, s: f64) -> Array {
        self.owned_unary_kernel(|a, o| kernels::gt_scalar(a, s, o))
    }
    /// `>=` scalar mask.
    pub fn ge_scalar(&self, s: f64) -> Array {
        self.owned_unary_kernel(|a, o| kernels::ge_scalar(a, s, o))
    }

    /// Sum ignoring NaN.
    pub fn nansum(&self) -> f64 {
        kernels::nansum_slice(self.as_slice())
    }
    /// Mean ignoring NaN (error if all NaN / empty count).
    pub fn nanmean(&self) -> Result<f64> {
        kernels::nanmean_slice(self.as_slice())
            .ok_or_else(|| Error::Shape("nanmean of empty/all-NaN array".into()))
    }
    /// Min ignoring NaN.
    pub fn nanmin(&self) -> Result<f64> {
        kernels::nanmin_slice(self.as_slice())
            .ok_or_else(|| Error::Shape("nanmin of empty/all-NaN array".into()))
    }
    /// Max ignoring NaN.
    pub fn nanmax(&self) -> Result<f64> {
        kernels::nanmax_slice(self.as_slice())
            .ok_or_else(|| Error::Shape("nanmax of empty/all-NaN array".into()))
    }
    /// Variance ignoring NaN; `ddof` as in NumPy.
    pub fn nanvar(&self, ddof: usize) -> Result<f64> {
        kernels::nanvar_slice(self.as_slice(), ddof).ok_or_else(|| {
            Error::Shape(format!("nanvar requires enough non-NaN points (ddof={ddof})"))
        })
    }
    /// Std-dev ignoring NaN.
    pub fn nanstd(&self, ddof: usize) -> Result<f64> {
        Ok(self.nanvar(ddof)?.sqrt())
    }

    /// Rank-1 half-open slice `[start, end)` (0-based) as a view.
    pub fn slice(&self, start: usize, end: usize) -> Result<ArrayView<'_>> {
        if self.rank() != 1 {
            return Err(Error::Shape(
                "slice requires rank-1; use rows/row for matrices".into(),
            ));
        }
        if start > end || end > self.len() {
            return Err(Error::Index(format!(
                "slice [{start}, {end}) out of range for len {}",
                self.len()
            )));
        }
        Ok(ArrayView::from_shape_slice(
            Shape::from_len(end - start),
            &self.as_slice()[start..end],
        ))
    }

    /// Contiguous row range `[start, end)` as a rank-2 view (0-based).
    pub fn rows(&self, start: usize, end: usize) -> Result<ArrayView<'_>> {
        if self.rank() != 2 {
            return Err(Error::Shape("rows requires rank-2".into()));
        }
        let (m, n) = (self.dims()[0], self.dims()[1]);
        if start > end || end > m {
            return Err(Error::Index(format!(
                "rows [{start}, {end}) out of range for {m} rows"
            )));
        }
        let off = start * n;
        let len = (end - start) * n;
        Ok(ArrayView::from_shape_slice(
            Shape::matrix(end - start, n)?,
            &self.as_slice()[off..off + len],
        ))
    }

    /// Single row `i` as rank-1 view (0-based).
    pub fn row(&self, i: usize) -> Result<ArrayView<'_>> {
        if self.rank() != 2 {
            return Err(Error::Shape("row requires rank-2".into()));
        }
        let (m, n) = (self.dims()[0], self.dims()[1]);
        if i >= m {
            return Err(Error::Index(format!("row {i} out of range for {m} rows")));
        }
        Ok(ArrayView::from_shape_slice(
            Shape::from_len(n),
            &self.as_slice()[i * n..(i + 1) * n],
        ))
    }

    /// Column `j` as **owned** rank-1 (copy; columns are not contiguous in row-major).
    pub fn col(&self, j: usize) -> Result<Array> {
        if self.rank() != 2 {
            return Err(Error::Shape("col requires rank-2".into()));
        }
        let (m, n) = (self.dims()[0], self.dims()[1]);
        if j >= n {
            return Err(Error::Index(format!("col {j} out of range for {n} cols")));
        }
        let mut data = pool::take_uninit(m);
        let src = self.as_slice();
        for i in 0..m {
            data[i] = src[i * n + j];
        }
        Ok(Self::from_parts(Shape::from_len(m), data))
    }

    // --- M6: axis reductions, quant helpers, argsort/take, any/all ---

    fn rank2_dims(&self) -> Result<(usize, usize)> {
        if self.rank() != 2 {
            return Err(Error::Shape("axis reductions require rank-2".into()));
        }
        Ok((self.dims()[0], self.dims()[1]))
    }

    /// Sum along `axis` (0 or 1) for rank-2; result is rank-1.
    pub fn sum_axis(&self, axis: usize) -> Result<Array> {
        let (m, n) = self.rank2_dims()?;
        match axis {
            0 => {
                let mut out = pool::take_uninit(n);
                kernels::axis0_sum(m, n, self.as_slice(), &mut out);
                Ok(Self::from_parts(Shape::from_len(n), out))
            }
            1 => {
                let mut out = pool::take_uninit(m);
                kernels::axis1_sum(m, n, self.as_slice(), &mut out);
                Ok(Self::from_parts(Shape::from_len(m), out))
            }
            _ => Err(Error::Shape(format!("axis {axis} out of range for rank-2"))),
        }
    }

    /// Mean along `axis` for rank-2.
    pub fn mean_axis(&self, axis: usize) -> Result<Array> {
        let (m, n) = self.rank2_dims()?;
        let s = self.sum_axis(axis)?;
        let denom = match axis {
            0 => m as f64,
            1 => n as f64,
            _ => unreachable!(),
        };
        if denom == 0.0 {
            return Err(Error::Shape("mean_axis of empty dimension".into()));
        }
        Ok(s.mul_scalar(1.0 / denom))
    }

    /// Min along `axis` for rank-2.
    pub fn min_axis(&self, axis: usize) -> Result<Array> {
        let (m, n) = self.rank2_dims()?;
        if m == 0 || n == 0 {
            return Err(Error::Shape("min_axis of empty dimension".into()));
        }
        match axis {
            0 => {
                let mut out = pool::take_uninit(n);
                kernels::axis0_min(m, n, self.as_slice(), &mut out);
                Ok(Self::from_parts(Shape::from_len(n), out))
            }
            1 => {
                let mut out = pool::take_uninit(m);
                kernels::axis1_min(m, n, self.as_slice(), &mut out);
                Ok(Self::from_parts(Shape::from_len(m), out))
            }
            _ => Err(Error::Shape(format!("axis {axis} out of range for rank-2"))),
        }
    }

    /// Max along `axis` for rank-2.
    pub fn max_axis(&self, axis: usize) -> Result<Array> {
        let (m, n) = self.rank2_dims()?;
        if m == 0 || n == 0 {
            return Err(Error::Shape("max_axis of empty dimension".into()));
        }
        match axis {
            0 => {
                let mut out = pool::take_uninit(n);
                kernels::axis0_max(m, n, self.as_slice(), &mut out);
                Ok(Self::from_parts(Shape::from_len(n), out))
            }
            1 => {
                let mut out = pool::take_uninit(m);
                kernels::axis1_max(m, n, self.as_slice(), &mut out);
                Ok(Self::from_parts(Shape::from_len(m), out))
            }
            _ => Err(Error::Shape(format!("axis {axis} out of range for rank-2"))),
        }
    }

    /// Variance along `axis` for rank-2 (`ddof` as NumPy).
    pub fn var_axis(&self, axis: usize, ddof: usize) -> Result<Array> {
        let (m, n) = self.rank2_dims()?;
        let mean = self.mean_axis(axis)?;
        let count = match axis {
            0 => m,
            1 => n,
            _ => return Err(Error::Shape(format!("axis {axis} out of range for rank-2"))),
        };
        if count <= ddof {
            return Err(Error::Shape(format!(
                "var_axis requires size > ddof (size={count}, ddof={ddof})"
            )));
        }
        let mut out = pool::take_uninit(mean.len());
        let src = self.as_slice();
        let mu = mean.as_slice();
        match axis {
            0 => {
                for j in 0..n {
                    let mut ss = 0.0;
                    for i in 0..m {
                        let d = src[i * n + j] - mu[j];
                        ss += d * d;
                    }
                    out[j] = ss / (count - ddof) as f64;
                }
            }
            1 => {
                for i in 0..m {
                    let mut ss = 0.0;
                    for j in 0..n {
                        let d = src[i * n + j] - mu[i];
                        ss += d * d;
                    }
                    out[i] = ss / (count - ddof) as f64;
                }
            }
            _ => unreachable!(),
        }
        Ok(Self::from_parts(Shape::from_len(mean.len()), out))
    }

    /// Std-dev along `axis` for rank-2.
    pub fn std_axis(&self, axis: usize, ddof: usize) -> Result<Array> {
        let v = self.var_axis(axis, ddof)?;
        Ok(v.sqrt())
    }

    /// True if any element is nonzero and non-NaN.
    pub fn any(&self) -> bool {
        kernels::any_slice(self.as_slice())
    }

    /// True if every element is nonzero and non-NaN (empty → false).
    pub fn all(&self) -> bool {
        kernels::all_slice(self.as_slice())
    }

    /// `any` along axis for rank-2 → 0/1 rank-1 mask.
    pub fn any_axis(&self, axis: usize) -> Result<Array> {
        let (m, n) = self.rank2_dims()?;
        match axis {
            0 => {
                let mut out = pool::take_uninit(n);
                kernels::axis0_any(m, n, self.as_slice(), &mut out);
                Ok(Self::from_parts(Shape::from_len(n), out))
            }
            1 => {
                let mut out = pool::take_uninit(m);
                kernels::axis1_any(m, n, self.as_slice(), &mut out);
                Ok(Self::from_parts(Shape::from_len(m), out))
            }
            _ => Err(Error::Shape(format!("axis {axis} out of range for rank-2"))),
        }
    }

    /// `all` along axis for rank-2 → 0/1 rank-1 mask.
    pub fn all_axis(&self, axis: usize) -> Result<Array> {
        let (m, n) = self.rank2_dims()?;
        match axis {
            0 => {
                let mut out = pool::take_uninit(n);
                kernels::axis0_all(m, n, self.as_slice(), &mut out);
                Ok(Self::from_parts(Shape::from_len(n), out))
            }
            1 => {
                let mut out = pool::take_uninit(m);
                kernels::axis1_all(m, n, self.as_slice(), &mut out);
                Ok(Self::from_parts(Shape::from_len(m), out))
            }
            _ => Err(Error::Shape(format!("axis {axis} out of range for rank-2"))),
        }
    }

    /// Argsort: 0-based indices as rank-1 `f64` (dense storage has no integer dtype).
    pub fn argsort(&self, descending: bool) -> Result<Array> {
        if self.rank() != 1 {
            return Err(Error::Shape("argsort requires rank-1".into()));
        }
        let n = self.len();
        let mut idx = vec![0usize; n];
        kernels::argsort_indices(self.as_slice(), descending, &mut idx);
        let mut data = pool::take_uninit(n);
        for i in 0..n {
            data[i] = idx[i] as f64;
        }
        Ok(Self::from_parts(Shape::from_len(n), data))
    }

    /// Gather elements of a rank-1 array by 0-based indices (values truncated toward zero).
    pub fn take(&self, indices: &Array) -> Result<Array> {
        if self.rank() != 1 {
            return Err(Error::Shape("take requires rank-1 source".into()));
        }
        if indices.rank() != 1 {
            return Err(Error::Shape("take indices must be rank-1".into()));
        }
        let src = self.as_slice();
        let n = src.len();
        let mut data = pool::take_uninit(indices.len());
        for (k, &raw) in indices.as_slice().iter().enumerate() {
            if raw.is_nan() || raw < 0.0 {
                return Err(Error::Index(format!("take index {raw} invalid")));
            }
            let i = raw as usize;
            if i >= n {
                return Err(Error::Index(format!("take index {i} out of range for len {n}")));
            }
            data[k] = src[i];
        }
        Ok(Self::from_parts(Shape::from_len(indices.len()), data))
    }

    /// Outer product of two rank-1 arrays → rank-2 `(len(a), len(b))`.
    pub fn outer(a: &Array, b: &Array) -> Result<Array> {
        if a.rank() != 1 || b.rank() != 1 {
            return Err(Error::Shape("outer requires two rank-1 arrays".into()));
        }
        let m = a.len();
        let n = b.len();
        let mut data = pool::take_uninit(m * n);
        let av = a.as_slice();
        let bv = b.as_slice();
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = av[i] * bv[j];
            }
        }
        Ok(Self::from_parts(Shape::matrix(m, n)?, data))
    }

    /// NumPy-style `diag`: vector → square matrix, or matrix → main diagonal vector.
    pub fn diag(a: &Array) -> Result<Array> {
        match a.rank() {
            1 => {
                let n = a.len();
                let mut data = pool::take_uninit(n * n);
                data.fill(0.0);
                let v = a.as_slice();
                for i in 0..n {
                    data[i * n + i] = v[i];
                }
                Ok(Self::from_parts(Shape::matrix(n, n)?, data))
            }
            2 => {
                let (m, n) = (a.dims()[0], a.dims()[1]);
                let k = m.min(n);
                let mut data = pool::take_uninit(k);
                let src = a.as_slice();
                for i in 0..k {
                    data[i] = src[i * n + i];
                }
                Ok(Self::from_parts(Shape::from_len(k), data))
            }
            _ => Err(Error::Shape("diag requires rank 1 or 2".into())),
        }
    }

    /// Main diagonal of a rank-2 matrix (alias of vector side of [`diag`]).
    pub fn diagonal(&self) -> Result<Array> {
        if self.rank() != 2 {
            return Err(Error::Shape("diagonal requires rank-2".into()));
        }
        Self::diag(self)
    }

    /// Trace of a square rank-2 matrix.
    pub fn trace(&self) -> Result<f64> {
        if self.rank() != 2 {
            return Err(Error::Shape("trace requires rank-2".into()));
        }
        let (m, n) = (self.dims()[0], self.dims()[1]);
        if m != n {
            return Err(Error::Shape("trace requires a square matrix".into()));
        }
        let src = self.as_slice();
        let mut s = 0.0;
        for i in 0..m {
            s += src[i * n + i];
        }
        Ok(s)
    }

    /// Sample/population covariance. **Variables in rows** (NumPy `rowvar=True`).
    ///
    /// `a` is `d × n` (d variables, n observations). Returns `d × d`. Default use `ddof=1`.
    pub fn cov(a: &Array, ddof: usize) -> Result<Array> {
        let x = match a.rank() {
            1 => a.reshape(vec![1, a.len()])?,
            2 => a.clone(),
            _ => return Err(Error::Shape("cov requires rank 1 or 2".into())),
        };
        let (d, n) = (x.dims()[0], x.dims()[1]);
        if n <= ddof {
            return Err(Error::Shape(format!(
                "cov requires n > ddof (n={n}, ddof={ddof})"
            )));
        }
        // Center rows
        let means = x.mean_axis(1)?; // length d
        let mu = means.as_slice();
        let src = x.as_slice();
        let mut centered = pool::take_uninit(d * n);
        for i in 0..d {
            for j in 0..n {
                centered[i * n + j] = src[i * n + j] - mu[i];
            }
        }
        let xc = Self::from_parts(Shape::matrix(d, n)?, centered);
        // (1/(n-ddof)) * X @ Xᵀ  via matmul_at(X, X) = XᵀX wrong; need X Xᵀ
        // matmul(X, Xᵀ): X is d×n, Xᵀ is n×d → d×d
        let xt = crate::linalg::transpose(&xc)?;
        let g = crate::linalg::matmul(&xc, &xt)?;
        let scale = 1.0 / (n - ddof) as f64;
        Ok(g.mul_scalar(scale))
    }

    /// Pearson correlation matrix; variables in rows (NumPy `rowvar=True`).
    pub fn corrcoef(a: &Array) -> Result<Array> {
        let c = Self::cov(a, 1)?;
        let d = c.dims()[0];
        let src = c.as_slice();
        let mut out = pool::take_uninit(d * d);
        for i in 0..d {
            let sii = src[i * d + i].sqrt();
            for j in 0..d {
                let sjj = src[j * d + j].sqrt();
                let denom = sii * sjj;
                out[i * d + j] = if denom == 0.0 || denom.is_nan() {
                    f64::NAN
                } else {
                    src[i * d + j] / denom
                };
            }
        }
        Ok(Self::from_parts(Shape::matrix(d, d)?, out))
    }

}

impl PartialEq for Array {
    /// Structural equality: same shape and elementwise equal, with **NaN == NaN**
    /// (unlike IEEE floating compare). Useful for tests; not a math relation.
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


#[cfg(test)]
mod tests {
    #[test]
    fn m6_axis_and_cov() {
        let m = Array::from_shape_slice(vec![2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(m.sum_axis(0).unwrap().as_slice(), &[5., 7., 9.]);
        assert_eq!(m.sum_axis(1).unwrap().as_slice(), &[6., 15.]);
        assert_eq!(m.min_axis(1).unwrap().as_slice(), &[1., 4.]);
        let mask = Array::from_shape_slice(vec![2, 2], &[0., 1., 0., 0.]).unwrap();
        assert!(mask.any());
        assert!(!mask.all());
        assert_eq!(mask.any_axis(0).unwrap().as_slice(), &[0., 1.]);

        let v = Array::from_shape_slice(vec![4], &[3., 1., 4., 2.]).unwrap();
        let idx = v.argsort(false).unwrap();
        assert_eq!(idx.as_slice(), &[1., 3., 0., 2.]);
        let taken = v.take(&idx).unwrap();
        assert_eq!(taken.as_slice(), &[1., 2., 3., 4.]);

        let a = Array::from_shape_slice(vec![2], &[1., 2.]).unwrap();
        let b = Array::from_shape_slice(vec![3], &[3., 4., 5.]).unwrap();
        let o = Array::outer(&a, &b).unwrap();
        assert_eq!(o.dims(), &[2, 3]);
        assert_eq!(o.as_slice()[0], 3.0);
        assert_eq!(Array::diag(&a).unwrap().dims(), &[2, 2]);
        assert_eq!(o.diagonal().unwrap().as_slice()[0], 3.0);
        assert!((Array::eye(2).unwrap().trace().unwrap() - 2.0).abs() < 1e-12);

        // cov: 2 vars, 3 obs
        let x = Array::from_shape_slice(vec![2, 3], &[1., 2., 3., 2., 4., 6.]).unwrap();
        let c = Array::cov(&x, 1).unwrap();
        assert_eq!(c.dims(), &[2, 2]);
        // var of [1,2,3] sample = 1, var of [2,4,6] = 4, cov = 2
        assert!((c.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-9);
        assert!((c.get(&[1, 1]).unwrap() - 4.0).abs() < 1e-9);
        assert!((c.get(&[0, 1]).unwrap() - 2.0).abs() < 1e-9);
        let r = Array::corrcoef(&x).unwrap();
        assert!((r.get(&[0, 1]).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn broadcast_and_compare() {
        let m = Array::from_shape_slice(vec![2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
        let row = Array::from_shape_slice(vec![3], &[10., 20., 30.]).unwrap();
        let s = m.add(&row).unwrap();
        assert_eq!(s.as_slice(), &[11., 22., 33., 14., 25., 36.]);
        let mask = m.lt_scalar(4.0);
        assert_eq!(mask.as_slice(), &[1., 1., 1., 0., 0., 0.]);
        let col = Array::from_shape_slice(vec![2, 1], &[1., 2.]).unwrap();
        let s2 = m.mul(&col).unwrap();
        assert_eq!(s2.as_slice(), &[1., 2., 3., 8., 10., 12.]);
    }

    #[test]
    fn nan_reductions_and_slices() {
        let a = Array::from_shape_slice(vec![4], &[1., f64::NAN, 3., 5.]).unwrap();
        assert!((a.nansum() - 9.0).abs() < 1e-12);
        assert!((a.nanmean().unwrap() - 3.0).abs() < 1e-12);
        assert_eq!(a.nanmin().unwrap(), 1.0);
        assert_eq!(a.nanmax().unwrap(), 5.0);
        let v = a.slice(1, 3).unwrap();
        assert_eq!(v.as_slice()[1], 3.0);
        let m = Array::from_shape_slice(vec![3, 2], &[1., 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(m.row(1).unwrap().as_slice(), &[3., 4.]);
        assert_eq!(m.col(0).unwrap().as_slice(), &[1., 3., 5.]);
        assert_eq!(m.rows(0, 2).unwrap().dims(), &[2, 2]);
    }

    #[test]
    fn ufuncs_and_var() {
        let a = Array::from_shape_slice(vec![3], &[-1., 4., 9.]).unwrap();
        assert_eq!(a.abs().as_slice(), &[1., 4., 9.]);
        assert!((a.sqrt().get(&[1]).unwrap() - 2.0).abs() < 1e-12);
        let b = Array::from_shape_slice(vec![3], &[1., 2., 3.]).unwrap();
        assert!((b.var(0).unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(b.argmin().unwrap(), 0);
        assert_eq!(b.argmax().unwrap(), 2);
        assert_eq!(b.cumsum().as_slice(), &[1., 3., 6.]);
        let c = Array::from_shape_slice(vec![2], &[1., f64::NAN]).unwrap();
        assert_eq!(c.isnan().as_slice(), &[0., 1.]);
    }


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
