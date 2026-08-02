//! Owned dense `i64` arrays (row-major, n-D). M7 — correctness-first.

use std::sync::Arc;

use arrow_array::{Array as ArrowArray, Int64Array};

use super::dtype::DType;
use super::kernels_i64 as kernels;
use super::pool_i64 as pool;
use super::shape::{broadcast_shapes, Shape};
use super::Array;
use crate::error::{Error, Result};

/// Cache-blocked out-of-place transpose: `dst` is `cols × rows` row-major.
fn blocked_transpose_i64(src: &[i64], rows: usize, cols: usize, dst: &mut [i64]) {
    const BS: usize = 16;
    for i0 in (0..rows).step_by(BS) {
        for j0 in (0..cols).step_by(BS) {
            let i1 = (i0 + BS).min(rows);
            let j1 = (j0 + BS).min(cols);
            for i in i0..i1 {
                for j in j0..j1 {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
        }
    }
}


fn gcd_i64(a: i64, b: i64) -> i64 {
    let mut a = a.unsigned_abs();
    let mut b = b.unsigned_abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    // Result fits in i64 except gcd(i64::MIN, i64::MIN) == 2^63; clamp.
    i64::try_from(a).unwrap_or(i64::MAX)
}

fn lcm_i64(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }
    let g = gcd_i64(a, b);
    // wrapping |a|/g*|b|
    let au = a.unsigned_abs();
    let bu = b.unsigned_abs();
    let gu = g.unsigned_abs();
    let prod = au / gu * bu;
    i64::try_from(prod).unwrap_or(i64::MAX)
}


#[derive(Clone, Copy)]
enum BroadcastOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BroadcastOp {
    fn apply_same(self, a: &[i64], b: &[i64], out: &mut [i64]) {
        match self {
            Self::Add => kernels::add_slices(a, b, out),
            Self::Sub => kernels::sub_slices(a, b, out),
            Self::Mul => kernels::mul_slices(a, b, out),
            Self::Div => kernels::div_slices(a, b, out),
        }
    }
    fn apply_row(self, m: usize, n: usize, a: &[i64], row: &[i64], out: &mut [i64]) {
        match self {
            Self::Add => kernels::add_matrix_row(m, n, a, row, out),
            Self::Sub => kernels::sub_matrix_row(m, n, a, row, out),
            Self::Mul => kernels::mul_matrix_row(m, n, a, row, out),
            Self::Div => kernels::div_matrix_row(m, n, a, row, out),
        }
    }
    fn apply_col(self, m: usize, n: usize, a: &[i64], col: &[i64], out: &mut [i64]) {
        match self {
            Self::Add => kernels::add_matrix_col(m, n, a, col, out),
            Self::Sub => kernels::sub_matrix_col(m, n, a, col, out),
            Self::Mul => kernels::mul_matrix_col(m, n, a, col, out),
            Self::Div => kernels::div_matrix_col(m, n, a, col, out),
        }
    }
}

/// Owned dense n-D array of `i64` values in row-major order.
///
/// M7 integer surface for keys and exact integer columns. Dense LA stays on
/// [`Array`] (`f64`). Arithmetic uses wrapping ops; integer division truncates
/// toward zero; division by zero yields `0` (no panic).
#[derive(Debug)]
pub struct ArrayI64 {
    shape: Shape,
    data: Arc<Vec<i64>>,
}

impl Drop for ArrayI64 {
    fn drop(&mut self) {
        if let Some(inner) = Arc::get_mut(&mut self.data) {
            let buf = std::mem::take(inner);
            pool::recycle(buf);
        }
    }
}

impl Clone for ArrayI64 {
    fn clone(&self) -> Self {
        let mut data = pool::take_uninit(self.len());
        data.copy_from_slice(self.as_slice());
        Self::from_parts(self.shape.clone(), data)
    }
}

impl ArrayI64 {
    /// Element type ([`DType::I64`]).
    #[inline]
    /// `dtype` (see `f64` [`Array`] counterpart).
    pub fn dtype(&self) -> DType {
        DType::I64
    }

    #[inline]
    /// `shape` (see `f64` [`Array`] counterpart).
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    #[inline]
    /// `dims` (see `f64` [`Array`] counterpart).
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }
    #[inline]
    /// `rank` (see `f64` [`Array`] counterpart).
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }
    #[inline]
    /// `len` (see `f64` [`Array`] counterpart).
    pub fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    /// `is_empty` (see `f64` [`Array`] counterpart).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    #[inline]
    /// `as_slice` (see `f64` [`Array`] counterpart).
    pub fn as_slice(&self) -> &[i64] {
        self.data.as_ref()
    }
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn buffer_strong_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }
    #[inline]
    /// `as_mut_slice` (see `f64` [`Array`] counterpart).
    pub fn as_mut_slice(&mut self) -> &mut [i64] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// `from_shape_vec` (see `f64` [`Array`] counterpart).
    pub fn from_shape_vec(shape: impl Into<Vec<usize>>, data: Vec<i64>) -> Result<Self> {
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

    #[inline]
    pub(crate) fn from_parts(shape: Shape, data: Vec<i64>) -> Self {
        debug_assert_eq!(data.len(), shape.numel());
        Self {
            shape,
            data: Arc::new(data),
        }
    }

    /// `from_shape_slice` (see `f64` [`Array`] counterpart).
    pub fn from_shape_slice(shape: impl Into<Vec<usize>>, data: &[i64]) -> Result<Self> {
        Self::from_shape_vec(shape, data.to_vec())
    }

    /// `zeros` (see `f64` [`Array`] counterpart).
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        Ok(Self::from_parts(shape.clone(), pool::take_zeroed(shape.numel())))
    }

    /// `ones` (see `f64` [`Array`] counterpart).
    pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        Ok(Self::from_parts(shape.clone(), pool::take_filled(shape.numel(), 1)))
    }

    /// `full` (see `f64` [`Array`] counterpart).
    pub fn full(shape: impl Into<Vec<usize>>, value: i64) -> Result<Self> {
        let shape = Shape::new(shape)?;
        Ok(Self::from_parts(shape.clone(), pool::take_filled(shape.numel(), value)))
    }

    /// Rank-1 range `[start, stop)` with step `1`.
    pub fn arange(start: i64, stop: i64) -> Result<Self> {
        Self::arange_step(start, stop, 1)
    }

    /// Rank-1 range `[start, stop)` with the given step (must be non-zero).
    pub fn arange_step(start: i64, stop: i64, step: i64) -> Result<Self> {
        if step == 0 {
            return Err(Error::Shape("arange step must be non-zero".into()));
        }
        if (step > 0 && start >= stop) || (step < 0 && start <= stop) {
            return Ok(Self::from_parts(Shape::from_len(0), Vec::new()));
        }
        // Estimate length carefully
        let n = if step > 0 {
            let span = (stop - start) as i128;
            ((span + step as i128 - 1) / step as i128) as usize
        } else {
            let span = (start - stop) as i128;
            let st = (-step) as i128;
            ((span + st - 1) / st) as usize
        };
        let mut data = pool::take_uninit(n);
        let mut x = start;
        let mut written = 0usize;
        for i in 0..n {
            if (step > 0 && x >= stop) || (step < 0 && x <= stop) {
                break;
            }
            data[i] = x;
            written = i + 1;
            x = x.wrapping_add(step);
        }
        data.truncate(written);
        Ok(Self::from_parts(Shape::from_len(data.len()), data))
    }

    /// `get` (see `f64` [`Array`] counterpart).
    pub fn get(&self, indices: &[usize]) -> Result<i64> {
        Ok(self.data[self.shape.offset(indices)?])
    }

    /// `set` (see `f64` [`Array`] counterpart).
    pub fn set(&mut self, indices: &[usize], value: i64) -> Result<()> {
        let off = self.shape.offset(indices)?;
        self.as_mut_slice()[off] = value;
        Ok(())
    }

    /// `fill` (see `f64` [`Array`] counterpart).
    pub fn fill(&mut self, value: i64) {
        self.as_mut_slice().fill(value);
    }

    /// `reshape` (see `f64` [`Array`] counterpart).
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

    /// `copy` (see `f64` [`Array`] counterpart).
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// `eye` (see `f64` [`Array`] counterpart).
    pub fn eye(n: usize) -> Result<Self> {
        let shape = Shape::matrix(n, n)?;
        let mut data = pool::take_zeroed(shape.numel());
        for i in 0..n {
            data[i * n + i] = 1;
        }
        Ok(Self::from_parts(shape, data))
    }

    /// `sum` (see `f64` [`Array`] counterpart).
    pub fn sum(&self) -> i64 {
        kernels::sum_slice(self.as_slice())
    }

    /// Mean as `f64` (integer arrays often need a real mean).
    pub fn mean(&self) -> Result<f64> {
        if self.is_empty() {
            return Err(Error::Shape("mean of empty array".into()));
        }
        Ok(self.sum() as f64 / self.len() as f64)
    }

    /// `min` (see `f64` [`Array`] counterpart).
    pub fn min(&self) -> Result<i64> {
        kernels::min_slice(self.as_slice()).ok_or_else(|| Error::Shape("min of empty array".into()))
    }

    /// `max` (see `f64` [`Array`] counterpart).
    pub fn max(&self) -> Result<i64> {
        kernels::max_slice(self.as_slice()).ok_or_else(|| Error::Shape("max of empty array".into()))
    }

    /// Cast to `f64` [`Array`] (exact for integers in the float mantissa range).
    pub fn to_f64(&self) -> Array {
        let mut data = super::pool::take_uninit(self.len());
        for (i, &x) in self.as_slice().iter().enumerate() {
            data[i] = x as f64;
        }
        Array::from_parts(self.shape.clone(), data)
    }

    /// Build from `f64` array by truncating toward zero.
    pub fn from_f64(a: &Array) -> Self {
        let mut data = pool::take_uninit(a.len());
        for (i, &x) in a.as_slice().iter().enumerate() {
            data[i] = x as i64;
        }
        Self::from_parts(a.shape().clone(), data)
    }

    /// `to_arrow` (see `f64` [`Array`] counterpart).
    pub fn to_arrow(&self) -> Int64Array {
        Int64Array::from(self.as_slice().to_vec())
    }

    /// `from_arrow` (see `f64` [`Array`] counterpart).
    pub fn from_arrow(array: &Int64Array, shape: impl Into<Vec<usize>>) -> Result<Self> {
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
        let data: Vec<i64> = array.values().iter().copied().collect();
        Ok(Self::from_parts(shape, data))
    }

    fn same_shape(&self, other: &ArrayI64) -> Result<()> {
        if !self.shape.same_as(other.shape()) {
            return Err(Error::Shape(format!(
                "shape mismatch: {} vs {}",
                self.shape, other.shape
            )));
        }
        Ok(())
    }

    /// `broadcast_to` (see `f64` [`Array`] counterpart).
    pub fn broadcast_to(&self, dims: impl Into<Vec<usize>>) -> Result<Self> {
        let target = Shape::new(dims)?;
        if self.shape.same_as(&target) {
            return Ok(self.clone());
        }
        let out_dims = broadcast_shapes(self.dims(), target.dims())?;
        if out_dims.as_slice() != target.dims() {
            return Err(Error::Shape(format!(
                "cannot broadcast {} to {}",
                self.shape, target
            )));
        }
        // Materialize by walking multi-indices
        let n = target.numel();
        let mut data = pool::take_uninit(n);
        let src = self.as_slice();
        let src_dims = self.dims();
        let tgt = target.dims();
        let rank = tgt.len();
        let mut idx = vec![0usize; rank];
        for flat in 0..n {
            // map target index into source (size-1 axes → 0)
            let mut src_idx = vec![0usize; src_dims.len()];
            let sr = src_dims.len();
            for i in 0..sr {
                let ti = rank - sr + i;
                src_idx[i] = if src_dims[i] == 1 { 0 } else { idx[ti] };
            }
            // handle rank padding when src rank < target
            if sr < rank {
                // src is right-aligned
                for i in 0..sr {
                    let ti = rank - sr + i;
                    src_idx[i] = if src_dims[i] == 1 { 0 } else { idx[ti] };
                }
            }
            let off = self.shape.offset(&src_idx)?;
            data[flat] = src[off];
            // increment idx
            for d in (0..rank).rev() {
                idx[d] += 1;
                if idx[d] < tgt[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(Self::from_parts(target, data))
    }

    fn owned_from_kernel<F>(&self, other: &ArrayI64, f: F) -> Result<ArrayI64>
    where
        F: FnOnce(&[i64], &[i64], &mut [i64]),
    {
        if self.shape.same_as(other.shape()) {
            let mut data = pool::take_uninit(self.len());
            f(self.as_slice(), other.as_slice(), &mut data);
            return Ok(Self::from_parts(self.shape.clone(), data));
        }
        let out_dims = broadcast_shapes(self.dims(), other.dims())?;
        let out_shape = Shape::new(out_dims)?;
        let left = self.broadcast_to(out_shape.dims())?;
        let right = other.broadcast_to(out_shape.dims())?;
        let mut data = pool::take_uninit(left.len());
        f(left.as_slice(), right.as_slice(), &mut data);
        Ok(Self::from_parts(out_shape, data))
    }

    fn owned_binary_broadcast(&self, other: &ArrayI64, op: BroadcastOp) -> Result<ArrayI64> {
        if self.shape.same_as(other.shape()) {
            let mut data = pool::take_uninit(self.len());
            op.apply_same(self.as_slice(), other.as_slice(), &mut data);
            return Ok(Self::from_parts(self.shape.clone(), data));
        }
        if self.rank() == 2 {
            let (m, n) = (self.dims()[0], self.dims()[1]);
            if (other.rank() == 1 && other.len() == n)
                || (other.rank() == 2 && other.dims() == [1, n])
            {
                let mut data = pool::take_uninit(m * n);
                op.apply_row(m, n, self.as_slice(), other.as_slice(), &mut data);
                return Ok(Self::from_parts(self.shape.clone(), data));
            }
            if other.rank() == 2 && other.dims() == [m, 1] {
                let mut data = pool::take_uninit(m * n);
                op.apply_col(m, n, self.as_slice(), other.as_slice(), &mut data);
                return Ok(Self::from_parts(self.shape.clone(), data));
            }
        }
        self.owned_from_kernel(other, |a, b, o| op.apply_same(a, b, o))
    }

    fn owned_unary<F>(&self, f: F) -> ArrayI64
    where
        F: FnOnce(&[i64], &mut [i64]),
    {
        let mut data = pool::take_uninit(self.len());
        f(self.as_slice(), &mut data);
        Self::from_parts(self.shape.clone(), data)
    }

    /// `add` (see `f64` [`Array`] counterpart).
    pub fn add(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_binary_broadcast(other, BroadcastOp::Add)
    }
    /// `sub` (see `f64` [`Array`] counterpart).
    pub fn sub(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_binary_broadcast(other, BroadcastOp::Sub)
    }
    /// `mul` (see `f64` [`Array`] counterpart).
    pub fn mul(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_binary_broadcast(other, BroadcastOp::Mul)
    }
    /// `div` (see `f64` [`Array`] counterpart).
    pub fn div(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_binary_broadcast(other, BroadcastOp::Div)
    }

    /// Write `self + other` into `out` (same shape; no alloc).
    pub fn add_out(&self, other: &ArrayI64, out: &mut ArrayI64) -> Result<()> {
        self.same_shape(other)?;
        self.same_shape(out)?;
        kernels::add_slices(self.as_slice(), other.as_slice(), out.as_mut_slice());
        Ok(())
    }
    /// `sub_out`.
    pub fn sub_out(&self, other: &ArrayI64, out: &mut ArrayI64) -> Result<()> {
        self.same_shape(other)?;
        self.same_shape(out)?;
        kernels::sub_slices(self.as_slice(), other.as_slice(), out.as_mut_slice());
        Ok(())
    }
    /// `mul_out`.
    pub fn mul_out(&self, other: &ArrayI64, out: &mut ArrayI64) -> Result<()> {
        self.same_shape(other)?;
        self.same_shape(out)?;
        kernels::mul_slices(self.as_slice(), other.as_slice(), out.as_mut_slice());
        Ok(())
    }
    /// `div_out`.
    pub fn div_out(&self, other: &ArrayI64, out: &mut ArrayI64) -> Result<()> {
        self.same_shape(other)?;
        self.same_shape(out)?;
        kernels::div_slices(self.as_slice(), other.as_slice(), out.as_mut_slice());
        Ok(())
    }
    /// `neg_out`.
    pub fn neg_out(&self, out: &mut ArrayI64) -> Result<()> {
        self.same_shape(out)?;
        kernels::neg_slice(self.as_slice(), out.as_mut_slice());
        Ok(())
    }
    /// `out = abs(self)`.
    pub fn abs_out(&self, out: &mut ArrayI64) -> Result<()> {
        self.same_shape(out)?;
        kernels::abs_slice(self.as_slice(), out.as_mut_slice());
        Ok(())
    }

    /// `neg` (see `f64` [`Array`] counterpart).
    pub fn neg(&self) -> ArrayI64 {
        self.owned_unary(kernels::neg_slice)
    }
    /// `abs` (see `f64` [`Array`] counterpart).
    pub fn abs(&self) -> ArrayI64 {
        self.owned_unary(kernels::abs_slice)
    }

    /// `add_assign_arr` (see `f64` [`Array`] counterpart).
    pub fn add_assign_arr(&mut self, other: &ArrayI64) -> Result<()> {
        self.same_shape(other)?;
        kernels::add_assign_slices(self.as_mut_slice(), other.as_slice());
        Ok(())
    }
    /// `sub_assign_arr` (see `f64` [`Array`] counterpart).
    pub fn sub_assign_arr(&mut self, other: &ArrayI64) -> Result<()> {
        self.same_shape(other)?;
        kernels::sub_assign_slices(self.as_mut_slice(), other.as_slice());
        Ok(())
    }
    /// `mul_assign_arr` (see `f64` [`Array`] counterpart).
    pub fn mul_assign_arr(&mut self, other: &ArrayI64) -> Result<()> {
        self.same_shape(other)?;
        kernels::mul_assign_slices(self.as_mut_slice(), other.as_slice());
        Ok(())
    }
    /// `div_assign_arr` (see `f64` [`Array`] counterpart).
    pub fn div_assign_arr(&mut self, other: &ArrayI64) -> Result<()> {
        self.same_shape(other)?;
        kernels::div_assign_slices(self.as_mut_slice(), other.as_slice());
        Ok(())
    }

    pub(crate) fn add_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| kernels::add_scalar(a, s, o))
    }
    pub(crate) fn sub_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| kernels::sub_scalar(a, s, o))
    }
    pub(crate) fn mul_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| kernels::mul_scalar(a, s, o))
    }
    pub(crate) fn div_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| kernels::div_scalar(a, s, o))
    }

    /// `eq` (see `f64` [`Array`] counterpart).
    pub fn eq(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_from_kernel(other, kernels::eq_slices)
    }
    /// `ne` (see `f64` [`Array`] counterpart).
    pub fn ne(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_from_kernel(other, kernels::ne_slices)
    }
    /// `lt` (see `f64` [`Array`] counterpart).
    pub fn lt(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_from_kernel(other, kernels::lt_slices)
    }
    /// `le` (see `f64` [`Array`] counterpart).
    pub fn le(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_from_kernel(other, kernels::le_slices)
    }
    /// `gt` (see `f64` [`Array`] counterpart).
    pub fn gt(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_from_kernel(other, kernels::gt_slices)
    }
    /// `ge` (see `f64` [`Array`] counterpart).
    pub fn ge(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_from_kernel(other, kernels::ge_slices)
    }

    /// `cumsum` (see `f64` [`Array`] counterpart).
    pub fn cumsum(&self) -> ArrayI64 {
        self.owned_unary(kernels::cumsum_slice)
    }

    /// `argmin` (see `f64` [`Array`] counterpart).
    pub fn argmin(&self) -> Result<usize> {
        kernels::argmin_slice(self.as_slice()).ok_or_else(|| Error::Shape("argmin of empty".into()))
    }
    /// `argmax` (see `f64` [`Array`] counterpart).
    pub fn argmax(&self) -> Result<usize> {
        kernels::argmax_slice(self.as_slice()).ok_or_else(|| Error::Shape("argmax of empty".into()))
    }

    /// `any` (see `f64` [`Array`] counterpart).
    pub fn any(&self) -> bool {
        kernels::any_slice(self.as_slice())
    }
    /// `all` (see `f64` [`Array`] counterpart).
    pub fn all(&self) -> bool {
        kernels::all_slice(self.as_slice())
    }

    fn rank2_dims(&self) -> Result<(usize, usize)> {
        if self.rank() != 2 {
            return Err(Error::Shape(format!(
                "expected rank-2, got rank {}",
                self.rank()
            )));
        }
        Ok((self.dims()[0], self.dims()[1]))
    }

    /// `sum_axis` (see `f64` [`Array`] counterpart).
    pub fn sum_axis(&self, axis: usize) -> Result<ArrayI64> {
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

    /// `mean_axis` (see `f64` [`Array`] counterpart).
    pub fn mean_axis(&self, axis: usize) -> Result<Array> {
        // mean as f64 array
        let s = self.sum_axis(axis)?;
        let (m, n) = self.rank2_dims()?;
        let denom = match axis {
            0 => m as f64,
            1 => n as f64,
            _ => unreachable!(),
        };
        if denom == 0.0 {
            return Err(Error::Shape("mean_axis of empty dimension".into()));
        }
        let mut data = super::pool::take_uninit(s.len());
        for (i, &x) in s.as_slice().iter().enumerate() {
            data[i] = x as f64 / denom;
        }
        Ok(Array::from_parts(s.shape().clone(), data))
    }

    /// `min_axis` (see `f64` [`Array`] counterpart).
    pub fn min_axis(&self, axis: usize) -> Result<ArrayI64> {
        let (m, n) = self.rank2_dims()?;
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

    /// `max_axis` (see `f64` [`Array`] counterpart).
    pub fn max_axis(&self, axis: usize) -> Result<ArrayI64> {
        let (m, n) = self.rank2_dims()?;
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

    /// Argsort indices as `i64` array (0-based). Rank-1 only.
    pub fn argsort(&self, descending: bool) -> Result<ArrayI64> {
        if self.rank() != 1 {
            return Err(Error::Shape("argsort requires rank-1".into()));
        }
        let n = self.len();
        let mut idx = vec![0usize; n];
        kernels::argsort_indices(self.as_slice(), descending, &mut idx);
        let mut data = pool::take_uninit(n);
        for i in 0..n {
            data[i] = idx[i] as i64;
        }
        Ok(Self::from_parts(Shape::from_len(n), data))
    }

    /// Take elements by 0-based integer indices (rank-1 source and indices).
    pub fn take(&self, indices: &ArrayI64) -> Result<ArrayI64> {
        if self.rank() != 1 {
            return Err(Error::Shape("take requires rank-1 source".into()));
        }
        if indices.rank() != 1 {
            return Err(Error::Shape("take indices must be rank-1".into()));
        }
        let n = indices.len();
        let mut data = pool::take_uninit(n);
        let src = self.as_slice();
        for (i, &ix) in indices.as_slice().iter().enumerate() {
            if ix < 0 || ix as usize >= self.len() {
                return Err(Error::Index(format!("take index {ix} out of range")));
            }
            data[i] = src[ix as usize];
        }
        Ok(Self::from_parts(Shape::from_len(n), data))
    }

    /// Flat 0-based indices of nonzero elements.
    pub fn nonzero(&self) -> ArrayI64 {
        let mut idx = Vec::new();
        for (i, &v) in self.as_slice().iter().enumerate() {
            if v != 0 {
                idx.push(i as i64);
            }
        }
        let n = idx.len();
        Self::from_parts(Shape::from_len(n), idx)
    }

    /// Select rank-1 elements where mask is nonzero.
    pub fn compress(&self, mask: &ArrayI64) -> Result<ArrayI64> {
        if self.rank() != 1 || mask.rank() != 1 {
            return Err(Error::Shape("compress requires rank-1 array and mask".into()));
        }
        if self.len() != mask.len() {
            return Err(Error::Shape("compress length mismatch".into()));
        }
        let mut out = Vec::new();
        for i in 0..self.len() {
            if mask.as_slice()[i] != 0 {
                out.push(self.as_slice()[i]);
            }
        }
        let n = out.len();
        Ok(Self::from_parts(Shape::from_len(n), out))
    }

    /// Scatter values at 0-based indices.
    pub fn put(&mut self, indices: &ArrayI64, values: &ArrayI64) -> Result<()> {
        if self.rank() != 1 || indices.rank() != 1 || values.rank() != 1 {
            return Err(Error::Shape("put requires rank-1 dest, indices, values".into()));
        }
        if indices.len() != values.len() {
            return Err(Error::Shape("put indices/values length mismatch".into()));
        }
        let n = self.len();
        let dst = self.as_mut_slice();
        for k in 0..indices.len() {
            let i = indices.as_slice()[k];
            if i < 0 {
                return Err(Error::Index(format!("put index {i} invalid")));
            }
            let i = i as usize;
            if i >= n {
                return Err(Error::Index(format!("put index {i} out of range")));
            }
            dst[i] = values.as_slice()[k];
        }
        Ok(())
    }

    /// Assign where mask nonzero.
    pub fn put_mask(&mut self, mask: &ArrayI64, values: &ArrayI64) -> Result<()> {
        if self.rank() != 1 || mask.rank() != 1 || values.rank() != 1 {
            return Err(Error::Shape("put_mask requires rank-1 args".into()));
        }
        if self.len() != mask.len() {
            return Err(Error::Shape("put_mask length mismatch".into()));
        }
        let count = mask.as_slice().iter().filter(|&&x| x != 0).count();
        let val = values.as_slice();
        if val.len() != 1 && val.len() != count {
            return Err(Error::Shape(format!(
                "put_mask values len {} != true count {count} (or 1)",
                val.len()
            )));
        }
        let dst = self.as_mut_slice();
        let m = mask.as_slice();
        let mut t = 0usize;
        for i in 0..dst.len() {
            if m[i] != 0 {
                dst[i] = if val.len() == 1 { val[0] } else { val[t] };
                t += 1;
            }
        }
        Ok(())
    }


    /// `slice` (see `f64` [`Array`] counterpart).
    pub fn slice(&self, start: usize, stop: usize) -> Result<ArrayI64> {
        if self.rank() != 1 {
            return Err(Error::Shape("slice expects rank-1".into()));
        }
        let n = self.len();
        if start > stop || stop > n {
            return Err(Error::Index(format!(
                "slice [{start}, {stop}) invalid for len {n}"
            )));
        }
        let data = self.as_slice()[start..stop].to_vec();
        Ok(Self::from_parts(Shape::from_len(data.len()), data))
    }

    /// `rows` (see `f64` [`Array`] counterpart).
    pub fn rows(&self, start: usize, stop: usize) -> Result<ArrayI64> {
        let (m, n) = self.rank2_dims()?;
        if start > stop || stop > m {
            return Err(Error::Index(format!(
                "rows [{start}, {stop}) invalid for {m} rows"
            )));
        }
        let data = self.as_slice()[start * n..stop * n].to_vec();
        Ok(Self::from_parts(Shape::matrix(stop - start, n)?, data))
    }

    /// `row` (see `f64` [`Array`] counterpart).
    pub fn row(&self, i: usize) -> Result<ArrayI64> {
        let (m, n) = self.rank2_dims()?;
        if i >= m {
            return Err(Error::Index(format!("row {i} out of range for {m} rows")));
        }
        let data = self.as_slice()[i * n..(i + 1) * n].to_vec();
        Ok(Self::from_parts(Shape::from_len(n), data))
    }

    /// `col` (see `f64` [`Array`] counterpart).
    pub fn col(&self, j: usize) -> Result<ArrayI64> {
        let (m, n) = self.rank2_dims()?;
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

    /// `transpose` (see `f64` [`Array`] counterpart).
    /// Transpose rank-1 → `(1,n)` row matrix; rank-2 uses blocked out-of-place transpose.
    pub fn transpose(&self) -> Result<ArrayI64> {
        match self.rank() {
            1 => {
                let n = self.len();
                let mut data = pool::take_uninit(n);
                data.copy_from_slice(self.as_slice());
                Ok(Self::from_parts(Shape::matrix(1, n)?, data))
            }
            2 => {
                let (rows, cols) = (self.dims()[0], self.dims()[1]);
                let mut data = pool::take_uninit(rows * cols);
                blocked_transpose_i64(self.as_slice(), rows, cols, &mut data);
                Ok(Self::from_parts(Shape::matrix(cols, rows)?, data))
            }
            r => Err(Error::Shape(format!(
                "transpose expects rank 1 or 2, got rank {r}"
            ))),
        }
    }


    /// `diagonal` (see `f64` [`Array`] counterpart).
    pub fn diagonal(&self) -> Result<ArrayI64> {
        let (m, n) = self.rank2_dims()?;
        let k = m.min(n);
        let mut data = pool::take_uninit(k);
        let src = self.as_slice();
        for i in 0..k {
            data[i] = src[i * n + i];
        }
        Ok(Self::from_parts(Shape::from_len(k), data))
    }

    /// `trace` (see `f64` [`Array`] counterpart).
    pub fn trace(&self) -> Result<i64> {
        Ok(self.diagonal()?.sum())
    }

    /// Vector → diagonal matrix, or matrix → diagonal vector.
    pub fn diag(a: &ArrayI64) -> Result<ArrayI64> {
        match a.rank() {
            1 => {
                let n = a.len();
                let mut data = pool::take_zeroed(n * n);
                for i in 0..n {
                    data[i * n + i] = a.as_slice()[i];
                }
                Ok(Self::from_parts(Shape::matrix(n, n)?, data))
            }
            2 => a.diagonal(),
            r => Err(Error::Shape(format!("diag expects rank 1 or 2, got {r}"))),
        }
    }

    /// `outer` (see `f64` [`Array`] counterpart).
    pub fn outer(a: &ArrayI64, b: &ArrayI64) -> Result<ArrayI64> {
        if a.rank() != 1 || b.rank() != 1 {
            return Err(Error::Shape("outer expects two rank-1 arrays".into()));
        }
        let m = a.len();
        let n = b.len();
        let mut data = pool::take_uninit(m * n);
        let aa = a.as_slice();
        let bb = b.as_slice();
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = aa[i].wrapping_mul(bb[j]);
            }
        }
        Ok(Self::from_parts(Shape::matrix(m, n)?, data))
    }

    /// Sign: −1, 0, +1.
    pub fn sign(&self) -> ArrayI64 {
        self.owned_unary(kernels::sign_slice)
    }

    /// Clip values to `[lo, hi]`.
    pub fn clip(&self, lo: i64, hi: i64) -> Result<ArrayI64> {
        if lo > hi {
            return Err(Error::shape(format!("clip lo {lo} > hi {hi}")));
        }
        Ok(self.owned_unary(|a, o| kernels::clip_slice(a, lo, hi, o)))
    }

    /// `where(cond, x, y)` — nonzero `cond` selects `x`, else `y` (same shape).
    pub fn where_cond(cond: &ArrayI64, x: &ArrayI64, y: &ArrayI64) -> Result<ArrayI64> {
        cond.same_shape(x)?;
        cond.same_shape(y)?;
        let mut data = pool::take_uninit(cond.len());
        kernels::where_slices(cond.as_slice(), x.as_slice(), y.as_slice(), &mut data);
        Ok(Self::from_parts(cond.shape.clone(), data))
    }

    /// Sample/population variance as `f64` (`ddof` like NumPy).
    pub fn var(&self, ddof: usize) -> Result<f64> {
        let n = self.len();
        if n <= ddof {
            return Err(Error::shape(format!(
                "var requires len > ddof (len={n}, ddof={ddof})"
            )));
        }
        let mean = self.sum() as f64 / n as f64;
        let mut acc = 0.0f64;
        for &x in self.as_slice() {
            let d = x as f64 - mean;
            acc += d * d;
        }
        Ok(acc / (n - ddof) as f64)
    }

    /// Standard deviation as `f64`.
    pub fn std(&self, ddof: usize) -> Result<f64> {
        Ok(self.var(ddof)?.sqrt())
    }

    /// Median as `f64` (even length averages the two centers). Empty → error.
    pub fn median(&self) -> Result<f64> {
        if self.is_empty() {
            return Err(Error::Shape("median of empty array".into()));
        }
        let mut v: Vec<f64> = self.as_slice().iter().map(|&x| x as f64).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        super::kernels::quantile_sorted(&v, 0.5)
            .ok_or_else(|| Error::Shape("median of empty array".into()))
    }

    /// Linear quantile `q ∈ [0, 1]` as `f64`. Empty → error.
    pub fn quantile(&self, q: f64) -> Result<f64> {
        if !(0.0..=1.0).contains(&q) || !q.is_finite() {
            return Err(Error::Shape(format!("quantile q must be in [0, 1], got {q}")));
        }
        if self.is_empty() {
            return Err(Error::Shape("quantile of empty array".into()));
        }
        let mut v: Vec<f64> = self.as_slice().iter().map(|&x| x as f64).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(super::kernels::quantile_sorted(&v, q).unwrap())
    }

    /// Several quantiles as rank-1 `f64` (same semantics as [`Array::quantiles`](super::Array::quantiles)).
    pub fn quantiles(&self, qs: &[f64]) -> Result<Array> {
        for &q in qs {
            if !(0.0..=1.0).contains(&q) || !q.is_finite() {
                return Err(Error::Shape(format!("quantile q must be in [0, 1], got {q}")));
            }
        }
        if self.is_empty() {
            return Err(Error::Shape("quantiles of empty array".into()));
        }
        let mut v: Vec<f64> = self.as_slice().iter().map(|&x| x as f64).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut data = super::pool::take_uninit(qs.len());
        for (i, &q) in qs.iter().enumerate() {
            data[i] = super::kernels::quantile_sorted(&v, q).unwrap();
        }
        Ok(Array::from_parts(Shape::from_len(qs.len()), data))
    }

    /// Median along axis for rank-2 → rank-1 `f64`.
    pub fn median_axis(&self, axis: usize) -> Result<Array> {
        let (m, n) = self.rank2_dims()?;
        let src = self.as_slice();
        match axis {
            0 => {
                let mut data = super::pool::take_uninit(n);
                let mut col = vec![0.0f64; m];
                for j in 0..n {
                    for i in 0..m {
                        col[i] = src[i * n + j] as f64;
                    }
                    data[j] = super::kernels::median_slice(&col).ok_or_else(|| {
                        Error::Shape("median_axis of empty".into())
                    })?;
                }
                Ok(Array::from_parts(Shape::from_len(n), data))
            }
            1 => {
                let mut data = super::pool::take_uninit(m);
                for i in 0..m {
                    let mut row: Vec<f64> = src[i * n..(i + 1) * n].iter().map(|&x| x as f64).collect();
                    data[i] = super::kernels::median_slice(&row).ok_or_else(|| {
                        Error::Shape("median_axis of empty".into())
                    })?;
                }
                Ok(Array::from_parts(Shape::from_len(m), data))
            }
            _ => Err(Error::Shape(format!("axis {axis} out of range for rank-2"))),
        }
    }

    /// Quantile along axis for rank-2 → rank-1 `f64`.
    pub fn quantile_axis(&self, axis: usize, q: f64) -> Result<Array> {
        if !(0.0..=1.0).contains(&q) || !q.is_finite() {
            return Err(Error::Shape(format!("quantile q must be in [0, 1], got {q}")));
        }
        let (m, n) = self.rank2_dims()?;
        let src = self.as_slice();
        match axis {
            0 => {
                let mut data = super::pool::take_uninit(n);
                let mut col = vec![0.0f64; m];
                for j in 0..n {
                    for i in 0..m {
                        col[i] = src[i * n + j] as f64;
                    }
                    col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    data[j] = super::kernels::quantile_sorted(&col, q).ok_or_else(|| {
                        Error::Shape("quantile_axis of empty".into())
                    })?;
                }
                Ok(Array::from_parts(Shape::from_len(n), data))
            }
            1 => {
                let mut data = super::pool::take_uninit(m);
                for i in 0..m {
                    let mut row: Vec<f64> = src[i * n..(i + 1) * n].iter().map(|&x| x as f64).collect();
                    row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    data[i] = super::kernels::quantile_sorted(&row, q).ok_or_else(|| {
                        Error::Shape("quantile_axis of empty".into())
                    })?;
                }
                Ok(Array::from_parts(Shape::from_len(m), data))
            }
            _ => Err(Error::Shape(format!("axis {axis} out of range for rank-2"))),
        }
    }



    /// Concatenate along `axis` (rank 1–2).
    pub fn concatenate(axis: usize, parts: &[&ArrayI64]) -> Result<ArrayI64> {
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
            let cols_out = shape.dims()[1];
            if axis == 0 {
                let mut row = 0usize;
                for p in parts {
                    let pr = p.dims()[0];
                    let pc = p.dims()[1];
                    for i in 0..pr {
                        let src = &p.as_slice()[i * pc..(i + 1) * pc];
                        data[(row + i) * cols_out..(row + i) * cols_out + pc].copy_from_slice(src);
                    }
                    row += pr;
                }
            } else {
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

    /// Stack rank-1 arrays along a new axis (0 or 1).
    pub fn stack(axis: usize, parts: &[&ArrayI64]) -> Result<ArrayI64> {
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
                let mut data = pool::take_uninit(k * n);
                for (i, p) in parts.iter().enumerate() {
                    data[i * n..(i + 1) * n].copy_from_slice(p.as_slice());
                }
                Ok(Self::from_parts(Shape::matrix(k, n)?, data))
            }
            1 => {
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

    /// `any` along axis for rank-2 → 0/1 rank-1 mask.
    pub fn any_axis(&self, axis: usize) -> Result<ArrayI64> {
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
    pub fn all_axis(&self, axis: usize) -> Result<ArrayI64> {
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

    /// Consume and reshape without copying the value buffer.
    pub fn into_reshape(mut self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        if shape.numel() != self.len() {
            return Err(Error::Shape(format!(
                "cannot reshape array of {} elements into {}",
                self.len(),
                shape
            )));
        }
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


    /// Variance along axis for rank-2 → rank-1 `f64` array.
    pub fn var_axis(&self, axis: usize, ddof: usize) -> Result<Array> {
        let (m, n) = self.rank2_dims()?;
        let (len_out, count) = match axis {
            0 => (n, m),
            1 => (m, n),
            _ => return Err(Error::Shape(format!("axis {axis} out of range for rank-2"))),
        };
        if count <= ddof {
            return Err(Error::shape(format!(
                "var_axis requires size > ddof (size={count}, ddof={ddof})"
            )));
        }
        let means = self.mean_axis(axis)?;
        let mut data = super::pool::take_uninit(len_out);
        let src = self.as_slice();
        let mu = means.as_slice();
        match axis {
            0 => {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for i in 0..m {
                        let d = src[i * n + j] as f64 - mu[j];
                        acc += d * d;
                    }
                    data[j] = acc / (count - ddof) as f64;
                }
            }
            1 => {
                for i in 0..m {
                    let mut acc = 0.0f64;
                    for j in 0..n {
                        let d = src[i * n + j] as f64 - mu[i];
                        acc += d * d;
                    }
                    data[i] = acc / (count - ddof) as f64;
                }
            }
            _ => unreachable!(),
        }
        Ok(Array::from_parts(Shape::from_len(len_out), data))
    }

    /// Std-dev along axis for rank-2 → rank-1 `f64` array.
    pub fn std_axis(&self, axis: usize, ddof: usize) -> Result<Array> {
        let v = self.var_axis(axis, ddof)?;
        let mut data = super::pool::take_uninit(v.len());
        for (i, &x) in v.as_slice().iter().enumerate() {
            data[i] = x.sqrt();
        }
        Ok(Array::from_parts(v.shape().clone(), data))
    }

    /// Element-wise power with non-negative integer exponent (wrapping).
    pub fn power_scalar(&self, exp: u32) -> ArrayI64 {
        self.owned_unary(|a, out| {
            for i in 0..a.len() {
                out[i] = a[i].wrapping_pow(exp);
            }
        })
    }

    /// Compare to scalar → 0/1 mask.
    pub fn eq_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| kernels::eq_scalar(a, s, o))
    }
    /// `ne_scalar`.
    pub fn ne_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = if a[i] != s { 1 } else { 0 };
            }
        })
    }
    /// `lt_scalar`.
    pub fn lt_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = if a[i] < s { 1 } else { 0 };
            }
        })
    }
    /// `le_scalar`.
    pub fn le_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = if a[i] <= s { 1 } else { 0 };
            }
        })
    }
    /// `gt_scalar`.
    pub fn gt_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = if a[i] > s { 1 } else { 0 };
            }
        })
    }
    /// `ge_scalar`.
    pub fn ge_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = if a[i] >= s { 1 } else { 0 };
            }
        })
    }


    /// Bitwise AND (elementwise).
    pub fn bitand(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_from_kernel(other, kernels::bitand_slices)
    }
    /// Bitwise OR (elementwise).
    pub fn bitor(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_from_kernel(other, kernels::bitor_slices)
    }
    /// Bitwise XOR (elementwise).
    pub fn bitxor(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_from_kernel(other, kernels::bitxor_slices)
    }
    /// Bitwise NOT (elementwise).
    pub fn bitnot(&self) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = !a[i];
            }
        })
    }
    /// Left shift by scalar (logical on bits; Rust `<<` for `i64`).
    pub fn shift_left(&self, bits: u32) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = a[i].wrapping_shl(bits);
            }
        })
    }
    /// Arithmetic right shift by scalar.
    pub fn shift_right(&self, bits: u32) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = a[i].wrapping_shr(bits);
            }
        })
    }
    /// Remainder `self % other` (Rust `%`; divisor 0 → 0).
    pub fn rem(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.owned_from_kernel(other, kernels::rem_slices)
    }
    /// `rem_scalar`.
    pub fn rem_scalar(&self, s: i64) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = if s == 0 { 0 } else { a[i] % s };
            }
        })
    }

    /// Sorted unique values (rank-1). Stable order = sorted ascending.
    pub fn unique(&self) -> Result<ArrayI64> {
        if self.rank() != 1 {
            return Err(Error::Shape("unique requires rank-1".into()));
        }
        let mut v = self.as_slice().to_vec();
        v.sort_unstable();
        v.dedup();
        Ok(Self::from_parts(Shape::from_len(v.len()), v))
    }

    /// `(unique_values, counts)` for rank-1 (sorted uniques).
    pub fn unique_counts(&self) -> Result<(ArrayI64, ArrayI64)> {
        if self.rank() != 1 {
            return Err(Error::Shape("unique_counts requires rank-1".into()));
        }
        let mut v = self.as_slice().to_vec();
        v.sort_unstable();
        if v.is_empty() {
            return Ok((
                Self::from_parts(Shape::from_len(0), Vec::new()),
                Self::from_parts(Shape::from_len(0), Vec::new()),
            ));
        }
        let mut vals = Vec::new();
        let mut counts = Vec::new();
        let mut cur = v[0];
        let mut c: i64 = 1;
        for &x in &v[1..] {
            if x == cur {
                c = c.wrapping_add(1);
            } else {
                vals.push(cur);
                counts.push(c);
                cur = x;
                c = 1;
            }
        }
        vals.push(cur);
        counts.push(c);
        let n = vals.len();
        Ok((
            Self::from_parts(Shape::from_len(n), vals),
            Self::from_parts(Shape::from_len(n), counts),
        ))
    }

    /// Membership test: for each element of `self`, 1 if in `test_elements` (rank-1), else 0.
    pub fn isin(&self, test_elements: &ArrayI64) -> Result<ArrayI64> {
        if test_elements.rank() != 1 {
            return Err(Error::Shape("isin test_elements must be rank-1".into()));
        }
        let set: std::collections::BTreeSet<i64> =
            test_elements.as_slice().iter().copied().collect();
        let mut data = pool::take_uninit(self.len());
        for (i, &x) in self.as_slice().iter().enumerate() {
            data[i] = if set.contains(&x) { 1 } else { 0 };
        }
        Ok(Self::from_parts(self.shape.clone(), data))
    }

    /// Count occurrences of non-negative integers in rank-1 `self` into bins `0..minlength` (or auto).
    /// Negative values error. Matches NumPy `bincount` without weights.
    pub fn bincount(&self, minlength: usize) -> Result<ArrayI64> {
        if self.rank() != 1 {
            return Err(Error::Shape("bincount requires rank-1".into()));
        }
        let mut max_v: i64 = -1;
        for &x in self.as_slice() {
            if x < 0 {
                return Err(Error::Shape("bincount does not accept negative values".into()));
            }
            if x > max_v {
                max_v = x;
            }
        }
        let n = if max_v < 0 {
            minlength
        } else {
            (max_v as usize + 1).max(minlength)
        };
        let mut data = pool::take_zeroed(n);
        for &x in self.as_slice() {
            let i = x as usize;
            data[i] = data[i].wrapping_add(1);
        }
        Ok(Self::from_parts(Shape::from_len(n), data))
    }

    /// Insertion indices to maintain order (rank-1 sorted `self` is not required;
    /// we sort a copy of `self` for search — NumPy requires sorted `a`. **We require
    /// `self` sorted ascending**; unsorted is an error if not monotonic.
    pub fn searchsorted(&self, v: i64, side_right: bool) -> Result<usize> {
        if self.rank() != 1 {
            return Err(Error::Shape("searchsorted requires rank-1".into()));
        }
        let a = self.as_slice();
        for w in a.windows(2) {
            if w[0] > w[1] {
                return Err(Error::Shape("searchsorted requires non-decreasing array".into()));
            }
        }
        let r = if side_right {
            a.partition_point(|&x| x <= v)
        } else {
            a.partition_point(|&x| x < v)
        };
        Ok(r)
    }

    /// Searchsorted for each element of `values` (rank-1) → rank-1 indices.
    pub fn searchsorted_array(&self, values: &ArrayI64, side_right: bool) -> Result<ArrayI64> {
        if values.rank() != 1 {
            return Err(Error::Shape("searchsorted values must be rank-1".into()));
        }
        let mut data = pool::take_uninit(values.len());
        for (i, &v) in values.as_slice().iter().enumerate() {
            data[i] = self.searchsorted(v, side_right)? as i64;
        }
        Ok(Self::from_parts(Shape::from_len(values.len()), data))
    }

    /// Return a sorted copy (rank-1). `descending` optional.
    pub fn sort(&self, descending: bool) -> Result<ArrayI64> {
        if self.rank() != 1 {
            return Err(Error::Shape("sort requires rank-1".into()));
        }
        let mut v = self.as_slice().to_vec();
        if descending {
            v.sort_unstable_by(|a, b| b.cmp(a));
        } else {
            v.sort_unstable();
        }
        Ok(Self::from_parts(Shape::from_len(v.len()), v))
    }


    /// Shared view of the whole array.
    #[inline]
    /// `view`.
    pub fn view(&self) -> super::ArrayViewI64<'_> {
        super::ArrayViewI64::from_shape_slice(self.shape.clone(), self.data.as_ref())
    }

    /// Mutable view of the whole array (COW if buffer shared).
    #[inline]
    /// `view_mut`.
    pub fn view_mut(&mut self) -> super::ArrayViewMutI64<'_> {
        let shape = self.shape.clone();
        super::ArrayViewMutI64::from_shape_slice(shape, self.as_mut_slice())
    }

    /// Element-wise power with non-negative integer exponents (wrapping).
    /// Negative exponents error.
    pub fn power(&self, exponents: &ArrayI64) -> Result<ArrayI64> {
        self.same_shape(exponents)?;
        let mut data = pool::take_uninit(self.len());
        for i in 0..self.len() {
            let e = exponents.as_slice()[i];
            if e < 0 {
                return Err(Error::Shape("power: negative exponent".into()));
            }
            if e > u32::MAX as i64 {
                return Err(Error::Shape("power: exponent too large".into()));
            }
            data[i] = self.as_slice()[i].wrapping_pow(e as u32);
        }
        Ok(Self::from_parts(self.shape.clone(), data))
    }

    /// `(quotients, remainders)` for elementwise Euclidean-style Rust `/` and `%`.
    /// Divisor 0 → quot 0, rem 0.
    pub fn divmod(&self, other: &ArrayI64) -> Result<(ArrayI64, ArrayI64)> {
        self.same_shape(other)?;
        let mut q = pool::take_uninit(self.len());
        let mut r = pool::take_uninit(self.len());
        let a = self.as_slice();
        let b = other.as_slice();
        for i in 0..a.len() {
            if b[i] == 0 {
                q[i] = 0;
                r[i] = 0;
            } else {
                q[i] = a[i] / b[i];
                r[i] = a[i] % b[i];
            }
        }
        Ok((
            Self::from_parts(self.shape.clone(), q),
            Self::from_parts(self.shape.clone(), r),
        ))
    }

    /// Element-wise GCD (`0` if both zero; always non-negative result).
    pub fn gcd(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.same_shape(other)?;
        let mut data = pool::take_uninit(self.len());
        for i in 0..self.len() {
            data[i] = gcd_i64(self.as_slice()[i], other.as_slice()[i]);
        }
        Ok(Self::from_parts(self.shape.clone(), data))
    }

    /// Element-wise LCM (wrapping); `0` if either arg is 0.
    pub fn lcm(&self, other: &ArrayI64) -> Result<ArrayI64> {
        self.same_shape(other)?;
        let mut data = pool::take_uninit(self.len());
        for i in 0..self.len() {
            data[i] = lcm_i64(self.as_slice()[i], other.as_slice()[i]);
        }
        Ok(Self::from_parts(self.shape.clone(), data))
    }

    /// Population count (number of `1` bits) per element.
    pub fn count_ones(&self) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = a[i].count_ones() as i64;
            }
        })
    }

    /// Number of leading zeros per element (`i64::leading_zeros`).
    pub fn leading_zeros(&self) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = a[i].leading_zeros() as i64;
            }
        })
    }

    /// Number of trailing zeros per element.
    pub fn trailing_zeros(&self) -> ArrayI64 {
        self.owned_unary(|a, o| {
            for i in 0..a.len() {
                o[i] = a[i].trailing_zeros() as i64;
            }
        })
    }

    /// Deep copy alias matching `Array::to_owned_array`.
    pub fn to_owned_array(&self) -> ArrayI64 {
        self.clone()
    }
}

impl PartialEq for ArrayI64 {
    fn eq(&self, other: &Self) -> bool {
        self.shape.same_as(other.shape()) && self.as_slice() == other.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_arith_cast() {
        let a = ArrayI64::from_shape_slice(vec![3], &[1, 2, 3]).unwrap();
        let b = ArrayI64::from_shape_slice(vec![3], &[10, 20, 30]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.as_slice(), &[11, 22, 33]);
        assert_eq!(a.sum(), 6);
        let f = a.to_f64();
        assert_eq!(f.as_slice(), &[1.0, 2.0, 3.0]);
        let back = ArrayI64::from_f64(&f);
        assert_eq!(back.as_slice(), a.as_slice());
    }

    #[test]
    fn arange_and_axis() {
        let a = ArrayI64::arange(0, 5).unwrap();
        assert_eq!(a.as_slice(), &[0, 1, 2, 3, 4]);
        let m = ArrayI64::from_shape_slice(vec![2, 3], &[1, 2, 3, 4, 5, 6]).unwrap();
        let s = m.sum_axis(0).unwrap();
        assert_eq!(s.as_slice(), &[5, 7, 9]);
        let s1 = m.sum_axis(1).unwrap();
        assert_eq!(s1.as_slice(), &[6, 15]);
    }

    #[test]
    fn argsort_take_broadcast_row() {
        let a = ArrayI64::from_shape_slice(vec![4], &[3, 1, 4, 2]).unwrap();
        let idx = a.argsort(false).unwrap();
        assert_eq!(idx.as_slice(), &[1, 3, 0, 2]);
        let t = a.take(&idx).unwrap();
        assert_eq!(t.as_slice(), &[1, 2, 3, 4]);
        let m = ArrayI64::from_shape_slice(vec![2, 3], &[1, 1, 1, 2, 2, 2]).unwrap();
        let row = ArrayI64::from_shape_slice(vec![3], &[10, 20, 30]).unwrap();
        let r = m.add(&row).unwrap();
        assert_eq!(r.as_slice(), &[11, 21, 31, 12, 22, 32]);
    }

    #[test]
    fn concat_stack_where_sign() {
        let a = ArrayI64::from_shape_slice(vec![2], &[1, 2]).unwrap();
        let b = ArrayI64::from_shape_slice(vec![2], &[3, 4]).unwrap();
        let c = ArrayI64::concatenate(0, &[&a, &b]).unwrap();
        assert_eq!(c.as_slice(), &[1, 2, 3, 4]);
        let s = ArrayI64::stack(0, &[&a, &b]).unwrap();
        assert_eq!(s.dims(), &[2, 2]);
        let cond = ArrayI64::from_shape_slice(vec![2], &[1, 0]).unwrap();
        let w = ArrayI64::where_cond(&cond, &a, &b).unwrap();
        assert_eq!(w.as_slice(), &[1, 4]);
        assert_eq!(ArrayI64::from_shape_slice(vec![3], &[-2, 0, 5]).unwrap().sign().as_slice(), &[-1, 0, 1]);
        assert_eq!(a.var(0).unwrap(), 0.25);
    }

    #[test]
    fn unique_isin_bincount_bits() {
        let a = ArrayI64::from_shape_slice(vec![6], &[3, 1, 2, 1, 3, 2]).unwrap();
        assert_eq!(a.unique().unwrap().as_slice(), &[1, 2, 3]);
        let (u, c) = a.unique_counts().unwrap();
        assert_eq!(u.as_slice(), &[1, 2, 3]);
        assert_eq!(c.as_slice(), &[2, 2, 2]);
        let m = a.isin(&ArrayI64::from_shape_slice(vec![2], &[1, 9]).unwrap()).unwrap();
        assert_eq!(m.as_slice(), &[0, 1, 0, 1, 0, 0]);
        let b = ArrayI64::from_shape_slice(vec![5], &[0, 1, 1, 3, 0]).unwrap().bincount(0).unwrap();
        assert_eq!(b.as_slice(), &[2, 2, 0, 1]);
        let s = ArrayI64::from_shape_slice(vec![4], &[1, 3, 5, 7]).unwrap();
        assert_eq!(s.searchsorted(4, false).unwrap(), 2);
        assert_eq!(s.searchsorted(3, true).unwrap(), 2);
        assert_eq!(a.sort(false).unwrap().as_slice(), &[1, 1, 2, 2, 3, 3]);
        let x = ArrayI64::from_shape_slice(vec![2], &[0b1100, 0b1010]).unwrap();
        let y = ArrayI64::from_shape_slice(vec![2], &[0b1010, 0b1100]).unwrap();
        assert_eq!(x.bitand(&y).unwrap().as_slice(), &[0b1000, 0b1000]);
        assert_eq!(x.rem_scalar(5).as_slice(), &[2, 0]); // 12%5=2, 10%5=0
    }

    #[test]
    fn view_power_divmod_gcd() {
        let mut a = ArrayI64::from_shape_slice(vec![2, 2], &[2, 4, 6, 8]).unwrap();
        assert_eq!(a.view().get(&[0, 1]).unwrap(), 4);
        a.view_mut().set(&[1, 0], 7).unwrap();
        assert_eq!(a.get(&[1, 0]).unwrap(), 7);
        let e = ArrayI64::from_shape_slice(vec![3], &[2, 3, 1]).unwrap();
        let b = ArrayI64::from_shape_slice(vec![3], &[2, 2, 5]).unwrap();
        assert_eq!(b.power(&e).unwrap().as_slice(), &[4, 8, 5]);
        let (q, r) = ArrayI64::from_shape_slice(vec![2], &[17, 20])
            .unwrap()
            .divmod(&ArrayI64::from_shape_slice(vec![2], &[5, 6]).unwrap())
            .unwrap();
        assert_eq!(q.as_slice(), &[3, 3]);
        assert_eq!(r.as_slice(), &[2, 2]);
        let g = ArrayI64::from_shape_slice(vec![2], &[12, 7])
            .unwrap()
            .gcd(&ArrayI64::from_shape_slice(vec![2], &[8, 0]).unwrap())
            .unwrap();
        assert_eq!(g.as_slice(), &[4, 7]);
        let l = ArrayI64::from_shape_slice(vec![1], &[4])
            .unwrap()
            .lcm(&ArrayI64::from_shape_slice(vec![1], &[6]).unwrap())
            .unwrap();
        assert_eq!(l.as_slice(), &[12]);
        assert_eq!(
            ArrayI64::from_shape_slice(vec![2], &[0b1011, 0])
                .unwrap()
                .count_ones()
                .as_slice(),
            &[3, 0]
        );
        // empty + matmul wrap smoke
        let empty = ArrayI64::zeros(vec![0]).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.unique().unwrap().len(), 0);
        let big = ArrayI64::full(vec![2, 2], i64::MAX).unwrap();
        let _ = crate::linalg::i64_ops::matmul(&big, &ArrayI64::eye(2).unwrap()).unwrap();
        // searchsorted unsorted errors
        assert!(ArrayI64::from_shape_slice(vec![3], &[1, 3, 2])
            .unwrap()
            .searchsorted(2, false)
            .is_err());
        // scalar ops
        let s = &ArrayI64::from_shape_slice(vec![2], &[1, 2]).unwrap() + 10;
        assert_eq!(s.as_slice(), &[11, 12]);
    }

    #[test]
    fn arrow_roundtrip() {
        let a = ArrayI64::from_shape_slice(vec![2, 2], &[1, 2, 3, 4]).unwrap();
        let ar = a.to_arrow();
        let b = ArrayI64::from_arrow(&ar, vec![2, 2]).unwrap();
        assert_eq!(a, b);
    }
}
