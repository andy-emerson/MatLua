//! Owned dense `i64` arrays (row-major, n-D). M7 — correctness-first.

use std::sync::Arc;

use arrow_array::{Array as ArrowArray, Int64Array};

use super::dtype::DType;
use super::kernels_i64 as kernels;
use super::pool_i64 as pool;
use super::shape::{broadcast_shapes, Shape};
use super::Array;
use crate::error::{Error, Result};

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

    /// Argsort indices as `i64` array (0-based).
    pub fn argsort(&self, descending: bool) -> Result<ArrayI64> {
        let n = self.len();
        let mut idx = vec![0usize; n];
        kernels::argsort_indices(self.as_slice(), descending, &mut idx);
        let mut data = pool::take_uninit(n);
        for i in 0..n {
            data[i] = idx[i] as i64;
        }
        Ok(Self::from_parts(Shape::from_len(n), data))
    }

    /// Take elements by 0-based integer indices (rank-1).
    pub fn take(&self, indices: &ArrayI64) -> Result<ArrayI64> {
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
                let src = self.as_slice();
                let mut data = pool::take_uninit(rows * cols);
                for i in 0..rows {
                    for j in 0..cols {
                        data[j * rows + i] = src[i * cols + j];
                    }
                }
                Ok(Self::from_parts(Shape::matrix(cols, rows)?, data))
            }
            r => Err(Error::Shape(format!(
                "transpose expects rank 1 or 2, got {r}"
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
    fn arrow_roundtrip() {
        let a = ArrayI64::from_shape_slice(vec![2, 2], &[1, 2, 3, 4]).unwrap();
        let ar = a.to_arrow();
        let b = ArrayI64::from_arrow(&ar, vec![2, 2]).unwrap();
        assert_eq!(a, b);
    }
}
