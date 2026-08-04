//! Convert between MatLua [`Array`](crate::Array) (row-major) and faer matrices.
//!
//! Inputs: zero-copy [`MatRef`] over contiguous row-major storage.
//! Outputs: owned row-major [`Array`] (copy out).

use faer::{Mat, MatRef};

use crate::array::{Array, Shape};
use crate::error::{Error, Result};

/// Interpret an array as a matrix for dense LA.
///
/// - rank 2: shape `(m, n)` → `m × n` matrix  
/// - rank 1: shape `(n,)` → `n × 1` column vector
pub(crate) fn array_as_matrix_dims(a: &Array) -> Result<(usize, usize)> {
    match a.rank() {
        1 => Ok((a.dims()[0], 1)),
        2 => Ok((a.dims()[0], a.dims()[1])),
        r => Err(Error::shape(format!(
            "linear algebra expects rank 1 or 2, got rank {r}"
        ))),
    }
}

/// Zero-copy faer view over a contiguous row-major MatLua array.
///
/// Rank-1 arrays are viewed as `n × 1` columns.
pub(crate) fn array_as_mat_ref(a: &Array) -> Result<MatRef<'_, f64>> {
    let (nrows, ncols) = array_as_matrix_dims(a)?;
    let data = a.as_slice();
    let n = nrows.saturating_mul(ncols);
    if data.len() != n {
        return Err(Error::shape("internal layout length mismatch"));
    }
    // faer panics if nrows*ncols != len; empty 0×k / k×0 with len 0 is fine.
    Ok(MatRef::from_row_major_slice(data, nrows, ncols))
}

/// Column-major scratch copy of an array, for faer **factorization** inputs.
///
/// faer's panel kernels are column-major-optimized; a row-major `MatRef`
/// makes every column access strided. The O(mn) blocked transpose here is
/// amortized against O(n³) factorization work — analyzed (DESIGN §3.26),
/// and measured on the 2026-08 bench container: LU −26%, full QR −8% at
/// n=2048 vs row-major views. GEMM paths keep zero-copy views (packing
/// already absorbs layout).
///
/// The buffer comes from the thread-local pool and returns to it on drop, so
/// repeated factorizations reuse capacity instead of churning the allocator.
pub(crate) struct ColMajor {
    buf: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl ColMajor {
    /// Zero-copy faer view over the column-major scratch copy.
    #[inline]
    pub(crate) fn view(&self) -> MatRef<'_, f64> {
        MatRef::from_column_major_slice(&self.buf, self.rows, self.cols)
    }
}

impl Drop for ColMajor {
    fn drop(&mut self) {
        crate::array::pool_recycle(std::mem::take(&mut self.buf));
    }
}

/// Build a [`ColMajor`] scratch copy of `a` (see the type's documentation).
pub(crate) fn array_to_colmajor(a: &Array) -> Result<ColMajor> {
    let (rows, cols) = array_as_matrix_dims(a)?;
    let src = a.as_slice();
    if src.len() != rows.saturating_mul(cols) {
        return Err(Error::shape("internal layout length mismatch"));
    }
    let mut buf = crate::array::pool_try_take_uninit(src.len())?;
    // dst[j*rows + i] = src[i*cols + j] — the shared blocked transpose
    // produces exactly the column-major image.
    super::blocked_transpose(src, rows, cols, &mut buf);
    Ok(ColMajor { buf, rows, cols })
}

/// Copy a faer matrix view into a MatLua row-major [`Array`].
///
/// - `n × 1` → rank-1 shape `(n,)` when `prefer_vector` is true
/// - `1 × n` → rank-1 shape `(n,)` when `prefer_vector` is true
/// - otherwise rank-2 shape `(m, n)`
///
/// The copy is a 32×32 **tiled gather**: faer results are column-major, and
/// faer's `copy_from` into a row-major dest degrades to a strided
/// element-by-element walk (measured: ~40% of the whole user-visible `qr`
/// time at n=2048 on the 2026-08 bench container). Tiles keep both sides
/// cache-resident whatever the source strides.
pub(crate) fn matref_to_array(m: MatRef<'_, f64>, prefer_vector: bool) -> Result<Array> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let n = nrows.saturating_mul(ncols);
    let mut data = crate::array::pool_try_take_uninit(n)?;
    if n > 0 {
        const BS: usize = 32;
        let rs = m.row_stride();
        let cs = m.col_stride();
        let p = m.as_ptr();
        let mut i0 = 0;
        while i0 < nrows {
            let i1 = (i0 + BS).min(nrows);
            let mut j0 = 0;
            while j0 < ncols {
                let j1 = (j0 + BS).min(ncols);
                for i in i0..i1 {
                    let row = i * ncols;
                    let base = i as isize * rs;
                    for j in j0..j1 {
                        // SAFETY: i < nrows and j < ncols, so the offset is a
                        // valid element of `m` by MatRef's own invariant.
                        data[row + j] = unsafe { *p.offset(base + j as isize * cs) };
                    }
                }
                j0 = j1;
            }
            i0 = i1;
        }
    }
    if prefer_vector && ncols == 1 {
        Ok(Array::from_parts(Shape::from_len(nrows), data))
    } else if prefer_vector && nrows == 1 {
        Ok(Array::from_parts(Shape::from_len(ncols), data))
    } else {
        Ok(Array::from_parts(Shape::matrix(nrows, ncols)?, data))
    }
}

/// Copy a faer owned [`Mat`] into a MatLua row-major [`Array`].
pub(crate) fn mat_to_array(m: &Mat<f64>, prefer_vector: bool) -> Result<Array> {
    matref_to_array(m.as_ref(), prefer_vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_ref_views_row_major_without_reordering() {
        // Row-major [1,2,3,4] as 2×2 is [[1,2],[3,4]].
        let a = Array::from_shape_slice(vec![2, 2], &[1., 2., 3., 4.]).unwrap();
        let m = array_as_mat_ref(&a).unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn rank1_is_column() {
        let a = Array::from_shape_slice(vec![3], &[1., 2., 3.]).unwrap();
        let m = array_as_mat_ref(&a).unwrap();
        assert_eq!((m.nrows(), m.ncols()), (3, 1));
        assert_eq!(m[(2, 0)], 3.0);
    }
}
