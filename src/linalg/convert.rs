//! Convert between MatLua [`Array`](crate::Array) (row-major) and faer matrices.
//!
//! Inputs: zero-copy [`MatRef`] over contiguous row-major storage.
//! Outputs: owned row-major [`Array`] (copy out).

use faer::{Mat, MatMut, MatRef};

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

/// Copy a faer matrix view into a MatLua row-major [`Array`].
///
/// - `n × 1` → rank-1 shape `(n,)` when `prefer_vector` is true  
/// - `1 × n` → rank-1 shape `(n,)` when `prefer_vector` is true  
/// - otherwise rank-2 shape `(m, n)`
pub(crate) fn matref_to_array(m: MatRef<'_, f64>, prefer_vector: bool) -> Result<Array> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let n = nrows.saturating_mul(ncols);
    let mut data = vec![0.0; n];
    if n > 0 {
        let mut out = MatMut::from_row_major_slice_mut(&mut data, nrows, ncols);
        out.copy_from(m);
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
