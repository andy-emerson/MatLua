//! Convert between MatLua [`Array`](crate::Array) (row-major) and faer [`Mat`].

use faer::Mat;

use crate::array::Array;
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

/// Copy a MatLua array into a faer column-major [`Mat`].
pub(crate) fn array_to_mat(a: &Array) -> Result<Mat<f64>> {
    let (nrows, ncols) = array_as_matrix_dims(a)?;
    let data = a.as_slice();
    if data.len() != nrows.saturating_mul(ncols) {
        return Err(Error::shape("internal layout length mismatch"));
    }
    // Row-major source → (i, j) at i * ncols + j
    Ok(Mat::from_fn(nrows, ncols, |i, j| data[i * ncols + j]))
}

/// Copy a faer [`Mat`] into a MatLua row-major [`Array`].
///
/// - `n × 1` → rank-1 shape `(n,)` when `prefer_vector` is true  
/// - otherwise rank-2 shape `(m, n)`
pub(crate) fn mat_to_array(m: &Mat<f64>, prefer_vector: bool) -> Result<Array> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let mut data = vec![0.0; nrows * ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            data[i * ncols + j] = m[(i, j)];
        }
    }
    if prefer_vector && ncols == 1 {
        Array::from_shape_vec(vec![nrows], data)
    } else if prefer_vector && nrows == 1 {
        Array::from_shape_vec(vec![ncols], data)
    } else {
        Array::from_shape_vec(vec![nrows, ncols], data)
    }
}

/// Copy a matrix-like faer view into an owned MatLua array (always rank-2 unless 0-size).
pub(crate) fn matref_to_array(m: faer::MatRef<'_, f64>) -> Result<Array> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let mut data = vec![0.0; nrows * ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            data[i * ncols + j] = m[(i, j)];
        }
    }
    Array::from_shape_vec(vec![nrows, ncols], data)
}
