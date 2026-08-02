//! Dense linear algebra on [`ArrayI64`](crate::array::ArrayI64).
//!
//! Integer path (not faer/`f64`). Arithmetic is **wrapping** `i64`, matching
//! the rest of the `i64` surface. Mathematically, integer×integer→integer;
//! fixed-width storage may wrap.

use crate::array::{pool_i64, ArrayI64, Shape};
use crate::error::{Error, Result};

/// Interpret rank-1 as column vector `(n, 1)`; rank-2 as matrix.
fn as_matrix_dims(a: &ArrayI64) -> Result<(usize, usize)> {
    match a.rank() {
        1 => Ok((a.len(), 1)),
        2 => Ok((a.dims()[0], a.dims()[1])),
        r => Err(Error::shape(format!(
            "linalg expects rank 1 or 2, got rank {r}"
        ))),
    }
}

fn matmul_result(data: Vec<i64>, rows: usize, cols: usize, prefer_vec: bool) -> Result<ArrayI64> {
    if prefer_vec && cols == 1 {
        Ok(ArrayI64::from_parts(Shape::from_len(rows), data))
    } else if prefer_vec && rows == 1 {
        Ok(ArrayI64::from_parts(Shape::from_len(cols), data))
    } else {
        Ok(ArrayI64::from_parts(Shape::matrix(rows, cols)?, data))
    }
}

/// Matrix product `a @ b` with wrapping `i64` accumulation.
///
/// Rank-1 operands are columns. Result is rank-1 when an operand was rank-1 and
/// the product is a single column (or row×col → scalar length-1 as rank-1).
pub fn matmul(a: &ArrayI64, b: &ArrayI64) -> Result<ArrayI64> {
    let (am, an) = as_matrix_dims(a)?;
    let (bm, bn) = as_matrix_dims(b)?;
    if an != bm {
        return Err(Error::shape(format!(
            "matmul shape mismatch: ({am}, {an}) vs ({bm}, {bn})"
        )));
    }
    let prefer_vec = b.rank() == 1 || (a.rank() == 1 && bn == 1);
    let mut data = pool_i64::take_uninit(am.saturating_mul(bn));
    let aa = a.as_slice();
    let bb = b.as_slice();
    // a is am×an row-major; b is bm×bn with bm==an
    for i in 0..am {
        for j in 0..bn {
            let mut s: i64 = 0;
            for k in 0..an {
                let av = aa[i * an + k];
                let bv = if b.rank() == 1 {
                    // b is column: index k
                    bb[k]
                } else {
                    bb[k * bn + j]
                };
                s = s.wrapping_add(av.wrapping_mul(bv));
            }
            data[i * bn + j] = s;
        }
    }
    // When b is rank-1, bn=1, layout is fine as am×1
    matmul_result(data, am, bn, prefer_vec)
}

/// `aᵀ @ b` with wrapping `i64`.
pub fn matmul_at(a: &ArrayI64, b: &ArrayI64) -> Result<ArrayI64> {
    let (am, an) = as_matrix_dims(a)?;
    let (bm, bn) = as_matrix_dims(b)?;
    if am != bm {
        return Err(Error::shape(format!(
            "matmul_at shape mismatch: a is ({am}, {an}), b is ({bm}, {bn})"
        )));
    }
    let prefer_vec = b.rank() == 1;
    // result an × bn; aᵀ is an×am
    let mut data = pool_i64::take_uninit(an.saturating_mul(bn));
    let aa = a.as_slice();
    let bb = b.as_slice();
    for i in 0..an {
        for j in 0..bn {
            let mut s: i64 = 0;
            for k in 0..am {
                let av = aa[k * an + i]; // a[k,i]
                let bv = if b.rank() == 1 {
                    bb[k]
                } else {
                    bb[k * bn + j]
                };
                s = s.wrapping_add(av.wrapping_mul(bv));
            }
            data[i * bn + j] = s;
        }
    }
    matmul_result(data, an, bn, prefer_vec)
}

/// `a @ bᵀ` with wrapping `i64`.
pub fn matmul_bt(a: &ArrayI64, b: &ArrayI64) -> Result<ArrayI64> {
    let (am, an) = as_matrix_dims(a)?;
    let (bm, bn) = as_matrix_dims(b)?;
    if an != bn {
        return Err(Error::shape(format!(
            "matmul_bt shape mismatch: a is ({am}, {an}), b is ({bm}, {bn}); need equal column counts"
        )));
    }
    // result am × bm
    let mut data = pool_i64::take_uninit(am.saturating_mul(bm));
    let aa = a.as_slice();
    let bb = b.as_slice();
    for i in 0..am {
        for j in 0..bm {
            let mut s: i64 = 0;
            for k in 0..an {
                let av = aa[i * an + k];
                let bv = bb[j * bn + k]; // b[j,k]
                s = s.wrapping_add(av.wrapping_mul(bv));
            }
            data[i * bm + j] = s;
        }
    }
    Ok(ArrayI64::from_parts(Shape::matrix(am, bm)?, data))
}

/// Dot product of two rank-1 arrays (wrapping `i64`).
pub fn dot(a: &ArrayI64, b: &ArrayI64) -> Result<i64> {
    if a.rank() != 1 || b.rank() != 1 {
        return Err(Error::shape("dot expects two rank-1 arrays"));
    }
    if a.len() != b.len() {
        return Err(Error::shape(format!(
            "dot length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let mut s: i64 = 0;
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        s = s.wrapping_add(x.wrapping_mul(*y));
    }
    Ok(s)
}

/// Euclidean (Frobenius) norm as `f64` (sqrt of sum of squares; squares wrap in `i64` then cast).
pub fn norm(a: &ArrayI64) -> Result<f64> {
    let mut ss: i64 = 0;
    for &x in a.as_slice() {
        ss = ss.wrapping_add(x.wrapping_mul(x));
    }
    Ok((ss as f64).sqrt())
}

/// Transpose (delegates to [`ArrayI64::transpose`]).
pub fn transpose(a: &ArrayI64) -> Result<ArrayI64> {
    a.transpose()
}

/// Identity (delegates to [`ArrayI64::eye`]).
pub fn eye(n: usize) -> Result<ArrayI64> {
    ArrayI64::eye(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::ArrayI64;

    #[test]
    fn matmul_2x2_and_vec() {
        let a = ArrayI64::from_shape_slice(vec![2, 2], &[1, 2, 3, 4]).unwrap();
        let b = ArrayI64::from_shape_slice(vec![2, 2], &[5, 6, 7, 8]).unwrap();
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.as_slice(), &[19, 22, 43, 50]);
        let v = ArrayI64::from_shape_slice(vec![2], &[1, 1]).unwrap();
        let av = matmul(&a, &v).unwrap();
        assert_eq!(av.rank(), 1);
        assert_eq!(av.as_slice(), &[3, 7]);
    }

    #[test]
    fn matmul_at_bt_dot() {
        let x = ArrayI64::from_shape_slice(vec![3, 2], &[1, 0, 1, 1, 1, 2]).unwrap();
        let y = ArrayI64::from_shape_slice(vec![3], &[1, 2, 3]).unwrap();
        let xty = matmul_at(&x, &y).unwrap();
        // Xᵀ y = [1+2+3, 0+2+6] = [6, 8]
        assert_eq!(xty.as_slice(), &[6, 8]);
        let a = ArrayI64::from_shape_slice(vec![2, 3], &[1, 2, 3, 4, 5, 6]).unwrap();
        let b = ArrayI64::from_shape_slice(vec![2, 3], &[1, 0, 0, 0, 1, 0]).unwrap();
        let abt = matmul_bt(&a, &b).unwrap();
        // a @ bᵀ : 2×2
        assert_eq!(abt.dims(), &[2, 2]);
        let d = dot(
            &ArrayI64::from_shape_slice(vec![3], &[1, 2, 3]).unwrap(),
            &ArrayI64::from_shape_slice(vec![3], &[4, 5, 6]).unwrap(),
        )
        .unwrap();
        assert_eq!(d, 32);
    }
}
