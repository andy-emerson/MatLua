//! Dense linear algebra on [`ArrayI64`](crate::array::ArrayI64).
//!
//! Integer path (not faer/`f64`). Arithmetic is **wrapping** `i64`, matching
//! the rest of the `i64` surface. Mathematically, integer×integer→integer;
//! fixed-width storage may wrap.
//!
//! M7.c: blocked/`ikj` matmul and unrolled dot for cache locality (still wrapping).

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


/// Blocked wrapping GEMM into zeroed `data` (am×bn row-major).
/// Large products parallelize over output rows (Rayon).
fn gemm_blocked(am: usize, an: usize, bn: usize, aa: &[i64], bb: &[i64], data: &mut [i64]) {
    const BS: usize = 48;
    let flops = am.saturating_mul(an).saturating_mul(bn);
    // Parallel threshold ~ 64³ work units
    if flops >= 64 * 64 * 64 && am >= 8 {
        use rayon::prelude::*;
        data.par_chunks_mut(bn).enumerate().for_each(|(i, c_row)| {
            let mut k0 = 0;
            while k0 < an {
                let k1 = (k0 + BS).min(an);
                for k in k0..k1 {
                    let aik = aa[i * an + k];
                    if aik == 0 {
                        continue;
                    }
                    let b_row = &bb[k * bn..(k + 1) * bn];
                    let mut j = 0;
                    while j + 4 <= bn {
                        c_row[j] = c_row[j].wrapping_add(aik.wrapping_mul(b_row[j]));
                        c_row[j + 1] = c_row[j + 1].wrapping_add(aik.wrapping_mul(b_row[j + 1]));
                        c_row[j + 2] = c_row[j + 2].wrapping_add(aik.wrapping_mul(b_row[j + 2]));
                        c_row[j + 3] = c_row[j + 3].wrapping_add(aik.wrapping_mul(b_row[j + 3]));
                        j += 4;
                    }
                    while j < bn {
                        c_row[j] = c_row[j].wrapping_add(aik.wrapping_mul(b_row[j]));
                        j += 1;
                    }
                }
                k0 = k1;
            }
        });
        return;
    }
    let mut i0 = 0;
    while i0 < am {
        let i1 = (i0 + BS).min(am);
        let mut j0 = 0;
        while j0 < bn {
            let j1 = (j0 + BS).min(bn);
            let mut k0 = 0;
            while k0 < an {
                let k1 = (k0 + BS).min(an);
                for i in i0..i1 {
                    let c_row = &mut data[i * bn + j0..i * bn + j1];
                    for k in k0..k1 {
                        let aik = aa[i * an + k];
                        if aik == 0 {
                            continue;
                        }
                        let b_row = &bb[k * bn + j0..k * bn + j1];
                        let mut j = 0;
                        let w = j1 - j0;
                        while j + 4 <= w {
                            c_row[j] = c_row[j].wrapping_add(aik.wrapping_mul(b_row[j]));
                            c_row[j + 1] = c_row[j + 1].wrapping_add(aik.wrapping_mul(b_row[j + 1]));
                            c_row[j + 2] = c_row[j + 2].wrapping_add(aik.wrapping_mul(b_row[j + 2]));
                            c_row[j + 3] = c_row[j + 3].wrapping_add(aik.wrapping_mul(b_row[j + 3]));
                            j += 4;
                        }
                        while j < w {
                            c_row[j] = c_row[j].wrapping_add(aik.wrapping_mul(b_row[j]));
                            j += 1;
                        }
                    }
                }
                k0 = k1;
            }
            j0 = j1;
        }
        i0 = i1;
    }
}


/// Matrix product `a @ b` with wrapping `i64` accumulation.
///
/// Rank-1 operands are columns. Result is rank-1 when an operand was rank-1 and
/// the product is a single column (or row×col → scalar length-1 as rank-1).
///
/// Uses `i–k–j` accumulation (row of `a` streams; inner walk over columns of `b`)
/// so `b`'s rows stay hot in cache for modest dense sizes.
pub fn matmul(a: &ArrayI64, b: &ArrayI64) -> Result<ArrayI64> {
    let (am, an) = as_matrix_dims(a)?;
    let (bm, bn) = as_matrix_dims(b)?;
    if an != bm {
        return Err(Error::shape(format!(
            "matmul shape mismatch: ({am}, {an}) vs ({bm}, {bn})"
        )));
    }
    let prefer_vec = b.rank() == 1 || (a.rank() == 1 && bn == 1);
    let n_out = am.saturating_mul(bn);
    let mut data = pool_i64::take_zeroed(n_out);
    let aa = a.as_slice();
    let bb = b.as_slice();

    if b.rank() == 1 {
        // a (am×an) @ b (an×1)
        for i in 0..am {
            let mut s: i64 = 0;
            let row = &aa[i * an..(i + 1) * an];
            let mut k = 0;
            while k + 4 <= an {
                s = s.wrapping_add(row[k].wrapping_mul(bb[k]));
                s = s.wrapping_add(row[k + 1].wrapping_mul(bb[k + 1]));
                s = s.wrapping_add(row[k + 2].wrapping_mul(bb[k + 2]));
                s = s.wrapping_add(row[k + 3].wrapping_mul(bb[k + 3]));
                k += 4;
            }
            while k < an {
                s = s.wrapping_add(row[k].wrapping_mul(bb[k]));
                k += 1;
            }
            data[i] = s;
        }
    } else {
        gemm_blocked(am, an, bn, aa, bb, &mut data);
    }
    matmul_result(data, am, bn, prefer_vec)
}

/// GEMM into preallocated rank-2 `out` with shape `(am, bn)`. Wrapping `i64`.
/// Does not collapse to rank-1 even when `b` is a column vector (`bn == 1`).
pub fn matmul_out(a: &ArrayI64, b: &ArrayI64, out: &mut ArrayI64) -> Result<()> {
    let (am, an) = as_matrix_dims(a)?;
    let (bm, bn) = as_matrix_dims(b)?;
    if an != bm {
        return Err(Error::shape(format!(
            "matmul shape mismatch: ({am}, {an}) vs ({bm}, {bn})"
        )));
    }
    if out.rank() != 2 || out.dims() != [am, bn] {
        return Err(Error::shape(format!(
            "matmul_out expects out shape ({am}, {bn}), got {:?}",
            out.dims()
        )));
    }
    let aa = a.as_slice();
    let bb = b.as_slice();
    let data = out.as_mut_slice();
    data.fill(0);
    if b.rank() == 1 {
        for i in 0..am {
            let mut s: i64 = 0;
            let row = &aa[i * an..(i + 1) * an];
            for k in 0..an {
                s = s.wrapping_add(row[k].wrapping_mul(bb[k]));
            }
            data[i] = s;
        }
    } else {
        gemm_blocked(am, an, bn, aa, bb, data);
    }
    Ok(())
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
    let mut data = pool_i64::take_zeroed(an.saturating_mul(bn));
    let aa = a.as_slice();
    let bb = b.as_slice();
    // C[i,j] = sum_k a[k,i] * b[k,j]  — stream k outer for a's columns
    if b.rank() == 1 {
        for i in 0..an {
            let mut s: i64 = 0;
            for k in 0..am {
                s = s.wrapping_add(aa[k * an + i].wrapping_mul(bb[k]));
            }
            data[i] = s;
        }
    } else {
        for k in 0..am {
            let b_row = &bb[k * bn..(k + 1) * bn];
            for i in 0..an {
                let aki = aa[k * an + i];
                if aki == 0 {
                    continue;
                }
                let c_row = &mut data[i * bn..(i + 1) * bn];
                for j in 0..bn {
                    c_row[j] = c_row[j].wrapping_add(aki.wrapping_mul(b_row[j]));
                }
            }
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
    let mut data = pool_i64::take_zeroed(am.saturating_mul(bm));
    let aa = a.as_slice();
    let bb = b.as_slice();
    for i in 0..am {
        let a_row = &aa[i * an..(i + 1) * an];
        for j in 0..bm {
            let b_row = &bb[j * bn..(j + 1) * bn];
            let mut s: i64 = 0;
            let mut k = 0;
            while k + 4 <= an {
                s = s.wrapping_add(a_row[k].wrapping_mul(b_row[k]));
                s = s.wrapping_add(a_row[k + 1].wrapping_mul(b_row[k + 1]));
                s = s.wrapping_add(a_row[k + 2].wrapping_mul(b_row[k + 2]));
                s = s.wrapping_add(a_row[k + 3].wrapping_mul(b_row[k + 3]));
                k += 4;
            }
            while k < an {
                s = s.wrapping_add(a_row[k].wrapping_mul(b_row[k]));
                k += 1;
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
    let x = a.as_slice();
    let y = b.as_slice();
    let mut s: i64 = 0;
    let mut i = 0;
    let n = x.len();
    while i + 4 <= n {
        s = s.wrapping_add(x[i].wrapping_mul(y[i]));
        s = s.wrapping_add(x[i + 1].wrapping_mul(y[i + 1]));
        s = s.wrapping_add(x[i + 2].wrapping_mul(y[i + 2]));
        s = s.wrapping_add(x[i + 3].wrapping_mul(y[i + 3]));
        i += 4;
    }
    while i < n {
        s = s.wrapping_add(x[i].wrapping_mul(y[i]));
        i += 1;
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
        assert_eq!(xty.as_slice(), &[6, 8]);
        let a = ArrayI64::from_shape_slice(vec![2, 3], &[1, 2, 3, 4, 5, 6]).unwrap();
        let b = ArrayI64::from_shape_slice(vec![2, 3], &[1, 0, 0, 0, 1, 0]).unwrap();
        let abt = matmul_bt(&a, &b).unwrap();
        assert_eq!(abt.dims(), &[2, 2]);
        let d = dot(
            &ArrayI64::from_shape_slice(vec![3], &[1, 2, 3]).unwrap(),
            &ArrayI64::from_shape_slice(vec![3], &[4, 5, 6]).unwrap(),
        )
        .unwrap();
        assert_eq!(d, 32);
    }

    #[test]
    fn matmul_larger_identity() {
        let a = ArrayI64::from_shape_slice(vec![4, 3], &(1..=12).collect::<Vec<_>>()).unwrap();
        let i = ArrayI64::eye(3).unwrap();
        let c = matmul(&a, &i).unwrap();
        assert_eq!(c.as_slice(), a.as_slice());
    }
}
