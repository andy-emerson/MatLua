//! Dense linear algebra on rank-1/2 [`Array`] values (faer-backed).
//!
//! Public functions take and return MatLua [`Array`]s. Inputs are viewed as
//! faer [`MatRef`](faer::MatRef) over contiguous row-major storage (zero-copy).
//! [`matmul`] writes GEMM output straight into a row-major buffer; other
//! results still pack out from faer into owned row-major arrays.
//!
//! # Rank conventions
//!
//! | Input | Interpretation |
//! |-------|----------------|
//! | rank 2, shape `(m, n)` | `m × n` matrix |
//! | rank 1, shape `(n,)` | `n × 1` column vector |
//!
//! Matrix×vector [`matmul`] returns rank-1. [`solve`] preserves the rank style of `b`.
//! Use [`dot`] for the inner product of two equal-length vectors.

mod convert;

use faer::linalg::matmul::matmul as faer_matmul;
use faer::linalg::solvers::Solve;
use faer::{get_global_parallelism, Accum, MatMut, Par, Side};

use crate::array::kernels;
use crate::array::{Array, Shape};
use crate::error::{Error, Result};

use convert::{array_as_mat_ref, array_as_matrix_dims, mat_to_array, matref_to_array};

/// Parallelism for GEMM: sequential for tiny products, otherwise faer's global
/// setting (Rayon by default when the `rayon` feature is on).
#[inline]
fn matmul_par(m: usize, n: usize, k: usize) -> Par {
    // ~ n³ work proxy; below ~128³ rayon overhead often dominates.
    let work = (m as u64)
        .saturating_mul(n as u64)
        .saturating_mul(k as u64);
    if work < 128u64 * 128 * 128 {
        Par::Seq
    } else {
        get_global_parallelism()
    }
}

/// Build an owned Array from a filled row-major buffer produced by GEMM.
#[inline]
fn matmul_result(
    data: Vec<f64>,
    nrows: usize,
    ncols: usize,
    prefer_vector: bool,
) -> Result<Array> {
    if prefer_vector && ncols == 1 {
        Ok(Array::from_parts(Shape::from_len(nrows), data))
    } else if prefer_vector && nrows == 1 {
        Ok(Array::from_parts(Shape::from_len(ncols), data))
    } else {
        Ok(Array::from_parts(Shape::matrix(nrows, ncols)?, data))
    }
}

/// Transpose a rank-1 or rank-2 array.
///
/// Rank-1 inputs become shape `(1, n)` row matrices (rank 2).
/// Rank-2 uses a blocked out-of-place transpose into a pooled buffer (O(mn),
/// cache-friendly for large n — not faer pack-out).
pub fn transpose(a: &Array) -> Result<Array> {
    match a.rank() {
        1 => {
            // Column (n,) → row matrix (1, n).
            let n = a.len();
            let mut data = crate::array::pool_take_uninit(n);
            data.copy_from_slice(a.as_slice());
            Ok(Array::from_parts(Shape::matrix(1, n)?, data))
        }
        2 => {
            let rows = a.dims()[0];
            let cols = a.dims()[1];
            let src = a.as_slice();
            let mut data = crate::array::pool_take_uninit(rows.saturating_mul(cols));
            blocked_transpose(src, rows, cols, &mut data);
            Ok(Array::from_parts(Shape::matrix(cols, rows)?, data))
        }
        r => Err(Error::shape(format!(
            "transpose expects rank 1 or 2, got rank {r}"
        ))),
    }
}

/// Cache-blocked out-of-place transpose: `dst` is `cols × rows` row-major.
/// Block size chosen for L1; still O(rows*cols) and better for large matrices.
fn blocked_transpose(src: &[f64], rows: usize, cols: usize, dst: &mut [f64]) {
    debug_assert_eq!(src.len(), rows.saturating_mul(cols));
    debug_assert_eq!(dst.len(), rows.saturating_mul(cols));
    const BS: usize = 32;
    let mut i0 = 0;
    while i0 < rows {
        let i1 = (i0 + BS).min(rows);
        let mut j0 = 0;
        while j0 < cols {
            let j1 = (j0 + BS).min(cols);
            for i in i0..i1 {
                let src_row = i * cols;
                for j in j0..j1 {
                    // dst is cols×rows: index (j, i) → j * rows + i
                    dst[j * rows + i] = src[src_row + j];
                }
            }
            j0 = j1;
        }
        i0 = i1;
    }
}

/// Matrix product `a @ b` (math notation).
///
/// Shapes: `(m, k) × (k, n) → (m, n)`. Rank-1 operands are treated as columns.
/// A matrix–vector product returns a rank-1 vector.
///
/// Implementation (P6): GEMM writes **directly** into a pre-sized row-major
/// buffer (no intermediate owned faer `Mat` + pack-out). Large products use
/// faer's global parallelism (Rayon by default).
pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
    let (am, an) = array_as_matrix_dims(a)?;
    let (bm, bn) = array_as_matrix_dims(b)?;
    if an != bm {
        return Err(Error::shape(format!(
            "matmul shape mismatch: ({am}, {an}) vs ({bm}, {bn})"
        )));
    }
    let lhs = array_as_mat_ref(a)?;
    let rhs = array_as_mat_ref(b)?;
    // Collapse n×1 (or 1×1 from row@col) to rank-1 when an operand was a vector.
    let prefer_vec = b.rank() == 1 || (a.rank() == 1 && bn == 1);

    let n_out = am.saturating_mul(bn);
    let mut data = vec![0.0; n_out];
    if n_out > 0 {
        let mut dst = MatMut::from_row_major_slice_mut(&mut data, am, bn);
        faer_matmul(
            &mut dst,
            Accum::Replace,
            lhs,
            rhs,
            1.0,
            matmul_par(am, bn, an),
        );
    }
    matmul_result(data, am, bn, prefer_vec)
}

/// Dot product of two rank-1 arrays of equal length.
pub fn dot(a: &Array, b: &Array) -> Result<f64> {
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
    Ok(kernels::dot_slice(a.as_slice(), b.as_slice()))
}

/// Solve `a x = b` for square `a` using LU with partial pivoting (faer).
///
/// - `a` must be rank-2 square
/// - `b` may be rank-1 `(n,)` or rank-2 `(n, k)`
/// - result matches `b`'s rank convention (vector if `b` was rank-1)
pub fn solve(a: &Array, b: &Array) -> Result<Array> {
    let (n, m) = array_as_matrix_dims(a)?;
    if n != m {
        return Err(Error::shape(format!(
            "solve requires square coefficient matrix, got ({n}, {m})"
        )));
    }
    let (bn, bk) = array_as_matrix_dims(b)?;
    if bn != n {
        return Err(Error::shape(format!(
            "solve rhs rows {bn} != matrix order {n}"
        )));
    }
    let am = array_as_mat_ref(a)?;
    let bm = array_as_mat_ref(b)?;
    let lu = am.partial_piv_lu();
    let x = lu.solve(bm);
    mat_to_array(&x, b.rank() == 1 && bk == 1)
}

/// Cholesky factor `L` of a symmetric positive-definite matrix (`A = L Lᵀ`).
///
/// Uses the lower triangle of `a`. Returns lower-triangular `L` as rank-2.
pub fn cholesky(a: &Array) -> Result<Array> {
    let (n, m) = array_as_matrix_dims(a)?;
    if n != m {
        return Err(Error::shape("cholesky requires a square matrix"));
    }
    if a.rank() != 2 {
        return Err(Error::shape("cholesky requires a rank-2 matrix"));
    }
    let am = array_as_mat_ref(a)?;
    let llt = am
        .llt(Side::Lower)
        .map_err(|e| Error::linalg(format!("cholesky failed: {e:?}")))?;
    matref_to_array(llt.L(), false)
}

/// Thin QR decomposition: returns `(Q, R)` with `A = Q R`.
///
/// For an `m × n` matrix with `m ≥ n`, `Q` is `m × n` and `R` is `n × n`.
pub fn qr(a: &Array) -> Result<(Array, Array)> {
    if a.rank() != 2 {
        return Err(Error::shape("qr requires a rank-2 matrix"));
    }
    let am = array_as_mat_ref(a)?;
    let qr = am.qr();
    let q = qr.compute_thin_Q();
    let r = qr.thin_R().to_owned();
    Ok((mat_to_array(&q, false)?, mat_to_array(&r, false)?))
}

/// Thin SVD: returns `(U, s, V)` where `s` is rank-1 singular values
/// (nonincreasing, nonnegative) and `A ≈ U diag(s) Vᵀ` with thin factors.
pub fn svd(a: &Array) -> Result<(Array, Array, Array)> {
    if a.rank() != 2 {
        return Err(Error::shape("svd requires a rank-2 matrix"));
    }
    let am = array_as_mat_ref(a)?;
    let svd = am
        .thin_svd()
        .map_err(|e| Error::linalg(format!("svd failed: {e:?}")))?;
    let u = matref_to_array(svd.U(), false)?;
    let v = matref_to_array(svd.V(), false)?;
    let s_diag = svd.S();
    let n = s_diag.dim();
    let col = s_diag.column_vector();
    let mut s = Vec::with_capacity(n);
    for i in 0..n {
        s.push(col[i]);
    }
    let s = Array::from_parts(crate::array::Shape::from_len(n), s);
    Ok((u, s, v))
}

/// Frobenius norm of a rank-1 or rank-2 array.
pub fn norm(a: &Array) -> Result<f64> {
    let _ = array_as_matrix_dims(a)?;
    Ok(kernels::sum_sq_slice(a.as_slice()).sqrt())
}

/// Identity matrix of order `n`.
#[inline]
pub fn eye(n: usize) -> Result<Array> {
    Array::eye(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_and_transpose() {
        let a = Array::from_shape_slice(vec![2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
        let b = Array::from_shape_slice(vec![3, 2], &[1., 0., 0., 1., 1., 1.]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.dims(), &[2, 2]);
        assert_eq!(c.as_slice(), &[4., 5., 10., 11.]);
        let at = transpose(&a).unwrap();
        assert_eq!(at.dims(), &[3, 2]);
        assert_eq!(at.get(&[2, 1]).unwrap(), 6.);
    }

    #[test]
    fn matmul_matrix_vector_is_rank1() {
        let m = Array::from_shape_slice(vec![2, 2], &[1., 2., 3., 4.]).unwrap();
        let v = Array::from_shape_slice(vec![2], &[1., 1.]).unwrap();
        let y = matmul(&m, &v).unwrap();
        assert_eq!(y.rank(), 1);
        assert_eq!(y.as_slice(), &[3., 7.]);
    }

    #[test]
    fn solve_identity_system() {
        let a = eye(3).unwrap();
        let b = Array::from_shape_slice(vec![3], &[1.5, -2.0, 0.25]).unwrap();
        let x = solve(&a, &b).unwrap();
        assert_eq!(x.rank(), 1);
        assert_eq!(x.as_slice(), b.as_slice());
    }

    #[test]
    fn solve_2x2() {
        let a = Array::from_shape_slice(vec![2, 2], &[3., 1., 1., 2.]).unwrap();
        let b = Array::from_shape_slice(vec![2], &[9., 8.]).unwrap();
        let x = solve(&a, &b).unwrap();
        assert!((x.get(&[0]).unwrap() - 2.0).abs() < 1e-10);
        assert!((x.get(&[1]).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn cholesky_roundtrip_product() {
        let a = Array::from_shape_slice(vec![2, 2], &[2.0, 0.5, 0.5, 1.0]).unwrap();
        let l = cholesky(&a).unwrap();
        let lt = transpose(&l).unwrap();
        let recon = matmul(&l, &lt).unwrap();
        for (x, y) in recon.as_slice().iter().zip(a.as_slice()) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn qr_reconstructs() {
        let a = Array::from_shape_slice(vec![3, 2], &[1., 2., 3., 4., 5., 6.]).unwrap();
        let (q, r) = qr(&a).unwrap();
        let recon = matmul(&q, &r).unwrap();
        assert_eq!(recon.dims(), a.dims());
        for (x, y) in recon.as_slice().iter().zip(a.as_slice()) {
            assert!((x - y).abs() < 1e-9, "{x} vs {y}");
        }
    }

    #[test]
    fn svd_singular_values_sorted() {
        let a = Array::from_shape_slice(vec![2, 2], &[3., 0., 0., 2.]).unwrap();
        let (_u, s, _v) = svd(&a).unwrap();
        assert!(s.get(&[0]).unwrap() >= s.get(&[1]).unwrap());
        assert!((s.get(&[0]).unwrap() - 3.0).abs() < 1e-8);
        assert!((s.get(&[1]).unwrap() - 2.0).abs() < 1e-8);
    }

    #[test]
    fn dot_and_norm() {
        let a = Array::from_shape_slice(vec![3], &[1., 2., 3.]).unwrap();
        let b = Array::from_shape_slice(vec![3], &[4., 5., 6.]).unwrap();
        assert!((dot(&a, &b).unwrap() - 32.0).abs() < 1e-12);
        assert!((norm(&a).unwrap() - (14.0f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn normal_equations_style() {
        let x = Array::from_shape_slice(
            vec![4, 2],
            &[
                1., 0., //
                1., 1., //
                1., 2., //
                1., 3.,
            ],
        )
        .unwrap();
        let y = Array::from_shape_slice(vec![4], &[1., 3., 5., 7.]).unwrap();
        let xt = transpose(&x).unwrap();
        let xtx = matmul(&xt, &x).unwrap();
        let xty = matmul(&xt, &y.reshape(vec![4, 1]).unwrap()).unwrap();
        let beta = solve(&xtx, &xty).unwrap();
        assert!((beta.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-9);
        assert!((beta.get(&[1, 0]).unwrap() - 2.0).abs() < 1e-9);
    }
}
