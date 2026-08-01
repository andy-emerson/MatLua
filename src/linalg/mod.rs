//! Dense linear algebra on rank-1/2 [`Array`] values (faer-backed).
//!
//! All public functions take and return MatLua [`Array`]s. Internally values are
//! copied into faer [`Mat`](faer::Mat) (column-major) and results are copied
//! back to row-major owned arrays. View/copy rules for zero-copy faer views can
//! be refined later; M2 prioritizes a correct owned API.
//!
//! # Rank conventions
//!
//! | Input | Interpretation |
//! |-------|----------------|
//! | rank 2, shape `(m, n)` | `m × n` matrix |
//! | rank 1, shape `(n,)` | `n × 1` column vector |
//!
//! Use [`dot`] for the inner product of two equal-length vectors.

mod convert;

use faer::linalg::solvers::Solve;
use faer::Side;

use crate::array::Array;
use crate::error::{Error, Result};

use convert::{array_as_matrix_dims, array_to_mat, mat_to_array, matref_to_array};

/// Transpose a rank-1 or rank-2 array.
///
/// Rank-1 inputs become shape `(1, n)` row matrices (rank 2).
pub fn transpose(a: &Array) -> Result<Array> {
    let m = array_to_mat(a)?;
    mat_to_array(&m.transpose().to_owned(), false)
}

/// Matrix product `a @ b` (math notation).
///
/// Shapes: `(m, k) × (k, n) → (m, n)`. Rank-1 operands are treated as columns.
pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
    let (am, an) = array_as_matrix_dims(a)?;
    let (bm, bn) = array_as_matrix_dims(b)?;
    if an != bm {
        return Err(Error::shape(format!(
            "matmul shape mismatch: ({am}, {an}) vs ({bm}, {bn})"
        )));
    }
    let am_mat = array_to_mat(a)?;
    let bm_mat = array_to_mat(b)?;
    let c = &am_mat * &bm_mat;
    let prefer_vec = a.rank() == 1 && b.rank() == 2 && bn == 1;
    mat_to_array(&c, prefer_vec)
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
    Ok(a
        .as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(&x, &y)| x * y)
        .sum())
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
    let am = array_to_mat(a)?;
    let bm = array_to_mat(b)?;
    let lu = am.partial_piv_lu();
    let x = lu.solve(&bm);
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
    let am = array_to_mat(a)?;
    let llt = am
        .llt(Side::Lower)
        .map_err(|e| Error::linalg(format!("cholesky failed: {e:?}")))?;
    matref_to_array(llt.L())
}

/// Thin QR decomposition: returns `(Q, R)` with `A = Q R`.
///
/// For an `m × n` matrix with `m ≥ n`, `Q` is `m × n` and `R` is `n × n`.
pub fn qr(a: &Array) -> Result<(Array, Array)> {
    if a.rank() != 2 {
        return Err(Error::shape("qr requires a rank-2 matrix"));
    }
    let am = array_to_mat(a)?;
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
    let am = array_to_mat(a)?;
    let svd = am
        .thin_svd()
        .map_err(|e| Error::linalg(format!("svd failed: {e:?}")))?;
    let u = matref_to_array(svd.U())?;
    let v = matref_to_array(svd.V())?;
    let s_diag = svd.S();
    let n = s_diag.dim();
    let col = s_diag.column_vector();
    let mut s = Vec::with_capacity(n);
    for i in 0..n {
        s.push(col[i]);
    }
    let s = Array::from_shape_vec(vec![n], s)?;
    Ok((u, s, v))
}

/// Frobenius norm of a rank-1 or rank-2 array.
pub fn norm(a: &Array) -> Result<f64> {
    let _ = array_as_matrix_dims(a)?;
    Ok(a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt())
}

/// Identity matrix of order `n`.
pub fn eye(n: usize) -> Result<Array> {
    let mut a = Array::zeros(vec![n, n])?;
    for i in 0..n {
        a.set(&[i, i], 1.0)?;
    }
    Ok(a)
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
