//! Dense linear algebra on rank-1/2 [`Array`] values (faer-backed).
//! Integer inputs to real solvers: [`from_i64`] (promote to `f64`, NumPy-style).
//!
//! Public functions take and return MatLua [`Array`]s. Inputs are viewed as
//! faer [`MatRef`](faer::MatRef) over contiguous row-major storage (zero-copy).
//! [`matmul`] / [`matmul_at`] / [`matmul_bt`] write GEMM into row-major buffers;
//! [`solve`] factors then solves **in place** on a row-major RHS copy (no
//! faer-owned Mat out).
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
pub mod i64_ops;
pub mod from_i64;

use faer::linalg::matmul::matmul as faer_matmul;
use faer::linalg::solvers::{Solve, SolveLstsq};
use faer::{get_global_parallelism, Accum, MatMut, Par, Side};

use crate::array::kernels;
use crate::array::{Array, Shape};
use crate::error::{Error, Result};

use convert::{array_as_mat_ref, array_as_matrix_dims, mat_to_array, matref_to_array};

/// Parallelism for GEMM: sequential for tiny products, otherwise faer's global
/// setting (typically Rayon with default faer features).
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
/// GEMM writes **directly** into a pre-sized row-major buffer (no intermediate
/// owned faer `Mat` + pack-out). Large products use faer's global parallelism
/// (typically Rayon with default faer features).
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
    let mut data = crate::array::pool_take_uninit(n_out);
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

/// Matrix product `aᵀ @ b`.
///
/// Shapes: `(m, k)ᵀ × (m, n) → (k, n)` i.e. `a` is `m×k`, `b` is `m×n`.
/// Rank-1 `b` is a column; result is rank-1 when `b` is rank-1.
///
/// Numerically matches [`matmul`]`(`[`transpose`]`(a), b)`. Implementation:
/// - general `a`/`b`: GEMM over a transposed [`MatRef`](faer::MatRef) (no owned `aᵀ`);
/// - same-buffer gram `aᵀa` with feature count `k ≥ 512`: blocked materialize of
///   `aᵀ` then dest-GEMM (faster than pure transposed views at large `k`).
pub fn matmul_at(a: &Array, b: &Array) -> Result<Array> {
    let (am, an) = array_as_matrix_dims(a)?;
    let (bm, bn) = array_as_matrix_dims(b)?;
    if am != bm {
        return Err(Error::shape(format!(
            "matmul_at shape mismatch: a is ({am}, {an}) so aᵀ is ({an}, {am}), b is ({bm}, {bn})"
        )));
    }
    // AᵀA (same buffer): for large feature count, materialize Aᵀ (blocked) then
    // dest-GEMM. Small k keeps a pure transposed MatRef GEMM (cheaper).
    if std::ptr::eq(a.as_slice().as_ptr(), b.as_slice().as_ptr())
        && a.len() == b.len()
        && a.rank() == 2
        && b.rank() == 2
        && an == bn
        && an >= 512
    {
        let at = transpose(a)?;
        return matmul(&at, a);
    }
    let lhs = array_as_mat_ref(a)?.transpose();
    let rhs = array_as_mat_ref(b)?;
    let prefer_vec = b.rank() == 1;
    let n_out = an.saturating_mul(bn);
    let mut data = crate::array::pool_take_uninit(n_out);
    if n_out > 0 {
        let mut dst = MatMut::from_row_major_slice_mut(&mut data, an, bn);
        faer_matmul(
            &mut dst,
            Accum::Replace,
            lhs,
            rhs,
            1.0,
            matmul_par(an, bn, am),
        );
    }
    matmul_result(data, an, bn, prefer_vec)
}

/// Matrix product `a @ bᵀ`.
///
/// Shapes: `(m, k) × (n, k)ᵀ → (m, n)` i.e. both `a` and `b` have `k` columns.
/// Rank-1 `a` is a row-as-column style via the usual rank-1 column view; prefer
/// rank-2 operands for clarity.
///
/// Numerically matches [`matmul`]`(a, `[`transpose`]`(b))`. Uses a transposed
/// [`MatRef`](faer::MatRef) on `b` (no owned `bᵀ`) except for large same-buffer
/// gram `a aᵀ` with observation count `k ≥ 512`, which materializes `aᵀ` once
/// then dest-GEMMs (mirror of [`matmul_at`]'s large-`k` path).
///
/// Primary consumer: [`Array::cov`] (row-variable gram \(XX^\top\)).
pub fn matmul_bt(a: &Array, b: &Array) -> Result<Array> {
    let (am, an) = array_as_matrix_dims(a)?;
    let (bm, bn) = array_as_matrix_dims(b)?;
    if an != bn {
        return Err(Error::shape(format!(
            "matmul_bt shape mismatch: a is ({am}, {an}), b is ({bm}, {bn}); need equal column counts for a @ bᵀ"
        )));
    }
    // AAᵀ (same buffer): large k (shared dimension) → materialize Aᵀ then A @ Aᵀ.
    if std::ptr::eq(a.as_slice().as_ptr(), b.as_slice().as_ptr())
        && a.len() == b.len()
        && a.rank() == 2
        && b.rank() == 2
        && am == bm
        && an >= 512
    {
        let at = transpose(a)?;
        return matmul(a, &at);
    }
    let lhs = array_as_mat_ref(a)?;
    let rhs = array_as_mat_ref(b)?.transpose();
    let prefer_vec = a.rank() == 1;
    let n_out = am.saturating_mul(bm);
    let mut data = crate::array::pool_take_uninit(n_out);
    if n_out > 0 {
        let mut dst = MatMut::from_row_major_slice_mut(&mut data, am, bm);
        faer_matmul(
            &mut dst,
            Accum::Replace,
            lhs,
            rhs,
            1.0,
            matmul_par(am, bm, an),
        );
    }
    matmul_result(data, am, bm, prefer_vec)
}

/// Normal equations: `solve(XᵀX, Xᵀy)` via [`matmul_at`].
///
/// - `x`: rank-2 `(m, k)` design matrix  
/// - `y`: rank-1 `(m,)` or rank-2 `(m, n)`  
/// - returns coefficients with the same rank style as `y`
///
/// Same result as `solve(matmul(transpose(x), x), matmul(transpose(x), y))`.
/// Large `XᵀX` may materialize `Xᵀ` once internally (see [`matmul_at`]).
pub fn normal_eq(x: &Array, y: &Array) -> Result<Array> {
    if x.rank() != 2 {
        return Err(Error::shape("normal_eq expects rank-2 design matrix X"));
    }
    let xtx = matmul_at(x, x)?;
    let xty = matmul_at(x, y)?;
    solve(&xtx, &xty)
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
    let lu = am.partial_piv_lu();
    // Dest-pack: copy RHS into owned row-major buffer and solve in place
    // (avoids faer-owned Mat + second pack-out).
    let prefer_vec = b.rank() == 1 && bk == 1;
    let n_out = bn.saturating_mul(bk);
    let mut data = crate::array::pool_take_uninit(n_out);
    if n_out > 0 {
        data.copy_from_slice(b.as_slice());
        let mut rhs = MatMut::from_row_major_slice_mut(&mut data, bn, bk);
        lu.solve_in_place(&mut rhs);
    }
    matmul_result(data, bn, bk, prefer_vec)
}

/// Least squares: minimize `‖a x − b‖₂` (overdetermined / square).
///
/// - `a`: rank-2 `(m, n)` with **`m ≥ n`**
/// - `b`: rank-1 `(m,)` or rank-2 `(m, k)`
/// - returns coefficients rank-1 or `(n, k)` matching `b`'s style
///
/// Uses faer column-pivoted QR least squares. Not an alias of [`normal_eq`]
/// (prefer this for multi-factor / noisy systems). Underdetermined `m < n`
/// is rejected — use [`pinv`] and matmul for that shape class.
pub fn lstsq(a: &Array, b: &Array) -> Result<Array> {
    if a.rank() != 2 {
        return Err(Error::shape("lstsq expects rank-2 coefficient matrix"));
    }
    let (m, n) = array_as_matrix_dims(a)?;
    if m < n {
        return Err(Error::shape(format!(
            "lstsq requires m >= n (got {m}×{n}); use pinv for underdetermined systems"
        )));
    }
    let (bm, bk) = array_as_matrix_dims(b)?;
    if bm != m {
        return Err(Error::shape(format!(
            "lstsq rhs rows {bm} != matrix rows {m}"
        )));
    }
    let am = array_as_mat_ref(a)?;
    let bm_ref = array_as_mat_ref(b)?;
    let qr = am.col_piv_qr();
    let x = qr.solve_lstsq(bm_ref);
    mat_to_array(&x, b.rank() == 1 && bk == 1)
}

/// Symmetric eigendecomposition: `a = v diag(w) vᵀ` (lower triangle of `a` used).
///
/// Returns `(w, v)` where `w` is rank-1 eigenvalues in **nondecreasing** order
/// and `v` is rank-2 eigenvectors as columns (NumPy `eigh` shape style).
pub fn eigh(a: &Array) -> Result<(Array, Array)> {
    if a.rank() != 2 {
        return Err(Error::shape("eigh requires a rank-2 matrix"));
    }
    let (n, m) = array_as_matrix_dims(a)?;
    if n != m {
        return Err(Error::shape("eigh requires a square matrix"));
    }
    let am = array_as_mat_ref(a)?;
    let evd = am
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| Error::linalg(format!("eigh failed: {e:?}")))?;
    let v = matref_to_array(evd.U(), false)?;
    let s_diag = evd.S();
    let dim = s_diag.dim();
    let col = s_diag.column_vector();
    let mut w = crate::array::pool_take_uninit(dim);
    for i in 0..dim {
        w[i] = col[i];
    }
    let w = Array::from_parts(Shape::from_len(dim), w);
    Ok((w, v))
}

/// Moore–Penrose pseudoinverse via full SVD (faer).
///
/// Shape: `(m, n) → (n, m)`. Suitable for rank-deficient and underdetermined
/// systems; for tall full-rank least squares prefer [`lstsq`].
pub fn pinv(a: &Array) -> Result<Array> {
    if a.rank() != 2 {
        return Err(Error::shape("pinv requires a rank-2 matrix"));
    }
    let am = array_as_mat_ref(a)?;
    let svd = am
        .svd()
        .map_err(|e| Error::linalg(format!("pinv svd failed: {e:?}")))?;
    let p = svd.pseudoinverse();
    mat_to_array(&p, false)
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
    let mut s = crate::array::pool_take_uninit(n);
    for i in 0..n {
        s[i] = col[i];
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
    fn matmul_at_matches_transpose_matmul() {
        let a = Array::from_shape_slice(vec![3, 2], &[1., 2., 3., 4., 5., 6.]).unwrap();
        let b = Array::from_shape_slice(vec![3, 2], &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]).unwrap();
        let short = matmul_at(&a, &b).unwrap();
        let long = matmul(&transpose(&a).unwrap(), &b).unwrap();
        assert_eq!(short.shape().dims(), long.shape().dims());
        for (x, y) in short.as_slice().iter().zip(long.as_slice()) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
        let v = Array::from_shape_slice(vec![3], &[1., 0., -1.]).unwrap();
        let short_v = matmul_at(&a, &v).unwrap();
        let long_v = matmul(&transpose(&a).unwrap(), &v).unwrap();
        assert_eq!(short_v.rank(), 1);
        for (x, y) in short_v.as_slice().iter().zip(long_v.as_slice()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn matmul_bt_matches_matmul_transpose() {
        let a = Array::from_shape_slice(vec![2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
        let b = Array::from_shape_slice(vec![2, 3], &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]).unwrap();
        let short = matmul_bt(&a, &b).unwrap();
        let long = matmul(&a, &transpose(&b).unwrap()).unwrap();
        assert_eq!(short.dims(), long.dims());
        for (x, y) in short.as_slice().iter().zip(long.as_slice()) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
        // Same-buffer AAᵀ
        let g = matmul_bt(&a, &a).unwrap();
        let g_long = matmul(&a, &transpose(&a).unwrap()).unwrap();
        for (x, y) in g.as_slice().iter().zip(g_long.as_slice()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn normal_eq_matches_composed_solve() {
        // Small least-squares style: X 4×2, y 4
        let x = Array::from_shape_slice(
            vec![4, 2],
            &[1., 0., 1., 1., 1., 2., 1., 3.],
        )
        .unwrap();
        let y = Array::from_shape_slice(vec![4], &[1., 2., 3., 4.]).unwrap();
        let short = normal_eq(&x, &y).unwrap();
        let long = solve(
            &matmul(&transpose(&x).unwrap(), &x).unwrap(),
            &matmul(&transpose(&x).unwrap(), &y).unwrap(),
        )
        .unwrap();
        assert_eq!(short.rank(), 1);
        for (a, b) in short.as_slice().iter().zip(long.as_slice()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
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
    fn lstsq_overdetermined_matches_normal_eq_full_rank() {
        // X 4×2, y 4 — full column rank
        let x = Array::from_shape_slice(
            vec![4, 2],
            &[1., 0., 1., 1., 1., 2., 1., 3.],
        )
        .unwrap();
        let y = Array::from_shape_slice(vec![4], &[1., 2., 3., 4.]).unwrap();
        let beta = lstsq(&x, &y).unwrap();
        let beta_ne = normal_eq(&x, &y).unwrap();
        assert_eq!(beta.rank(), 1);
        for (a, b) in beta.as_slice().iter().zip(beta_ne.as_slice()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn eigh_identity() {
        let a = Array::eye(3).unwrap();
        let (w, v) = eigh(&a).unwrap();
        assert_eq!(w.rank(), 1);
        assert_eq!(v.dims(), &[3, 3]);
        for i in 0..3 {
            assert!((w.get(&[i]).unwrap() - 1.0).abs() < 1e-10);
        }
        // V should be orthogonal: VᵀV ≈ I
        let vt = transpose(&v).unwrap();
        let g = matmul(&vt, &v).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((g.get(&[i, j]).unwrap() - e).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn pinv_tall_left_inverse_product() {
        // Full column rank 3×2 → pinv is 2×3, pinv @ A ≈ I₂
        let a = Array::from_shape_slice(vec![3, 2], &[1., 0., 0., 1., 1., 1.]).unwrap();
        let p = pinv(&a).unwrap();
        assert_eq!(p.dims(), &[2, 3]);
        let i = matmul(&p, &a).unwrap();
        assert!((i.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-8);
        assert!((i.get(&[1, 1]).unwrap() - 1.0).abs() < 1e-8);
        assert!(i.get(&[0, 1]).unwrap().abs() < 1e-8);
        assert!(i.get(&[1, 0]).unwrap().abs() < 1e-8);
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
