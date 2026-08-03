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

use convert::{array_as_mat_ref, array_as_matrix_dims, array_to_colmajor, mat_to_array, matref_to_array};

/// Parallelism for GEMM: sequential for tiny products, otherwise faer's global
/// setting (typically Rayon with default faer features).
#[inline]
fn matmul_par(m: usize, n: usize, k: usize) -> Par {
    // ~ n³ work proxy. The 128³ cutoff is empirical (M7.c bench host, 2026-07;
    // unverified elsewhere) — see DESIGN §3.26.
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
    // Tile so inner writes stream along destination rows (unit stride on dst).
    // Analyzed (DESIGN §3.26): a 32×32 f64 tile is 8 KB read + 8 KB written,
    // 16 KB total — comfortably inside a 32 KB L1d (smallest common size on
    // x86-64/aarch64) with room left for streaming.
    const BS: usize = 32;
    let mut j0 = 0;
    while j0 < cols {
        let j1 = (j0 + BS).min(cols);
        let mut i0 = 0;
        while i0 < rows {
            let i1 = (i0 + BS).min(rows);
            for j in j0..j1 {
                let dst_row = j * rows;
                let mut i = i0;
                while i + 4 <= i1 {
                    // src[i, j] = src[i*cols + j]
                    dst[dst_row + i] = src[i * cols + j];
                    dst[dst_row + i + 1] = src[(i + 1) * cols + j];
                    dst[dst_row + i + 2] = src[(i + 2) * cols + j];
                    dst[dst_row + i + 3] = src[(i + 3) * cols + j];
                    i += 4;
                }
                while i < i1 {
                    dst[dst_row + i] = src[i * cols + j];
                    i += 1;
                }
            }
            i0 = i1;
        }
        j0 = j1;
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

/// GEMM into a preallocated `out` with shape `(am, bn)` (rank-2). Does not collapse to rank-1.
pub fn matmul_out(a: &Array, b: &Array, out: &mut Array) -> Result<()> {
    let (am, an) = array_as_matrix_dims(a)?;
    let (bm, bn) = array_as_matrix_dims(b)?;
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
    let lhs = array_as_mat_ref(a)?;
    let rhs = array_as_mat_ref(b)?;
    if am * bn > 0 {
        let mut dst = MatMut::from_row_major_slice_mut(out.as_mut_slice(), am, bn);
        faer_matmul(
            &mut dst,
            Accum::Replace,
            lhs,
            rhs,
            1.0,
            matmul_par(am, bn, an),
        );
    }
    Ok(())
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
    // The 512 crossover is empirical (M7.c bench host, 2026-07; unverified
    // elsewhere) — see DESIGN §3.26.
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
    // The 512 crossover is empirical (M7.c bench host, 2026-07; unverified
    // elsewhere) — see DESIGN §3.26.
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

/// Unblocked row-major LU with partial pivoting + in-place solve of a single
/// RHS (LAPACK `dgesv` shape, no blocking). Returns `false` on an exact zero
/// pivot; the caller falls back to the faer path so singular behavior stays
/// identical to the large-n route.
fn lu_solve_unblocked(n: usize, a: &mut [f64], x: &mut [f64]) -> bool {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(x.len(), n);
    // Pivot rows are swapped eagerly, and the RHS permutation is applied at
    // the moment each pivot is chosen (equivalent to LAPACK's ipiv replay).
    for k in 0..n {
        let mut p = k;
        let mut best = a[k * n + k].abs();
        for i in k + 1..n {
            let v = a[i * n + k].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if best == 0.0 {
            return false;
        }
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
            x.swap(k, p);
        }
        let akk = a[k * n + k];
        for i in k + 1..n {
            let l = a[i * n + k] / akk;
            a[i * n + k] = l;
            let (top, bot) = a.split_at_mut(i * n);
            let arow = &top[k * n + k + 1..k * n + n];
            let irow = &mut bot[k + 1..n];
            for j in 0..irow.len() {
                irow[j] -= l * arow[j];
            }
        }
    }
    for k in 0..n {
        for i in k + 1..n {
            x[i] -= a[i * n + k] * x[k];
        }
    }
    for k in (0..n).rev() {
        x[k] /= a[k * n + k];
        for i in 0..k {
            x[i] -= a[i * n + k] * x[k];
        }
    }
    true
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
    // Small-system fast path: LAPACK-style unblocked LU on a row-major copy
    // (no blocked-machinery or layout-conversion overhead). Crossover is
    // empirical (2026-08 bench container: unblocked wins 1.5–3.6× at
    // n ≤ 192, ties ~384, loses at 512; cutoff kept conservative pending
    // another host class — DESIGN §3.26). Restricted to single-RHS; exact
    // zero pivot falls through to the faer path so singular behavior is
    // unchanged. Agrees with faer to machine epsilon on well-conditioned
    // systems (same pivoting strategy).
    if n <= 192 && bk == 1 && n > 0 {
        let mut lu = crate::array::pool_take_uninit(n * n);
        lu.copy_from_slice(a.as_slice());
        let mut x = crate::array::pool_take_uninit(n);
        x.copy_from_slice(b.as_slice());
        if lu_solve_unblocked(n, &mut lu, &mut x) {
            crate::array::pool_recycle(lu);
            let prefer_vec = b.rank() == 1;
            return matmul_result(x, n, 1, prefer_vec);
        }
        // Exact zero pivot: return both scratch buffers and take the faer path.
        crate::array::pool_recycle(lu);
        crate::array::pool_recycle(x);
    }
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
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
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
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
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
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
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
    let svd = am
        .svd()
        .map_err(|e| Error::linalg(format!("pinv svd failed: {e:?}")))?;
    let p = svd.pseudoinverse();
    mat_to_array(&p, false)
}

/// Solve `a x = b` for symmetric positive-definite `a` via Cholesky (`LLᵀ`).
///
/// Uses the lower triangle of `a`. About half the flops of the LU [`solve`]
/// on SPD systems; errors with [`Error::Linalg`] when `a` is not positive
/// definite. `b` may be rank-1 `(n,)` or rank-2 `(n, k)`; the result matches
/// `b`'s rank convention. (Added for TallyDB-class hosts: factor-only
/// [`cholesky`] existed, but their windows need the solve.)
pub fn cholesky_solve(a: &Array, b: &Array) -> Result<Array> {
    let (n, m) = array_as_matrix_dims(a)?;
    if a.rank() != 2 || n != m {
        return Err(Error::shape("cholesky_solve requires a square rank-2 matrix"));
    }
    let (bn, bk) = array_as_matrix_dims(b)?;
    if bn != n {
        return Err(Error::shape(format!(
            "cholesky_solve rhs rows {bn} != matrix order {n}"
        )));
    }
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
    let llt = am
        .llt(Side::Lower)
        .map_err(|e| Error::linalg(format!("cholesky_solve: not positive definite: {e:?}")))?;
    // Dest-pack like `solve`: copy RHS row-major and solve in place.
    let prefer_vec = b.rank() == 1 && bk == 1;
    let n_out = bn.saturating_mul(bk);
    let mut data = crate::array::pool_take_uninit(n_out);
    if n_out > 0 {
        data.copy_from_slice(b.as_slice());
        let mut rhs = MatMut::from_row_major_slice_mut(&mut data, bn, bk);
        llt.solve_in_place(&mut rhs);
    }
    matmul_result(data, bn, bk, prefer_vec)
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
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
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
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
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
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
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
    fn cholesky_solve_matches_lu_solve_on_spd() {
        let n = 24usize;
        // Diagonally dominant symmetric => SPD.
        let mut d = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = 0.1 * (((i * 7 + j * 3) % 11) as f64) / 11.0;
                d[i * n + j] = if i == j { v + n as f64 } else { v };
            }
        }
        // symmetrize
        for i in 0..n {
            for j in 0..i {
                let m = 0.5 * (d[i * n + j] + d[j * n + i]);
                d[i * n + j] = m;
                d[j * n + i] = m;
            }
        }
        let a = Array::from_shape_slice(vec![n, n], &d).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let bv = Array::from_shape_slice(vec![n], &b).unwrap();
        let x_chol = cholesky_solve(&a, &bv).unwrap();
        let x_lu = solve(&a, &bv).unwrap();
        assert_eq!(x_chol.rank(), 1);
        for i in 0..n {
            assert!((x_chol.as_slice()[i] - x_lu.as_slice()[i]).abs() < 1e-9);
        }
        // Non-SPD input must error, not return garbage.
        let mut nd = d.clone();
        nd[0] = -100.0;
        let bad = Array::from_shape_slice(vec![n, n], &nd).unwrap();
        assert!(cholesky_solve(&bad, &bv).is_err());
    }

    #[test]
    fn small_solve_path_matches_faer_path() {
        // n=100 takes the unblocked path for the rank-1 rhs; a 2-column rhs
        // forces the faer path on the same system. Column 0 must agree to
        // machine-epsilon scale (same pivoting strategy).
        let n = 100usize;
        let mut data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = (((i * 31 + j * 17) % 1000) as f64) / 1000.0 - 0.5;
                data[i * n + j] = if i == j { v + n as f64 } else { v };
            }
        }
        let a = Array::from_shape_slice(vec![n, n], &data).unwrap();
        let b1: Vec<f64> = (0..n).map(|i| (i % 13) as f64).collect();
        let mut b2 = vec![0.0f64; n * 2];
        for i in 0..n {
            b2[i * 2] = b1[i];
            b2[i * 2 + 1] = (i % 7) as f64;
        }
        let bv = Array::from_shape_slice(vec![n], &b1).unwrap();
        let bm = Array::from_shape_slice(vec![n, 2], &b2).unwrap();
        let x_small = solve(&a, &bv).unwrap();
        let x_faer = solve(&a, &bm).unwrap();
        for i in 0..n {
            let d = (x_small.as_slice()[i] - x_faer.as_slice()[i * 2]).abs();
            assert!(d < 1e-10, "row {i}: {d}");
        }
    }

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
    fn det_slogdet_rank_cond() {
        let a = Array::from_shape_slice(vec![2, 2], &[1., 2., 3., 4.]).unwrap();
        let d = det(&a).unwrap();
        assert!((d - (-2.0)).abs() < 1e-10, "det={d}");
        let (sign, logabs) = slogdet(&a).unwrap();
        assert!((sign - (-1.0)).abs() < 1e-12);
        assert!((logabs - 2.0_f64.ln()).abs() < 1e-10);
        assert_eq!(matrix_rank(&a, None).unwrap(), 2);
        let singular = Array::from_shape_slice(vec![2, 2], &[1., 2., 2., 4.]).unwrap();
        assert_eq!(matrix_rank(&singular, None).unwrap(), 1);
        let id = Array::eye(3).unwrap();
        assert!((cond(&id).unwrap() - 1.0).abs() < 1e-9);
        let (wr, wi) = eigvals(&id).unwrap();
        assert_eq!(wr.len(), 3);
        assert!(wi.as_slice().iter().all(|&x| x.abs() < 1e-12));
        assert!(wr.as_slice().iter().all(|&x| (x - 1.0).abs() < 1e-9));
    }

    #[test]
    fn eig_identity_real() {
        let id = Array::eye(2).unwrap();
        let (wr, wi, vr, vi) = eig(&id).unwrap();
        assert!(wr.as_slice().iter().all(|&x| (x - 1.0).abs() < 1e-8));
        assert!(wi.as_slice().iter().all(|&x| x.abs() < 1e-10));
        assert_eq!(vr.dims(), &[2, 2]);
        assert!(vi.as_slice().iter().all(|&x| x.abs() < 1e-10));
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

// ----- M7.b diagnostics (f64) -----

/// Determinant of a square rank-2 matrix (via partial-pivoted LU).
pub fn det(a: &Array) -> Result<f64> {
    let (sign, logabs) = slogdet(a)?;
    if sign == 0.0 {
        return Ok(0.0);
    }
    Ok(sign * logabs.exp())
}

/// Sign and log of absolute determinant: `(sign, log|det|)`.
///
/// `sign` is `-1.0`, `0.0`, or `1.0`. When the matrix is singular, returns
/// `(0.0, -∞)`. Matches NumPy `slogdet` for real matrices.
pub fn slogdet(a: &Array) -> Result<(f64, f64)> {
    if a.rank() != 2 {
        return Err(Error::shape("slogdet requires a rank-2 matrix"));
    }
    let (n, m) = array_as_matrix_dims(a)?;
    if n != m {
        return Err(Error::shape(format!(
            "slogdet requires a square matrix, got ({n}, {m})"
        )));
    }
    if n == 0 {
        return Ok((1.0, 0.0)); // det of 0×0 is 1 by convention
    }
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
    let lu = am.partial_piv_lu();
    let u = lu.U();
    let p = lu.P();
    let mut sign = perm_sign(p.arrays().0);
    let mut logabs = 0.0_f64;
    for i in 0..n {
        let d = u[(i, i)];
        if d == 0.0 || !d.is_finite() {
            return Ok((0.0, f64::NEG_INFINITY));
        }
        if d < 0.0 {
            sign = -sign;
        }
        logabs += d.abs().ln();
    }
    Ok((sign, logabs))
}

/// Sign of a permutation given as forward image `fwd[i] = π(i)`.
fn perm_sign(fwd: &[usize]) -> f64 {
    let n = fwd.len();
    let mut seen = vec![false; n];
    let mut sign = 1.0_f64;
    for i in 0..n {
        if seen[i] {
            continue;
        }
        let mut j = i;
        let mut cycle_len = 0usize;
        while !seen[j] {
            seen[j] = true;
            j = fwd[j];
            cycle_len += 1;
        }
        // cycle length k contributes (-1)^{k-1}
        if cycle_len > 0 && (cycle_len - 1) % 2 == 1 {
            sign = -sign;
        }
    }
    sign
}

fn singular_values_vec(a: &Array) -> Result<Vec<f64>> {
    if a.rank() != 2 {
        return Err(Error::shape("expected rank-2 matrix"));
    }
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
    let s = am
        .singular_values()
        .map_err(|e| Error::linalg(format!("singular_values failed: {e:?}")))?;
    Ok(s)
}

/// Default numerical rank tolerance: `max(m, n) * eps * σ_max` (NumPy-style).
pub fn matrix_rank(a: &Array, tol: Option<f64>) -> Result<usize> {
    let (m, n) = array_as_matrix_dims(a)?;
    if a.rank() != 2 {
        return Err(Error::shape("matrix_rank requires a rank-2 matrix"));
    }
    if m == 0 || n == 0 {
        return Ok(0);
    }
    let s = singular_values_vec(a)?;
    if s.is_empty() {
        return Ok(0);
    }
    let smax = s[0].abs(); // nonincreasing nonnegative
    let thresh = tol.unwrap_or_else(|| {
        let eps = f64::EPSILON;
        (m.max(n) as f64) * eps * smax
    });
    Ok(s.iter().filter(|&&v| v > thresh).count())
}

/// 2-norm condition number `σ_max / σ_min` (via SVD).
///
/// Returns `+∞` if the matrix is rank-deficient (smallest singular value is 0).
/// Empty shape errors; for non-square uses the rectangular 2-norm condition.
pub fn cond(a: &Array) -> Result<f64> {
    if a.rank() != 2 {
        return Err(Error::shape("cond requires a rank-2 matrix"));
    }
    let (m, n) = array_as_matrix_dims(a)?;
    if m == 0 || n == 0 {
        return Err(Error::shape("cond of empty matrix is undefined"));
    }
    let s = singular_values_vec(a)?;
    let smax = s.first().copied().unwrap_or(0.0);
    let smin = s.last().copied().unwrap_or(0.0);
    if smin == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(smax / smin)
}

/// Eigenvalues of a square matrix as real/imag parts `(wr, wi)` (rank-1 each).
///
/// Real matrices may produce complex conjugate pairs; there is no complex dtype
/// yet, so parts are split (NumPy would use complex). Order matches faer.
pub fn eigvals(a: &Array) -> Result<(Array, Array)> {
    if a.rank() != 2 {
        return Err(Error::shape("eigvals requires a rank-2 matrix"));
    }
    let (n, m) = array_as_matrix_dims(a)?;
    if n != m {
        return Err(Error::shape("eigvals requires a square matrix"));
    }
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
    let vals = am
        .eigenvalues()
        .map_err(|e| Error::linalg(format!("eigvals failed: {e:?}")))?;
    let mut re = crate::array::pool_take_uninit(n);
    let mut im = crate::array::pool_take_uninit(n);
    for (i, z) in vals.iter().enumerate() {
        re[i] = z.re;
        im[i] = z.im;
    }
    Ok((
        Array::from_parts(Shape::from_len(n), re),
        Array::from_parts(Shape::from_len(n), im),
    ))
}

/// Right eigendecomposition: `(wr, wi, vr_re, vr_im)`.
///
/// `A v_j = λ_j v_j` with `λ_j = wr[j] + i wi[j]` and column `j` of
/// `vr_re + i vr_im` the eigenvector. Complex results are split into real arrays
/// (no `c64` dtype yet).
pub fn eig(a: &Array) -> Result<(Array, Array, Array, Array)> {
    if a.rank() != 2 {
        return Err(Error::shape("eig requires a rank-2 matrix"));
    }
    let (n, m) = array_as_matrix_dims(a)?;
    if n != m {
        return Err(Error::shape("eig requires a square matrix"));
    }
    // Factorization input: column-major copy (see `ColMajor`).
    let a_cm = array_to_colmajor(a)?;
    let am = a_cm.view();
    let evd = am
        .eigen()
        .map_err(|e| Error::linalg(format!("eig failed: {e:?}")))?;
    let u = evd.U(); // Complex matrix n×n
    let s = evd.S();
    let mut wr = crate::array::pool_take_uninit(n);
    let mut wi = crate::array::pool_take_uninit(n);
    let col = s.column_vector();
    for i in 0..n {
        wr[i] = col[i].re;
        wi[i] = col[i].im;
    }
    let mut vre = crate::array::pool_take_uninit(n * n);
    let mut vim = crate::array::pool_take_uninit(n * n);
    for i in 0..n {
        for j in 0..n {
            let z = u[(i, j)];
            vre[i * n + j] = z.re;
            vim[i * n + j] = z.im;
        }
    }
    Ok((
        Array::from_parts(Shape::from_len(n), wr),
        Array::from_parts(Shape::from_len(n), wi),
        Array::from_parts(Shape::matrix(n, n)?, vre),
        Array::from_parts(Shape::matrix(n, n)?, vim),
    ))
}
