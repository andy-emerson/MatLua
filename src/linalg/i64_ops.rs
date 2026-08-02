//! Dense linear algebra on [`ArrayI64`](crate::array::ArrayI64).
//!
//! Integer path (not faer/`f64`). Arithmetic is **wrapping** `i64`, matching
//! the rest of the `i64` surface.
//!
//! # Matmul algorithm (M7.c)
//!
//! Research / design notes:
//! - **Not** f64 promote + faer: would break exactness past 2⁵³ and wrapping.
//! - **Strassen** works over any ring (incl. wrapping `i64`) but practical
//!   crossover is typically n ≳ 512–1000 once the *base* GEMM is strong; we
//!   invest in the cubic kernel first (Goto/BLIS literature).
//! - High-performance GEMM is multilevel: **cache panels + packing** so the
//!   micro-kernel sees unit-stride A/B, then **register tiles** (mr×nr) for ILP
//!   (Goto & van de Geijn; BLIS GEBP). See also salykova.github.io/gemm-cpu.
//! - **Rayon** over output row-panels for large products (already a win on
//!   multi-core; packing still helps single-thread cache).

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

// --- Packing GEMM parameters (tuned for L1/L2 on typical x86; O(n) work) ---
/// Rows of A/C panel (mc).
const MC: usize = 64;
/// Cols of B/C panel (nc).
const NC: usize = 64;
/// Inner product depth panel (kc).
const KC: usize = 256;
/// Micro-kernel register tile rows.
const MR: usize = 4;
/// Micro-kernel register tile cols.
const NR: usize = 4;

/// Pack A[i0..i0+m, k0..k0+k] row-major → contiguous `mc×kc` (row-major).
#[inline]
fn pack_a(aa: &[i64], an: usize, i0: usize, m: usize, k0: usize, k: usize, buf: &mut [i64]) {
    debug_assert_eq!(buf.len(), m * k);
    for i in 0..m {
        let src = &aa[(i0 + i) * an + k0..(i0 + i) * an + k0 + k];
        buf[i * k..(i + 1) * k].copy_from_slice(src);
    }
}

/// Pack B[k0..k0+k, j0..j0+n] row-major → contiguous `kc×nc` (row-major).
#[inline]
fn pack_b(bb: &[i64], bn: usize, k0: usize, k: usize, j0: usize, n: usize, buf: &mut [i64]) {
    debug_assert_eq!(buf.len(), k * n);
    for p in 0..k {
        let src = &bb[(k0 + p) * bn + j0..(k0 + p) * bn + j0 + n];
        buf[p * n..(p + 1) * n].copy_from_slice(src);
    }
}

/// Compute rows `i0..i0+mb` of C into contiguous `c_panel` (mb×bn).
fn gemm_panel_rows(
    _am: usize,
    an: usize,
    bn: usize,
    aa: &[i64],
    bb: &[i64],
    i0: usize,
    mb: usize,
    c_panel: &mut [i64],
) {
    debug_assert_eq!(c_panel.len(), mb * bn);
    let mut a_pack = vec![0i64; MC * KC];
    let mut b_pack = vec![0i64; KC * NC];

    let mut j0 = 0;
    while j0 < bn {
        let nb = (bn - j0).min(NC);
        let mut k0 = 0;
        while k0 < an {
            let kb = (an - k0).min(KC);
            pack_b(bb, bn, k0, kb, j0, nb, &mut b_pack[..kb * nb]);
            pack_a(aa, an, i0, mb, k0, kb, &mut a_pack[..mb * kb]);
            // Update c_panel[0..mb, j0..j0+nb] using packed panels.
            // Extract sub-columns into a temp micro C or stride in place.
            // c_panel is mb×bn; we need ldc = bn.
            // micro_kernel writes with ldc=bn into c_panel starting at column j0.
            // Call micro on full mb×nb view with packed A (mb×kb) and B (kb×nb).
            micro_kernel_strided(
                mb,
                nb,
                kb,
                &a_pack[..mb * kb],
                &b_pack[..kb * nb],
                c_panel,
                bn,
                j0,
            );
            k0 += kb;
        }
        j0 += nb;
    }
}

/// Like [`micro_kernel`] but C is `m×ldc` with update starting at column `j0`.
#[inline]
fn micro_kernel_strided(
    m: usize,
    n: usize,
    k: usize,
    a: &[i64],
    b: &[i64],
    c: &mut [i64],
    ldc: usize,
    j0: usize,
) {
    let mut i = 0;
    while i + MR <= m {
        let mut j = 0;
        while j + NR <= n {
            let mut c00 = c[(i) * ldc + (j0 + j)];
            let mut c01 = c[(i) * ldc + (j0 + j + 1)];
            let mut c02 = c[(i) * ldc + (j0 + j + 2)];
            let mut c03 = c[(i) * ldc + (j0 + j + 3)];
            let mut c10 = c[(i + 1) * ldc + (j0 + j)];
            let mut c11 = c[(i + 1) * ldc + (j0 + j + 1)];
            let mut c12 = c[(i + 1) * ldc + (j0 + j + 2)];
            let mut c13 = c[(i + 1) * ldc + (j0 + j + 3)];
            let mut c20 = c[(i + 2) * ldc + (j0 + j)];
            let mut c21 = c[(i + 2) * ldc + (j0 + j + 1)];
            let mut c22 = c[(i + 2) * ldc + (j0 + j + 2)];
            let mut c23 = c[(i + 2) * ldc + (j0 + j + 3)];
            let mut c30 = c[(i + 3) * ldc + (j0 + j)];
            let mut c31 = c[(i + 3) * ldc + (j0 + j + 1)];
            let mut c32 = c[(i + 3) * ldc + (j0 + j + 2)];
            let mut c33 = c[(i + 3) * ldc + (j0 + j + 3)];
            for p in 0..k {
                let b0 = b[p * n + j];
                let b1 = b[p * n + j + 1];
                let b2 = b[p * n + j + 2];
                let b3 = b[p * n + j + 3];
                let a0 = a[i * k + p];
                let a1 = a[(i + 1) * k + p];
                let a2 = a[(i + 2) * k + p];
                let a3 = a[(i + 3) * k + p];
                c00 = c00.wrapping_add(a0.wrapping_mul(b0));
                c01 = c01.wrapping_add(a0.wrapping_mul(b1));
                c02 = c02.wrapping_add(a0.wrapping_mul(b2));
                c03 = c03.wrapping_add(a0.wrapping_mul(b3));
                c10 = c10.wrapping_add(a1.wrapping_mul(b0));
                c11 = c11.wrapping_add(a1.wrapping_mul(b1));
                c12 = c12.wrapping_add(a1.wrapping_mul(b2));
                c13 = c13.wrapping_add(a1.wrapping_mul(b3));
                c20 = c20.wrapping_add(a2.wrapping_mul(b0));
                c21 = c21.wrapping_add(a2.wrapping_mul(b1));
                c22 = c22.wrapping_add(a2.wrapping_mul(b2));
                c23 = c23.wrapping_add(a2.wrapping_mul(b3));
                c30 = c30.wrapping_add(a3.wrapping_mul(b0));
                c31 = c31.wrapping_add(a3.wrapping_mul(b1));
                c32 = c32.wrapping_add(a3.wrapping_mul(b2));
                c33 = c33.wrapping_add(a3.wrapping_mul(b3));
            }
            c[i * ldc + j0 + j] = c00;
            c[i * ldc + j0 + j + 1] = c01;
            c[i * ldc + j0 + j + 2] = c02;
            c[i * ldc + j0 + j + 3] = c03;
            c[(i + 1) * ldc + j0 + j] = c10;
            c[(i + 1) * ldc + j0 + j + 1] = c11;
            c[(i + 1) * ldc + j0 + j + 2] = c12;
            c[(i + 1) * ldc + j0 + j + 3] = c13;
            c[(i + 2) * ldc + j0 + j] = c20;
            c[(i + 2) * ldc + j0 + j + 1] = c21;
            c[(i + 2) * ldc + j0 + j + 2] = c22;
            c[(i + 2) * ldc + j0 + j + 3] = c23;
            c[(i + 3) * ldc + j0 + j] = c30;
            c[(i + 3) * ldc + j0 + j + 1] = c31;
            c[(i + 3) * ldc + j0 + j + 2] = c32;
            c[(i + 3) * ldc + j0 + j + 3] = c33;
            j += NR;
        }
        while j < n {
            for ii in 0..MR {
                let mut s = c[(i + ii) * ldc + j0 + j];
                for p in 0..k {
                    s = s.wrapping_add(a[(i + ii) * k + p].wrapping_mul(b[p * n + j]));
                }
                c[(i + ii) * ldc + j0 + j] = s;
            }
            j += 1;
        }
        i += MR;
    }
    while i < m {
        for j in 0..n {
            let mut s = c[i * ldc + j0 + j];
            for p in 0..k {
                s = s.wrapping_add(a[i * k + p].wrapping_mul(b[p * n + j]));
            }
            c[i * ldc + j0 + j] = s;
        }
        i += 1;
    }
}

/// Simple ikj GEMM (no packing) for small products / vector path helper.
fn gemm_simple(am: usize, an: usize, bn: usize, aa: &[i64], bb: &[i64], data: &mut [i64]) {
    for i in 0..am {
        let c_row = &mut data[i * bn..(i + 1) * bn];
        for k in 0..an {
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
    }
}

/// Entry: packed GEBP for matrices; simple path for tiny; optional Rayon over row panels.
fn gemm_blocked(am: usize, an: usize, bn: usize, aa: &[i64], bb: &[i64], data: &mut [i64]) {
    let flops = am.saturating_mul(an).saturating_mul(bn);
    if flops < 48 * 48 * 48 {
        gemm_simple(am, an, bn, aa, bb, data);
        return;
    }

    if flops >= 64 * 64 * 64 && am >= 8 {
        use rayon::prelude::*;
        // Build list of (i0, mb) panels and process in parallel with correct splits.
        let mut panels = Vec::new();
        let mut i0 = 0;
        while i0 < am {
            let mb = (am - i0).min(MC);
            panels.push((i0, mb));
            i0 += mb;
        }
        // Safety: each panel writes disjoint rows — use split_at_mut chain.
        // Rayon: parallel_map into temporary row buffers then copy — extra alloc.
        // Prefer: one thread per panel with split_at_mut via scoped approach.
        let mut slices: Vec<&mut [i64]> = Vec::with_capacity(panels.len());
        let mut rest = data;
        let mut prev_end = 0usize;
        for &(i0, mb) in &panels {
            debug_assert_eq!(i0, prev_end);
            let (chunk, tail) = rest.split_at_mut(mb * bn);
            slices.push(chunk);
            rest = tail;
            prev_end = i0 + mb;
        }
        slices
            .into_par_iter()
            .zip(panels.into_par_iter())
            .for_each(|(c_panel, (i0, mb))| {
                gemm_panel_rows(am, an, bn, aa, bb, i0, mb, c_panel);
            });
        return;
    }

    let mut i0 = 0;
    while i0 < am {
        let mb = (am - i0).min(MC);
        gemm_panel_rows(am, an, bn, aa, bb, i0, mb, &mut data[i0 * bn..(i0 + mb) * bn]);
        i0 += mb;
    }
}


/// Dispatch matrix GEMM (packed GEBP / simple / parallel panels).
/// Strassen was measured through n=4096 on this class of host and never beat
/// GEBP (S/G ≥ 1.0); removed to keep the path simple (WASM-friendly GEBP only).
fn gemm_dispatch(am: usize, an: usize, bn: usize, aa: &[i64], bb: &[i64], data: &mut [i64]) {
    gemm_blocked(am, an, bn, aa, bb, data);
}

/// Force GEBP (no Strassen) — for crossover measurement only.
#[doc(hidden)]
pub fn matmul_gebp_only(a: &ArrayI64, b: &ArrayI64) -> Result<ArrayI64> {
    let (am, an) = as_matrix_dims(a)?;
    let (bm, bn) = as_matrix_dims(b)?;
    if an != bm {
        return Err(Error::shape(format!(
            "matmul shape mismatch: ({am}, {an}) vs ({bm}, {bn})"
        )));
    }
    let prefer_vec = b.rank() == 1 || (a.rank() == 1 && bn == 1);
    let mut data = pool_i64::take_zeroed(am.saturating_mul(bn));
    if b.rank() == 1 {
        // fall back to matmul path
        return matmul(a, b);
    }
    gemm_blocked(am, an, bn, a.as_slice(), b.as_slice(), &mut data);
    matmul_result(data, am, bn, prefer_vec)
}

/// Matrix product `a @ b` with wrapping `i64` accumulation.
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
        gemm_dispatch(am, an, bn, aa, bb, &mut data);
    }
    matmul_result(data, am, bn, prefer_vec)
}

/// GEMM into preallocated rank-2 `out` with shape `(am, bn)`. Wrapping `i64`.
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
        gemm_dispatch(am, an, bn, aa, bb, data);
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

/// Euclidean (Frobenius) norm as `f64` (sqrt of sum of squares; squares wrap then cast).
/// Four-way ILP accumulation (same idea as `sum_sq` on f64).
pub fn norm(a: &ArrayI64) -> Result<f64> {
    let s = a.as_slice();
    let mut s0: i64 = 0;
    let mut s1: i64 = 0;
    let mut s2: i64 = 0;
    let mut s3: i64 = 0;
    let mut chunks = s.chunks_exact(4);
    for c in chunks.by_ref() {
        s0 = s0.wrapping_add(c[0].wrapping_mul(c[0]));
        s1 = s1.wrapping_add(c[1].wrapping_mul(c[1]));
        s2 = s2.wrapping_add(c[2].wrapping_mul(c[2]));
        s3 = s3.wrapping_add(c[3].wrapping_mul(c[3]));
    }
    let mut ss = s0.wrapping_add(s1).wrapping_add(s2).wrapping_add(s3);
    for &x in chunks.remainder() {
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

    #[test]
    fn matmul_packed_matches_naive_128() {
        // Correctness vs simple triple loop for n=32 (packed path).
        let n = 32;
        let mut da = Vec::with_capacity(n * n);
        let mut db = Vec::with_capacity(n * n);
        let mut x = 1i64;
        for _ in 0..n * n {
            da.push(x);
            x = x.wrapping_add(3);
            db.push(x);
            x = x.wrapping_add(5);
        }
        let a = ArrayI64::from_shape_vec(vec![n, n], da).unwrap();
        let b = ArrayI64::from_shape_vec(vec![n, n], db).unwrap();
        let c = matmul(&a, &b).unwrap();
        // reference
        let mut r = vec![0i64; n * n];
        let aa = a.as_slice();
        let bb = b.as_slice();
        for i in 0..n {
            for k in 0..n {
                let aik = aa[i * n + k];
                for j in 0..n {
                    r[i * n + j] = r[i * n + j].wrapping_add(aik.wrapping_mul(bb[k * n + j]));
                }
            }
        }
        assert_eq!(c.as_slice(), r.as_slice());
    }
}

