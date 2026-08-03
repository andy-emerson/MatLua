//! Dense linear algebra on [`ArrayI64`](crate::array::ArrayI64).
//!
//! Integer path (not faer/`f64`). Arithmetic is **wrapping** `i64`, matching
//! the rest of the `i64` surface.
//!
//! # Matmul algorithm (M7.c)
//!
//! Research notes (GotoBLAS / BLIS GEBP; portable integer GEMM):
//! - **Not** f64 promote + faer: breaks exactness past 2⁵³ and wrapping semantics.
//! - NumPy `int64 @ int64` has **no BLAS backend** (OpenBLAS/MKL are float); it
//!   falls off hard with n (cache). That is why fair NumPy times at n=4096 can
//!   fail to finish on small hosts — not because MatLua is “missing” matmul.
//! - MatLua uses **packing + mc×kc / kc×nc panels + mr×nr micro-kernel** (GEBP).
//!   Micro-kernel is **8×8** wrapping `i64` muls (wider tile ⇒ more ops per load).
//!   No AVX-512 `vpmullq` (not portable to WASM / older x86).
//! - Parallelism: split output row-panels when runtime `available_parallelism`
//!   and total flops justify ≥1 panel per thread (work rule, not a fixed n table).
//! - Strassen over rings is valid but was slower than this base kernel through
//!   n=4096 on measured hosts; kept off until the cubic kernel is stronger.

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

// --- Packing GEMM (Goto/BLIS GEBP structure) ---------------------------------
//
// Algorithm (portable; no host-specific timing thresholds):
// 1. Panel C by mc rows × nc cols (GEBP).
// 2. Pack A (mc×kc) and B (kc×nc) so the micro-kernel sees unit-stride loads.
// 3. Micro-kernel: mr×nr register tile (8×8) of rank-1 updates over k.
//
// Why not only 4×4: larger tiles raise arithmetic intensity (ops per load) —
// standard BLIS/Goto design, independent of one machine’s cache sizes.
// Why not AVX-512 vpmullq: not portable (WASM / older x86); we stay on
// wrapping scalar i64 muls inside a wider software tile.
// Panel sizes target L2-resident packs for 64-bit elements (literature defaults).

/// Rows of A/C panel (mc).
const MC: usize = 64;
/// Cols of B/C panel (nc).
const NC: usize = 64;
/// Inner depth panel (kc).
const KC: usize = 128;
/// Micro-kernel rows (mr).
const MR: usize = 8;
/// Micro-kernel cols (nr).
const NR: usize = 8;

/// Pack A[i0.., k0..] row-major → row-major mc×kc (contiguous rows of A).
#[inline]
fn pack_a(aa: &[i64], an: usize, i0: usize, m: usize, k0: usize, k: usize, buf: &mut [i64]) {
    for i in 0..m {
        let src = &aa[(i0 + i) * an + k0..(i0 + i) * an + k0 + k];
        buf[i * k..(i + 1) * k].copy_from_slice(src);
    }
}

/// Pack B[k0.., j0..] → **panel-major for NR micro-panels**: for each jr in 0..n
/// step NR, store kc×nr' contiguous (matches micro-kernel B loads).
#[inline]
fn pack_b_nr(bb: &[i64], bn: usize, k0: usize, k: usize, j0: usize, n: usize, buf: &mut [i64]) {
    let mut off = 0;
    let mut jr = 0;
    while jr < n {
        let nr = (n - jr).min(NR);
        for p in 0..k {
            let src = &bb[(k0 + p) * bn + (j0 + jr)..(k0 + p) * bn + (j0 + jr) + nr];
            buf[off + p * nr..off + p * nr + nr].copy_from_slice(src);
        }
        off += k * nr;
        jr += nr;
    }
}

/// 8×8 wrapping micro-kernel: C[i..i+mr, j..j+nr] += A[i.., 0..k] * B_panel[0..k, 0..nr]
/// A is row-major (mr rows × k); B is k×nr contiguous.
#[inline]
fn micro_8x8(
    k: usize,
    a: &[i64], // leading row stride = k; at least 8 rows
    b: &[i64], // k × 8
    c: &mut [i64],
    ldc: usize,
    i0: usize,
    j0: usize,
) {
    // Load 8×8 C tile into registers
    let mut c00 = c[i0 * ldc + j0];
    let mut c01 = c[i0 * ldc + j0 + 1];
    let mut c02 = c[i0 * ldc + j0 + 2];
    let mut c03 = c[i0 * ldc + j0 + 3];
    let mut c04 = c[i0 * ldc + j0 + 4];
    let mut c05 = c[i0 * ldc + j0 + 5];
    let mut c06 = c[i0 * ldc + j0 + 6];
    let mut c07 = c[i0 * ldc + j0 + 7];
    let mut c10 = c[(i0 + 1) * ldc + j0];
    let mut c11 = c[(i0 + 1) * ldc + j0 + 1];
    let mut c12 = c[(i0 + 1) * ldc + j0 + 2];
    let mut c13 = c[(i0 + 1) * ldc + j0 + 3];
    let mut c14 = c[(i0 + 1) * ldc + j0 + 4];
    let mut c15 = c[(i0 + 1) * ldc + j0 + 5];
    let mut c16 = c[(i0 + 1) * ldc + j0 + 6];
    let mut c17 = c[(i0 + 1) * ldc + j0 + 7];
    let mut c20 = c[(i0 + 2) * ldc + j0];
    let mut c21 = c[(i0 + 2) * ldc + j0 + 1];
    let mut c22 = c[(i0 + 2) * ldc + j0 + 2];
    let mut c23 = c[(i0 + 2) * ldc + j0 + 3];
    let mut c24 = c[(i0 + 2) * ldc + j0 + 4];
    let mut c25 = c[(i0 + 2) * ldc + j0 + 5];
    let mut c26 = c[(i0 + 2) * ldc + j0 + 6];
    let mut c27 = c[(i0 + 2) * ldc + j0 + 7];
    let mut c30 = c[(i0 + 3) * ldc + j0];
    let mut c31 = c[(i0 + 3) * ldc + j0 + 1];
    let mut c32 = c[(i0 + 3) * ldc + j0 + 2];
    let mut c33 = c[(i0 + 3) * ldc + j0 + 3];
    let mut c34 = c[(i0 + 3) * ldc + j0 + 4];
    let mut c35 = c[(i0 + 3) * ldc + j0 + 5];
    let mut c36 = c[(i0 + 3) * ldc + j0 + 6];
    let mut c37 = c[(i0 + 3) * ldc + j0 + 7];
    let mut c40 = c[(i0 + 4) * ldc + j0];
    let mut c41 = c[(i0 + 4) * ldc + j0 + 1];
    let mut c42 = c[(i0 + 4) * ldc + j0 + 2];
    let mut c43 = c[(i0 + 4) * ldc + j0 + 3];
    let mut c44 = c[(i0 + 4) * ldc + j0 + 4];
    let mut c45 = c[(i0 + 4) * ldc + j0 + 5];
    let mut c46 = c[(i0 + 4) * ldc + j0 + 6];
    let mut c47 = c[(i0 + 4) * ldc + j0 + 7];
    let mut c50 = c[(i0 + 5) * ldc + j0];
    let mut c51 = c[(i0 + 5) * ldc + j0 + 1];
    let mut c52 = c[(i0 + 5) * ldc + j0 + 2];
    let mut c53 = c[(i0 + 5) * ldc + j0 + 3];
    let mut c54 = c[(i0 + 5) * ldc + j0 + 4];
    let mut c55 = c[(i0 + 5) * ldc + j0 + 5];
    let mut c56 = c[(i0 + 5) * ldc + j0 + 6];
    let mut c57 = c[(i0 + 5) * ldc + j0 + 7];
    let mut c60 = c[(i0 + 6) * ldc + j0];
    let mut c61 = c[(i0 + 6) * ldc + j0 + 1];
    let mut c62 = c[(i0 + 6) * ldc + j0 + 2];
    let mut c63 = c[(i0 + 6) * ldc + j0 + 3];
    let mut c64 = c[(i0 + 6) * ldc + j0 + 4];
    let mut c65 = c[(i0 + 6) * ldc + j0 + 5];
    let mut c66 = c[(i0 + 6) * ldc + j0 + 6];
    let mut c67 = c[(i0 + 6) * ldc + j0 + 7];
    let mut c70 = c[(i0 + 7) * ldc + j0];
    let mut c71 = c[(i0 + 7) * ldc + j0 + 1];
    let mut c72 = c[(i0 + 7) * ldc + j0 + 2];
    let mut c73 = c[(i0 + 7) * ldc + j0 + 3];
    let mut c74 = c[(i0 + 7) * ldc + j0 + 4];
    let mut c75 = c[(i0 + 7) * ldc + j0 + 5];
    let mut c76 = c[(i0 + 7) * ldc + j0 + 6];
    let mut c77 = c[(i0 + 7) * ldc + j0 + 7];

    for p in 0..k {
        let b0 = b[p * 8];
        let b1 = b[p * 8 + 1];
        let b2 = b[p * 8 + 2];
        let b3 = b[p * 8 + 3];
        let b4 = b[p * 8 + 4];
        let b5 = b[p * 8 + 5];
        let b6 = b[p * 8 + 6];
        let b7 = b[p * 8 + 7];
        let a0 = a[0 * k + p];
        c00 = c00.wrapping_add(a0.wrapping_mul(b0));
        c01 = c01.wrapping_add(a0.wrapping_mul(b1));
        c02 = c02.wrapping_add(a0.wrapping_mul(b2));
        c03 = c03.wrapping_add(a0.wrapping_mul(b3));
        c04 = c04.wrapping_add(a0.wrapping_mul(b4));
        c05 = c05.wrapping_add(a0.wrapping_mul(b5));
        c06 = c06.wrapping_add(a0.wrapping_mul(b6));
        c07 = c07.wrapping_add(a0.wrapping_mul(b7));
        let a1 = a[1 * k + p];
        c10 = c10.wrapping_add(a1.wrapping_mul(b0));
        c11 = c11.wrapping_add(a1.wrapping_mul(b1));
        c12 = c12.wrapping_add(a1.wrapping_mul(b2));
        c13 = c13.wrapping_add(a1.wrapping_mul(b3));
        c14 = c14.wrapping_add(a1.wrapping_mul(b4));
        c15 = c15.wrapping_add(a1.wrapping_mul(b5));
        c16 = c16.wrapping_add(a1.wrapping_mul(b6));
        c17 = c17.wrapping_add(a1.wrapping_mul(b7));
        let a2 = a[2 * k + p];
        c20 = c20.wrapping_add(a2.wrapping_mul(b0));
        c21 = c21.wrapping_add(a2.wrapping_mul(b1));
        c22 = c22.wrapping_add(a2.wrapping_mul(b2));
        c23 = c23.wrapping_add(a2.wrapping_mul(b3));
        c24 = c24.wrapping_add(a2.wrapping_mul(b4));
        c25 = c25.wrapping_add(a2.wrapping_mul(b5));
        c26 = c26.wrapping_add(a2.wrapping_mul(b6));
        c27 = c27.wrapping_add(a2.wrapping_mul(b7));
        let a3 = a[3 * k + p];
        c30 = c30.wrapping_add(a3.wrapping_mul(b0));
        c31 = c31.wrapping_add(a3.wrapping_mul(b1));
        c32 = c32.wrapping_add(a3.wrapping_mul(b2));
        c33 = c33.wrapping_add(a3.wrapping_mul(b3));
        c34 = c34.wrapping_add(a3.wrapping_mul(b4));
        c35 = c35.wrapping_add(a3.wrapping_mul(b5));
        c36 = c36.wrapping_add(a3.wrapping_mul(b6));
        c37 = c37.wrapping_add(a3.wrapping_mul(b7));
        let a4 = a[4 * k + p];
        c40 = c40.wrapping_add(a4.wrapping_mul(b0));
        c41 = c41.wrapping_add(a4.wrapping_mul(b1));
        c42 = c42.wrapping_add(a4.wrapping_mul(b2));
        c43 = c43.wrapping_add(a4.wrapping_mul(b3));
        c44 = c44.wrapping_add(a4.wrapping_mul(b4));
        c45 = c45.wrapping_add(a4.wrapping_mul(b5));
        c46 = c46.wrapping_add(a4.wrapping_mul(b6));
        c47 = c47.wrapping_add(a4.wrapping_mul(b7));
        let a5 = a[5 * k + p];
        c50 = c50.wrapping_add(a5.wrapping_mul(b0));
        c51 = c51.wrapping_add(a5.wrapping_mul(b1));
        c52 = c52.wrapping_add(a5.wrapping_mul(b2));
        c53 = c53.wrapping_add(a5.wrapping_mul(b3));
        c54 = c54.wrapping_add(a5.wrapping_mul(b4));
        c55 = c55.wrapping_add(a5.wrapping_mul(b5));
        c56 = c56.wrapping_add(a5.wrapping_mul(b6));
        c57 = c57.wrapping_add(a5.wrapping_mul(b7));
        let a6 = a[6 * k + p];
        c60 = c60.wrapping_add(a6.wrapping_mul(b0));
        c61 = c61.wrapping_add(a6.wrapping_mul(b1));
        c62 = c62.wrapping_add(a6.wrapping_mul(b2));
        c63 = c63.wrapping_add(a6.wrapping_mul(b3));
        c64 = c64.wrapping_add(a6.wrapping_mul(b4));
        c65 = c65.wrapping_add(a6.wrapping_mul(b5));
        c66 = c66.wrapping_add(a6.wrapping_mul(b6));
        c67 = c67.wrapping_add(a6.wrapping_mul(b7));
        let a7 = a[7 * k + p];
        c70 = c70.wrapping_add(a7.wrapping_mul(b0));
        c71 = c71.wrapping_add(a7.wrapping_mul(b1));
        c72 = c72.wrapping_add(a7.wrapping_mul(b2));
        c73 = c73.wrapping_add(a7.wrapping_mul(b3));
        c74 = c74.wrapping_add(a7.wrapping_mul(b4));
        c75 = c75.wrapping_add(a7.wrapping_mul(b5));
        c76 = c76.wrapping_add(a7.wrapping_mul(b6));
        c77 = c77.wrapping_add(a7.wrapping_mul(b7));
    }

    c[i0 * ldc + j0] = c00;
    c[i0 * ldc + j0 + 1] = c01;
    c[i0 * ldc + j0 + 2] = c02;
    c[i0 * ldc + j0 + 3] = c03;
    c[i0 * ldc + j0 + 4] = c04;
    c[i0 * ldc + j0 + 5] = c05;
    c[i0 * ldc + j0 + 6] = c06;
    c[i0 * ldc + j0 + 7] = c07;
    c[(i0 + 1) * ldc + j0] = c10;
    c[(i0 + 1) * ldc + j0 + 1] = c11;
    c[(i0 + 1) * ldc + j0 + 2] = c12;
    c[(i0 + 1) * ldc + j0 + 3] = c13;
    c[(i0 + 1) * ldc + j0 + 4] = c14;
    c[(i0 + 1) * ldc + j0 + 5] = c15;
    c[(i0 + 1) * ldc + j0 + 6] = c16;
    c[(i0 + 1) * ldc + j0 + 7] = c17;
    c[(i0 + 2) * ldc + j0] = c20;
    c[(i0 + 2) * ldc + j0 + 1] = c21;
    c[(i0 + 2) * ldc + j0 + 2] = c22;
    c[(i0 + 2) * ldc + j0 + 3] = c23;
    c[(i0 + 2) * ldc + j0 + 4] = c24;
    c[(i0 + 2) * ldc + j0 + 5] = c25;
    c[(i0 + 2) * ldc + j0 + 6] = c26;
    c[(i0 + 2) * ldc + j0 + 7] = c27;
    c[(i0 + 3) * ldc + j0] = c30;
    c[(i0 + 3) * ldc + j0 + 1] = c31;
    c[(i0 + 3) * ldc + j0 + 2] = c32;
    c[(i0 + 3) * ldc + j0 + 3] = c33;
    c[(i0 + 3) * ldc + j0 + 4] = c34;
    c[(i0 + 3) * ldc + j0 + 5] = c35;
    c[(i0 + 3) * ldc + j0 + 6] = c36;
    c[(i0 + 3) * ldc + j0 + 7] = c37;
    c[(i0 + 4) * ldc + j0] = c40;
    c[(i0 + 4) * ldc + j0 + 1] = c41;
    c[(i0 + 4) * ldc + j0 + 2] = c42;
    c[(i0 + 4) * ldc + j0 + 3] = c43;
    c[(i0 + 4) * ldc + j0 + 4] = c44;
    c[(i0 + 4) * ldc + j0 + 5] = c45;
    c[(i0 + 4) * ldc + j0 + 6] = c46;
    c[(i0 + 4) * ldc + j0 + 7] = c47;
    c[(i0 + 5) * ldc + j0] = c50;
    c[(i0 + 5) * ldc + j0 + 1] = c51;
    c[(i0 + 5) * ldc + j0 + 2] = c52;
    c[(i0 + 5) * ldc + j0 + 3] = c53;
    c[(i0 + 5) * ldc + j0 + 4] = c54;
    c[(i0 + 5) * ldc + j0 + 5] = c55;
    c[(i0 + 5) * ldc + j0 + 6] = c56;
    c[(i0 + 5) * ldc + j0 + 7] = c57;
    c[(i0 + 6) * ldc + j0] = c60;
    c[(i0 + 6) * ldc + j0 + 1] = c61;
    c[(i0 + 6) * ldc + j0 + 2] = c62;
    c[(i0 + 6) * ldc + j0 + 3] = c63;
    c[(i0 + 6) * ldc + j0 + 4] = c64;
    c[(i0 + 6) * ldc + j0 + 5] = c65;
    c[(i0 + 6) * ldc + j0 + 6] = c66;
    c[(i0 + 6) * ldc + j0 + 7] = c67;
    c[(i0 + 7) * ldc + j0] = c70;
    c[(i0 + 7) * ldc + j0 + 1] = c71;
    c[(i0 + 7) * ldc + j0 + 2] = c72;
    c[(i0 + 7) * ldc + j0 + 3] = c73;
    c[(i0 + 7) * ldc + j0 + 4] = c74;
    c[(i0 + 7) * ldc + j0 + 5] = c75;
    c[(i0 + 7) * ldc + j0 + 6] = c76;
    c[(i0 + 7) * ldc + j0 + 7] = c77;
}

/// Generic remainder micro-kernel (any m,n ≤ MR,NR not full tile).
#[inline]
fn micro_edge(
    m: usize,
    n: usize,
    k: usize,
    a: &[i64],
    b: &[i64],
    c: &mut [i64],
    ldc: usize,
    i0: usize,
    j0: usize,
    b_nr: usize,
) {
    for ii in 0..m {
        for jj in 0..n {
            let mut s = c[(i0 + ii) * ldc + (j0 + jj)];
            for p in 0..k {
                s = s.wrapping_add(a[ii * k + p].wrapping_mul(b[p * b_nr + jj]));
            }
            c[(i0 + ii) * ldc + (j0 + jj)] = s;
        }
    }
}

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
    // Pack buffers sized for full panels (reused across k/j loops).
    let mut a_pack = vec![0i64; MC * KC];
    let mut b_pack = vec![0i64; KC * NC];

    let mut j0 = 0;
    while j0 < bn {
        let nb = (bn - j0).min(NC);
        let mut k0 = 0;
        while k0 < an {
            let kb = (an - k0).min(KC);
            pack_a(aa, an, i0, mb, k0, kb, &mut a_pack[..mb * kb]);
            pack_b_nr(bb, bn, k0, kb, j0, nb, &mut b_pack[..kb * nb]);

            // Walk B micro-panels and A row tiles
            let mut jr = 0;
            let mut b_off = 0;
            while jr < nb {
                let nr = (nb - jr).min(NR);
                let mut ir = 0;
                while ir < mb {
                    let mr = (mb - ir).min(MR);
                    if mr == MR && nr == NR {
                        micro_8x8(
                            kb,
                            &a_pack[ir * kb..],
                            &b_pack[b_off..b_off + kb * NR],
                            c_panel,
                            bn,
                            ir,
                            j0 + jr,
                        );
                    } else {
                        micro_edge(
                            mr,
                            nr,
                            kb,
                            &a_pack[ir * kb..(ir + mr) * kb],
                            &b_pack[b_off..b_off + kb * nr],
                            c_panel,
                            bn,
                            ir,
                            j0 + jr,
                            nr,
                        );
                    }
                    ir += mr;
                }
                b_off += kb * nr;
                jr += nr;
            }
            k0 += kb;
        }
        j0 += nb;
    }
}

/// Simple ikj GEMM for tiny products (packing overhead not amortized).
fn gemm_simple(am: usize, an: usize, bn: usize, aa: &[i64], bb: &[i64], data: &mut [i64]) {
    for i in 0..am {
        let c_row = &mut data[i * bn..(i + 1) * bn];
        for p in 0..an {
            let aik = aa[i * an + p];
            if aik == 0 {
                continue;
            }
            let b_row = &bb[p * bn..(p + 1) * bn];
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

/// Parallelize only when there are ≥2 row-panels **and** enough work that
/// splitting is meaningful: `flops / nthreads` above one full panel product.
/// Uses runtime `available_parallelism` (not a fixed n-threshold from one host).
fn gemm_blocked(am: usize, an: usize, bn: usize, aa: &[i64], bb: &[i64], data: &mut [i64]) {
    let flops = (am as u64).saturating_mul(an as u64).saturating_mul(bn as u64);
    // Tiny: no packing.
    if flops < (48u64 * 48 * 48) {
        gemm_simple(am, an, bn, aa, bb, data);
        return;
    }

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);
    // One panel ≈ MC×an×bn ops; want ≥1 panel of work per thread to parallelize.
    let panel_work = (MC as u64).saturating_mul(an as u64).saturating_mul(bn as u64);
    let want_par = nthreads >= 2
        && am >= MC * 2
        && flops >= panel_work.saturating_mul(nthreads as u64);

    if want_par {
        use rayon::prelude::*;
        let mut panels = Vec::new();
        let mut i0 = 0;
        while i0 < am {
            let mb = (am - i0).min(MC);
            panels.push((i0, mb));
            i0 += mb;
        }
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
    fn matmul_packed_matches_naive_96() {
        // n not multiple of MR/NR — edges of 8×8 micro-kernel.
        let n = 96;
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

