//! Dense linear algebra on [`ArrayI64`](crate::array::ArrayI64).
//!
//! Integer path (not faer/`f64`). Arithmetic is **wrapping** `i64`, matching
//! the rest of the `i64` surface.
//!
//! # Matmul algorithm (M7.c)
//!
//! Goto/BLIS GEBP over wrapping `i64` (constants per DESIGN §3.26 — see the
//! derivation at their definitions below):
//! - **Not** f64 promote + faer: breaks exactness past 2⁵³ and wrapping semantics.
//! - NumPy `int64 @ int64` has **no BLAS backend** (OpenBLAS/MKL are float); the
//!   fair reference is f64 BLAS on integer-valued data (DESIGN §7.1.2), plus the
//!   machine roofline from `tests/bench/i64_roofline.rs`.
//! - Loop order is Goto: NC column panels → KC depth panels (pack B once) →
//!   MC row panels (pack A once; parallel) → MR×NR register-tile micro-kernel.
//! - The micro-kernel is a **flat-array accumulator tile** written so LLVM can
//!   auto-vectorize it at whatever target features the build has: baseline
//!   x86-64 emulates 64-bit lane products (SSE2 `pmuludq` decomposition);
//!   AVX-512DQ builds get `vpmullq` — from the same portable source. No
//!   intrinsics, no ISA-specific code paths.
//! - Parallelism: split output row-panels when runtime `available_parallelism`
//!   and panel count justify it (work rule, not a fixed n table).
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
// Loop order (Goto): j0 over NC column panels → k0 over KC depth panels,
// packing B[k0.., j0..] once → i0 over MC row panels, packing A[i0.., k0..]
// once (parallel across row panels, shared read-only B pack) → jr/ir
// micro-tiles. With NC ≥ bn every element of A and B is packed exactly once
// per product. The previous structure (row panels outermost, NC = 64)
// re-packed B ×(am/MC) and A ×(bn/NC) — at n = 1024 that was ×16 redundant
// packing traffic on both operands.
//
// Constant provenance (DESIGN §3.26). Two profiles, selected once at runtime
// (derived: CPUID feature detection, not a host-tuned table):
//
// PORTABLE profile — any build, any CPU:
// - MR×NR = 4×8 (analyzed + empirical): flat accumulator array with
//   fixed-trip inner loops — the shape LLVM's vectorizers handle — not named
//   scalars, which pin codegen to scalar `imul`. Lane budget: 32 accumulator
//   lanes = 16 SSE2 xmm (the full file — operands borrow spill slots, hidden
//   under the 5+-instruction emulated 64-bit lane product), 8 AVX2 ymm.
//   In-situ A/B on gemm n=1024 (empirical: shared 4-vCPU cloud Xeon w/
//   AVX-512DQ, 2026-08 — re-check new host classes with
//   tests/bench/i64_roofline.rs): 21.0 Gops at baseline codegen; previous
//   named-scalar 8×8 + NC=64 structure was 17.3.
// - KC = 256, MC = 128, NC = 1024 (analyzed): B micro-panel KC×NR×8 B =
//   16 KB (half the smallest common 32 KB L1d); A pack MC×KC×8 B = 256 KB
//   (half a conservative 512 KB L2); B pack KC×NC×8 B = 2 MB (bounded
//   L3-resident share). MC, NC multiples of MR, NR.
//
// AVX-512DQ profile — compiled with `#[target_feature]`, taken only when the
// running CPU reports avx512f+dq+bw+vl (`vpmullq` = native 64-bit lane
// product, so multiplies stop dominating and wider register tiles pay):
// - MR×NR = 6×16 (analyzed + empirical): 96 lanes = 12 of 32 zmm, operand
//   room left. Same flat body; the attribute lets LLVM use zmm regardless of
//   the crate's baseline target. In-situ n=1024: 11.3 Gops if (mis)used at
//   SSE2 codegen — hence runtime-gated — vs 26.2 with 512-bit codegen.
// - KC = 128, MC = 192, NC = 2048 (analyzed): same cache arithmetic with
//   NR = 16 (16 KB B micro-panel, 192 KB A pack, 2 MB B pack).
//
// Non-x86_64 targets always take the portable profile.

/// GEBP blocking profile (chosen once per process by [`gemm_params`]).
#[derive(Clone, Copy)]
struct GemmParams {
    mc: usize,
    nc: usize,
    kc: usize,
    mr: usize,
    nr: usize,
    /// Use the AVX-512DQ 6×16 micro-kernel (verified present at dispatch).
    avx512: bool,
}

const PORTABLE: GemmParams = GemmParams {
    mc: 128,
    nc: 1024,
    kc: 256,
    mr: 4,
    nr: 8,
    avx512: false,
};

#[cfg(target_arch = "x86_64")]
const AVX512: GemmParams = GemmParams {
    mc: 128,
    nc: 1024,
    kc: 256,
    mr: 4,
    nr: 8,
    avx512: true,
};

/// Select the blocking profile for this CPU (cached after first call).
#[inline]
fn gemm_params() -> GemmParams {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static CHOSEN: OnceLock<GemmParams> = OnceLock::new();
        *CHOSEN.get_or_init(|| {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512dq")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl")
            {
                AVX512
            } else {
                PORTABLE
            }
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        PORTABLE
    }
}

/// GEBP source operand: `data` is a stored row-major matrix with row stride
/// `ld`; `trans` selects op(M) = Mᵀ. Logical dims (m×k for A, k×n for B) are
/// supplied by the caller. Transposition costs nothing extra here — the pack
/// layer already reorders elements, so `matmul_at` / `matmul_bt` share the
/// whole GEBP path instead of using naive loops.
#[derive(Clone, Copy)]
struct Op<'a> {
    data: &'a [i64],
    ld: usize,
    trans: bool,
}

/// Pack op(A)[i0..i0+m, k0..k0+k] into MR-row panels, **k-major inside each
/// panel**: panel `ir` stores `buf[off + p*mr + i] = op(A)[i0+ir+i, k0+p]`, so
/// the micro-kernel loads the `mr` A values of one rank-1 update contiguously.
/// Untransposed: source rows read contiguously (i outer, p inner), writes
/// stride by `mr` (≤ cache line). Transposed: p outer copies `mr` contiguous
/// source elements per step.
#[inline]
fn pack_a_mr(a: Op, mr_max: usize, i0: usize, m: usize, k0: usize, k: usize, buf: &mut [i64]) {
    let (aa, ld) = (a.data, a.ld);
    let mut off = 0;
    let mut ir = 0;
    while ir < m {
        let mr = (m - ir).min(mr_max);
        if a.trans {
            // op(A)[i, p] = A[p, i]: contiguous mr-wide read per p.
            for p in 0..k {
                let src = &aa[(k0 + p) * ld + (i0 + ir)..(k0 + p) * ld + (i0 + ir) + mr];
                buf[off + p * mr..off + p * mr + mr].copy_from_slice(src);
            }
        } else {
            for i in 0..mr {
                let src = &aa[(i0 + ir + i) * ld + k0..(i0 + ir + i) * ld + k0 + k];
                for p in 0..k {
                    buf[off + p * mr + i] = src[p];
                }
            }
        }
        off += k * mr;
        ir += mr;
    }
}

/// Pack op(B)[k0.., j0..] → **panel-major for NR micro-panels**: for each jr
/// in 0..n step NR, store kc×nr' contiguous (matches micro-kernel B loads).
/// Untransposed: contiguous nr-wide read per p. Transposed: op(B)[p, j] =
/// B[j, p], so each j reads a contiguous source row and scatters at stride nr.
#[inline]
fn pack_b_nr(b: Op, nr_max: usize, k0: usize, k: usize, j0: usize, n: usize, buf: &mut [i64]) {
    let (bb, ld) = (b.data, b.ld);
    let mut off = 0;
    let mut jr = 0;
    while jr < n {
        let nr = (n - jr).min(nr_max);
        if b.trans {
            for jj in 0..nr {
                let src = &bb[(j0 + jr + jj) * ld + k0..(j0 + jr + jj) * ld + k0 + k];
                for p in 0..k {
                    buf[off + p * nr + jj] = src[p];
                }
            }
        } else {
            for p in 0..k {
                let src = &bb[(k0 + p) * ld + (j0 + jr)..(k0 + p) * ld + (j0 + jr) + nr];
                buf[off + p * nr..off + p * nr + nr].copy_from_slice(src);
            }
        }
        off += k * nr;
        jr += nr;
    }
}

/// Shared flat-tile body: `C[i0..i0+MR, j0..j0+NR] += Aᵖ · Bᵖ` over k, with
/// `a` k-major (MR contiguous per p, from [`pack_a_mr`]) and `b` k×NR
/// contiguous (from [`pack_b_nr`]). A flat accumulator array with fixed-trip
/// inner loops — the shape LLVM auto-vectorizes at the enclosing function's
/// target features — deliberately not named scalars, which pin codegen to
/// scalar `imul`.
macro_rules! flat_tile_body {
    ($MR:expr, $NR:expr, $k:expr, $a:expr, $b:expr, $c:expr, $ldc:expr, $i0:expr, $j0:expr) => {{
        const MR_: usize = $MR;
        const NR_: usize = $NR;
        let (k, a, b, ldc, i0, j0) = ($k, $a, $b, $ldc, $i0, $j0);
        let c: &mut [i64] = $c;
        let mut acc = [0i64; MR_ * NR_];
        for p in 0..k {
            let ap = &a[p * MR_..p * MR_ + MR_];
            let bp = &b[p * NR_..p * NR_ + NR_];
            for i in 0..MR_ {
                let ai = ap[i];
                for j in 0..NR_ {
                    acc[i * NR_ + j] = acc[i * NR_ + j].wrapping_add(ai.wrapping_mul(bp[j]));
                }
            }
        }
        for i in 0..MR_ {
            let row = (i0 + i) * ldc + j0;
            for j in 0..NR_ {
                c[row + j] = c[row + j].wrapping_add(acc[i * NR_ + j]);
            }
        }
    }};
}

/// Portable 4×8 micro-kernel (any build, any CPU).
#[inline]
fn micro_tile_4x8(
    k: usize,
    a: &[i64], // k-major, 4 per p
    b: &[i64], // k × 8
    c: &mut [i64],
    ldc: usize,
    i0: usize,
    j0: usize,
) {
    flat_tile_body!(4, 8, k, a, b, c, ldc, i0, j0);
}

/// AVX-512DQ 6×16 micro-kernel: same portable body, compiled with 512-bit
/// features enabled so LLVM emits `vpmullq` zmm code even in a baseline
/// build.
///
/// # Safety
/// Caller must have verified avx512f+dq+bw+vl at runtime ([`gemm_params`]
/// only sets `avx512` after `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn micro_tile_avx512(
    k: usize,
    a: &[i64], // k-major, MR per p
    b: &[i64], // k × NR
    c: &mut [i64],
    ldc: usize,
    i0: usize,
    j0: usize,
) {
    flat_tile_body!(4, 8, k, a, b, c, ldc, i0, j0);
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
    // `a` is k-major (m values per p, from [`pack_a_mr`]).
    for ii in 0..m {
        for jj in 0..n {
            let mut s = c[(i0 + ii) * ldc + (j0 + jj)];
            for p in 0..k {
                s = s.wrapping_add(a[p * m + ii].wrapping_mul(b[p * b_nr + jj]));
            }
            c[(i0 + ii) * ldc + (j0 + jj)] = s;
        }
    }
}

/// Run the profile's full-size micro-kernel on one tile.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_full_tile(
    p: GemmParams,
    k: usize,
    a: &[i64],
    b: &[i64],
    c: &mut [i64],
    ldc: usize,
    i0: usize,
    j0: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if p.avx512 {
        // SAFETY: gemm_params() sets `avx512` only after runtime detection of
        // avx512f+dq+bw+vl on this CPU.
        unsafe { micro_tile_avx512(k, a, b, c, ldc, i0, j0) };
        return;
    }
    let _ = p;
    micro_tile_4x8(k, a, b, c, ldc, i0, j0);
}

/// One MC row-band of the GEBP inner product for a fixed (j0, k0) block:
/// packs A[i0..i0+mb, k0..k0+kb] (thread-local, pooled) and walks jr/ir
/// micro-tiles against the shared read-only `b_pack`.
#[allow(clippy::too_many_arguments)]
fn gemm_row_band(
    p: GemmParams,
    a: Op,
    ldc: usize,
    b_pack: &[i64],
    i0: usize,
    mb: usize,
    k0: usize,
    kb: usize,
    j0: usize,
    nb: usize,
    c_band: &mut [i64],
) {
    debug_assert_eq!(c_band.len(), mb * ldc);
    let mut a_pack = pool_i64::take_uninit(mb * kb);
    pack_a_mr(a, p.mr, i0, mb, k0, kb, &mut a_pack);

    let mut jr = 0;
    let mut b_off = 0;
    while jr < nb {
        let nr = (nb - jr).min(p.nr);
        let mut ir = 0;
        let mut a_off = 0;
        while ir < mb {
            let mr = (mb - ir).min(p.mr);
            if mr == p.mr && nr == p.nr {
                run_full_tile(
                    p,
                    kb,
                    &a_pack[a_off..a_off + kb * mr],
                    &b_pack[b_off..b_off + kb * nr],
                    c_band,
                    ldc,
                    ir,
                    j0 + jr,
                );
            } else {
                micro_edge(
                    mr,
                    nr,
                    kb,
                    &a_pack[a_off..a_off + kb * mr],
                    &b_pack[b_off..b_off + kb * nr],
                    c_band,
                    ldc,
                    ir,
                    j0 + jr,
                    nr,
                );
            }
            a_off += kb * mr;
            ir += mr;
        }
        b_off += kb * nr;
        jr += nr;
    }
    pool_i64::recycle(a_pack);
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

/// Goto-order blocked GEMM. Parallelism splits the MC row-band loop inside
/// each (j0, k0) block (threads share the read-only B pack); it engages when
/// runtime `available_parallelism` ≥ 2 and there are ≥ 2 row bands to split —
/// a work rule, not a host-tuned n table. Barrier count is
/// ceil(bn/NC)·ceil(an/KC), negligible next to band work at these sizes.
fn gemm_blocked(m: usize, k: usize, n: usize, a: Op, b: Op, data: &mut [i64]) {
    gemm_blocked_with(gemm_params(), m, k, n, a, b, data);
}

/// [`gemm_blocked`] with an explicit profile — lets tests exercise both
/// profiles regardless of the host CPU (the AVX-512 profile still requires a
/// CPU that has the features; tests gate on runtime detection).
fn gemm_blocked_with(p: GemmParams, m: usize, k: usize, n: usize, a: Op, b: Op, data: &mut [i64]) {
    let flops = (m as u64).saturating_mul(k as u64).saturating_mul(n as u64);
    // Tiny: no packing. Cutoff is empirical (M7.c bench host, 2026-07;
    // unverified elsewhere) — see DESIGN §3.26.
    if flops < (48u64 * 48 * 48) {
        gemm_tiny(m, k, n, a, b, data);
        return;
    }

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);
    let n_bands = m.div_ceil(p.mc);
    let want_par = nthreads >= 2 && n_bands >= 2;

    let mut b_pack = pool_i64::take_uninit(p.kc * p.nc.min(n));

    let mut j0 = 0;
    while j0 < n {
        let nb = (n - j0).min(p.nc);
        let mut k0 = 0;
        while k0 < k {
            let kb = (k - k0).min(p.kc);
            pack_b_nr(b, p.nr, k0, kb, j0, nb, &mut b_pack[..kb * nb]);

            if want_par {
                use rayon::prelude::*;
                // Carve C into disjoint row bands once per (j0, k0) block.
                let mut bands = Vec::with_capacity(n_bands);
                let mut i0 = 0;
                let mut rest: &mut [i64] = &mut data[..];
                while i0 < m {
                    let mb = (m - i0).min(p.mc);
                    let (chunk, tail) = rest.split_at_mut(mb * n);
                    bands.push((i0, mb, chunk));
                    rest = tail;
                    i0 += mb;
                }
                let bp = &b_pack;
                bands.into_par_iter().for_each(|(i0, mb, c_band)| {
                    gemm_row_band(p, a, n, bp, i0, mb, k0, kb, j0, nb, c_band);
                });
            } else {
                let mut i0 = 0;
                while i0 < m {
                    let mb = (m - i0).min(p.mc);
                    gemm_row_band(
                        p,
                        a,
                        n,
                        &b_pack,
                        i0,
                        mb,
                        k0,
                        kb,
                        j0,
                        nb,
                        &mut data[i0 * n..(i0 + mb) * n],
                    );
                    i0 += mb;
                }
            }
            k0 += kb;
        }
        j0 += nb;
    }
    pool_i64::recycle(b_pack);
}

/// Tiny-product fallback (no packing): dispatch on transposition. The
/// untransposed arm is the unrolled ikj loop; the transposed arms are the
/// cache-sensible naive orders (kpj for AᵀB, row-dot for ABᵀ).
fn gemm_tiny(m: usize, k: usize, n: usize, a: Op, b: Op, data: &mut [i64]) {
    match (a.trans, b.trans) {
        (false, false) => gemm_simple(m, k, n, a.data, b.data, data),
        (true, false) => {
            // C[i, j] += A[p, i] * B[p, j], streaming rows of both.
            for p in 0..k {
                let a_row = &a.data[p * a.ld..p * a.ld + m];
                let b_row = &b.data[p * b.ld..p * b.ld + n];
                for i in 0..m {
                    let api = a_row[i];
                    if api == 0 {
                        continue;
                    }
                    let c_row = &mut data[i * n..(i + 1) * n];
                    for j in 0..n {
                        c_row[j] = c_row[j].wrapping_add(api.wrapping_mul(b_row[j]));
                    }
                }
            }
        }
        (false, true) => {
            // C[i, j] += A[i, p] * B[j, p]: row-dot per output element.
            for i in 0..m {
                let a_row = &a.data[i * a.ld..i * a.ld + k];
                for j in 0..n {
                    let b_row = &b.data[j * b.ld..j * b.ld + k];
                    let mut s: i64 = 0;
                    for p in 0..k {
                        s = s.wrapping_add(a_row[p].wrapping_mul(b_row[p]));
                    }
                    data[i * n + j] = data[i * n + j].wrapping_add(s);
                }
            }
        }
        (true, true) => {
            // Not reachable from the public face; keep a correct fallback.
            for i in 0..m {
                for j in 0..n {
                    let mut s: i64 = 0;
                    for p in 0..k {
                        s = s.wrapping_add(
                            a.data[p * a.ld + i].wrapping_mul(b.data[j * b.ld + p]),
                        );
                    }
                    data[i * n + j] = data[i * n + j].wrapping_add(s);
                }
            }
        }
    }
}

/// Dispatch matrix GEMM (packed GEBP / tiny / parallel bands), untransposed.
/// Strassen was measured through n=4096 on this class of host and never beat
/// GEBP (S/G ≥ 1.0); removed to keep the path simple (WASM-friendly GEBP only).
fn gemm_dispatch(am: usize, an: usize, bn: usize, aa: &[i64], bb: &[i64], data: &mut [i64]) {
    let a = Op { data: aa, ld: an, trans: false };
    let b = Op { data: bb, ld: bn, trans: false };
    gemm_blocked(am, an, bn, a, b, data);
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
    gemm_dispatch(am, an, bn, a.as_slice(), b.as_slice(), &mut data);
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
        // Aᵀx = Σₖ x[k]·A[k, :] — stream A rows contiguously (axpy order),
        // instead of a strided column dot per output element.
        for kk in 0..am {
            let bk = bb[kk];
            if bk == 0 {
                continue;
            }
            let row = &aa[kk * an..(kk + 1) * an];
            for i in 0..an {
                data[i] = data[i].wrapping_add(bk.wrapping_mul(row[i]));
            }
        }
    } else {
        // Full GEBP path; transposition is absorbed by the pack layer.
        let a_op = Op { data: aa, ld: an, trans: true };
        let b_op = Op { data: bb, ld: bn, trans: false };
        gemm_blocked(an, am, bn, a_op, b_op, &mut data);
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
    // Full GEBP path; Bᵀ is absorbed by the pack layer.
    let a_op = Op { data: aa, ld: an, trans: false };
    let b_op = Op { data: bb, ld: bn, trans: true };
    gemm_blocked(am, an, bm, a_op, b_op, &mut data);
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
    fn gemm_profiles_match_naive() {
        // Exercise BOTH blocking profiles explicitly (dispatch would pin the
        // suite to whichever profile this host selects). AVX-512 profile runs
        // only where the CPU reports the features it needs.
        let (m, k, n) = (67usize, 131usize, 45usize);
        let da = wrapheavy(m * k, 41);
        let db = wrapheavy(k * n, 43);
        let r = naive_matmul(m, k, n, &da, &db);
        let a = Op { data: &da, ld: k, trans: false };
        let b = Op { data: &db, ld: n, trans: false };

        let mut c = vec![0i64; m * n];
        gemm_blocked_with(PORTABLE, m, k, n, a, b, &mut c);
        assert_eq!(c, r, "PORTABLE profile mismatch");

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
        {
            let mut c = vec![0i64; m * n];
            gemm_blocked_with(AVX512, m, k, n, a, b, &mut c);
            assert_eq!(c, r, "AVX512 profile mismatch");
        }
    }

    /// Reference ijk matmul (wrapping), for exactness checks of the packed path.
    fn naive_matmul(m: usize, k: usize, n: usize, aa: &[i64], bb: &[i64]) -> Vec<i64> {
        let mut r = vec![0i64; m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = aa[i * k + p];
                for j in 0..n {
                    r[i * n + j] = r[i * n + j].wrapping_add(aip.wrapping_mul(bb[p * n + j]));
                }
            }
        }
        r
    }

    fn wrapheavy(len: usize, seed: i64) -> Vec<i64> {
        let mut v = Vec::with_capacity(len);
        let mut x = seed;
        for _ in 0..len {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(x);
        }
        v
    }

    #[test]
    fn matmul_packed_matches_naive_rectangular_edges() {
        // (m, k, n) chosen to exercise: MR/NR edge tiles (67 % 6, 45 % 16),
        // multiple KC panels (k = 131 > 128), and wrap-heavy values.
        let (m, k, n) = (67usize, 131usize, 45usize);
        let a = ArrayI64::from_shape_vec(vec![m, k], wrapheavy(m * k, 3)).unwrap();
        let b = ArrayI64::from_shape_vec(vec![k, n], wrapheavy(k * n, 7)).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(
            c.as_slice(),
            naive_matmul(m, k, n, a.as_slice(), b.as_slice()).as_slice()
        );
    }

    #[test]
    fn matmul_packed_matches_naive_parallel_bands() {
        // m > 2*MC forces the parallel row-band path on multi-core hosts;
        // k > KC forces accumulation across depth panels.
        let (m, k, n) = (390usize, 130usize, 70usize);
        let a = ArrayI64::from_shape_vec(vec![m, k], wrapheavy(m * k, 11)).unwrap();
        let b = ArrayI64::from_shape_vec(vec![k, n], wrapheavy(k * n, 13)).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(
            c.as_slice(),
            naive_matmul(m, k, n, a.as_slice(), b.as_slice()).as_slice()
        );
    }

    #[test]
    fn matmul_at_gebp_matches_naive() {
        // AᵀB with A (am×an): logical (an, am, bn) product; edges off all
        // tile multiples; large enough for the packed (non-tiny) path.
        let (am, an, bn) = (131usize, 67usize, 45usize);
        let a = ArrayI64::from_shape_vec(vec![am, an], wrapheavy(am * an, 17)).unwrap();
        let b = ArrayI64::from_shape_vec(vec![am, bn], wrapheavy(am * bn, 19)).unwrap();
        let c = matmul_at(&a, &b).unwrap();
        let (aa, bb) = (a.as_slice(), b.as_slice());
        let mut r = vec![0i64; an * bn];
        for p in 0..am {
            for i in 0..an {
                for j in 0..bn {
                    r[i * bn + j] = r[i * bn + j]
                        .wrapping_add(aa[p * an + i].wrapping_mul(bb[p * bn + j]));
                }
            }
        }
        assert_eq!(c.as_slice(), r.as_slice());
        // Rank-1 axpy path.
        let x = ArrayI64::from_shape_vec(vec![am], wrapheavy(am, 23)).unwrap();
        let atx = matmul_at(&a, &x).unwrap();
        let mut rv = vec![0i64; an];
        for p in 0..am {
            for i in 0..an {
                rv[i] = rv[i].wrapping_add(aa[p * an + i].wrapping_mul(x.as_slice()[p]));
            }
        }
        assert_eq!(atx.as_slice(), rv.as_slice());
    }

    #[test]
    fn matmul_bt_gebp_matches_naive() {
        // ABᵀ with shared k = 131 (> KC edge unaffected but non-multiple),
        // parallel-band m on multi-core hosts (390 > 2·MC).
        let (am, k, bm) = (390usize, 131usize, 53usize);
        let a = ArrayI64::from_shape_vec(vec![am, k], wrapheavy(am * k, 29)).unwrap();
        let b = ArrayI64::from_shape_vec(vec![bm, k], wrapheavy(bm * k, 31)).unwrap();
        let c = matmul_bt(&a, &b).unwrap();
        let (aa, bb) = (a.as_slice(), b.as_slice());
        let mut r = vec![0i64; am * bm];
        for i in 0..am {
            for j in 0..bm {
                let mut s = 0i64;
                for p in 0..k {
                    s = s.wrapping_add(aa[i * k + p].wrapping_mul(bb[j * k + p]));
                }
                r[i * bm + j] = s;
            }
        }
        assert_eq!(c.as_slice(), r.as_slice());
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

