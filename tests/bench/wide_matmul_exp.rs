//! Wide-value i64 matmul candidates (experiment, Human-directed 2026-08-04).
//!
//! Above the f64-promote bound (`k·max|A|·max|B| > 2⁵³`) the shipped path is
//! the exact wrapping GEBP. Two exact candidates compete to beat it there:
//!
//! - **i32-pack GEBP**: when every input fits in `i32`, each 32×32→64
//!   widening product is exact and wrapping i64 sums are exact mod 2⁶⁴ for
//!   **any** k — no magnitude bound at all, unlike the 2⁵³ promote guard.
//!   Also halves pack bytes (i32 packs), doubling effective L1/L2 depth.
//! - **Strassen over the wrapping ring**: exact (rings have no float
//!   stability problem); trades 1/8 of the multiplies per level for O(n²)
//!   wrapping adds and extra memory traffic.
//!
//! Whatever measures best wins (Human ruling); losers get recorded in DESIGN
//! with their numbers. Exactness is asserted against the shipped kernel
//! before any timing is trusted.
//!
//! ```text
//! cargo test --release --features lua --test wide_matmul_exp -- --run
//! ```
//!
//! Output: `exp\t<name>\t<n>\t<ms>` rows plus a verdict summary.

use std::env;
use std::hint::black_box;
use std::time::Instant;

use matlua::array::ArrayI64;
use matlua::linalg::i64_ops;

// --- Data ---------------------------------------------------------------------

/// Wide-but-i32 values: |v| ≤ 1e9 < 2³¹, while k·max|A|·max|B| ≈ 4·10²¹ ≫ 2⁵³
/// at n = 4096 — squarely in the exact-GEBP regime, and eligible for the
/// i32 candidate.
fn wide_i32_vals(n: usize, mul: i64, add: i64) -> Vec<i64> {
    (0..n * n)
        .map(|i| (i as i64).wrapping_mul(mul).wrapping_add(add).rem_euclid(2_000_000_001) - 1_000_000_000)
        .collect()
}

// --- Candidate 1: i32-pack GEBP (single-thread prototype) ---------------------
//
// Same Goto structure and 4×8 tile shape as the shipped kernel; packs are
// `i32`, the tile widens each operand to i64 (`as i64` = sign extension; on
// AVX-512 the products come from 32-bit source lanes). KC doubled vs the i64
// profile (i32 packs are half the bytes, same L1 footprint).

const KC_I32: usize = 512;
const MC: usize = 128;
const NC: usize = 1024;
const MR: usize = 4;
const NR: usize = 8;

fn pack_a_i32(aa: &[i64], ld: usize, i0: usize, m: usize, k0: usize, k: usize, buf: &mut [i32]) {
    let mut off = 0;
    let mut ir = 0;
    while ir < m {
        let mr = (m - ir).min(MR);
        for p in 0..k {
            for i in 0..mr {
                buf[off + p * mr + i] = aa[(i0 + ir + i) * ld + k0 + p] as i32;
            }
        }
        off += mr * k;
        ir += mr;
    }
}

fn pack_b_i32(bb: &[i64], ld: usize, k0: usize, k: usize, j0: usize, n: usize, buf: &mut [i32]) {
    let mut off = 0;
    let mut jr = 0;
    while jr < n {
        let nr = (n - jr).min(NR);
        for p in 0..k {
            for j in 0..nr {
                buf[off + p * nr + j] = bb[(k0 + p) * ld + j0 + jr + j] as i32;
            }
        }
        off += nr * k;
        jr += nr;
    }
}

macro_rules! tile_i32_body {
    ($ap:expr, $bp:expr, $k:expr, $c:expr, $ldc:expr) => {{
        let (ap, bp, k, c, ldc): (&[i32], &[i32], usize, &mut [i64], usize) =
            ($ap, $bp, $k, $c, $ldc);
        let mut acc = [[0i64; NR]; MR];
        for p in 0..k {
            let av = &ap[p * MR..p * MR + MR];
            let bv = &bp[p * NR..p * NR + NR];
            for i in 0..MR {
                let ai = av[i] as i64;
                for j in 0..NR {
                    acc[i][j] = acc[i][j].wrapping_add(ai.wrapping_mul(bv[j] as i64));
                }
            }
        }
        for i in 0..MR {
            for j in 0..NR {
                c[i * ldc + j] = c[i * ldc + j].wrapping_add(acc[i][j]);
            }
        }
    }};
}

fn tile_i32(ap: &[i32], bp: &[i32], k: usize, c: &mut [i64], ldc: usize) {
    tile_i32_body!(ap, bp, k, c, ldc);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn tile_i32_avx512(ap: &[i32], bp: &[i32], k: usize, c: &mut [i64], ldc: usize) {
    tile_i32_body!(ap, bp, k, c, ldc);
}

#[cfg(target_arch = "x86_64")]
fn avx512_ok() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}
#[cfg(not(target_arch = "x86_64"))]
fn avx512_ok() -> bool {
    false
}

/// Full-tile-only prototype: caller guarantees n % 4 == 0 and n % 8 == 0
/// (experiment sizes are powers of two). Single thread.
fn gemm_i32(n: usize, aa: &[i64], bb: &[i64], isa: bool, out: &mut [i64]) {
    out.fill(0);
    let mut bpack = vec![0i32; KC_I32 * NC];
    let mut apack = vec![0i32; MC * KC_I32];
    let mut j0 = 0;
    while j0 < n {
        let nb = (n - j0).min(NC);
        let mut k0 = 0;
        while k0 < n {
            let kb = (n - k0).min(KC_I32);
            pack_b_i32(bb, n, k0, kb, j0, nb, &mut bpack);
            let mut i0 = 0;
            while i0 < n {
                let mb = (n - i0).min(MC);
                pack_a_i32(aa, n, i0, mb, k0, kb, &mut apack);
                let mut jr = 0;
                while jr < nb {
                    let bp = &bpack[jr * kb..(jr + NR) * kb];
                    let mut ir = 0;
                    while ir < mb {
                        let ap = &apack[ir * kb..(ir + MR) * kb];
                        let c0 = (i0 + ir) * n + j0 + jr;
                        let c = &mut out[c0..c0 + (MR - 1) * n + NR];
                        if isa {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                tile_i32_avx512(ap, bp, kb, c, n);
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            tile_i32(ap, bp, kb, c, n);
                        } else {
                            tile_i32(ap, bp, kb, c, n);
                        }
                        ir += MR;
                    }
                    jr += NR;
                }
                i0 += mb;
            }
            k0 += kb;
        }
        j0 += nb;
    }
}

// --- Candidate 2: Strassen over the wrapping ring -----------------------------
//
// Flat row-major Vec<i64> quadrant split; leaf calls the shipped
// `i64_ops::matmul` (parallel GEBP — wide values keep it off the promote
// path, but even if a wrapped intermediate landed range-safe the promote is
// bit-identical, so exactness holds either way).

const STRASSEN_LEAF: usize = 1024;

fn madd(a: &[i64], b: &[i64]) -> Vec<i64> {
    a.iter().zip(b).map(|(x, y)| x.wrapping_add(*y)).collect()
}
fn msub(a: &[i64], b: &[i64]) -> Vec<i64> {
    a.iter().zip(b).map(|(x, y)| x.wrapping_sub(*y)).collect()
}

fn split(a: &[i64], n: usize) -> [Vec<i64>; 4] {
    let h = n / 2;
    let mut q = [
        Vec::with_capacity(h * h),
        Vec::with_capacity(h * h),
        Vec::with_capacity(h * h),
        Vec::with_capacity(h * h),
    ];
    for i in 0..h {
        q[0].extend_from_slice(&a[i * n..i * n + h]);
        q[1].extend_from_slice(&a[i * n + h..(i + 1) * n]);
        q[2].extend_from_slice(&a[(i + h) * n..(i + h) * n + h]);
        q[3].extend_from_slice(&a[(i + h) * n + h..(i + h + 1) * n]);
    }
    q
}

fn join(c11: &[i64], c12: &[i64], c21: &[i64], c22: &[i64], n: usize) -> Vec<i64> {
    let h = n / 2;
    let mut c = vec![0i64; n * n];
    for i in 0..h {
        c[i * n..i * n + h].copy_from_slice(&c11[i * h..(i + 1) * h]);
        c[i * n + h..(i + 1) * n].copy_from_slice(&c12[i * h..(i + 1) * h]);
        c[(i + h) * n..(i + h) * n + h].copy_from_slice(&c21[i * h..(i + 1) * h]);
        c[(i + h) * n + h..(i + h + 1) * n].copy_from_slice(&c22[i * h..(i + 1) * h]);
    }
    c
}

fn leaf_matmul(a: &[i64], b: &[i64], n: usize) -> Vec<i64> {
    let am = ArrayI64::from_shape_slice(vec![n, n], a).unwrap();
    let bm = ArrayI64::from_shape_slice(vec![n, n], b).unwrap();
    i64_ops::matmul(&am, &bm).unwrap().as_slice().to_vec()
}

fn strassen(a: &[i64], b: &[i64], n: usize) -> Vec<i64> {
    strassen_with_leaf(a, b, n, STRASSEN_LEAF)
}

// --- Harness ------------------------------------------------------------------

fn time_best_ms(samples: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm
    let mut best = f64::INFINITY;
    for _ in 0..samples {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    if !env::args().any(|a| a == "--run") {
        println!("wide_matmul_exp: skipped (pass --run)");
        return;
    }
    let isa = avx512_ok();
    println!("exp\tavx512_available\t-\t{isa}");

    // Exactness first, at n=512 (wide values, above the promote bound for
    // this k: 512·1e9·1e9 ≈ 5·10²⁰ ≫ 2⁵³).
    {
        let n = 512;
        let aa = wide_i32_vals(n, 48271, 11);
        let bb = wide_i32_vals(n, 69621, 7);
        let reference = leaf_matmul(&aa, &bb, n);
        let mut c = vec![0i64; n * n];
        gemm_i32(n, &aa, &bb, false, &mut c);
        assert_eq!(c, reference, "i32 portable kernel mismatch");
        if isa {
            gemm_i32(n, &aa, &bb, true, &mut c);
            assert_eq!(c, reference, "i32 avx512 kernel mismatch");
        }
        // Strassen with a temporarily tiny leaf to force real recursion.
        let s = strassen_with_leaf(&aa, &bb, n, 128);
        assert_eq!(s, reference, "strassen mismatch");
        println!("exp\texactness\t{n}\tall candidates bit-identical");
    }

    for &n in &[1024usize, 2048, 4096] {
        let aa = wide_i32_vals(n, 48271, 11);
        let bb = wide_i32_vals(n, 69621, 7);
        let samples = if n >= 4096 { 3 } else { 5 };

        let am = ArrayI64::from_shape_slice(vec![n, n], &aa).unwrap();
        let bm = ArrayI64::from_shape_slice(vec![n, n], &bb).unwrap();
        let t = time_best_ms(samples, || {
            black_box(i64_ops::matmul(&am, &bm).unwrap());
        });
        println!("exp\tshipped_gebp_par\t{n}\t{t:.3}");

        let mut c = vec![0i64; n * n];
        let t = time_best_ms(samples, || {
            gemm_i32(n, &aa, &bb, false, &mut c);
            black_box(&c);
        });
        println!("exp\ti32_gebp_1thread_portable\t{n}\t{t:.3}");
        if isa {
            let t = time_best_ms(samples, || {
                gemm_i32(n, &aa, &bb, true, &mut c);
                black_box(&c);
            });
            println!("exp\ti32_gebp_1thread_avx512\t{n}\t{t:.3}");
        }

        if n > STRASSEN_LEAF {
            let t = time_best_ms(samples, || {
                black_box(strassen(&aa, &bb, n));
            });
            println!("exp\tstrassen_leaf1024\t{n}\t{t:.3}");
        }
    }
    println!("exp\tdone\t-\t-");
}

fn strassen_with_leaf(a: &[i64], b: &[i64], n: usize, leaf: usize) -> Vec<i64> {
    if n <= leaf {
        return leaf_matmul(a, b, n);
    }
    let h = n / 2;
    let [a11, a12, a21, a22] = split(a, n);
    let [b11, b12, b21, b22] = split(b, n);
    let m1 = strassen_with_leaf(&madd(&a11, &a22), &madd(&b11, &b22), h, leaf);
    let m2 = strassen_with_leaf(&madd(&a21, &a22), &b11, h, leaf);
    let m3 = strassen_with_leaf(&a11, &msub(&b12, &b22), h, leaf);
    let m4 = strassen_with_leaf(&a22, &msub(&b21, &b11), h, leaf);
    let m5 = strassen_with_leaf(&madd(&a11, &a12), &b22, h, leaf);
    let m6 = strassen_with_leaf(&msub(&a21, &a11), &madd(&b11, &b12), h, leaf);
    let m7 = strassen_with_leaf(&msub(&a12, &a22), &madd(&b21, &b22), h, leaf);
    let c11 = madd(&msub(&madd(&m1, &m4), &m5), &m7);
    let c12 = madd(&m3, &m5);
    let c21 = madd(&m2, &m4);
    let c22 = madd(&madd(&msub(&m1, &m2), &m3), &m6);
    join(&c11, &c12, &c21, &c22, n)
}
