//! Cached runtime ISA detection for kernel dispatch (derived — CPUID at
//! first use, not a host-tuned table; DESIGN §3.26). The `#[target_feature]`
//! twin-kernel pattern and its measured effects live with each kernel.

/// True when the CPU supports the avx512f+dq+bw+vl set that our
/// `#[target_feature]` kernel twins enable. Always false off x86_64.
#[inline]
pub(crate) fn avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512dq")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// --- Micro-calibration -------------------------------------------------------
//
// CPUID says whether AVX-512 instructions are *executable*, not whether they
// are *fast right now*: 512-bit execution can be dynamically downclocked
// (frequency licensing, co-tenant pressure on shared cloud hosts, or
// migration to weaker silicon). Observed on the 2026-08 bench container: the
// 512-bit MAC tile dropped to ~1/4 throughput for hours while scalar
// throughput stayed normal — static CPUID dispatch would have picked the
// slower kernel the whole time. So path *choice* uses [`avx512_fast`]: CPUID
// plus a one-time ~1 ms race of the same MAC tile compiled both ways,
// requiring the 512-bit twin to win by ≥5% (healthy hosts show ≥1.5×; the
// margin only filters ties). Derived, not host-tuned (DESIGN §3.26).
// Limitation: the verdict is cached per process, so a process started during
// a throttled window keeps the portable path until restart — acceptable, as
// both paths are exact and the portable path is the safe default.

/// Wrapping-i64 4×8 MAC tile used as the calibration workload (the same
/// shape as the GEMM micro-kernel; also a proxy for the elementwise twins).
#[inline(never)]
fn cal_tile(k: usize, a: &[i64], b: &[i64], acc: &mut [i64; 32]) {
    for p in 0..k {
        let ap = &a[p * 4..p * 4 + 4];
        let bp = &b[p * 8..p * 8 + 8];
        for i in 0..4 {
            let ai = ap[i];
            for j in 0..8 {
                acc[i * 8 + j] = acc[i * 8 + j].wrapping_add(ai.wrapping_mul(bp[j]));
            }
        }
    }
}

/// 512-bit twin of [`cal_tile`] (same body).
///
/// # Safety
/// Call only when [`avx512`] is true.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn cal_tile_avx512(k: usize, a: &[i64], b: &[i64], acc: &mut [i64; 32]) {
    for p in 0..k {
        let ap = &a[p * 4..p * 4 + 4];
        let bp = &b[p * 8..p * 8 + 8];
        for i in 0..4 {
            let ai = ap[i];
            for j in 0..8 {
                acc[i * 8 + j] = acc[i * 8 + j].wrapping_add(ai.wrapping_mul(bp[j]));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn measure_best_of<F: FnMut()>(rounds: usize, mut f: F) -> std::time::Duration {
    let mut best = std::time::Duration::MAX;
    for _ in 0..rounds {
        let t = std::time::Instant::now();
        f();
        best = best.min(t.elapsed());
    }
    best
}

/// True when AVX-512 is present **and** measurably faster right now (see the
/// module block comment). This is the selector all kernel dispatch uses;
/// [`avx512`] alone remains the executability/safety gate.
#[inline]
pub(crate) fn avx512_fast() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static FAST: OnceLock<bool> = OnceLock::new();
        *FAST.get_or_init(|| {
            if !avx512() {
                return false;
            }
            const K: usize = 256;
            const REPS: usize = 60;
            let mut x = 0x9E3779B97F4A7C15u64 as i64;
            let mut fill = |v: &mut Vec<i64>, n: usize| {
                for _ in 0..n {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    v.push(x);
                }
            };
            let mut a = Vec::with_capacity(K * 4);
            let mut b = Vec::with_capacity(K * 8);
            fill(&mut a, K * 4);
            fill(&mut b, K * 8);
            let mut acc = [0i64; 32];
            // Warm both, then best-of-3 rounds of REPS tiles each (~1 ms total).
            cal_tile(K, &a, &b, &mut acc);
            // SAFETY: avx512() verified above.
            unsafe { cal_tile_avx512(K, &a, &b, &mut acc) };
            let t_base = measure_best_of(3, || {
                for _ in 0..REPS {
                    cal_tile(K, std::hint::black_box(&a), std::hint::black_box(&b), &mut acc);
                }
            });
            let t_isa = measure_best_of(3, || {
                for _ in 0..REPS {
                    // SAFETY: avx512() verified above.
                    unsafe {
                        cal_tile_avx512(
                            K,
                            std::hint::black_box(&a),
                            std::hint::black_box(&b),
                            &mut acc,
                        )
                    };
                }
            });
            std::hint::black_box(acc);
            // Require a ≥5% win for the 512-bit path.
            t_isa.as_nanos() * 100 < t_base.as_nanos() * 95
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fast_implies_executable() {
        if super::avx512_fast() {
            assert!(super::avx512());
        }
    }
}
