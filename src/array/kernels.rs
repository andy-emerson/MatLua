//! Contiguous `f64` kernels for elementwise ops and reductions.
//!
//! Specialized arithmetic (no `Fn` closures) and index loops over dense
//! slices so LLVM can auto-vectorize. Slices must be the same length where
//! binary.

// ISA-dispatched elementwise kernels: one portable loop body, compiled twice.
// The plain version autovectorizes at the build's baseline target (SSE2 on
// default x86-64); the `#[target_feature]` twin is the same body compiled
// with AVX-512 enabled and is taken when the running CPU has it (same
// pattern as the GEMM profiles in `linalg::i64_ops`). Measured on the
// 2026-08 bench container: +52% on L2-resident 256² operands, +19% at 1024²
// (memory-bound sizes converge). No intrinsics; twins share the source body.
macro_rules! isa_binary_f64 {
    ($name:ident, $avx:ident, $op:tt) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
        unsafe fn $avx(a: &[f64], b: &[f64], out: &mut [f64]) {
            for i in 0..a.len() {
                out[i] = a[i] $op b[i];
            }
        }
        #[inline]
        pub(crate) fn $name(a: &[f64], b: &[f64], out: &mut [f64]) {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), out.len());
            #[cfg(target_arch = "x86_64")]
            if crate::array::isa::avx512_fast() {
                // SAFETY: features verified by isa::avx512().
                unsafe { $avx(a, b, out) };
                return;
            }
            for i in 0..a.len() {
                out[i] = a[i] $op b[i];
            }
        }
    };
}

macro_rules! isa_scalar_f64 {
    ($name:ident, $avx:ident, $op:tt) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
        unsafe fn $avx(a: &[f64], s: f64, out: &mut [f64]) {
            for i in 0..a.len() {
                out[i] = a[i] $op s;
            }
        }
        /// `out[i] = a[i] `op` s` (ISA-dispatched; see module block comment).
        #[inline]
        pub(crate) fn $name(a: &[f64], s: f64, out: &mut [f64]) {
            debug_assert_eq!(a.len(), out.len());
            #[cfg(target_arch = "x86_64")]
            if crate::array::isa::avx512_fast() {
                // SAFETY: features verified by isa::avx512().
                unsafe { $avx(a, s, out) };
                return;
            }
            for i in 0..a.len() {
                out[i] = a[i] $op s;
            }
        }
    };
}

isa_binary_f64!(add_slices, add_slices_avx512, +);
isa_binary_f64!(sub_slices, sub_slices_avx512, -);
isa_binary_f64!(mul_slices, mul_slices_avx512, *);
isa_binary_f64!(div_slices, div_slices_avx512, /);

#[inline]
pub(crate) fn add_assign_slices(a: &mut [f64], b: &[f64]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

#[inline]
pub(crate) fn sub_assign_slices(a: &mut [f64], b: &[f64]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] -= b[i];
    }
}

#[inline]
pub(crate) fn mul_assign_slices(a: &mut [f64], b: &[f64]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

#[inline]
pub(crate) fn div_assign_slices(a: &mut [f64], b: &[f64]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] /= b[i];
    }
}

#[inline]
pub(crate) fn neg_slice(a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = -a[i];
    }
}

isa_scalar_f64!(add_scalar, add_scalar_avx512, +);

isa_scalar_f64!(sub_scalar, sub_scalar_avx512, -);

/// `out[i] = s - a[i]`
#[inline]
pub(crate) fn scalar_sub(a: &[f64], s: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = s - a[i];
    }
}

isa_scalar_f64!(mul_scalar, mul_scalar_avx512, *);

/// `out[i] = a[i] / s`
#[inline]
pub(crate) fn div_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] / s;
    }
}

/// `out[i] = s / a[i]`
#[inline]
pub(crate) fn scalar_div(a: &[f64], s: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = s / a[i];
    }
}

/// Dot product with four accumulators.
#[inline]
pub(crate) fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let n = a.len();
    let mut i = 0;
    while i + 4 <= n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = s0 + s1 + s2 + s3;
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Shared parallel-reduction rule (derived, DESIGN §3.26): go parallel when
/// each rayon task gets at least QUANTUM elements — 2²⁰ elements ≈ 8 MB ≈
/// ~1 ms of memory-bound work against tens-of-µs spawn/join cost.
const REDUCE_QUANTUM: usize = 1 << 20;

#[inline]
fn reduce_par_ok(len: usize) -> bool {
    len >= 2 * REDUCE_QUANTUM
        && std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            >= 2
}

/// Sequential sum, 8 independent accumulators (ILP/autovec).
#[inline]
fn sum_seq(a: &[f64]) -> f64 {
    let mut s = [0.0f64; 8];
    let mut chunks = a.chunks_exact(8);
    for c in chunks.by_ref() {
        for j in 0..8 {
            s[j] += c[j];
        }
    }
    let mut t = s.iter().sum::<f64>();
    for &x in chunks.remainder() {
        t += x;
    }
    t
}

/// AVX-512 twin of [`sum_seq`] (same body; 512-bit codegen).
///
/// # Safety
/// Caller must have verified the features ([`crate::array::isa::avx512`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn sum_seq_avx512(a: &[f64]) -> f64 {
    let mut s = [0.0f64; 8];
    let mut chunks = a.chunks_exact(8);
    for c in chunks.by_ref() {
        for j in 0..8 {
            s[j] += c[j];
        }
    }
    let mut t = s.iter().sum::<f64>();
    for &x in chunks.remainder() {
        t += x;
    }
    t
}

#[inline]
fn sum_dispatch(a: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if crate::array::isa::avx512_fast() {
        // SAFETY: features verified by isa::avx512().
        return unsafe { sum_seq_avx512(a) };
    }
    sum_seq(a)
}

/// Sum: ISA-dispatched, parallel above the reduction quantum. Summation was
/// already reassociated (ILP lanes), so chunked reduction keeps the same
/// rounding class. `mean` and the axis reductions sit on this.
#[inline]
pub(crate) fn sum_slice(a: &[f64]) -> f64 {
    if reduce_par_ok(a.len()) {
        use rayon::prelude::*;
        return a.par_chunks(REDUCE_QUANTUM).map(sum_dispatch).sum();
    }
    sum_dispatch(a)
}

/// Sequential sum of squares, 8 independent accumulators (ILP/autovec).
#[inline]
fn sum_sq_seq(a: &[f64]) -> f64 {
    let mut s = [0.0f64; 8];
    let mut chunks = a.chunks_exact(8);
    for c in chunks.by_ref() {
        for j in 0..8 {
            s[j] += c[j] * c[j];
        }
    }
    let mut t = s.iter().sum::<f64>();
    for &x in chunks.remainder() {
        t += x * x;
    }
    t
}

/// AVX-512 twin of [`sum_sq_seq`] (same body; 512-bit codegen).
///
/// # Safety
/// Caller must have verified the features ([`crate::array::isa::avx512`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn sum_sq_seq_avx512(a: &[f64]) -> f64 {
    let mut s = [0.0f64; 8];
    let mut chunks = a.chunks_exact(8);
    for c in chunks.by_ref() {
        for j in 0..8 {
            s[j] += c[j] * c[j];
        }
    }
    let mut t = s.iter().sum::<f64>();
    for &x in chunks.remainder() {
        t += x * x;
    }
    t
}

#[inline]
fn sum_sq_dispatch(a: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if crate::array::isa::avx512_fast() {
        // SAFETY: features verified by isa::avx512().
        return unsafe { sum_sq_seq_avx512(a) };
    }
    sum_sq_seq(a)
}

/// Sum of squares (for Frobenius norm). ISA-dispatched, and parallel when
/// each thread gets at least QUANTUM elements — derived (DESIGN §3.26):
/// rayon spawn/join costs tens of µs; 2²⁰ elements ≈ 8 MB ≈ ~1 ms of
/// memory-bound work, so overhead stays under a few percent. The summation
/// was already reassociated (8-lane ILP), so chunked reduction does not
/// change the rounding class.
#[inline]
pub(crate) fn sum_sq_slice(a: &[f64]) -> f64 {
    if reduce_par_ok(a.len()) {
        use rayon::prelude::*;
        return a.par_chunks(REDUCE_QUANTUM).map(sum_sq_dispatch).sum();
    }
    sum_sq_dispatch(a)
}

/// Min/max over non-empty slices (Rust `f64::min`/`max` NaN handling —
/// unchanged). Eight ILP accumulators; ISA-dispatched twins; parallel above
/// the shared reduction quantum. Chunk results combine with the same
/// operator, so semantics match the sequential loop.
macro_rules! isa_minmax_f64 {
    ($seq:ident, $avx:ident, $disp:ident, $pub_fn:ident, $m:ident) => {
        #[inline]
        fn $seq(a: &[f64]) -> f64 {
            let mut m = [a[0]; 8];
            let mut chunks = a[1..].chunks_exact(8);
            for c in chunks.by_ref() {
                for j in 0..8 {
                    m[j] = m[j].$m(c[j]);
                }
            }
            let mut r = a[0];
            for j in 0..8 {
                r = r.$m(m[j]);
            }
            for &x in chunks.remainder() {
                r = r.$m(x);
            }
            r
        }
        /// # Safety
        /// Caller must have verified the features (`isa::avx512`).
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
        unsafe fn $avx(a: &[f64]) -> f64 {
            let mut m = [a[0]; 8];
            let mut chunks = a[1..].chunks_exact(8);
            for c in chunks.by_ref() {
                for j in 0..8 {
                    m[j] = m[j].$m(c[j]);
                }
            }
            let mut r = a[0];
            for j in 0..8 {
                r = r.$m(m[j]);
            }
            for &x in chunks.remainder() {
                r = r.$m(x);
            }
            r
        }
        #[inline]
        fn $disp(a: &[f64]) -> f64 {
            #[cfg(target_arch = "x86_64")]
            if crate::array::isa::avx512_fast() {
                // SAFETY: features verified by isa::avx512().
                return unsafe { $avx(a) };
            }
            $seq(a)
        }
        #[inline]
        pub(crate) fn $pub_fn(a: &[f64]) -> Option<f64> {
            if a.is_empty() {
                return None;
            }
            if reduce_par_ok(a.len()) {
                use rayon::prelude::*;
                return a
                    .par_chunks(REDUCE_QUANTUM)
                    .map($disp)
                    .reduce_with(|x, y| x.$m(y));
            }
            Some($disp(a))
        }
    };
}

isa_minmax_f64!(min_seq, min_seq_avx512, min_dispatch, min_slice, min);
isa_minmax_f64!(max_seq, max_seq_avx512, max_dispatch, max_slice, max);

#[inline]
pub(crate) fn abs_slice(a: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i].abs();
    }
}

#[inline]
pub(crate) fn sqrt_slice(a: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i].sqrt();
    }
}

#[inline]
pub(crate) fn exp_slice(a: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i].exp();
    }
}

#[inline]
pub(crate) fn log_slice(a: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i].ln();
    }
}

#[inline]
pub(crate) fn log1p_slice(a: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i].ln_1p();
    }
}

#[inline]
pub(crate) fn sign_slice(a: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        let x = a[i];
        out[i] = if x.is_nan() {
            f64::NAN
        } else if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
}

#[inline]
pub(crate) fn power_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i].powf(b[i]);
    }
}

#[inline]
pub(crate) fn power_scalar(a: &[f64], p: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i].powf(p);
    }
}

#[inline]
pub(crate) fn clip_slice(a: &[f64], lo: f64, hi: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        let x = a[i];
        out[i] = if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        };
    }
}

#[inline]
pub(crate) fn isnan_slice(a: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i].is_nan() { 1.0 } else { 0.0 };
    }
}

#[inline]
pub(crate) fn isfinite_slice(a: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i].is_finite() { 1.0 } else { 0.0 };
    }
}

#[inline]
pub(crate) fn where_slices(cond: &[f64], x: &[f64], y: &[f64], out: &mut [f64]) {
    for i in 0..cond.len() {
        out[i] = if cond[i] != 0.0 && !cond[i].is_nan() {
            x[i]
        } else {
            y[i]
        };
    }
}

#[inline]
pub(crate) fn cumsum_slice(a: &[f64], out: &mut [f64]) {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i];
        out[i] = s;
    }
}

#[inline]
pub(crate) fn argmin_slice(a: &[f64]) -> Option<usize> {
    if a.is_empty() {
        return None;
    }
    let mut best = f64::INFINITY;
    let mut best_i = 0usize;
    for (i, &x) in a.iter().enumerate() {
        if x < best {
            best = x;
            best_i = i;
        }
    }
    if best.is_infinite() && a.iter().all(|x| x.is_nan()) {
        Some(0)
    } else {
        Some(best_i)
    }
}

#[inline]
pub(crate) fn argmax_slice(a: &[f64]) -> Option<usize> {
    if a.is_empty() {
        return None;
    }
    let mut best = f64::NEG_INFINITY;
    let mut best_i = 0usize;
    for (i, &x) in a.iter().enumerate() {
        if x > best {
            best = x;
            best_i = i;
        }
    }
    if best.is_infinite() && a.iter().all(|x| x.is_nan()) {
        Some(0)
    } else {
        Some(best_i)
    }
}

/// Population or sample variance; `ddof` is degrees of freedom subtract.
#[inline]
pub(crate) fn var_slice(a: &[f64], ddof: usize) -> Option<f64> {
    let n = a.len();
    if n == 0 || n <= ddof {
        return None;
    }
    let mean = a.iter().sum::<f64>() / n as f64;
    let mut ss = 0.0;
    for &x in a {
        let d = x - mean;
        ss += d * d;
    }
    Some(ss / (n - ddof) as f64)
}


// --- Compares → 0/1 masks (IEEE: NaN compares false) ---

#[inline]
pub(crate) fn eq_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] == b[i] { 1.0 } else { 0.0 };
    }
}
#[inline]
pub(crate) fn ne_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] != b[i] { 1.0 } else { 0.0 };
    }
}
#[inline]
pub(crate) fn lt_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] < b[i] { 1.0 } else { 0.0 };
    }
}
#[inline]
pub(crate) fn le_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] <= b[i] { 1.0 } else { 0.0 };
    }
}
#[inline]
pub(crate) fn gt_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] > b[i] { 1.0 } else { 0.0 };
    }
}
#[inline]
pub(crate) fn ge_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] >= b[i] { 1.0 } else { 0.0 };
    }
}

#[inline]
pub(crate) fn eq_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] == s { 1.0 } else { 0.0 };
    }
}
#[inline]
pub(crate) fn ne_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] != s { 1.0 } else { 0.0 };
    }
}
#[inline]
pub(crate) fn lt_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] < s { 1.0 } else { 0.0 };
    }
}
#[inline]
pub(crate) fn le_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] <= s { 1.0 } else { 0.0 };
    }
}
#[inline]
pub(crate) fn gt_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] > s { 1.0 } else { 0.0 };
    }
}
#[inline]
pub(crate) fn ge_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = if a[i] >= s { 1.0 } else { 0.0 };
    }
}

// --- NaN-skipping reductions ---

#[inline]
pub(crate) fn nansum_slice(a: &[f64]) -> f64 {
    let mut s = 0.0;
    for &x in a {
        if !x.is_nan() {
            s += x;
        }
    }
    s
}

#[inline]
pub(crate) fn nanmean_slice(a: &[f64]) -> Option<f64> {
    let mut s = 0.0;
    let mut n = 0usize;
    for &x in a {
        if !x.is_nan() {
            s += x;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some(s / n as f64)
    }
}

#[inline]
pub(crate) fn nanmin_slice(a: &[f64]) -> Option<f64> {
    let mut m = f64::INFINITY;
    let mut any = false;
    for &x in a {
        if !x.is_nan() && x < m {
            m = x;
            any = true;
        }
    }
    if any { Some(m) } else { None }
}

#[inline]
pub(crate) fn nanmax_slice(a: &[f64]) -> Option<f64> {
    let mut m = f64::NEG_INFINITY;
    let mut any = false;
    for &x in a {
        if !x.is_nan() && x > m {
            m = x;
            any = true;
        }
    }
    if any { Some(m) } else { None }
}

#[inline]
pub(crate) fn nanvar_slice(a: &[f64], ddof: usize) -> Option<f64> {
    let mut s = 0.0;
    let mut n = 0usize;
    for &x in a {
        if !x.is_nan() {
            s += x;
            n += 1;
        }
    }
    if n == 0 || n <= ddof {
        return None;
    }
    let mean = s / n as f64;
    let mut ss = 0.0;
    for &x in a {
        if !x.is_nan() {
            let d = x - mean;
            ss += d * d;
        }
    }
    Some(ss / (n - ddof) as f64)
}


// --- M6 axis reductions (rank-2, row-major) ---

#[inline]
pub(crate) fn axis0_sum(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    // sum over rows → length n
    debug_assert_eq!(out.len(), n);
    out.fill(0.0);
    for i in 0..m {
        let row = &a[i * n..(i + 1) * n];
        for j in 0..n {
            out[j] += row[j];
        }
    }
}

#[inline]
pub(crate) fn axis1_sum(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(out.len(), m);
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j];
        }
        out[i] = s;
    }
}

/// Fused mean over axis 0 (rows): one pass, write means of length n.
#[inline]
pub(crate) fn axis0_mean(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(out.len(), n);
    if m == 0 {
        out.fill(f64::NAN);
        return;
    }
    out.fill(0.0);
    for i in 0..m {
        let row = &a[i * n..(i + 1) * n];
        for j in 0..n {
            out[j] += row[j];
        }
    }
    let inv = 1.0 / m as f64;
    for j in 0..n {
        out[j] *= inv;
    }
}

/// Fused mean over axis 1 (cols): one pass, write means of length m.
#[inline]
pub(crate) fn axis1_mean(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(out.len(), m);
    if n == 0 {
        out.fill(f64::NAN);
        return;
    }
    let inv = 1.0 / n as f64;
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j];
        }
        out[i] = s * inv;
    }
}

/// Fused variance over axis 0: two-pass in one function (no intermediate Array).
#[inline]
pub(crate) fn axis0_var(m: usize, n: usize, a: &[f64], ddof: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), n);
    if m <= ddof {
        out.fill(f64::NAN);
        return;
    }
    let mut mean = vec![0.0; n];
    for i in 0..m {
        for j in 0..n {
            mean[j] += a[i * n + j];
        }
    }
    let inv = 1.0 / m as f64;
    for j in 0..n {
        mean[j] *= inv;
    }
    out.fill(0.0);
    for i in 0..m {
        for j in 0..n {
            let d = a[i * n + j] - mean[j];
            out[j] += d * d;
        }
    }
    let scale = 1.0 / (m - ddof) as f64;
    for j in 0..n {
        out[j] *= scale;
    }
}

/// Fused variance over axis 1 (no intermediate mean Array).
#[inline]
pub(crate) fn axis1_var(m: usize, n: usize, a: &[f64], ddof: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), m);
    if n <= ddof {
        out.fill(f64::NAN);
        return;
    }
    let scale = 1.0 / (n - ddof) as f64;
    let inv = 1.0 / n as f64;
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j];
        }
        let mu = s * inv;
        let mut ss = 0.0;
        for j in 0..n {
            let d = a[i * n + j] - mu;
            ss += d * d;
        }
        out[i] = ss * scale;
    }
}


// --- Fused broadcast binary (matrix ± row / col) ---

/// `out[i,j] = a[i,j] op row[j]` for rank-2 `m×n` and length-`n` row.
#[inline]
pub(crate) fn add_matrix_row(m: usize, n: usize, a: &[f64], row: &[f64], out: &mut [f64]) {
    debug_assert_eq!(row.len(), n);
    debug_assert_eq!(a.len(), m * n);
    for i in 0..m {
        let base = i * n;
        for j in 0..n {
            out[base + j] = a[base + j] + row[j];
        }
    }
}
#[inline]
pub(crate) fn sub_matrix_row(m: usize, n: usize, a: &[f64], row: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let base = i * n;
        for j in 0..n {
            out[base + j] = a[base + j] - row[j];
        }
    }
}
#[inline]
pub(crate) fn mul_matrix_row(m: usize, n: usize, a: &[f64], row: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let base = i * n;
        for j in 0..n {
            out[base + j] = a[base + j] * row[j];
        }
    }
}
#[inline]
pub(crate) fn div_matrix_row(m: usize, n: usize, a: &[f64], row: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let base = i * n;
        for j in 0..n {
            out[base + j] = a[base + j] / row[j];
        }
    }
}

/// `out[i,j] = a[i,j] op col[i]` for rank-2 `m×n` and length-`m` col.
#[inline]
pub(crate) fn add_matrix_col(m: usize, n: usize, a: &[f64], col: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let base = i * n;
        let c = col[i];
        for j in 0..n {
            out[base + j] = a[base + j] + c;
        }
    }
}
#[inline]
pub(crate) fn sub_matrix_col(m: usize, n: usize, a: &[f64], col: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let base = i * n;
        let c = col[i];
        for j in 0..n {
            out[base + j] = a[base + j] - c;
        }
    }
}
#[inline]
pub(crate) fn mul_matrix_col(m: usize, n: usize, a: &[f64], col: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let base = i * n;
        let c = col[i];
        for j in 0..n {
            out[base + j] = a[base + j] * c;
        }
    }
}
#[inline]
pub(crate) fn div_matrix_col(m: usize, n: usize, a: &[f64], col: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let base = i * n;
        let c = col[i];
        for j in 0..n {
            out[base + j] = a[base + j] / c;
        }
    }
}

#[inline]
pub(crate) fn axis0_min(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    out.fill(f64::INFINITY);
    for i in 0..m {
        for j in 0..n {
            let x = a[i * n + j];
            if x < out[j] {
                out[j] = x;
            }
        }
    }
}

#[inline]
pub(crate) fn axis1_min(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let mut mnv = f64::INFINITY;
        for j in 0..n {
            let x = a[i * n + j];
            if x < mnv {
                mnv = x;
            }
        }
        out[i] = mnv;
    }
}

#[inline]
pub(crate) fn axis0_max(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    out.fill(f64::NEG_INFINITY);
    for i in 0..m {
        for j in 0..n {
            let x = a[i * n + j];
            if x > out[j] {
                out[j] = x;
            }
        }
    }
}

#[inline]
pub(crate) fn axis1_max(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let mut mx = f64::NEG_INFINITY;
        for j in 0..n {
            let x = a[i * n + j];
            if x > mx {
                mx = x;
            }
        }
        out[i] = mx;
    }
}

#[inline]
pub(crate) fn truthy(x: f64) -> bool {
    x != 0.0 && !x.is_nan()
}

#[inline]
pub(crate) fn any_slice(a: &[f64]) -> bool {
    a.iter().any(|&x| truthy(x))
}

#[inline]
pub(crate) fn all_slice(a: &[f64]) -> bool {
    !a.is_empty() && a.iter().all(|&x| truthy(x))
}

#[inline]
pub(crate) fn axis0_any(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for i in 0..m {
        for j in 0..n {
            if truthy(a[i * n + j]) {
                out[j] = 1.0;
            }
        }
    }
}

#[inline]
pub(crate) fn axis1_any(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let mut v = 0.0;
        for j in 0..n {
            if truthy(a[i * n + j]) {
                v = 1.0;
                break;
            }
        }
        out[i] = v;
    }
}

#[inline]
pub(crate) fn axis0_all(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    out.fill(1.0);
    if m == 0 {
        out.fill(0.0);
        return;
    }
    for i in 0..m {
        for j in 0..n {
            if !truthy(a[i * n + j]) {
                out[j] = 0.0;
            }
        }
    }
}

#[inline]
pub(crate) fn axis1_all(m: usize, n: usize, a: &[f64], out: &mut [f64]) {
    for i in 0..m {
        let mut v = 1.0;
        if n == 0 {
            v = 0.0;
        } else {
            for j in 0..n {
                if !truthy(a[i * n + j]) {
                    v = 0.0;
                    break;
                }
            }
        }
        out[i] = v;
    }
}

/// Argsort indices (0-based) into `idx` of length n.
pub(crate) fn argsort_indices(a: &[f64], descending: bool, idx: &mut [usize]) {
    let n = a.len();
    debug_assert_eq!(idx.len(), n);
    for i in 0..n {
        idx[i] = i;
    }
    if descending {
        idx.sort_by(|&i, &j| a[j].partial_cmp(&a[i]).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap_or(std::cmp::Ordering::Equal));
    }
}


/// Linear quantile on a **sorted** slice (`q` in [0, 1]). Empty → None.
#[inline]
pub(crate) fn quantile_sorted(sorted: &[f64], q: f64) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(sorted[0]);
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (pos.ceil() as usize).min(n - 1);
    let h = pos - lo as f64;
    if lo == hi || h == 0.0 {
        Some(sorted[lo])
    } else {
        Some(sorted[lo].mul_add(1.0 - h, sorted[hi] * h))
    }
}

/// Median of unsorted data. Empty → None.
///
/// Odd length: `select_nth` (average O(n)). Even length: two order statistics
/// then average (still cheaper than a full sort for large n).
#[inline]
pub(crate) fn median_slice(a: &[f64]) -> Option<f64> {
    let n = a.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(a[0]);
    }
    let mut v = a.to_vec();
    if n % 2 == 1 {
        let mid = n / 2;
        let (_, val, _) = v.select_nth_unstable_by(mid, |x, y| {
            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(*val)
    } else {
        let hi = n / 2;
        // After select for hi, v[hi] is the upper middle; lower middle is max of left partition.
        v.select_nth_unstable_by(hi, |x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let upper = v[hi];
        let lower = v[..hi]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Some(0.5 * (lower + upper))
    }
}

