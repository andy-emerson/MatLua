//! Contiguous `f64` kernels for elementwise ops and reductions.
//!
//! Specialized arithmetic (no `Fn` closures) and index loops over dense
//! slices so LLVM can auto-vectorize. Slices must be the same length where
//! binary.

#[inline]
pub(crate) fn add_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

#[inline]
pub(crate) fn sub_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

#[inline]
pub(crate) fn mul_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

#[inline]
pub(crate) fn div_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] / b[i];
    }
}

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

/// `out[i] = a[i] + s`
#[inline]
pub(crate) fn add_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] + s;
    }
}

/// `out[i] = a[i] - s`
#[inline]
pub(crate) fn sub_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] - s;
    }
}

/// `out[i] = s - a[i]`
#[inline]
pub(crate) fn scalar_sub(a: &[f64], s: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = s - a[i];
    }
}

/// `out[i] = a[i] * s`
#[inline]
pub(crate) fn mul_scalar(a: &[f64], s: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] * s;
    }
}

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

/// Sum with four accumulators for better ILP / auto-vectorization.
#[inline]
pub(crate) fn sum_slice(a: &[f64]) -> f64 {
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut chunks = a.chunks_exact(4);
    for c in chunks.by_ref() {
        s0 += c[0];
        s1 += c[1];
        s2 += c[2];
        s3 += c[3];
    }
    let mut s = s0 + s1 + s2 + s3;
    for &x in chunks.remainder() {
        s += x;
    }
    s
}

/// Sum of squares (for Frobenius norm).
#[inline]
pub(crate) fn sum_sq_slice(a: &[f64]) -> f64 {
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut chunks = a.chunks_exact(4);
    for c in chunks.by_ref() {
        s0 += c[0] * c[0];
        s1 += c[1] * c[1];
        s2 += c[2] * c[2];
        s3 += c[3] * c[3];
    }
    let mut s = s0 + s1 + s2 + s3;
    for &x in chunks.remainder() {
        s += x * x;
    }
    s
}

/// Minimum over a dense slice.
///
/// NaN values are skipped relative to a `+∞` seed (not full IEEE `minNum`
/// semantics).
///
/// Cache-blocked reduction (same structure as [`max_slice`]): O(n) with
/// L1-friendly tiles so large n does not thrash on a single accumulator chain.
#[inline]
pub(crate) fn min_slice(a: &[f64]) -> Option<f64> {
    if a.is_empty() {
        return None;
    }
    const BLOCK: usize = 512;
    let mut global = f64::INFINITY;
    for block in a.chunks(BLOCK) {
        let mut m0 = f64::INFINITY;
        let mut m1 = f64::INFINITY;
        let mut m2 = f64::INFINITY;
        let mut m3 = f64::INFINITY;
        let mut chunks = block.chunks_exact(4);
        for c in chunks.by_ref() {
            if c[0] < m0 {
                m0 = c[0];
            }
            if c[1] < m1 {
                m1 = c[1];
            }
            if c[2] < m2 {
                m2 = c[2];
            }
            if c[3] < m3 {
                m3 = c[3];
            }
        }
        let mut m = m0;
        if m1 < m {
            m = m1;
        }
        if m2 < m {
            m = m2;
        }
        if m3 < m {
            m = m3;
        }
        for &x in chunks.remainder() {
            if x < m {
                m = x;
            }
        }
        if m < global {
            global = m;
        }
    }
    Some(global)
}

/// Maximum over a dense slice.
///
/// NaN values are skipped relative to a `−∞` seed (not full IEEE `maxNum`
/// semantics).
///
/// Cache-blocked reduction: maxima within L1-friendly blocks, then max of
/// block maxima. Still O(n); blocking helps large n (better cache use), not a
/// small-n-only trick.
#[inline]
pub(crate) fn max_slice(a: &[f64]) -> Option<f64> {
    if a.is_empty() {
        return None;
    }
    // ~4 KiB of f64s — stays friendly to L1 while giving the inner loop room.
    const BLOCK: usize = 512;
    let mut global = f64::NEG_INFINITY;
    for block in a.chunks(BLOCK) {
        let mut m0 = f64::NEG_INFINITY;
        let mut m1 = f64::NEG_INFINITY;
        let mut m2 = f64::NEG_INFINITY;
        let mut m3 = f64::NEG_INFINITY;
        let mut chunks = block.chunks_exact(4);
        for c in chunks.by_ref() {
            if c[0] > m0 {
                m0 = c[0];
            }
            if c[1] > m1 {
                m1 = c[1];
            }
            if c[2] > m2 {
                m2 = c[2];
            }
            if c[3] > m3 {
                m3 = c[3];
            }
        }
        let mut m = m0;
        if m1 > m {
            m = m1;
        }
        if m2 > m {
            m = m2;
        }
        if m3 > m {
            m = m3;
        }
        for &x in chunks.remainder() {
            if x > m {
                m = x;
            }
        }
        if m > global {
            global = m;
        }
    }
    Some(global)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_and_reductions() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = [0.0; 5];
        add_slices(&a, &b, &mut out);
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0, 55.0]);
        mul_slices(&a, &b, &mut out);
        assert_eq!(out, [10.0, 40.0, 90.0, 160.0, 250.0]);
        assert!((sum_slice(&a) - 15.0).abs() < 1e-12);
        assert_eq!(min_slice(&a), Some(1.0));
        assert_eq!(max_slice(&a), Some(5.0));
        assert_eq!(min_slice(&[]), None);
    }
}


// --- M5 Tier-1 ufuncs (IEEE NaN propagate unless noted) ---

#[inline]
pub(crate) fn abs_slice(a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].abs();
    }
}

#[inline]
pub(crate) fn sqrt_slice(a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].sqrt();
    }
}

#[inline]
pub(crate) fn exp_slice(a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].exp();
    }
}

#[inline]
pub(crate) fn log_slice(a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].ln();
    }
}

#[inline]
pub(crate) fn log1p_slice(a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].ln_1p();
    }
}

#[inline]
pub(crate) fn sign_slice(a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
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
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].powf(b[i]);
    }
}

#[inline]
pub(crate) fn power_scalar(a: &[f64], p: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].powf(p);
    }
}

#[inline]
pub(crate) fn clip_slice(a: &[f64], lo: f64, hi: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
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

/// 1.0 where NaN, else 0.0 (dense f64 mask; not a separate bool dtype).
#[inline]
pub(crate) fn isnan_slice(a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = if a[i].is_nan() { 1.0 } else { 0.0 };
    }
}

#[inline]
pub(crate) fn isfinite_slice(a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = if a[i].is_finite() { 1.0 } else { 0.0 };
    }
}

#[inline]
pub(crate) fn where_slices(cond: &[f64], a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(cond.len(), a.len());
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        // Nonzero (and not NaN) is true — Lua/C style; NaN condition → false branch.
        out[i] = if cond[i] != 0.0 && !cond[i].is_nan() {
            a[i]
        } else {
            b[i]
        };
    }
}

#[inline]
pub(crate) fn cumsum_slice(a: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
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
    let mut best_i = 0usize;
    let mut best = f64::INFINITY;
    for (i, &x) in a.iter().enumerate() {
        if x < best {
            best = x;
            best_i = i;
        }
    }
    // If all NaN, best stays ∞ — return 0 like a weak default; callers with all-NaN rare.
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
    let mut best_i = 0usize;
    let mut best = f64::NEG_INFINITY;
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

/// Population or sample variance from Welford; `ddof` is degrees of freedom subtract.
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
