//! Contiguous `f64` kernels for elementwise ops and reductions (P2).
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

/// Minimum over a dense slice (chunked compares for ILP / auto-vectorization).
#[inline]
pub(crate) fn min_slice(a: &[f64]) -> Option<f64> {
    if a.is_empty() {
        return None;
    }
    let mut m0 = f64::INFINITY;
    let mut m1 = f64::INFINITY;
    let mut m2 = f64::INFINITY;
    let mut m3 = f64::INFINITY;
    let mut chunks = a.chunks_exact(4);
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
    Some(m)
}

/// Maximum over a dense slice (chunked compares for ILP / auto-vectorization).
#[inline]
pub(crate) fn max_slice(a: &[f64]) -> Option<f64> {
    if a.is_empty() {
        return None;
    }
    let mut m0 = f64::NEG_INFINITY;
    let mut m1 = f64::NEG_INFINITY;
    let mut m2 = f64::NEG_INFINITY;
    let mut m3 = f64::NEG_INFINITY;
    let mut chunks = a.chunks_exact(4);
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
    Some(m)
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
