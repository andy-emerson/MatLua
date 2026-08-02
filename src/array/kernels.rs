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

/// Min with pairwise reduction (returns None if empty).
#[inline]
pub(crate) fn min_slice(a: &[f64]) -> Option<f64> {
    if a.is_empty() {
        return None;
    }
    // Blocked min for cache; NaN-aware: skip NaN like f64::min only if both NaN
    let mut best = f64::INFINITY;
    const BS: usize = 512;
    let mut i = 0;
    while i < a.len() {
        let end = (i + BS).min(a.len());
        let mut m0 = f64::INFINITY;
        let mut m1 = f64::INFINITY;
        let mut m2 = f64::INFINITY;
        let mut m3 = f64::INFINITY;
        let mut j = i;
        while j + 4 <= end {
            m0 = m0.min(a[j]);
            m1 = m1.min(a[j + 1]);
            m2 = m2.min(a[j + 2]);
            m3 = m3.min(a[j + 3]);
            j += 4;
        }
        while j < end {
            m0 = m0.min(a[j]);
            j += 1;
        }
        best = best.min(m0).min(m1).min(m2).min(m3);
        i = end;
    }
    if best.is_infinite() && a.iter().all(|x| x.is_nan()) {
        Some(f64::NAN)
    } else {
        Some(best)
    }
}

/// Max with pairwise reduction.
#[inline]
pub(crate) fn max_slice(a: &[f64]) -> Option<f64> {
    if a.is_empty() {
        return None;
    }
    let mut best = f64::NEG_INFINITY;
    const BS: usize = 512;
    let mut i = 0;
    while i < a.len() {
        let end = (i + BS).min(a.len());
        let mut m0 = f64::NEG_INFINITY;
        let mut m1 = f64::NEG_INFINITY;
        let mut m2 = f64::NEG_INFINITY;
        let mut m3 = f64::NEG_INFINITY;
        let mut j = i;
        while j + 4 <= end {
            m0 = m0.max(a[j]);
            m1 = m1.max(a[j + 1]);
            m2 = m2.max(a[j + 2]);
            m3 = m3.max(a[j + 3]);
            j += 4;
        }
        while j < end {
            m0 = m0.max(a[j]);
            j += 1;
        }
        best = best.max(m0).max(m1).max(m2).max(m3);
        i = end;
    }
    if best.is_infinite() && a.iter().all(|x| x.is_nan()) {
        Some(f64::NAN)
    } else {
        Some(best)
    }
}

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
