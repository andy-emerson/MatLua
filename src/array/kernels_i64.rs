//! Contiguous `i64` kernels (M7.c: unrolled bulk paths; wrapping arithmetic).
#![allow(dead_code)]

#[inline]
fn zip4(a: &[i64], b: &[i64], out: &mut [i64], f: impl Fn(i64, i64) -> i64) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(out.len(), n);
    let mut i = 0;
    while i + 4 <= n {
        out[i] = f(a[i], b[i]);
        out[i + 1] = f(a[i + 1], b[i + 1]);
        out[i + 2] = f(a[i + 2], b[i + 2]);
        out[i + 3] = f(a[i + 3], b[i + 3]);
        i += 4;
    }
    while i < n {
        out[i] = f(a[i], b[i]);
        i += 1;
    }
}

#[inline]
fn map4(a: &[i64], out: &mut [i64], f: impl Fn(i64) -> i64) {
    let n = a.len();
    debug_assert_eq!(out.len(), n);
    let mut i = 0;
    while i + 4 <= n {
        out[i] = f(a[i]);
        out[i + 1] = f(a[i + 1]);
        out[i + 2] = f(a[i + 2]);
        out[i + 3] = f(a[i + 3]);
        i += 4;
    }
    while i < n {
        out[i] = f(a[i]);
        i += 1;
    }
}

#[inline]
fn assign4(a: &mut [i64], b: &[i64], f: impl Fn(i64, i64) -> i64) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    let mut i = 0;
    while i + 4 <= n {
        a[i] = f(a[i], b[i]);
        a[i + 1] = f(a[i + 1], b[i + 1]);
        a[i + 2] = f(a[i + 2], b[i + 2]);
        a[i + 3] = f(a[i + 3], b[i + 3]);
        i += 4;
    }
    while i < n {
        a[i] = f(a[i], b[i]);
        i += 1;
    }
}

// ISA-dispatched twins: same portable body compiled with AVX-512 features
// and taken when the CPU has them (pattern and measured effects as in the
// f64 kernels and the GEMM profiles; i64 multiply additionally gains the
// native `vpmullq`). Wrapping i64 ops are exact under any lane order.
macro_rules! isa_binary_i64 {
    ($name:ident, $avx:ident, $f:expr) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
        unsafe fn $avx(a: &[i64], b: &[i64], out: &mut [i64]) {
            let f = $f;
            for i in 0..a.len() {
                out[i] = f(a[i], b[i]);
            }
        }
        #[inline]
        pub(crate) fn $name(a: &[i64], b: &[i64], out: &mut [i64]) {
            #[cfg(target_arch = "x86_64")]
            if crate::array::isa::avx512() {
                // SAFETY: features verified by isa::avx512().
                unsafe { $avx(a, b, out) };
                return;
            }
            zip4(a, b, out, $f);
        }
    };
}

isa_binary_i64!(add_slices, add_slices_avx512, i64::wrapping_add);
isa_binary_i64!(sub_slices, sub_slices_avx512, i64::wrapping_sub);
isa_binary_i64!(mul_slices, mul_slices_avx512, i64::wrapping_mul);
/// Truncating division; division by zero → 0 (no panic).
#[inline]
pub(crate) fn div_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    zip4(a, b, out, |x, y| if y == 0 { 0 } else { x / y });
}
#[inline]
pub(crate) fn add_assign_slices(a: &mut [i64], b: &[i64]) {
    assign4(a, b, i64::wrapping_add);
}
#[inline]
pub(crate) fn sub_assign_slices(a: &mut [i64], b: &[i64]) {
    assign4(a, b, i64::wrapping_sub);
}
#[inline]
pub(crate) fn mul_assign_slices(a: &mut [i64], b: &[i64]) {
    assign4(a, b, i64::wrapping_mul);
}
#[inline]
pub(crate) fn div_assign_slices(a: &mut [i64], b: &[i64]) {
    assign4(a, b, |x, y| if y == 0 { 0 } else { x / y });
}
#[inline]
pub(crate) fn neg_slice(a: &[i64], out: &mut [i64]) {
    map4(a, out, i64::wrapping_neg);
}
macro_rules! isa_scalar_i64 {
    ($name:ident, $avx:ident, $f:expr) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
        unsafe fn $avx(a: &[i64], s: i64, out: &mut [i64]) {
            let f = $f;
            for i in 0..a.len() {
                out[i] = f(a[i], s);
            }
        }
        #[inline]
        pub(crate) fn $name(a: &[i64], s: i64, out: &mut [i64]) {
            #[cfg(target_arch = "x86_64")]
            if crate::array::isa::avx512() {
                // SAFETY: features verified by isa::avx512().
                unsafe { $avx(a, s, out) };
                return;
            }
            map4(a, out, |x| ($f)(x, s));
        }
    };
}

isa_scalar_i64!(add_scalar, add_scalar_avx512, i64::wrapping_add);
isa_scalar_i64!(sub_scalar, sub_scalar_avx512, i64::wrapping_sub);
isa_scalar_i64!(mul_scalar, mul_scalar_avx512, i64::wrapping_mul);
#[inline]
pub(crate) fn div_scalar(a: &[i64], s: i64, out: &mut [i64]) {
    if s == 0 {
        out.fill(0);
        return;
    }
    map4(a, out, |x| x / s);
}
#[inline]
pub(crate) fn abs_slice(a: &[i64], out: &mut [i64]) {
    map4(a, out, i64::wrapping_abs);
}

/// Four independent accumulators for ILP.
#[inline]
pub(crate) fn sum_slice(a: &[i64]) -> i64 {
    let mut s0: i64 = 0;
    let mut s1: i64 = 0;
    let mut s2: i64 = 0;
    let mut s3: i64 = 0;
    let mut chunks = a.chunks_exact(4);
    for c in chunks.by_ref() {
        s0 = s0.wrapping_add(c[0]);
        s1 = s1.wrapping_add(c[1]);
        s2 = s2.wrapping_add(c[2]);
        s3 = s3.wrapping_add(c[3]);
    }
    let mut s = s0.wrapping_add(s1).wrapping_add(s2).wrapping_add(s3);
    for &x in chunks.remainder() {
        s = s.wrapping_add(x);
    }
    s
}
#[inline]
pub(crate) fn min_slice(a: &[i64]) -> Option<i64> {
    if a.is_empty() {
        return None;
    }
    let mut m0 = a[0];
    let mut m1 = a[0];
    let mut m2 = a[0];
    let mut m3 = a[0];
    let mut chunks = a[1..].chunks_exact(4);
    for c in chunks.by_ref() {
        m0 = m0.min(c[0]);
        m1 = m1.min(c[1]);
        m2 = m2.min(c[2]);
        m3 = m3.min(c[3]);
    }
    for &x in chunks.remainder() {
        m0 = m0.min(x);
    }
    Some(m0.min(m1).min(m2).min(m3))
}
#[inline]
pub(crate) fn max_slice(a: &[i64]) -> Option<i64> {
    if a.is_empty() {
        return None;
    }
    let mut m0 = a[0];
    let mut m1 = a[0];
    let mut m2 = a[0];
    let mut m3 = a[0];
    let mut chunks = a[1..].chunks_exact(4);
    for c in chunks.by_ref() {
        m0 = m0.max(c[0]);
        m1 = m1.max(c[1]);
        m2 = m2.max(c[2]);
        m3 = m3.max(c[3]);
    }
    for &x in chunks.remainder() {
        m0 = m0.max(x);
    }
    Some(m0.max(m1).max(m2).max(m3))
}
#[inline]
pub(crate) fn eq_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    zip4(a, b, out, |x, y| if x == y { 1 } else { 0 });
}
#[inline]
pub(crate) fn ne_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    zip4(a, b, out, |x, y| if x != y { 1 } else { 0 });
}
#[inline]
pub(crate) fn lt_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    zip4(a, b, out, |x, y| if x < y { 1 } else { 0 });
}
#[inline]
pub(crate) fn le_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    zip4(a, b, out, |x, y| if x <= y { 1 } else { 0 });
}
#[inline]
pub(crate) fn gt_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    zip4(a, b, out, |x, y| if x > y { 1 } else { 0 });
}
#[inline]
pub(crate) fn ge_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    zip4(a, b, out, |x, y| if x >= y { 1 } else { 0 });
}
#[inline]
pub(crate) fn eq_scalar(a: &[i64], s: i64, out: &mut [i64]) {
    map4(a, out, |x| if x == s { 1 } else { 0 });
}
#[inline]
pub(crate) fn cumsum_slice(a: &[i64], out: &mut [i64]) {
    let mut s: i64 = 0;
    for i in 0..a.len() {
        s = s.wrapping_add(a[i]);
        out[i] = s;
    }
}
#[inline]
pub(crate) fn argmin_slice(a: &[i64]) -> Option<usize> {
    if a.is_empty() {
        return None;
    }
    let mut best = 0usize;
    for i in 1..a.len() {
        if a[i] < a[best] {
            best = i;
        }
    }
    Some(best)
}
#[inline]
pub(crate) fn argmax_slice(a: &[i64]) -> Option<usize> {
    if a.is_empty() {
        return None;
    }
    let mut best = 0usize;
    for i in 1..a.len() {
        if a[i] > a[best] {
            best = i;
        }
    }
    Some(best)
}
#[inline]
pub(crate) fn truthy(x: i64) -> bool {
    x != 0
}
#[inline]
pub(crate) fn any_slice(a: &[i64]) -> bool {
    a.iter().any(|&x| truthy(x))
}
#[inline]
pub(crate) fn all_slice(a: &[i64]) -> bool {
    !a.is_empty() && a.iter().all(|&x| truthy(x))
}
#[inline]
pub(crate) fn axis0_sum(m: usize, n: usize, a: &[i64], out: &mut [i64]) {
    out.fill(0);
    for i in 0..m {
        for j in 0..n {
            out[j] = out[j].wrapping_add(a[i * n + j]);
        }
    }
}
#[inline]
pub(crate) fn axis1_sum(m: usize, n: usize, a: &[i64], out: &mut [i64]) {
    for i in 0..m {
        out[i] = sum_slice(&a[i * n..(i + 1) * n]);
    }
}
#[inline]
pub(crate) fn axis0_min(m: usize, n: usize, a: &[i64], out: &mut [i64]) {
    out.fill(i64::MAX);
    for i in 0..m {
        for j in 0..n {
            out[j] = out[j].min(a[i * n + j]);
        }
    }
}
#[inline]
pub(crate) fn axis1_min(m: usize, n: usize, a: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let mut v = i64::MAX;
        for j in 0..n {
            v = v.min(a[i * n + j]);
        }
        out[i] = v;
    }
}
#[inline]
pub(crate) fn axis0_max(m: usize, n: usize, a: &[i64], out: &mut [i64]) {
    out.fill(i64::MIN);
    for i in 0..m {
        for j in 0..n {
            out[j] = out[j].max(a[i * n + j]);
        }
    }
}
#[inline]
pub(crate) fn axis1_max(m: usize, n: usize, a: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let mut v = i64::MIN;
        for j in 0..n {
            v = v.max(a[i * n + j]);
        }
        out[i] = v;
    }
}
#[inline]
pub(crate) fn add_matrix_row(m: usize, n: usize, a: &[i64], row: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let ai = &a[i * n..(i + 1) * n];
        let oi = &mut out[i * n..(i + 1) * n];
        zip4(ai, row, oi, i64::wrapping_add);
    }
}
#[inline]
pub(crate) fn sub_matrix_row(m: usize, n: usize, a: &[i64], row: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let ai = &a[i * n..(i + 1) * n];
        let oi = &mut out[i * n..(i + 1) * n];
        zip4(ai, row, oi, i64::wrapping_sub);
    }
}
#[inline]
pub(crate) fn mul_matrix_row(m: usize, n: usize, a: &[i64], row: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let ai = &a[i * n..(i + 1) * n];
        let oi = &mut out[i * n..(i + 1) * n];
        zip4(ai, row, oi, i64::wrapping_mul);
    }
}
#[inline]
pub(crate) fn div_matrix_row(m: usize, n: usize, a: &[i64], row: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let ai = &a[i * n..(i + 1) * n];
        let oi = &mut out[i * n..(i + 1) * n];
        zip4(ai, row, oi, |x, y| if y == 0 { 0 } else { x / y });
    }
}
#[inline]
pub(crate) fn add_matrix_col(m: usize, n: usize, a: &[i64], col: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let base = i * n;
        let c = col[i];
        for j in 0..n {
            out[base + j] = a[base + j].wrapping_add(c);
        }
    }
}
#[inline]
pub(crate) fn sub_matrix_col(m: usize, n: usize, a: &[i64], col: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let base = i * n;
        let c = col[i];
        for j in 0..n {
            out[base + j] = a[base + j].wrapping_sub(c);
        }
    }
}
#[inline]
pub(crate) fn mul_matrix_col(m: usize, n: usize, a: &[i64], col: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let base = i * n;
        let c = col[i];
        for j in 0..n {
            out[base + j] = a[base + j].wrapping_mul(c);
        }
    }
}
#[inline]
pub(crate) fn div_matrix_col(m: usize, n: usize, a: &[i64], col: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let base = i * n;
        let c = col[i];
        for j in 0..n {
            out[base + j] = if c == 0 { 0 } else { a[base + j] / c };
        }
    }
}
pub(crate) fn argsort_indices(a: &[i64], descending: bool, idx: &mut [usize]) {
    let n = a.len();
    for i in 0..n {
        idx[i] = i;
    }
    if descending {
        idx.sort_by(|&i, &j| a[j].cmp(&a[i]));
    } else {
        idx.sort_by(|&i, &j| a[i].cmp(&a[j]));
    }
}

#[inline]
pub(crate) fn where_slices(cond: &[i64], x: &[i64], y: &[i64], out: &mut [i64]) {
    for i in 0..cond.len() {
        out[i] = if cond[i] != 0 { x[i] } else { y[i] };
    }
}
#[inline]
pub(crate) fn sign_slice(a: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i].signum();
    }
}
#[inline]
pub(crate) fn clip_slice(a: &[i64], lo: i64, hi: i64, out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i].clamp(lo, hi);
    }
}
#[inline]
pub(crate) fn axis0_any(m: usize, n: usize, a: &[i64], out: &mut [i64]) {
    out.fill(0);
    for i in 0..m {
        for j in 0..n {
            if a[i * n + j] != 0 {
                out[j] = 1;
            }
        }
    }
}
#[inline]
pub(crate) fn axis1_any(m: usize, n: usize, a: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let mut v = 0i64;
        for j in 0..n {
            if a[i * n + j] != 0 {
                v = 1;
                break;
            }
        }
        out[i] = v;
    }
}
#[inline]
pub(crate) fn axis0_all(m: usize, n: usize, a: &[i64], out: &mut [i64]) {
    out.fill(1);
    for i in 0..m {
        for j in 0..n {
            if a[i * n + j] == 0 {
                out[j] = 0;
            }
        }
    }
}
#[inline]
pub(crate) fn axis1_all(m: usize, n: usize, a: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let mut v = 1i64;
        for j in 0..n {
            if a[i * n + j] == 0 {
                v = 0;
                break;
            }
        }
        out[i] = v;
    }
}

#[inline]
pub(crate) fn bitand_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i] & b[i];
    }
}
#[inline]
pub(crate) fn bitor_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i] | b[i];
    }
}
#[inline]
pub(crate) fn bitxor_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i] ^ b[i];
    }
}
#[inline]
pub(crate) fn rem_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = if b[i] == 0 { 0 } else { a[i] % b[i] };
    }
}
