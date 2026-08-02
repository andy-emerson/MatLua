//! Contiguous `i64` kernels (correctness-first; not yet tuned).

#[inline]
pub(crate) fn add_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i].wrapping_add(b[i]);
    }
}
#[inline]
pub(crate) fn sub_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i].wrapping_sub(b[i]);
    }
}
#[inline]
pub(crate) fn mul_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i].wrapping_mul(b[i]);
    }
}
/// Truncating division (Rust `/`); division by zero yields 0 (document; avoid panic).
#[inline]
pub(crate) fn div_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = if b[i] == 0 { 0 } else { a[i] / b[i] };
    }
}
#[inline]
pub(crate) fn add_assign_slices(a: &mut [i64], b: &[i64]) {
    for i in 0..a.len() {
        a[i] = a[i].wrapping_add(b[i]);
    }
}
#[inline]
pub(crate) fn sub_assign_slices(a: &mut [i64], b: &[i64]) {
    for i in 0..a.len() {
        a[i] = a[i].wrapping_sub(b[i]);
    }
}
#[inline]
pub(crate) fn mul_assign_slices(a: &mut [i64], b: &[i64]) {
    for i in 0..a.len() {
        a[i] = a[i].wrapping_mul(b[i]);
    }
}
#[inline]
pub(crate) fn div_assign_slices(a: &mut [i64], b: &[i64]) {
    for i in 0..a.len() {
        a[i] = if b[i] == 0 { 0 } else { a[i] / b[i] };
    }
}
#[inline]
pub(crate) fn neg_slice(a: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i].wrapping_neg();
    }
}
#[inline]
pub(crate) fn add_scalar(a: &[i64], s: i64, out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i].wrapping_add(s);
    }
}
#[inline]
pub(crate) fn sub_scalar(a: &[i64], s: i64, out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i].wrapping_sub(s);
    }
}
#[inline]
pub(crate) fn mul_scalar(a: &[i64], s: i64, out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i].wrapping_mul(s);
    }
}
#[inline]
pub(crate) fn div_scalar(a: &[i64], s: i64, out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = if s == 0 { 0 } else { a[i] / s };
    }
}
#[inline]
pub(crate) fn abs_slice(a: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = a[i].wrapping_abs();
    }
}
#[inline]
pub(crate) fn sum_slice(a: &[i64]) -> i64 {
    let mut s: i64 = 0;
    for &x in a {
        s = s.wrapping_add(x);
    }
    s
}
#[inline]
pub(crate) fn min_slice(a: &[i64]) -> Option<i64> {
    a.iter().copied().min()
}
#[inline]
pub(crate) fn max_slice(a: &[i64]) -> Option<i64> {
    a.iter().copied().max()
}
#[inline]
pub(crate) fn eq_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = if a[i] == b[i] { 1 } else { 0 };
    }
}
#[inline]
pub(crate) fn ne_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = if a[i] != b[i] { 1 } else { 0 };
    }
}
#[inline]
pub(crate) fn lt_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = if a[i] < b[i] { 1 } else { 0 };
    }
}
#[inline]
pub(crate) fn le_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = if a[i] <= b[i] { 1 } else { 0 };
    }
}
#[inline]
pub(crate) fn gt_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = if a[i] > b[i] { 1 } else { 0 };
    }
}
#[inline]
pub(crate) fn ge_slices(a: &[i64], b: &[i64], out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = if a[i] >= b[i] { 1 } else { 0 };
    }
}
#[inline]
pub(crate) fn eq_scalar(a: &[i64], s: i64, out: &mut [i64]) {
    for i in 0..a.len() {
        out[i] = if a[i] == s { 1 } else { 0 };
    }
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
        let mut s: i64 = 0;
        for j in 0..n {
            s = s.wrapping_add(a[i * n + j]);
        }
        out[i] = s;
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
        let base = i * n;
        for j in 0..n {
            out[base + j] = a[base + j].wrapping_add(row[j]);
        }
    }
}
#[inline]
pub(crate) fn sub_matrix_row(m: usize, n: usize, a: &[i64], row: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let base = i * n;
        for j in 0..n {
            out[base + j] = a[base + j].wrapping_sub(row[j]);
        }
    }
}
#[inline]
pub(crate) fn mul_matrix_row(m: usize, n: usize, a: &[i64], row: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let base = i * n;
        for j in 0..n {
            out[base + j] = a[base + j].wrapping_mul(row[j]);
        }
    }
}
#[inline]
pub(crate) fn div_matrix_row(m: usize, n: usize, a: &[i64], row: &[i64], out: &mut [i64]) {
    for i in 0..m {
        let base = i * n;
        for j in 0..n {
            out[base + j] = if row[j] == 0 { 0 } else { a[base + j] / row[j] };
        }
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
