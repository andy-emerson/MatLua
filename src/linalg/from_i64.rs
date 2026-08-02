//! Real (`f64`) dense LA on integer inputs via promotion (NumPy-style).
//!
//! Integer×integer products stay on [`super::i64_ops`]. Solvers and factorizations
//! promote with [`ArrayI64::to_f64`](crate::array::ArrayI64::to_f64) and return
//! [`Array`](crate::array::Array) (`f64`). Values with magnitude above \(2^{53}\)
//! are not exact in `f64`.

use crate::array::{Array, ArrayI64};
use crate::error::Result;

#[inline]
fn f(a: &ArrayI64) -> Array {
    a.to_f64()
}

/// `solve` after promoting `a` and `b` to `f64`.
pub fn solve(a: &ArrayI64, b: &ArrayI64) -> Result<Array> {
    super::solve(&f(a), &f(b))
}

/// `lstsq` after promoting to `f64`.
pub fn lstsq(a: &ArrayI64, b: &ArrayI64) -> Result<Array> {
    super::lstsq(&f(a), &f(b))
}

/// `normal_eq` after promoting to `f64`.
pub fn normal_eq(x: &ArrayI64, y: &ArrayI64) -> Result<Array> {
    super::normal_eq(&f(x), &f(y))
}

/// `pinv` after promoting to `f64`.
pub fn pinv(a: &ArrayI64) -> Result<Array> {
    super::pinv(&f(a))
}

/// `eigh` after promoting to `f64` (eigenvalues and vectors as `f64` arrays).
pub fn eigh(a: &ArrayI64) -> Result<(Array, Array)> {
    super::eigh(&f(a))
}

/// `cholesky` after promoting to `f64`.
pub fn cholesky(a: &ArrayI64) -> Result<Array> {
    super::cholesky(&f(a))
}

/// `qr` after promoting to `f64`.
pub fn qr(a: &ArrayI64) -> Result<(Array, Array)> {
    super::qr(&f(a))
}

/// `svd` after promoting to `f64`.
pub fn svd(a: &ArrayI64) -> Result<(Array, Array, Array)> {
    super::svd(&f(a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::ArrayI64;

    #[test]
    fn solve_identity_i64_returns_f64() {
        let a = ArrayI64::eye(2).unwrap();
        let b = ArrayI64::from_shape_slice(vec![2], &[3, 4]).unwrap();
        let x = solve(&a, &b).unwrap();
        assert_eq!(x.dtype(), crate::array::DType::F64);
        assert!((x.as_slice()[0] - 3.0).abs() < 1e-12);
        assert!((x.as_slice()[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn eigh_symmetric_i64() {
        let a = ArrayI64::from_shape_slice(vec![2, 2], &[2, 0, 0, 3]).unwrap();
        let (w, _v) = eigh(&a).unwrap();
        assert_eq!(w.len(), 2);
        let mut ev = w.as_slice().to_vec();
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ev[0] - 2.0).abs() < 1e-10);
        assert!((ev[1] - 3.0).abs() < 1e-10);
    }
}
