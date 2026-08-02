//! Integration correctness checks on the public Rust API (no Lua).
//!
//! Unit tests also live next to modules under `src/`. This file guards the
//! crate surface as an external consumer would use it.

use matlua::array::Array;
use matlua::linalg;

#[test]
fn constructors_and_shape() {
    let z = Array::zeros(vec![2, 3]).unwrap();
    assert_eq!(z.dims(), &[2, 3]);
    assert_eq!(z.as_slice(), &[0., 0., 0., 0., 0., 0.]);
    let o = Array::ones(vec![2]).unwrap();
    assert_eq!(o.as_slice(), &[1., 1.]);
    let a = Array::arange(0.0, 4.0).unwrap();
    assert_eq!(a.as_slice(), &[0., 1., 2., 3.]);
}

#[test]
fn elementwise_and_reductions() {
    let a = Array::from_shape_slice(vec![2, 2], &[1., 2., 3., 4.]).unwrap();
    let b = Array::from_shape_slice(vec![2, 2], &[10., 20., 30., 40.]).unwrap();
    let c = Array::add(&a, &b).unwrap();
    assert_eq!(c.as_slice(), &[11., 22., 33., 44.]);
    assert!((a.sum() - 10.0).abs() < 1e-12);
    assert_eq!(a.min().unwrap(), 1.0);
    assert_eq!(a.max().unwrap(), 4.0);
}

#[test]
fn linalg_desk_path() {
    let a = Array::from_shape_slice(vec![2, 2], &[3., 1., 1., 2.]).unwrap();
    let b = Array::from_shape_slice(vec![2], &[9., 8.]).unwrap();
    let x = linalg::solve(&a, &b).unwrap();
    assert!((x.get(&[0]).unwrap() - 2.0).abs() < 1e-10);
    assert!((x.get(&[1]).unwrap() - 3.0).abs() < 1e-10);
    let y = linalg::matmul(&a, &x).unwrap();
    assert!((y.get(&[0]).unwrap() - 9.0).abs() < 1e-9);
    let n = linalg::norm(&a).unwrap();
    assert!((n - (3.0f64.hypot(1.0).hypot(1.0).hypot(2.0))).abs() < 1e-9);
}

#[test]
fn matmul_at_and_normal_eq() {
    use matlua::array::Array;
    use matlua::linalg::{matmul, matmul_at, normal_eq, solve, transpose};

    let x = Array::from_shape_slice(vec![4, 2], &[1., 0., 1., 1., 1., 2., 1., 3.]).unwrap();
    let y = Array::from_shape_slice(vec![4], &[0., 1., 2., 3.]).unwrap();
    let atb = matmul_at(&x, &y).unwrap();
    let long = matmul(&transpose(&x).unwrap(), &y).unwrap();
    assert_eq!(atb.as_slice(), long.as_slice());
    let beta = normal_eq(&x, &y).unwrap();
    let beta2 = solve(
        &matmul(&transpose(&x).unwrap(), &x).unwrap(),
        &matmul(&transpose(&x).unwrap(), &y).unwrap(),
    )
    .unwrap();
    for (a, b) in beta.as_slice().iter().zip(beta2.as_slice()) {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn arrow_roundtrip_and_reshape_cow() {
    let a = Array::from_shape_slice(vec![2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    let arrow = a.to_arrow();
    let b = Array::from_arrow(&arrow, vec![2, 3]).unwrap();
    assert_eq!(a.as_slice(), b.as_slice());

    let r = a.reshape(vec![3, 2]).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    // Shared until write: mutating reshape COWs.
    let mut w = r.reshape(vec![6]).unwrap();
    w.as_mut_slice()[0] = 99.0;
    assert_eq!(a.as_slice()[0], 1.0);
    assert_eq!(w.as_slice()[0], 99.0);
}

#[test]
fn decompositions_smoke() {
    let a = Array::from_shape_slice(vec![2, 2], &[2.0, 0.5, 0.5, 1.0]).unwrap();
    let l = linalg::cholesky(&a).unwrap();
    assert_eq!(l.dims(), &[2, 2]);
    let (q, r) = linalg::qr(&a).unwrap();
    let recon = linalg::matmul(&q, &r).unwrap();
    for (x, y) in recon.as_slice().iter().zip(a.as_slice()) {
        assert!((x - y).abs() < 1e-9);
    }
    let (_u, s, _v) = linalg::svd(&a).unwrap();
    assert!(s.get(&[0]).unwrap() >= s.get(&[1]).unwrap());
}

#[test]
fn matmul_at_gram_matches_transpose_path() {
    // Small k uses view path; correctness vs long path.
    let x = Array::from_shape_slice(
        vec![5, 3],
        &[
            1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 1., 1.,
        ],
    )
    .unwrap();
    let short = linalg::matmul_at(&x, &x).unwrap();
    let long = linalg::matmul(&linalg::transpose(&x).unwrap(), &x).unwrap();
    for (a, b) in short.as_slice().iter().zip(long.as_slice()) {
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    }
}

#[test]
fn lstsq_eigh_pinv_smoke() {
    let x = Array::from_shape_slice(vec![4, 2], &[1., 0., 1., 1., 1., 2., 1., 3.]).unwrap();
    let y = Array::from_shape_slice(vec![4], &[1., 3., 5., 7.]).unwrap();
    let b = linalg::lstsq(&x, &y).unwrap();
    assert_eq!(b.rank(), 1);
    assert_eq!(b.len(), 2);

    let s = Array::from_shape_slice(vec![2, 2], &[2., 0.5, 0.5, 1.]).unwrap();
    let (w, v) = linalg::eigh(&s).unwrap();
    assert_eq!(w.len(), 2);
    assert_eq!(v.dims(), &[2, 2]);
    assert!(w.get(&[0]).unwrap() <= w.get(&[1]).unwrap());

    let p = linalg::pinv(&x).unwrap();
    assert_eq!(p.dims(), &[2, 4]);
}

#[test]
fn m6_tier2_smoke() {
    let m = Array::from_shape_slice(vec![2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    assert_eq!(m.sum_axis(0).unwrap().as_slice(), &[5., 7., 9.]);
    let x = Array::from_shape_slice(vec![2, 3], &[1., 2., 3., 2., 4., 6.]).unwrap();
    let c = Array::cov(&x, 1).unwrap();
    assert!((c.get(&[0, 1]).unwrap() - 2.0).abs() < 1e-9);
    let v = Array::from_shape_slice(vec![3], &[3., 1., 2.]).unwrap();
    let idx = v.argsort(false).unwrap();
    assert_eq!(v.take(&idx).unwrap().as_slice(), &[1., 2., 3.]);
    assert_eq!(Array::diag(&v).unwrap().dims(), &[3, 3]);
}

#[test]
fn m7_i64_surface_smoke() {
    use matlua::{Array, ArrayI64, DType};
    let a = ArrayI64::from_shape_slice(vec![2, 3], &[1, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(a.dtype(), DType::I64);
    assert_eq!(a.sum(), 21);
    assert_eq!(a.sum_axis(0).unwrap().as_slice(), &[5, 7, 9]);
    let b = ArrayI64::arange(0, 4).unwrap();
    assert_eq!(b.as_slice(), &[0, 1, 2, 3]);
    let c = a.add(&ArrayI64::full(vec![2, 3], 1).unwrap()).unwrap();
    assert_eq!(c.as_slice()[0], 2);
    let f = a.to_f64();
    assert_eq!(f.dtype(), DType::F64);
    assert_eq!(f.as_slice()[0], 1.0);
    let back = f.to_i64();
    assert_eq!(back.as_slice(), a.as_slice());
    let idx = ArrayI64::from_shape_slice(vec![3], &[3, 1, 4]).unwrap().argsort(false).unwrap();
    assert_eq!(idx.as_slice(), &[1, 0, 2]);
    let ar = a.to_arrow();
    let a2 = ArrayI64::from_arrow(&ar, vec![2, 3]).unwrap();
    assert_eq!(a, a2);
    // f64 LA path unchanged
    let m = Array::eye(2).unwrap();
    let _ = matlua::linalg::solve(&m, &Array::from_shape_slice(vec![2], &[1., 2.]).unwrap()).unwrap();
}
