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
