//! Operator overloads for owned [`Array`](super::Array) (element-wise).
//!
//! Operators panic on shape mismatch so they stay expression-friendly.
//! Prefer [`Array::add`](super::Array::add) (and friends) for fallible APIs.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::Array;
use crate::error::Result;

fn panic_shape(op: &str, err: crate::error::Error) -> ! {
    panic!("matlua array {op}: {err}");
}

impl Add for &Array {
    type Output = Array;
    fn add(self, rhs: &Array) -> Array {
        self.binary_op(rhs, |a, b| a + b)
            .unwrap_or_else(|e| panic_shape("add", e))
    }
}

impl Add for Array {
    type Output = Array;
    fn add(self, rhs: Array) -> Array {
        Add::add(&self, &rhs)
    }
}

impl Add<&Array> for Array {
    type Output = Array;
    fn add(self, rhs: &Array) -> Array {
        Add::add(&self, rhs)
    }
}

impl Add<Array> for &Array {
    type Output = Array;
    fn add(self, rhs: Array) -> Array {
        Add::add(self, &rhs)
    }
}

impl AddAssign<&Array> for Array {
    fn add_assign(&mut self, rhs: &Array) {
        self.binary_op_assign(rhs, |a, b| a + b)
            .unwrap_or_else(|e| panic_shape("add_assign", e));
    }
}

impl AddAssign for Array {
    fn add_assign(&mut self, rhs: Array) {
        AddAssign::add_assign(self, &rhs);
    }
}

impl Sub for &Array {
    type Output = Array;
    fn sub(self, rhs: &Array) -> Array {
        self.binary_op(rhs, |a, b| a - b)
            .unwrap_or_else(|e| panic_shape("sub", e))
    }
}

impl Sub for Array {
    type Output = Array;
    fn sub(self, rhs: Array) -> Array {
        Sub::sub(&self, &rhs)
    }
}

impl Sub<&Array> for Array {
    type Output = Array;
    fn sub(self, rhs: &Array) -> Array {
        Sub::sub(&self, rhs)
    }
}

impl Sub<Array> for &Array {
    type Output = Array;
    fn sub(self, rhs: Array) -> Array {
        Sub::sub(self, &rhs)
    }
}

impl SubAssign<&Array> for Array {
    fn sub_assign(&mut self, rhs: &Array) {
        self.binary_op_assign(rhs, |a, b| a - b)
            .unwrap_or_else(|e| panic_shape("sub_assign", e));
    }
}

impl SubAssign for Array {
    fn sub_assign(&mut self, rhs: Array) {
        SubAssign::sub_assign(self, &rhs);
    }
}

impl Mul for &Array {
    type Output = Array;
    fn mul(self, rhs: &Array) -> Array {
        self.binary_op(rhs, |a, b| a * b)
            .unwrap_or_else(|e| panic_shape("mul", e))
    }
}

impl Mul for Array {
    type Output = Array;
    fn mul(self, rhs: Array) -> Array {
        Mul::mul(&self, &rhs)
    }
}

impl Mul<&Array> for Array {
    type Output = Array;
    fn mul(self, rhs: &Array) -> Array {
        Mul::mul(&self, rhs)
    }
}

impl Mul<Array> for &Array {
    type Output = Array;
    fn mul(self, rhs: Array) -> Array {
        Mul::mul(self, &rhs)
    }
}

impl MulAssign<&Array> for Array {
    fn mul_assign(&mut self, rhs: &Array) {
        self.binary_op_assign(rhs, |a, b| a * b)
            .unwrap_or_else(|e| panic_shape("mul_assign", e));
    }
}

impl MulAssign for Array {
    fn mul_assign(&mut self, rhs: Array) {
        MulAssign::mul_assign(self, &rhs);
    }
}

impl Div for &Array {
    type Output = Array;
    fn div(self, rhs: &Array) -> Array {
        self.binary_op(rhs, |a, b| a / b)
            .unwrap_or_else(|e| panic_shape("div", e))
    }
}

impl Div for Array {
    type Output = Array;
    fn div(self, rhs: Array) -> Array {
        Div::div(&self, &rhs)
    }
}

impl Div<&Array> for Array {
    type Output = Array;
    fn div(self, rhs: &Array) -> Array {
        Div::div(&self, rhs)
    }
}

impl Div<Array> for &Array {
    type Output = Array;
    fn div(self, rhs: Array) -> Array {
        Div::div(self, &rhs)
    }
}

impl DivAssign<&Array> for Array {
    fn div_assign(&mut self, rhs: &Array) {
        self.binary_op_assign(rhs, |a, b| a / b)
            .unwrap_or_else(|e| panic_shape("div_assign", e));
    }
}

impl DivAssign for Array {
    fn div_assign(&mut self, rhs: Array) {
        DivAssign::div_assign(self, &rhs);
    }
}

impl Neg for &Array {
    type Output = Array;
    fn neg(self) -> Array {
        Array::neg(self)
    }
}

impl Neg for Array {
    type Output = Array;
    fn neg(self) -> Array {
        Array::neg(&self)
    }
}

impl Add<f64> for &Array {
    type Output = Array;
    fn add(self, rhs: f64) -> Array {
        self.scalar_op(rhs, |a, s| a + s)
    }
}

impl Add<f64> for Array {
    type Output = Array;
    fn add(self, rhs: f64) -> Array {
        Add::add(&self, rhs)
    }
}

impl Add<&Array> for f64 {
    type Output = Array;
    fn add(self, rhs: &Array) -> Array {
        rhs.scalar_op(self, |a, s| s + a)
    }
}

impl Add<Array> for f64 {
    type Output = Array;
    fn add(self, rhs: Array) -> Array {
        Add::add(self, &rhs)
    }
}

impl Sub<f64> for &Array {
    type Output = Array;
    fn sub(self, rhs: f64) -> Array {
        self.scalar_op(rhs, |a, s| a - s)
    }
}

impl Sub<f64> for Array {
    type Output = Array;
    fn sub(self, rhs: f64) -> Array {
        Sub::sub(&self, rhs)
    }
}

impl Sub<&Array> for f64 {
    type Output = Array;
    fn sub(self, rhs: &Array) -> Array {
        rhs.scalar_op(self, |a, s| s - a)
    }
}

impl Sub<Array> for f64 {
    type Output = Array;
    fn sub(self, rhs: Array) -> Array {
        Sub::sub(self, &rhs)
    }
}

impl Mul<f64> for &Array {
    type Output = Array;
    fn mul(self, rhs: f64) -> Array {
        self.scalar_op(rhs, |a, s| a * s)
    }
}

impl Mul<f64> for Array {
    type Output = Array;
    fn mul(self, rhs: f64) -> Array {
        Mul::mul(&self, rhs)
    }
}

impl Mul<&Array> for f64 {
    type Output = Array;
    fn mul(self, rhs: &Array) -> Array {
        rhs.scalar_op(self, |a, s| s * a)
    }
}

impl Mul<Array> for f64 {
    type Output = Array;
    fn mul(self, rhs: Array) -> Array {
        Mul::mul(self, &rhs)
    }
}

impl Div<f64> for &Array {
    type Output = Array;
    fn div(self, rhs: f64) -> Array {
        self.scalar_op(rhs, |a, s| a / s)
    }
}

impl Div<f64> for Array {
    type Output = Array;
    fn div(self, rhs: f64) -> Array {
        Div::div(&self, rhs)
    }
}

impl Div<&Array> for f64 {
    type Output = Array;
    fn div(self, rhs: &Array) -> Array {
        rhs.scalar_op(self, |a, s| s / a)
    }
}

impl Div<Array> for f64 {
    type Output = Array;
    fn div(self, rhs: Array) -> Array {
        Div::div(self, &rhs)
    }
}

/// Fallible element-wise helpers (aliases of the inherent methods).
pub trait TryElemwise {
    /// Fallible add.
    fn try_add(&self, other: &Array) -> Result<Array>;
    /// Fallible sub.
    fn try_sub(&self, other: &Array) -> Result<Array>;
    /// Fallible mul.
    fn try_mul(&self, other: &Array) -> Result<Array>;
    /// Fallible div.
    fn try_div(&self, other: &Array) -> Result<Array>;
}

impl TryElemwise for Array {
    fn try_add(&self, other: &Array) -> Result<Array> {
        Array::add(self, other)
    }
    fn try_sub(&self, other: &Array) -> Result<Array> {
        Array::sub(self, other)
    }
    fn try_mul(&self, other: &Array) -> Result<Array> {
        Array::mul(self, other)
    }
    fn try_div(&self, other: &Array) -> Result<Array> {
        Array::div(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operators_match_methods() {
        let a = Array::from_shape_slice(vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let b = Array::from_shape_slice(vec![3], &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!((&a + &b).as_slice(), Array::add(&a, &b).unwrap().as_slice());
        assert_eq!((&a * 2.0).as_slice(), &[2.0, 4.0, 6.0]);
        assert_eq!((-&a).as_slice(), &[-1.0, -2.0, -3.0]);
    }
}
