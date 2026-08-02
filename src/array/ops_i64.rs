//! Operator overloads for [`ArrayI64`] (element-wise; panic on shape mismatch).

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::ArrayI64;

fn panic_shape(op: &str, err: crate::error::Error) -> ! {
    panic!("matlua ArrayI64 {op}: {err}");
}

impl Add for &ArrayI64 {
    type Output = ArrayI64;
    fn add(self, rhs: &ArrayI64) -> ArrayI64 {
        ArrayI64::add(self, rhs).unwrap_or_else(|e| panic_shape("add", e))
    }
}
impl Add for ArrayI64 {
    type Output = ArrayI64;
    fn add(self, rhs: ArrayI64) -> ArrayI64 {
        Add::add(&self, &rhs)
    }
}
impl Add<&ArrayI64> for ArrayI64 {
    type Output = ArrayI64;
    fn add(self, rhs: &ArrayI64) -> ArrayI64 {
        Add::add(&self, rhs)
    }
}
impl Add<ArrayI64> for &ArrayI64 {
    type Output = ArrayI64;
    fn add(self, rhs: ArrayI64) -> ArrayI64 {
        Add::add(self, &rhs)
    }
}
impl AddAssign<&ArrayI64> for ArrayI64 {
    fn add_assign(&mut self, rhs: &ArrayI64) {
        self.add_assign_arr(rhs).unwrap_or_else(|e| panic_shape("add_assign", e));
    }
}
impl AddAssign for ArrayI64 {
    fn add_assign(&mut self, rhs: ArrayI64) {
        AddAssign::add_assign(self, &rhs);
    }
}

impl Sub for &ArrayI64 {
    type Output = ArrayI64;
    fn sub(self, rhs: &ArrayI64) -> ArrayI64 {
        ArrayI64::sub(self, rhs).unwrap_or_else(|e| panic_shape("sub", e))
    }
}
impl Sub for ArrayI64 {
    type Output = ArrayI64;
    fn sub(self, rhs: ArrayI64) -> ArrayI64 {
        Sub::sub(&self, &rhs)
    }
}
impl Sub<&ArrayI64> for ArrayI64 {
    type Output = ArrayI64;
    fn sub(self, rhs: &ArrayI64) -> ArrayI64 {
        Sub::sub(&self, rhs)
    }
}
impl Sub<ArrayI64> for &ArrayI64 {
    type Output = ArrayI64;
    fn sub(self, rhs: ArrayI64) -> ArrayI64 {
        Sub::sub(self, &rhs)
    }
}
impl SubAssign<&ArrayI64> for ArrayI64 {
    fn sub_assign(&mut self, rhs: &ArrayI64) {
        self.sub_assign_arr(rhs).unwrap_or_else(|e| panic_shape("sub_assign", e));
    }
}
impl SubAssign for ArrayI64 {
    fn sub_assign(&mut self, rhs: ArrayI64) {
        SubAssign::sub_assign(self, &rhs);
    }
}

impl Mul for &ArrayI64 {
    type Output = ArrayI64;
    fn mul(self, rhs: &ArrayI64) -> ArrayI64 {
        ArrayI64::mul(self, rhs).unwrap_or_else(|e| panic_shape("mul", e))
    }
}
impl Mul for ArrayI64 {
    type Output = ArrayI64;
    fn mul(self, rhs: ArrayI64) -> ArrayI64 {
        Mul::mul(&self, &rhs)
    }
}
impl Mul<&ArrayI64> for ArrayI64 {
    type Output = ArrayI64;
    fn mul(self, rhs: &ArrayI64) -> ArrayI64 {
        Mul::mul(&self, rhs)
    }
}
impl Mul<ArrayI64> for &ArrayI64 {
    type Output = ArrayI64;
    fn mul(self, rhs: ArrayI64) -> ArrayI64 {
        Mul::mul(self, &rhs)
    }
}
impl MulAssign<&ArrayI64> for ArrayI64 {
    fn mul_assign(&mut self, rhs: &ArrayI64) {
        self.mul_assign_arr(rhs).unwrap_or_else(|e| panic_shape("mul_assign", e));
    }
}
impl MulAssign for ArrayI64 {
    fn mul_assign(&mut self, rhs: ArrayI64) {
        MulAssign::mul_assign(self, &rhs);
    }
}

impl Div for &ArrayI64 {
    type Output = ArrayI64;
    fn div(self, rhs: &ArrayI64) -> ArrayI64 {
        ArrayI64::div(self, rhs).unwrap_or_else(|e| panic_shape("div", e))
    }
}
impl Div for ArrayI64 {
    type Output = ArrayI64;
    fn div(self, rhs: ArrayI64) -> ArrayI64 {
        Div::div(&self, &rhs)
    }
}
impl Div<&ArrayI64> for ArrayI64 {
    type Output = ArrayI64;
    fn div(self, rhs: &ArrayI64) -> ArrayI64 {
        Div::div(&self, rhs)
    }
}
impl Div<ArrayI64> for &ArrayI64 {
    type Output = ArrayI64;
    fn div(self, rhs: ArrayI64) -> ArrayI64 {
        Div::div(self, &rhs)
    }
}
impl DivAssign<&ArrayI64> for ArrayI64 {
    fn div_assign(&mut self, rhs: &ArrayI64) {
        self.div_assign_arr(rhs).unwrap_or_else(|e| panic_shape("div_assign", e));
    }
}
impl DivAssign for ArrayI64 {
    fn div_assign(&mut self, rhs: ArrayI64) {
        DivAssign::div_assign(self, &rhs);
    }
}

impl Neg for &ArrayI64 {
    type Output = ArrayI64;
    fn neg(self) -> ArrayI64 {
        ArrayI64::neg(self)
    }
}
impl Neg for ArrayI64 {
    type Output = ArrayI64;
    fn neg(self) -> ArrayI64 {
        ArrayI64::neg(&self)
    }
}
