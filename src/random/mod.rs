//! Seedable PRNG for array constructors (M7.b).
//!
//! xoshiro256** (public domain / CC0-style algorithm by Blackman & Vigna) plus
//! Box–Muller normals. No external `rand` dependency so embed hosts stay lean.
//! Not cryptographic.

use crate::array::{Array, ArrayI64, Shape};
use crate::error::{Error, Result};
use std::sync::{Mutex, OnceLock};

fn global() -> &'static Mutex<Xoshiro256StarStar> {
    static GLOBAL: OnceLock<Mutex<Xoshiro256StarStar>> = OnceLock::new();
    GLOBAL.get_or_init(|| Mutex::new(Xoshiro256StarStar::from_seed(0x4d41_544c_5541_0001)))
}

/// Replace the global RNG seed (deterministic stream from `seed`).
pub fn seed(seed: u64) {
    *global().lock().unwrap() = Xoshiro256StarStar::from_seed(seed);
}

fn with_rng<R>(f: impl FnOnce(&mut Xoshiro256StarStar) -> R) -> R {
    let mut g = global().lock().unwrap();
    f(&mut g)
}

/// Uniform `f64` in \[0, 1).
pub fn random(shape: impl Into<Vec<usize>>) -> Result<Array> {
    let shape = Shape::new(shape)?;
    let n = shape.numel();
    let mut data = crate::array::pool_take_uninit(n);
    with_rng(|rng| {
        for x in &mut data {
            *x = rng.f64();
        }
    });
    Ok(Array::from_parts(shape, data))
}

/// Uniform `f64` in \[low, high).
pub fn uniform(shape: impl Into<Vec<usize>>, low: f64, high: f64) -> Result<Array> {
    if !(low < high) || !low.is_finite() || !high.is_finite() {
        return Err(Error::shape(format!(
            "uniform requires finite low < high (got {low}, {high})"
        )));
    }
    let span = high - low;
    let mut a = random(shape)?;
    for x in a.as_mut_slice() {
        *x = x.mul_add(span, low);
    }
    Ok(a)
}

/// Standard normal (mean 0, std 1), Box–Muller.
pub fn randn(shape: impl Into<Vec<usize>>) -> Result<Array> {
    let shape = Shape::new(shape)?;
    let n = shape.numel();
    let mut data = crate::array::pool_take_uninit(n);
    with_rng(|rng| {
        let mut i = 0;
        while i < n {
            let (z0, z1) = rng.normal_pair();
            data[i] = z0;
            i += 1;
            if i < n {
                data[i] = z1;
                i += 1;
            }
        }
    });
    Ok(Array::from_parts(shape, data))
}

/// Normal with mean `mu` and std `sigma` (`sigma > 0`).
pub fn normal(shape: impl Into<Vec<usize>>, mu: f64, sigma: f64) -> Result<Array> {
    if !(sigma > 0.0) || !sigma.is_finite() || !mu.is_finite() {
        return Err(Error::shape(format!(
            "normal requires finite mu and sigma > 0 (got mu={mu}, sigma={sigma})"
        )));
    }
    let mut a = randn(shape)?;
    for x in a.as_mut_slice() {
        *x = x.mul_add(sigma, mu);
    }
    Ok(a)
}

/// Integer uniform in `[low, high)` as `ArrayI64` (high exclusive).
pub fn integers(shape: impl Into<Vec<usize>>, low: i64, high: i64) -> Result<ArrayI64> {
    if low >= high {
        return Err(Error::shape(format!(
            "integers requires low < high (got {low}, {high})"
        )));
    }
    let shape = Shape::new(shape)?;
    let n = shape.numel();
    let span = (high as i128) - (low as i128);
    let mut data = crate::array::pool_i64::take_uninit(n);
    with_rng(|rng| {
        for x in &mut data {
            let u = rng.next_u64() as u128;
            let off = (u % span as u128) as i64;
            *x = low.wrapping_add(off);
        }
    });
    Ok(ArrayI64::from_parts(shape, data))
}

/// Sample `k` values with replacement from a rank-1 `f64` array.
pub fn choice(a: &Array, k: usize) -> Result<Array> {
    if a.rank() != 1 {
        return Err(Error::shape("choice expects rank-1 population"));
    }
    if a.is_empty() {
        return Err(Error::shape("choice from empty array"));
    }
    let n = a.len();
    let src = a.as_slice();
    let mut data = crate::array::pool_take_uninit(k);
    with_rng(|rng| {
        for x in &mut data {
            let j = (rng.next_u64() as usize) % n;
            *x = src[j];
        }
    });
    Ok(Array::from_parts(Shape::from_len(k), data))
}

/// Sample `k` values with replacement from a rank-1 `i64` array.
pub fn choice_i64(a: &ArrayI64, k: usize) -> Result<ArrayI64> {
    if a.rank() != 1 {
        return Err(Error::shape("choice expects rank-1 population"));
    }
    if a.is_empty() {
        return Err(Error::shape("choice from empty array"));
    }
    let n = a.len();
    let src = a.as_slice();
    let mut data = crate::array::pool_i64::take_uninit(k);
    with_rng(|rng| {
        for x in &mut data {
            let j = (rng.next_u64() as usize) % n;
            *x = src[j];
        }
    });
    Ok(ArrayI64::from_parts(Shape::from_len(k), data))
}

// ----- xoshiro256** -----

struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    fn from_seed(seed: u64) -> Self {
        // SplitMix64 expand single seed to 4 words (none zero).
        let mut sm = seed;
        let mut s = [0u64; 4];
        for i in 0..4 {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            s[i] = z ^ (z >> 31);
        }
        if s.iter().all(|&x| x == 0) {
            s[0] = 1;
        }
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in \[0, 1).
    #[inline]
    fn f64(&mut self) -> f64 {
        // upper 53 bits → [0, 1)
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn normal_pair(&mut self) -> (f64, f64) {
        // Box–Muller
        let mut u1 = self.f64();
        while u1 == 0.0 {
            u1 = self.f64();
        }
        let u2 = self.f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn seed_reproducible_uniform() {
        let _g = TEST_LOCK.lock().unwrap();
        seed(42);
        let a = random(vec![8]).unwrap();
        seed(42);
        let b = random(vec![8]).unwrap();
        assert_eq!(a.as_slice(), b.as_slice());
        seed(1);
        let c = random(vec![8]).unwrap();
        assert_ne!(a.as_slice(), c.as_slice());
    }

    #[test]
    fn integers_range() {
        let _g = TEST_LOCK.lock().unwrap();
        seed(7);
        let a = integers(vec![100], 10, 20).unwrap();
        assert!(a.as_slice().iter().all(|&x| (10..20).contains(&x)));
    }
}
