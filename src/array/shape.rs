//! Shape and row-major layout helpers (0-based, dense, contiguous).

use crate::error::{Error, Result};

/// Logical shape of an n-D array (length of each axis).
///
/// Layout is always **row-major (C-order)**: the last axis varies fastest.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Box<[usize]>,
}

impl Shape {
    /// Create a shape from axis lengths.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the product of dimensions overflows `usize`.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        let _ = numel_checked(&dims)?;
        Ok(Self {
            dims: dims.into_boxed_slice(),
        })
    }

    /// Rank-0 scalar shape `[]` (one element).
    pub fn scalar() -> Self {
        Self {
            dims: Box::new([]),
        }
    }

    /// Rank-1 shape.
    pub fn from_len(n: usize) -> Self {
        Self {
            dims: Box::new([n]),
        }
    }

    /// Axis lengths.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of axes.
    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        // Invariant: constructed only via checked paths.
        numel_checked(&self.dims).expect("shape numel overflowed after construction")
    }

    /// Row-major strides (elements, not bytes).
    pub fn strides(&self) -> Vec<usize> {
        row_major_strides(&self.dims)
    }

    /// Flat offset for a multi-index (0-based per axis).
    pub fn offset(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.rank() {
            return Err(Error::Shape(format!(
                "expected {} indices for rank {}, got {}",
                self.rank(),
                self.rank(),
                indices.len()
            )));
        }
        let strides = self.strides();
        let mut off = 0usize;
        for (i, (&idx, &stride)) in indices.iter().zip(strides.iter()).enumerate() {
            let dim = self.dims[i];
            if idx >= dim {
                return Err(Error::Index(format!(
                    "index {idx} out of bounds for axis {i} of length {dim}"
                )));
            }
            off = off
                .checked_add(idx.checked_mul(stride).ok_or_else(|| {
                    Error::Shape("offset overflow".into())
                })?)
                .ok_or_else(|| Error::Shape("offset overflow".into()))?;
        }
        Ok(off)
    }

    /// True if `other` has the same axis lengths.
    #[inline]
    pub fn same_as(&self, other: &Shape) -> bool {
        self.dims.as_ref() == other.dims.as_ref()
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        if self.rank() == 1 {
            write!(f, ",")?;
        }
        write!(f, ")")
    }
}

impl From<Shape> for Vec<usize> {
    fn from(s: Shape) -> Self {
        s.dims.into_vec()
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.dims
    }
}

/// Product of dimensions with overflow check.
pub fn numel_checked(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| Error::Shape("shape numel overflow".into()))
    })
}

/// Row-major strides for `dims`.
pub fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; dims.len()];
    let mut acc = 1usize;
    for i in (0..dims.len()).rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(dims[i]);
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_row_major() {
        let s = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(s.strides(), vec![12, 4, 1]);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.offset(&[1, 2, 3]).unwrap(), 1 * 12 + 2 * 4 + 3);
    }

    #[test]
    fn scalar_shape() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.offset(&[]).unwrap(), 0);
    }
}
