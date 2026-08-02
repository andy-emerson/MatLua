//! Shape and row-major layout helpers (0-based, dense, contiguous).

use crate::error::{Error, Result};

/// Logical shape of an n-D array (length of each axis).
///
/// Layout is always **row-major (C-order)**: the last axis varies fastest.
/// `numel` is cached at construction so hot paths avoid re-multiplying dims.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Box<[usize]>,
    numel: usize,
}

impl Shape {
    /// Create a shape from axis lengths.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the product of dimensions overflows `usize`.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        let numel = numel_checked(&dims)?;
        Ok(Self {
            dims: dims.into_boxed_slice(),
            numel,
        })
    }

    /// Rank-0 scalar shape `[]` (one element).
    pub fn scalar() -> Self {
        Self {
            dims: Box::new([]),
            numel: 1,
        }
    }

    /// Rank-1 shape.
    #[inline]
    pub fn from_len(n: usize) -> Self {
        Self {
            dims: Box::new([n]),
            numel: n,
        }
    }

    /// Rank-2 matrix shape `(rows, cols)` with overflow check.
    pub fn matrix(rows: usize, cols: usize) -> Result<Self> {
        let numel = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::Shape("shape numel overflow".into()))?;
        Ok(Self {
            dims: Box::new([rows, cols]),
            numel,
        })
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

    /// Total number of elements (cached).
    #[inline]
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Row-major strides (elements, not bytes).
    pub fn strides(&self) -> Vec<usize> {
        row_major_strides(&self.dims)
    }

    /// Flat offset for a multi-index (0-based per axis).
    ///
    /// Computes the offset without allocating a strides buffer.
    pub fn offset(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.rank() {
            return Err(Error::Shape(format!(
                "expected {} indices for rank {}, got {}",
                self.rank(),
                self.rank(),
                indices.len()
            )));
        }
        // Row-major: last axis varies fastest.
        let mut off = 0usize;
        let mut stride = 1usize;
        for i in (0..self.rank()).rev() {
            let idx = indices[i];
            let dim = self.dims[i];
            if idx >= dim {
                return Err(Error::Index(format!(
                    "index {idx} out of bounds for axis {i} of length {dim}"
                )));
            }
            off = off
                .checked_add(
                    idx.checked_mul(stride)
                        .ok_or_else(|| Error::Shape("offset overflow".into()))?,
                )
                .ok_or_else(|| Error::Shape("offset overflow".into()))?;
            stride = stride
                .checked_mul(dim)
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

    #[test]
    fn matrix_and_from_len() {
        let m = Shape::matrix(3, 4).unwrap();
        assert_eq!(m.dims(), &[3, 4]);
        assert_eq!(m.numel(), 12);
        assert_eq!(Shape::from_len(7).numel(), 7);
    }
}


/// NumPy-style broadcast of two dimension lists (right-aligned).
///
/// Each axis must match or one side is `1`. Ranks may differ (missing axes act as 1).
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = if i + a.len() < rank {
            1
        } else {
            a[i + a.len() - rank]
        };
        let db = if i + b.len() < rank {
            1
        } else {
            b[i + b.len() - rank]
        };
        if da == db {
            out.push(da);
        } else if da == 1 {
            out.push(db);
        } else if db == 1 {
            out.push(da);
        } else {
            return Err(Error::Shape(format!(
                "cannot broadcast shapes {a:?} and {b:?} (axis conflict {da} vs {db})"
            )));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod broadcast_tests {
    use super::*;

    #[test]
    fn broadcast_matrix_and_row() {
        let s = broadcast_shapes(&[3, 4], &[4]).unwrap();
        assert_eq!(s, vec![3, 4]);
        let s = broadcast_shapes(&[3, 4], &[3, 1]).unwrap();
        assert_eq!(s, vec![3, 4]);
        assert!(broadcast_shapes(&[3, 4], &[2, 4]).is_err());
    }
}
