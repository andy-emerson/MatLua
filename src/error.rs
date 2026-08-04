//! Public error type for MatLua.

use thiserror::Error;

/// Convenience result alias for MatLua operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by MatLua.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Catch-all for messages that do not yet have a dedicated variant.
    #[error("{0}")]
    Message(String),

    /// A shape, rank, or size constraint was violated.
    #[error("shape error: {0}")]
    Shape(String),

    /// An index was out of bounds (reported in the face that raised it).
    #[error("index error: {0}")]
    Index(String),

    /// Dense linear algebra failure (e.g. Cholesky of a non-SPD matrix).
    #[error("linear algebra error: {0}")]
    Linalg(String),

    /// The allocator refused a buffer request. There is **no MatLua-imposed
    /// size ceiling** — the limit is whatever the OS/cgroup grants the host
    /// process. Surfaces on the Lua face as a normal (pcall-able) error
    /// instead of aborting the host.
    #[error("allocation failure: {0}")]
    Alloc(String),
}

impl Error {
    /// Build a [`Error::Message`] from anything displayable.
    pub fn message(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
    }

    /// Build a [`Error::Shape`].
    pub fn shape(msg: impl Into<String>) -> Self {
        Self::Shape(msg.into())
    }

    /// Build a [`Error::Index`].
    pub fn index(msg: impl Into<String>) -> Self {
        Self::Index(msg.into())
    }

    /// Build a [`Error::Linalg`].
    pub fn linalg(msg: impl Into<String>) -> Self {
        Self::Linalg(msg.into())
    }

    /// Build a [`Error::Alloc`] for a refused buffer of `len` elements of
    /// `elem` bytes each.
    pub fn alloc(len: usize, elem: usize) -> Self {
        Self::Alloc(format!(
            "cannot allocate {len} elements ({} bytes requested)",
            len.saturating_mul(elem)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_displays() {
        let err = Error::message("smoke");
        assert_eq!(err.to_string(), "smoke");
    }
}
