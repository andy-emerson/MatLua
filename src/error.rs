//! Public error type for MatLua.

use thiserror::Error;

/// Convenience result alias for MatLua operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by MatLua.
///
/// Variants will grow with the API. M0 only reserves the surface.
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
}

impl Error {
    /// Build a [`Error::Message`] from anything displayable.
    pub fn message(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
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
