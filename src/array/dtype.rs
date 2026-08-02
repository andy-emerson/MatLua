//! Element types for dense MatLua arrays.

/// Physical element type of an owned array.
///
/// v0.1 quality bar remains [`DType::F64`] for dense LA. [`DType::I64`] (M7)
/// covers ordering keys and exact integer columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// IEEE-754 binary64 (default product bar).
    F64,
    /// Signed 64-bit integer.
    I64,
}

impl DType {
    /// Short name used on the Lua face and in errors (`"f64"`, `"i64"`).
    pub fn name(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::I64 => "i64",
        }
    }

    /// Parse a dtype name (case-insensitive). Accepts `f64`/`float64` and `i64`/`int64`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f64" | "float64" | "float" | "double" => Some(Self::F64),
            "i64" | "int64" | "int" | "integer" => Some(Self::I64),
            _ => None,
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
