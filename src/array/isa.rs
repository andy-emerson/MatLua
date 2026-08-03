//! Cached runtime ISA detection for kernel dispatch (derived — CPUID at
//! first use, not a host-tuned table; DESIGN §3.26). The `#[target_feature]`
//! twin-kernel pattern and its measured effects live with each kernel.

/// True when the CPU supports the avx512f+dq+bw+vl set that our
/// `#[target_feature]` kernel twins enable. Always false off x86_64.
#[inline]
pub(crate) fn avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512dq")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
