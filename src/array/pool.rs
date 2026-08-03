//! Thread-local recycle pool for owned `f64` buffers.
//!
//! Repeated same-size constructs (common in Lua scripts and microbenches) reuse
//! capacity instead of returning pages to the OS on every userdata `__gc`.

use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
use std::cell::RefCell;

const MIN_POOL_CAP: usize = 256;
const MAX_POOL_BUFFERS: usize = 24;

thread_local! {
    static POOL: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
}

/// Buffer of length `len`, every element set to `fill`.
#[inline]
pub(crate) fn take_filled(len: usize, fill: f64) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }
    let mut v = take_capacity(len);
    // Single touch: set_len then fill once (no zero-then-overwrite).
    unsafe {
        v.set_len(len);
    }
    v.fill(fill);
    v
}

/// Zero-filled buffer of length `len`.
///
/// Fresh buffers use `alloc_zeroed` (OS demand-zero pages — same cost model as
/// `numpy.zeros`). Pooled buffers may be dirty and are write-filled.
#[inline]
pub(crate) fn take_zeroed(len: usize) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }
    // Prefer a fresh zeroed allocation over recycling + fill: NumPy zeros does
    // not touch pages; write-filling 4096² is an order-of-magnitude tax.
    let layout = match Layout::array::<f64>(len) {
        Ok(l) => l,
        Err(_) => panic!("zeros length overflow"),
    };
    unsafe {
        let ptr = alloc_zeroed(layout) as *mut f64;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Vec::from_raw_parts(ptr, len, len)
    }
}

/// Length `len`, contents uninitialized — caller must write every element.
#[inline]
pub(crate) fn take_uninit(len: usize) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }
    let mut v = take_capacity(len);
    unsafe {
        v.set_len(len);
    }
    v
}

#[inline]
fn take_capacity(len: usize) -> Vec<f64> {
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let mut best: Option<usize> = None;
        for (i, v) in pool.iter().enumerate() {
            if v.capacity() >= len {
                best = Some(match best {
                    None => i,
                    Some(j) if v.capacity() < pool[j].capacity() => i,
                    Some(j) => j,
                });
            }
        }
        if let Some(i) = best {
            let mut v = pool.swap_remove(i);
            v.clear();
            v
        } else {
            Vec::with_capacity(len)
        }
    })
}

/// Return a unique buffer to the pool (or drop if small / pool full).
#[inline]
pub(crate) fn recycle(mut v: Vec<f64>) {
    if v.capacity() < MIN_POOL_CAP {
        return;
    }
    v.clear();
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() >= MAX_POOL_BUFFERS {
            return;
        }
        pool.push(v);
    });
}
