//! Thread-local recycle pool for owned `i64` buffers (correctness first; same policy as f64).

use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
use std::cell::RefCell;

// Empirical (M7.c bench host, 2026-07; unverified elsewhere) — see DESIGN §3.26.
// Not derived from any allocator or workload analysis; revisit under M9.
const MIN_POOL_CAP: usize = 256;
const MAX_POOL_BUFFERS: usize = 24;

thread_local! {
    static POOL: RefCell<Vec<Vec<i64>>> = const { RefCell::new(Vec::new()) };
}

#[inline]
pub(crate) fn take_filled(len: usize, fill: i64) -> Vec<i64> {
    if len == 0 {
        return Vec::new();
    }
    let mut v = take_capacity(len);
    unsafe {
        v.set_len(len);
    }
    v.fill(fill);
    v
}

/// Zero-filled via OS demand-zero pages (see f64 [`crate::array::pool::take_zeroed`]).
#[inline]
pub(crate) fn take_zeroed(len: usize) -> Vec<i64> {
    if len == 0 {
        return Vec::new();
    }
    let layout = match Layout::array::<i64>(len) {
        Ok(l) => l,
        Err(_) => panic!("zeros length overflow"),
    };
    unsafe {
        let ptr = alloc_zeroed(layout) as *mut i64;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Vec::from_raw_parts(ptr, len, len)
    }
}

/// Caller must write every element before reading.
#[inline]
pub(crate) fn take_uninit(len: usize) -> Vec<i64> {
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
fn take_capacity(len: usize) -> Vec<i64> {
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

#[inline]
pub(crate) fn recycle(mut v: Vec<i64>) {
    if v.capacity() < MIN_POOL_CAP {
        return;
    }
    v.clear();
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < MAX_POOL_BUFFERS {
            pool.push(v);
        }
    });
}
