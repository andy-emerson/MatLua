//! Thread-local recycle pool for owned `f64` buffers.
//!
//! Repeated same-size constructs (common in Lua scripts and microbenches) reuse
//! capacity instead of returning pages to the OS on every userdata `__gc`.

use crate::error::{Error, Result};
use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
use std::cell::RefCell;

// Empirical (M7.c bench host, 2026-07; unverified elsewhere) — see DESIGN §3.26.
// Not derived from any allocator or workload analysis; revisit under M9.
const MIN_POOL_CAP: usize = 256;
const MAX_POOL_BUFFERS: usize = 24;

thread_local! {
    static POOL: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
}

// The infallible take_filled / take_zeroed were removed once every
// Result-returning construction path moved to the try_ twins below (§3.27);
// only take_uninit keeps an infallible form, for plain-value contexts
// (Clone, casts, operator impls) whose OOM window is documented there.

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
    match try_take_capacity(len) {
        Ok(v) => v,
        // Same-size derived scratch keeps the historical abort-on-OOM until
        // the M10 boundary work; script-originated sizes go through try_*.
        Err(_) => match Layout::array::<f64>(len) {
            Ok(l) => handle_alloc_error(l),
            Err(_) => panic!("allocation length overflow"),
        },
    }
}

#[inline]
fn try_take_capacity(len: usize) -> Result<Vec<f64>> {
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
            Ok(v)
        } else {
            let mut v = Vec::new();
            v.try_reserve_exact(len)
                .map_err(|_| Error::alloc(len, size_of::<f64>()))?;
            Ok(v)
        }
    })
}

// --- Fallible twins (no MatLua size ceiling; failure → Error::Alloc) ---------
//
// Result-returning construction paths use these so a refused allocation
// surfaces as a catchable error instead of aborting the embedding host
// (DESIGN: allocation ruling 2026-08-04).

/// Fallible [`take_filled`].
#[inline]
pub(crate) fn try_take_filled(len: usize, fill: f64) -> Result<Vec<f64>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut v = try_take_capacity(len)?;
    unsafe {
        v.set_len(len);
    }
    v.fill(fill);
    Ok(v)
}

/// Fallible [`take_zeroed`] (OS demand-zero pages, null → error).
#[inline]
pub(crate) fn try_take_zeroed(len: usize) -> Result<Vec<f64>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let layout =
        Layout::array::<f64>(len).map_err(|_| Error::alloc(len, size_of::<f64>()))?;
    unsafe {
        let ptr = alloc_zeroed(layout) as *mut f64;
        if ptr.is_null() {
            return Err(Error::alloc(len, size_of::<f64>()));
        }
        Ok(Vec::from_raw_parts(ptr, len, len))
    }
}

/// Fallible [`take_uninit`] — caller must write every element.
#[inline]
pub(crate) fn try_take_uninit(len: usize) -> Result<Vec<f64>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut v = try_take_capacity(len)?;
    unsafe {
        v.set_len(len);
    }
    Ok(v)
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
