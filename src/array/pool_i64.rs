//! Thread-local recycle pool for owned `i64` buffers (correctness first; same policy as f64).

use crate::error::{Error, Result};
use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
use std::cell::RefCell;

// Empirical (M7.c bench host, 2026-07; unverified elsewhere) — see DESIGN §3.26.
// Not derived from any allocator or workload analysis; revisit under M9.
const MIN_POOL_CAP: usize = 256;
const MAX_POOL_BUFFERS: usize = 24;

thread_local! {
    static POOL: RefCell<Vec<Vec<i64>>> = const { RefCell::new(Vec::new()) };
}

// Infallible take_filled / take_zeroed removed with the §3.27 fallible-
// allocation ruling; see the f64 pool for the rationale note.

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
    match try_take_capacity(len) {
        Ok(v) => v,
        // Same-size derived scratch keeps the historical abort-on-OOM until
        // the M10 boundary work; script-originated sizes go through try_*.
        Err(_) => match Layout::array::<i64>(len) {
            Ok(l) => handle_alloc_error(l),
            Err(_) => panic!("allocation length overflow"),
        },
    }
}

#[inline]
fn try_take_capacity(len: usize) -> Result<Vec<i64>> {
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
                .map_err(|_| Error::alloc(len, size_of::<i64>()))?;
            Ok(v)
        }
    })
}

// --- Fallible twins (no MatLua size ceiling; failure → Error::Alloc) ---------

/// Fallible [`take_filled`].
#[inline]
pub(crate) fn try_take_filled(len: usize, fill: i64) -> Result<Vec<i64>> {
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
pub(crate) fn try_take_zeroed(len: usize) -> Result<Vec<i64>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let layout =
        Layout::array::<i64>(len).map_err(|_| Error::alloc(len, size_of::<i64>()))?;
    unsafe {
        let ptr = alloc_zeroed(layout) as *mut i64;
        if ptr.is_null() {
            return Err(Error::alloc(len, size_of::<i64>()));
        }
        Ok(Vec::from_raw_parts(ptr, len, len))
    }
}

/// Fallible [`take_uninit`] — caller must write every element.
#[inline]
pub(crate) fn try_take_uninit(len: usize) -> Result<Vec<i64>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut v = try_take_capacity(len)?;
    unsafe {
        v.set_len(len);
    }
    Ok(v)
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
