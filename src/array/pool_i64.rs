//! Thread-local recycle pool for owned `i64` buffers (correctness first; same policy as f64).

use std::cell::RefCell;

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

#[inline]
pub(crate) fn take_zeroed(len: usize) -> Vec<i64> {
    take_filled(len, 0)
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
