//! `ArrayI64` userdata helpers (1-based Lua face).

#![allow(non_snake_case)]

use std::ffi::CStr;
use std::os::raw::c_int;
use std::ptr;

use crate::array::ArrayI64;

use super::ffi::*;

pub const ARRAY_I64_MT: &CStr = c"matlua.ArrayI64";

/// Owned i64 array in Lua userdata.
pub struct ArrayI64Ud {
    pub array: ArrayI64,
}

#[inline]
pub unsafe fn check_array_i64(L: *mut lua_State, idx: c_int) -> *mut ArrayI64Ud {
    unsafe { luaL_checkudata(L, idx, ARRAY_I64_MT.as_ptr()) as *mut ArrayI64Ud }
}

#[inline]
pub unsafe fn test_array_i64(L: *mut lua_State, idx: c_int) -> *mut ArrayI64Ud {
    unsafe { luaL_testudata(L, idx, ARRAY_I64_MT.as_ptr()) as *mut ArrayI64Ud }
}

/// Push a new ArrayI64 userdata (moves the array).
///
/// Same GC-debt policy as [`super::ud::push_array`]: step when uniquely owning
/// a non-trivial buffer so dead userdata do not pile up on allocate-heavy faces
/// (`A+B`, transpose, …) — major driver of Lua/Rust ≫ 1 on elementwise.
pub unsafe fn push_array_i64(L: *mut lua_State, array: ArrayI64) {
    unsafe {
        let account = if array.buffer_strong_count() == 1 {
            array.len().saturating_mul(8)
        } else {
            0
        };
        let p = lua_newuserdatauv(L, std::mem::size_of::<ArrayI64Ud>(), 0) as *mut ArrayI64Ud;
        ptr::write(p, ArrayI64Ud { array });
        luaL_setmetatable(L, ARRAY_I64_MT.as_ptr());
        if account >= 64 * 1024 {
            let step_kb = ((account / 1024) as c_int).min(256);
            let _ = lua_gc(L, LUA_GCSTEP, step_kb);
        }
    }
}

/// Nested integer tables → [`ArrayI64`] (same rectangular rules as `array()`).
pub unsafe fn array_i64_from_table(L: *mut lua_State, idx: c_int) -> Result<ArrayI64, String> {
    let idx = if idx < 0 {
        unsafe { lua_gettop(L) + idx + 1 }
    } else {
        idx
    };
    if unsafe { lua_type(L, idx) } != LUA_TTABLE {
        return Err("array_i64() expects a table".into());
    }
    let dims = unsafe { infer_shape(L, idx)? };
    let shape = crate::array::Shape::new(dims).map_err(|e| e.to_string())?;
    let mut data = Vec::with_capacity(shape.numel());
    unsafe { fill_row_major_i64(L, idx, shape.dims(), 0, &mut data)? };
    if data.len() != shape.numel() {
        return Err(format!(
            "ragged table: expected {} numbers, got {}",
            shape.numel(),
            data.len()
        ));
    }
    Ok(ArrayI64::from_parts(shape, data))
}

unsafe fn infer_shape(L: *mut lua_State, idx: c_int) -> Result<Vec<usize>, String> {
    let mut shape = Vec::new();
    let mut cur = idx;
    let mut owned_depth = 0i32;
    loop {
        if shape.len() > 16 {
            if owned_depth > 0 {
                unsafe { lua_pop(L, owned_depth) };
            }
            return Err("array nesting too deep".into());
        }
        let len = unsafe { luaL_len(L, cur) } as usize;
        if len == 0 {
            shape.push(0);
            break;
        }
        unsafe { lua_rawgeti(L, cur, 1) };
        owned_depth += 1;
        let t = unsafe { lua_type(L, -1) };
        if t == LUA_TNUMBER {
            shape.push(len);
            break;
        } else if t == LUA_TTABLE {
            shape.push(len);
            cur = unsafe { lua_gettop(L) };
            continue;
        } else {
            unsafe { lua_pop(L, owned_depth) };
            return Err("array_i64 tables must contain numbers or nested tables".into());
        }
    }
    if owned_depth > 0 {
        unsafe { lua_pop(L, owned_depth) };
    }
    Ok(shape)
}

unsafe fn fill_row_major_i64(
    L: *mut lua_State,
    idx: c_int,
    shape: &[usize],
    axis: usize,
    out: &mut Vec<i64>,
) -> Result<(), String> {
    if axis >= shape.len() {
        return Err("internal: fill past rank".into());
    }
    let expected = shape[axis];
    let len = unsafe { luaL_len(L, idx) } as usize;
    if len != expected {
        return Err(format!(
            "ragged table at axis {axis}: expected length {expected}, got {len}"
        ));
    }
    if axis + 1 == shape.len() {
        for i in 1..=len as i64 {
            unsafe { lua_rawgeti(L, idx, i) };
            if unsafe { lua_type(L, -1) } != LUA_TNUMBER {
                unsafe { lua_pop(L, 1) };
                return Err("leaf entries must be numbers".into());
            }
            let v = unsafe { luaL_checkinteger(L, -1) };
            unsafe { lua_pop(L, 1) };
            out.push(v);
        }
        return Ok(());
    }
    for i in 1..=len as i64 {
        unsafe { lua_rawgeti(L, idx, i) };
        if unsafe { lua_type(L, -1) } != LUA_TTABLE {
            unsafe { lua_pop(L, 1) };
            return Err("expected nested table".into());
        }
        let child = unsafe { lua_gettop(L) };
        unsafe { fill_row_major_i64(L, child, shape, axis + 1, out)? };
        unsafe { lua_pop(L, 1) };
    }
    Ok(())
}
