//! Array userdata helpers (1-based Lua face over 0-based Rust arrays).

#![allow(non_snake_case)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr;

use crate::array::Array;

use super::ffi::*;
// lua_gc / LUA_GC* from ffi

pub const ARRAY_MT: &CStr = c"matlua.Array";

/// Owned array stored in Lua userdata.
pub struct ArrayUd {
    pub array: Array,
}

#[inline]
pub unsafe fn check_array(L: *mut lua_State, idx: c_int) -> *mut ArrayUd {
    unsafe { luaL_checkudata(L, idx, ARRAY_MT.as_ptr()) as *mut ArrayUd }
}

#[inline]
pub unsafe fn test_array(L: *mut lua_State, idx: c_int) -> *mut ArrayUd {
    unsafe { luaL_testudata(L, idx, ARRAY_MT.as_ptr()) as *mut ArrayUd }
}

/// Push a new Array userdata (moves the array).
pub unsafe fn push_array(L: *mut lua_State, array: Array) {
    unsafe {
        // External f64 buffer is invisible to Lua's GC debt. Step GC by the
        // payload size so large arrays are collected promptly (otherwise
        // memory balloons and major GC / allocator churn destroy face times).
        let nbytes = array.len().saturating_mul(8);
        let p = lua_newuserdatauv(L, std::mem::size_of::<ArrayUd>(), 0) as *mut ArrayUd;
        ptr::write(p, ArrayUd { array });
        luaL_setmetatable(L, ARRAY_MT.as_ptr());
        if nbytes >= 4096 {
            let step_kb = (nbytes / 1024) as c_int;
            let _ = lua_gc(L, LUA_GCSTEP, step_kb);
        }
    }
}

/// Raise a Lua error (longjmp). Marked as returning `c_int` for `lua_try!`.
///
/// Uses `lua_pushlstring` so we do not allocate a `CString` on the error path.
pub unsafe fn lua_error_msg(L: *mut lua_State, msg: &str) -> c_int {
    unsafe {
        // Lua copies the bytes; embedded NULs are allowed via pushlstring.
        lua_pushlstring(L, msg.as_ptr() as *const c_char, msg.len());
        lua_error(L)
    }
}

/// Max rank for stack-backed multi-index decoding (no heap on get/set).
pub const MAX_INDEX_RANK: usize = 8;

/// Convert 1-based Lua multi-index (stack args `from..=to` inclusive) into `buf`.
///
/// Returns the filled prefix of `buf` (length `rank`). Uses no heap allocation
/// when `rank <= MAX_INDEX_RANK` and `buf` is stack-provided.
pub unsafe fn indices_1_based(
    L: *mut lua_State,
    from: c_int,
    to: c_int,
    rank: usize,
    buf: &mut [usize; MAX_INDEX_RANK],
) -> Result<usize, String> {
    let n = (to - from + 1) as usize;
    if n != rank {
        return Err(format!("expected {rank} indices, got {n}"));
    }
    if rank > MAX_INDEX_RANK {
        return Err(format!(
            "rank {rank} exceeds Lua face index limit {MAX_INDEX_RANK}"
        ));
    }
    for i in 0..rank {
        let v = unsafe { luaL_checkinteger(L, from + i as c_int) };
        if v < 1 {
            return Err(format!("index must be >= 1, got {v}"));
        }
        buf[i] = (v as usize) - 1;
    }
    Ok(rank)
}

/// Read a shape from Lua: either a single integer-list table, or consecutive numbers.
pub unsafe fn shape_from_args(L: *mut lua_State, from: c_int) -> Result<Vec<usize>, String> {
    let top = unsafe { lua_gettop(L) };
    if from > top {
        return Err("missing shape arguments".into());
    }
    if unsafe { lua_type(L, from) } == LUA_TTABLE && from == top {
        return unsafe { shape_from_table(L, from) };
    }
    let mut dims = Vec::new();
    for i in from..=top {
        if !unsafe { lua_isnumber(L, i) } {
            return Err("shape dimensions must be numbers (or one table)".into());
        }
        let n = unsafe { luaL_checkinteger(L, i) };
        if n < 0 {
            return Err("shape dimensions must be non-negative".into());
        }
        dims.push(n as usize);
    }
    if dims.is_empty() {
        return Err("empty shape".into());
    }
    Ok(dims)
}

pub unsafe fn shape_from_table(L: *mut lua_State, idx: c_int) -> Result<Vec<usize>, String> {
    let len = unsafe { luaL_len(L, idx) };
    if len < 0 {
        return Err("invalid shape table".into());
    }
    let mut dims = Vec::with_capacity(len as usize);
    for i in 1..=len {
        unsafe { lua_rawgeti(L, idx, i) };
        if !unsafe { lua_isnumber(L, -1) } {
            unsafe { lua_pop(L, 1) };
            return Err("shape table entries must be numbers".into());
        }
        let n = unsafe { luaL_checkinteger(L, -1) };
        unsafe { lua_pop(L, 1) };
        if n < 0 {
            return Err("shape dimensions must be non-negative".into());
        }
        dims.push(n as usize);
    }
    Ok(dims)
}

/// Build an array from a rectangular nested Lua number table (copies).
pub unsafe fn array_from_table(L: *mut lua_State, idx: c_int) -> Result<Array, String> {
    let idx = if idx < 0 {
        unsafe { lua_gettop(L) + idx + 1 }
    } else {
        idx
    };
    if unsafe { lua_type(L, idx) } != LUA_TTABLE {
        return Err("array() expects a table".into());
    }

    let dims = unsafe { infer_shape(L, idx)? };
    let shape = crate::array::Shape::new(dims).map_err(|e| e.to_string())?;
    let mut data = Vec::with_capacity(shape.numel());
    unsafe { fill_row_major(L, idx, shape.dims(), 0, &mut data)? };
    if data.len() != shape.numel() {
        return Err(format!(
            "ragged table: expected {} numbers, got {}",
            shape.numel(),
            data.len()
        ));
    }
    Ok(Array::from_parts(shape, data))
}

/// Infer rectangular shape by following the first element at each nesting level.
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
            return Err("array tables must contain numbers or nested tables".into());
        }
    }
    if owned_depth > 0 {
        unsafe { lua_pop(L, owned_depth) };
    }
    Ok(shape)
}

unsafe fn fill_row_major(
    L: *mut lua_State,
    idx: c_int,
    shape: &[usize],
    axis: usize,
    out: &mut Vec<f64>,
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
            if !unsafe { lua_isnumber(L, -1) } {
                unsafe { lua_pop(L, 1) };
                return Err("leaf entries must be numbers".into());
            }
            let v = unsafe { luaL_checknumber(L, -1) };
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
        unsafe { fill_row_major(L, child, shape, axis + 1, out)? };
        unsafe { lua_pop(L, 1) };
    }
    Ok(())
}

pub unsafe fn push_shape_table(L: *mut lua_State, dims: &[usize]) {
    unsafe {
        lua_createtable(L, dims.len() as c_int, 0);
        for (i, &d) in dims.iter().enumerate() {
            lua_pushinteger(L, d as lua_Integer);
            lua_rawseti(L, -2, (i + 1) as lua_Integer);
        }
    }
}
