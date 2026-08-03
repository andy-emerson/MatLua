//! Host-embedder entry points: borrow engine buffers into Lua without a second
//! interpreter, and without MatLua freeing host memory.
//!
//! # Safety contract
//!
//! - The host keeps the buffer alive for as long as any Lua userdata from
//!   [`push_view_f64`] / [`push_view_i64`] (or derived copies) may still be
//!   reachable from the state.
//! - Views are **read-only** on the Lua face. Mutation requires
//!   `to_array()` / `to_array_i64()` (owned copy) first.
//! - These APIs do not longjmp; they return [`Result`]. Callers that then
//!   interact with Lua must still obey the longjmp/`Drop` rules (M10).

#![allow(non_snake_case)]

use std::os::raw::c_int;
use std::ptr;

use crate::array::{Array, ArrayI64, Shape};
use crate::error::{Error, Result};

use super::ffi::*;
use super::ud::push_array;
use super::ud_i64::push_array_i64;

/// Metatable name for read-only host `f64` views.
pub const VIEW_F64_MT: &std::ffi::CStr = c"matlua.ArrayView";
/// Metatable name for read-only host `i64` views.
pub const VIEW_I64_MT: &std::ffi::CStr = c"matlua.ArrayViewI64";

/// Read-only view over a host `f64` buffer (does **not** free on Drop).
pub struct HostViewF64 {
    /// Logical shape (row-major).
    pub shape: Shape,
    /// Pointer to host buffer (not freed by MatLua).
    pub data: *const f64,
    /// Element count (`shape.numel()`).
    pub len: usize,
}

/// Read-only view over a host `i64` buffer (does **not** free on Drop).
pub struct HostViewI64 {
    /// Logical shape (row-major).
    pub shape: Shape,
    /// Pointer to host buffer (not freed by MatLua).
    pub data: *const i64,
    /// Element count (`shape.numel()`).
    pub len: usize,
}

impl HostViewF64 {
    /// Borrow as a slice (host must guarantee validity).
    ///
    /// # Safety
    /// `data` must be valid for `len` reads for the lifetime of the returned slice.
    pub unsafe fn as_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    /// Copy into an owned [`Array`].
    pub fn to_owned(&self) -> Array {
        let mut data = crate::array::pool_take_uninit(self.len);
        unsafe {
            data.copy_from_slice(self.as_slice());
        }
        Array::from_parts(self.shape.clone(), data)
    }
}

impl HostViewI64 {
    /// # Safety
    /// `data` must be valid for `len` reads for the lifetime of the returned slice.
    pub unsafe fn as_slice(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    /// Copy into an owned [`ArrayI64`].
    pub fn to_owned(&self) -> ArrayI64 {
        let mut data = crate::array::pool_i64::take_uninit(self.len);
        unsafe {
            data.copy_from_slice(self.as_slice());
        }
        ArrayI64::from_parts(self.shape.clone(), data)
    }
}

/// Push a **copy** of host `f64` data as a normal owned `ml.array` userdata.
pub unsafe fn push_array_copy_f64(
    L: *mut lua_State,
    dims: impl Into<Vec<usize>>,
    data: &[f64],
) -> Result<()> {
    let shape = Shape::new(dims)?;
    if data.len() != shape.numel() {
        return Err(Error::shape(format!(
            "host buffer len {} != shape numel {}",
            data.len(),
            shape.numel()
        )));
    }
    let mut owned = crate::array::pool_take_uninit(data.len());
    owned.copy_from_slice(data);
    unsafe { push_array(L, Array::from_parts(shape, owned)) };
    Ok(())
}

/// Push a **copy** of host `i64` data as owned `ml.array_i64`.
pub unsafe fn push_array_copy_i64(
    L: *mut lua_State,
    dims: impl Into<Vec<usize>>,
    data: &[i64],
) -> Result<()> {
    let shape = Shape::new(dims)?;
    if data.len() != shape.numel() {
        return Err(Error::shape(format!(
            "host buffer len {} != shape numel {}",
            data.len(),
            shape.numel()
        )));
    }
    let mut owned = crate::array::pool_i64::take_uninit(data.len());
    owned.copy_from_slice(data);
    unsafe { push_array_i64(L, ArrayI64::from_parts(shape, owned)) };
    Ok(())
}

/// Push a read-only view over host memory (zero-copy). Does not free `data`.
///
/// # Safety
/// - `L` valid Lua state.
/// - `data` valid for `len` `f64`s while any resulting userdata is reachable.
/// - `len` must equal `numel(dims)`.
pub unsafe fn push_view_f64(
    L: *mut lua_State,
    dims: impl Into<Vec<usize>>,
    data: *const f64,
    len: usize,
) -> Result<()> {
    if data.is_null() && len != 0 {
        return Err(Error::shape("null host pointer with non-zero len"));
    }
    let shape = Shape::new(dims)?;
    if len != shape.numel() {
        return Err(Error::shape(format!(
            "host view len {len} != shape numel {}",
            shape.numel()
        )));
    }
    ensure_view_f64_mt(L);
    unsafe {
        let p = lua_newuserdatauv(L, std::mem::size_of::<HostViewF64>(), 0) as *mut HostViewF64;
        ptr::write(
            p,
            HostViewF64 {
                shape,
                data,
                len,
            },
        );
        luaL_setmetatable(L, VIEW_F64_MT.as_ptr());
    }
    Ok(())
}

/// Push a read-only view over host `i64` memory (zero-copy).
///
/// # Safety
/// Same as [`push_view_f64`], for `i64`.
pub unsafe fn push_view_i64(
    L: *mut lua_State,
    dims: impl Into<Vec<usize>>,
    data: *const i64,
    len: usize,
) -> Result<()> {
    if data.is_null() && len != 0 {
        return Err(Error::shape("null host pointer with non-zero len"));
    }
    let shape = Shape::new(dims)?;
    if len != shape.numel() {
        return Err(Error::shape(format!(
            "host view len {len} != shape numel {}",
            shape.numel()
        )));
    }
    ensure_view_i64_mt(L);
    unsafe {
        let p = lua_newuserdatauv(L, std::mem::size_of::<HostViewI64>(), 0) as *mut HostViewI64;
        ptr::write(
            p,
            HostViewI64 {
                shape,
                data,
                len,
            },
        );
        luaL_setmetatable(L, VIEW_I64_MT.as_ptr());
    }
    Ok(())
}

unsafe fn ensure_view_f64_mt(L: *mut lua_State) {
    unsafe {
        if luaL_newmetatable(L, VIEW_F64_MT.as_ptr()) == 0 {
            lua_pop(L, 1);
            return;
        }
        // __index methods table
        lua_newtable(L);
        let methods: [(&std::ffi::CStr, unsafe extern "C" fn(*mut lua_State) -> c_int); 5] = [
            (c"shape", v_f64_shape),
            (c"rank", v_f64_rank),
            (c"get", v_f64_get),
            (c"to_array", v_f64_to_array),
            (c"dtype", v_f64_dtype),
        ];
        for (name, f) in methods {
            lua_pushcfunction(L, Some(f));
            lua_setfield(L, -2, name.as_ptr());
        }
        lua_setfield(L, -2, c"__index".as_ptr());
        lua_pushcfunction(L, Some(v_f64_len));
        lua_setfield(L, -2, c"__len".as_ptr());
        // __gc drops the MatLua-owned Shape inside the view struct. The host's
        // data buffer is never freed here — the host owns that memory.
        lua_pushcfunction(L, Some(v_f64_gc));
        lua_setfield(L, -2, c"__gc".as_ptr());
        lua_pop(L, 1);
    }
}

unsafe extern "C" fn v_f64_gc(L: *mut lua_State) -> c_int {
    let ud = unsafe { check_view_f64(L, 1) };
    // SAFETY: __gc runs exactly once per userdata; the struct is not used after.
    unsafe { std::ptr::drop_in_place(ud) };
    0
}

unsafe fn ensure_view_i64_mt(L: *mut lua_State) {
    unsafe {
        if luaL_newmetatable(L, VIEW_I64_MT.as_ptr()) == 0 {
            lua_pop(L, 1);
            return;
        }
        lua_newtable(L);
        let methods: [(&std::ffi::CStr, unsafe extern "C" fn(*mut lua_State) -> c_int); 5] = [
            (c"shape", v_i64_shape),
            (c"rank", v_i64_rank),
            (c"get", v_i64_get),
            (c"to_array", v_i64_to_array),
            (c"dtype", v_i64_dtype),
        ];
        for (name, f) in methods {
            lua_pushcfunction(L, Some(f));
            lua_setfield(L, -2, name.as_ptr());
        }
        lua_setfield(L, -2, c"__index".as_ptr());
        lua_pushcfunction(L, Some(v_i64_len));
        lua_setfield(L, -2, c"__len".as_ptr());
        // __gc drops the MatLua-owned Shape; host data is never freed here.
        lua_pushcfunction(L, Some(v_i64_gc));
        lua_setfield(L, -2, c"__gc".as_ptr());
        lua_pop(L, 1);
    }
}

unsafe extern "C" fn v_i64_gc(L: *mut lua_State) -> c_int {
    let ud = unsafe { check_view_i64(L, 1) };
    // SAFETY: __gc runs exactly once per userdata; the struct is not used after.
    unsafe { std::ptr::drop_in_place(ud) };
    0
}

#[inline]
unsafe fn check_view_f64(L: *mut lua_State, idx: c_int) -> *mut HostViewF64 {
    unsafe { luaL_checkudata(L, idx, VIEW_F64_MT.as_ptr()) as *mut HostViewF64 }
}

#[inline]
unsafe fn check_view_i64(L: *mut lua_State, idx: c_int) -> *mut HostViewI64 {
    unsafe { luaL_checkudata(L, idx, VIEW_I64_MT.as_ptr()) as *mut HostViewI64 }
}

unsafe extern "C" fn v_f64_shape(L: *mut lua_State) -> c_int {
    let v = unsafe { &*check_view_f64(L, 1) };
    let dims = v.shape.dims();
    unsafe {
        lua_createtable(L, dims.len() as c_int, 0);
        for (i, &d) in dims.iter().enumerate() {
            lua_pushinteger(L, d as lua_Integer);
            lua_rawseti(L, -2, (i + 1) as lua_Integer);
        }
    }
    1
}

unsafe extern "C" fn v_f64_rank(L: *mut lua_State) -> c_int {
    let v = unsafe { &*check_view_f64(L, 1) };
    unsafe { lua_pushinteger(L, v.shape.rank() as lua_Integer) };
    1
}

unsafe extern "C" fn v_f64_len(L: *mut lua_State) -> c_int {
    let v = unsafe { &*check_view_f64(L, 1) };
    unsafe { lua_pushinteger(L, v.len as lua_Integer) };
    1
}

unsafe extern "C" fn v_f64_dtype(L: *mut lua_State) -> c_int {
    unsafe {
        lua_pushstring(L, c"f64".as_ptr());
    }
    1
}

unsafe extern "C" fn v_f64_get(L: *mut lua_State) -> c_int {
    let v = unsafe { &*check_view_f64(L, 1) };
    let top = unsafe { lua_gettop(L) };
    let rank = v.shape.rank();
    if top - 1 != rank as c_int {
        return super::ud::lua_error_msg(L, "get: wrong number of indices");
    }
    let mut idx = Vec::with_capacity(rank);
    for i in 0..rank {
        let one = unsafe { luaL_checkinteger(L, 2 + i as c_int) };
        if one < 1 {
            return super::ud::lua_error_msg(L, "index must be >= 1");
        }
        idx.push((one - 1) as usize);
    }
    let off = match v.shape.offset(&idx) {
        Ok(o) => o,
        Err(e) => return super::ud::lua_error_msg(L, &e.to_string()),
    };
    if off >= v.len {
        return super::ud::lua_error_msg(L, "index out of range");
    }
    let val = unsafe { *v.data.add(off) };
    unsafe { lua_pushnumber(L, val) };
    1
}

unsafe extern "C" fn v_f64_to_array(L: *mut lua_State) -> c_int {
    let v = unsafe { &*check_view_f64(L, 1) };
    let owned = v.to_owned();
    unsafe { push_array(L, owned) };
    1
}

unsafe extern "C" fn v_i64_shape(L: *mut lua_State) -> c_int {
    let v = unsafe { &*check_view_i64(L, 1) };
    let dims = v.shape.dims();
    unsafe {
        lua_createtable(L, dims.len() as c_int, 0);
        for (i, &d) in dims.iter().enumerate() {
            lua_pushinteger(L, d as lua_Integer);
            lua_rawseti(L, -2, (i + 1) as lua_Integer);
        }
    }
    1
}

unsafe extern "C" fn v_i64_rank(L: *mut lua_State) -> c_int {
    let v = unsafe { &*check_view_i64(L, 1) };
    unsafe { lua_pushinteger(L, v.shape.rank() as lua_Integer) };
    1
}

unsafe extern "C" fn v_i64_len(L: *mut lua_State) -> c_int {
    let v = unsafe { &*check_view_i64(L, 1) };
    unsafe { lua_pushinteger(L, v.len as lua_Integer) };
    1
}

unsafe extern "C" fn v_i64_dtype(L: *mut lua_State) -> c_int {
    unsafe {
        lua_pushstring(L, c"i64".as_ptr());
    }
    1
}

unsafe extern "C" fn v_i64_get(L: *mut lua_State) -> c_int {
    let v = unsafe { &*check_view_i64(L, 1) };
    let top = unsafe { lua_gettop(L) };
    let rank = v.shape.rank();
    if top - 1 != rank as c_int {
        return super::ud::lua_error_msg(L, "get: wrong number of indices");
    }
    let mut idx = Vec::with_capacity(rank);
    for i in 0..rank {
        let one = unsafe { luaL_checkinteger(L, 2 + i as c_int) };
        if one < 1 {
            return super::ud::lua_error_msg(L, "index must be >= 1");
        }
        idx.push((one - 1) as usize);
    }
    let off = match v.shape.offset(&idx) {
        Ok(o) => o,
        Err(e) => return super::ud::lua_error_msg(L, &e.to_string()),
    };
    if off >= v.len {
        return super::ud::lua_error_msg(L, "index out of range");
    }
    let val = unsafe { *v.data.add(off) };
    unsafe { lua_pushinteger(L, val as lua_Integer) };
    1
}

unsafe extern "C" fn v_i64_to_array(L: *mut lua_State) -> c_int {
    let v = unsafe { &*check_view_i64(L, 1) };
    let owned = v.to_owned();
    unsafe { push_array_i64(L, owned) };
    1
}
