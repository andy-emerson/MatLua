//! Minimal hand-rolled bindings to the Lua 5.4 C API (PUC).
//!
//! Only the surface MatLua needs is declared. Hosts may use their own Lua
//! build so long as it is ABI-compatible PUC 5.4.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::{c_char, c_double, c_int, c_void};

/// Opaque Lua state.
#[repr(C)]
pub struct lua_State {
    _private: [u8; 0],
}

pub type lua_Number = c_double;
pub type lua_Integer = i64;
pub type lua_CFunction = Option<unsafe extern "C" fn(L: *mut lua_State) -> c_int>;

pub const LUA_OK: c_int = 0;
pub const LUA_ERRRUN: c_int = 2;
pub const LUA_MULTRET: c_int = -1;

pub const LUA_TNONE: c_int = -1;
pub const LUA_TNIL: c_int = 0;
pub const LUA_TBOOLEAN: c_int = 1;
pub const LUA_TNUMBER: c_int = 3;
pub const LUA_TSTRING: c_int = 4;
pub const LUA_TTABLE: c_int = 5;
pub const LUA_TFUNCTION: c_int = 6;
pub const LUA_TUSERDATA: c_int = 7;

/// Lua 5.4 registry index (`-LUAI_MAXSTACK - 1000` with default maxstack).
pub const LUA_REGISTRYINDEX: c_int = -1_001_000;

#[repr(C)]
pub struct luaL_Reg {
    pub name: *const c_char,
    pub func: lua_CFunction,
}

unsafe extern "C" {
    pub fn luaL_newstate() -> *mut lua_State;
    pub fn lua_close(L: *mut lua_State);
    pub fn luaL_openlibs(L: *mut lua_State);

    pub fn lua_gettop(L: *mut lua_State) -> c_int;
    pub fn lua_settop(L: *mut lua_State, idx: c_int);
    pub fn lua_pushvalue(L: *mut lua_State, idx: c_int);
    pub fn lua_checkstack(L: *mut lua_State, n: c_int) -> c_int;

    pub fn lua_pushnil(L: *mut lua_State);
    pub fn lua_pushnumber(L: *mut lua_State, n: lua_Number);
    pub fn lua_pushinteger(L: *mut lua_State, n: lua_Integer);
    pub fn lua_pushboolean(L: *mut lua_State, b: c_int);
    pub fn lua_pushstring(L: *mut lua_State, s: *const c_char) -> *const c_char;
    pub fn lua_pushlstring(L: *mut lua_State, s: *const c_char, len: usize) -> *const c_char;
    pub fn lua_pushcclosure(L: *mut lua_State, f: lua_CFunction, n: c_int);
    pub fn lua_createtable(L: *mut lua_State, narr: c_int, nrec: c_int);

    pub fn lua_type(L: *mut lua_State, idx: c_int) -> c_int;
    pub fn lua_tonumberx(L: *mut lua_State, idx: c_int, isnum: *mut c_int) -> lua_Number;
    pub fn lua_tointegerx(L: *mut lua_State, idx: c_int, isnum: *mut c_int) -> lua_Integer;
    pub fn lua_tolstring(L: *mut lua_State, idx: c_int, len: *mut usize) -> *const c_char;
    pub fn lua_touserdata(L: *mut lua_State, idx: c_int) -> *mut c_void;

    pub fn lua_getfield(L: *mut lua_State, idx: c_int, k: *const c_char) -> c_int;
    pub fn lua_setfield(L: *mut lua_State, idx: c_int, k: *const c_char);
    pub fn lua_rawgeti(L: *mut lua_State, idx: c_int, n: lua_Integer) -> c_int;
    pub fn lua_rawseti(L: *mut lua_State, idx: c_int, n: lua_Integer);
    pub fn lua_setmetatable(L: *mut lua_State, objindex: c_int) -> c_int;
    pub fn lua_getglobal(L: *mut lua_State, name: *const c_char) -> c_int;
    pub fn lua_setglobal(L: *mut lua_State, name: *const c_char);

    pub fn lua_error(L: *mut lua_State) -> c_int;
    pub fn lua_pcallk(
        L: *mut lua_State,
        nargs: c_int,
        nresults: c_int,
        errfunc: c_int,
        ctx: isize,
        k: *const c_void,
    ) -> c_int;
    pub fn luaL_loadbufferx(
        L: *mut lua_State,
        buff: *const c_char,
        sz: usize,
        name: *const c_char,
        mode: *const c_char,
    ) -> c_int;

    pub fn lua_newuserdatauv(L: *mut lua_State, sz: usize, nuvalue: c_int) -> *mut c_void;
    pub fn luaL_newmetatable(L: *mut lua_State, tname: *const c_char) -> c_int;
    pub fn luaL_setmetatable(L: *mut lua_State, tname: *const c_char);
    pub fn luaL_checkudata(L: *mut lua_State, ud: c_int, tname: *const c_char) -> *mut c_void;
    pub fn luaL_testudata(L: *mut lua_State, ud: c_int, tname: *const c_char) -> *mut c_void;
    pub fn luaL_checknumber(L: *mut lua_State, arg: c_int) -> lua_Number;
    pub fn luaL_checkinteger(L: *mut lua_State, arg: c_int) -> lua_Integer;
    pub fn luaL_optnumber(L: *mut lua_State, arg: c_int, def: lua_Number) -> lua_Number;
    pub fn luaL_requiref(
        L: *mut lua_State,
        modname: *const c_char,
        openf: lua_CFunction,
        glb: c_int,
    );
    pub fn luaL_len(L: *mut lua_State, idx: c_int) -> lua_Integer;
}

#[inline]
pub unsafe fn lua_pop(L: *mut lua_State, n: c_int) {
    unsafe { lua_settop(L, -(n) - 1) }
}

#[inline]
pub unsafe fn lua_newtable(L: *mut lua_State) {
    unsafe { lua_createtable(L, 0, 0) }
}

#[inline]
pub unsafe fn lua_pushcfunction(L: *mut lua_State, f: lua_CFunction) {
    unsafe { lua_pushcclosure(L, f, 0) }
}

#[inline]
pub unsafe fn lua_pcall(L: *mut lua_State, nargs: c_int, nresults: c_int, errfunc: c_int) -> c_int {
    unsafe { lua_pcallk(L, nargs, nresults, errfunc, 0, std::ptr::null()) }
}

#[inline]
pub unsafe fn lua_tostring(L: *mut lua_State, idx: c_int) -> *const c_char {
    unsafe { lua_tolstring(L, idx, std::ptr::null_mut()) }
}

#[inline]
pub unsafe fn lua_isnumber(L: *mut lua_State, idx: c_int) -> bool {
    let mut isnum = 0;
    unsafe {
        lua_tonumberx(L, idx, &mut isnum);
    }
    isnum != 0
}
