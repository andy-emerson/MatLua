//! Hand-rolled Lua 5.4 face (host-owned `lua_State`).
//!
//! Enable with the `lua` Cargo feature. Users see a Lua library (`require
//! "matlua"` after registration); this module is registration and binding glue.
//!
//! # Indexing
//!
//! The Lua face is **1-based**. Rust [`Array`](crate::Array) APIs remain 0-based.
//!
//! # Host integration
//!
//! ```ignore
//! // L: *mut lua_State owned by the host
//! unsafe { matlua::lua::register(L) };
//! // scripts: local ml = require "matlua"
//! ```
//!
//! For tests and tools, [`Lua`] creates a private state with standard libraries
//! and MatLua preloaded (using the vendored PUC 5.4 build).

#![allow(non_snake_case)] // Lua C API uses `L` for lua_State*

mod api;
mod ffi;
mod ud;

use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;

use crate::error::{Error, Result};

use api::luaopen_matlua;
use ffi::*;

pub use ffi::lua_State;

/// Register MatLua as package `matlua` on an existing Lua 5.4 state.
///
/// Leaves the stack balanced. After this call, scripts may
/// `local ml = require "matlua"`.
///
/// # Safety
///
/// `L` must be a valid PUC Lua 5.4 `lua_State*`. The host retains ownership of
/// the state and must keep it alive for the duration of any MatLua userdata.
pub unsafe fn register(L: *mut lua_State) {
    unsafe {
        luaL_requiref(L, c"matlua".as_ptr(), Some(luaopen_matlua), 1);
        lua_pop(L, 1);
    }
}

/// Enable Lua 5.4 **generational** GC on a host state (recommended for MatLua).
///
/// Large arrays live on the Rust heap; without GC tuning, dead userdata pile up
/// and face-path allocators pay major GC / free storms. Safe to call more than once.
///
/// # Safety
/// `L` must be a valid Lua 5.4 state.
pub unsafe fn enable_generational_gc(L: *mut lua_State) {
    unsafe {
        let _ = lua_gc(L, LUA_GCGEN, 0, 0);
    }
}

/// C entry point compatible with `luaL_requiref` / `luaopen_*` conventions.
///
/// # Safety
///
/// `L` must be a valid Lua 5.4 state.
pub unsafe extern "C" fn luaopen_matlua_lib(L: *mut lua_State) -> c_int {
    unsafe { luaopen_matlua(L) }
}

/// Owned Lua state for tests and simple tools (not required for hosts).
pub struct Lua {
    state: *mut lua_State,
}

impl Lua {
    /// Create a new state, open standard libraries, and register MatLua.
    pub fn new() -> Result<Self> {
        let state = unsafe { luaL_newstate() };
        if state.is_null() {
            return Err(Error::message("luaL_newstate failed"));
        }
        unsafe {
            luaL_openlibs(state);
            // Generational GC: better for allocate-heavy numeric scripts (PUC lua.c default for interactive).
            let _ = lua_gc(state, LUA_GCGEN, 0, 0);
            register(state);
        }
        Ok(Self { state })
    }

    /// Borrow the raw state pointer.
    pub fn as_ptr(&self) -> *mut lua_State {
        self.state
    }

    /// Execute a chunk; errors become [`Error::Message`].
    pub fn do_string(&self, source: &str) -> Result<()> {
        let c = CString::new(source).map_err(|e| Error::message(e.to_string()))?;
        let name = c"=(matlua)";
        unsafe {
            let load = luaL_loadbufferx(
                self.state,
                c.as_ptr(),
                source.len(),
                name.as_ptr(),
                ptr::null(),
            );
            if load != LUA_OK {
                let msg = error_string(self.state);
                lua_pop(self.state, 1);
                return Err(Error::message(format!("lua load error: {msg}")));
            }
            let call = lua_pcall(self.state, 0, LUA_MULTRET, 0);
            if call != LUA_OK {
                let msg = error_string(self.state);
                lua_pop(self.state, 1);
                return Err(Error::message(format!("lua runtime error: {msg}")));
            }
        }
        Ok(())
    }

    /// Call a global function with no arguments (discards results).
    ///
    /// Used by benches so wall-clock timing does not include chunk compile cost.
    pub fn call_global(&self, name: &str) -> Result<()> {
        let c = CString::new(name).map_err(|e| Error::message(e.to_string()))?;
        unsafe {
            lua_getglobal(self.state, c.as_ptr());
            if lua_type(self.state, -1) != LUA_TFUNCTION {
                lua_pop(self.state, 1);
                return Err(Error::message(format!(
                    "global {name:?} is not a function"
                )));
            }
            let call = lua_pcall(self.state, 0, 0, 0);
            if call != LUA_OK {
                let msg = error_string(self.state);
                lua_pop(self.state, 1);
                return Err(Error::message(format!("lua runtime error: {msg}")));
            }
        }
        Ok(())
    }
}

impl Drop for Lua {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { lua_close(self.state) };
            self.state = ptr::null_mut();
        }
    }
}

unsafe fn error_string(state: *mut lua_State) -> String {
    unsafe {
        let s = lua_tostring(state, -1);
        if s.is_null() {
            return "(non-string error)".into();
        }
        std::ffi::CStr::from_ptr(s).to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_and_constructors() {
        let lua = Lua::new().unwrap();
        lua.do_string(
            r#"
            local ml = require "matlua"
            local z = ml.zeros(2, 3)
            assert(z:rank() == 2)
            assert(#z == 6)
            local sh = z:shape()
            assert(sh[1] == 2 and sh[2] == 3)
            z:set(2, 3, 4.5)
            assert(math.abs(z:get(2, 3) - 4.5) < 1e-12)
            "#,
        )
        .unwrap();
    }

    #[test]
    fn elementwise_and_matmul_solve() {
        let lua = Lua::new().unwrap();
        lua.do_string(
            r#"
            local ml = require "matlua"
            local a = ml.array({1, 2, 3})
            local b = a * 2 + 1
            assert(math.abs(b:get(1) - 3) < 1e-12)
            assert(math.abs(b:sum() - (3+5+7)) < 1e-12)

            local A = ml.array({{3, 1}, {1, 2}})
            local rhs = ml.array({9, 8})
            local x = ml.solve(A, rhs)
            assert(math.abs(x:get(1) - 2) < 1e-9)
            assert(math.abs(x:get(2) - 3) < 1e-9)

            local M = ml.array({{1, 2}, {3, 4}})
            local v = ml.array({1, 1})
            local y = ml.matmul(M, v)
            assert(math.abs(y:get(1) - 3) < 1e-12)
            assert(math.abs(y:get(2) - 7) < 1e-12)

            local X = ml.array({{1, 0}, {1, 1}, {1, 2}, {1, 3}})
            local yy = ml.array({1, 3, 5, 7})
            local beta = ml.solve(
              ml.matmul(X:transpose(), X),
              ml.matmul(X:transpose(), yy:reshape(4, 1))
            )
            assert(math.abs(beta:get(1, 1) - 1) < 1e-8)
            assert(math.abs(beta:get(2, 1) - 2) < 1e-8)
            "#,
        )
        .unwrap();
    }

    #[test]
    fn one_based_indexing_errors() {
        let lua = Lua::new().unwrap();
        let err = lua
            .do_string(
                r#"
                local ml = require "matlua"
                local a = ml.zeros(2)
                a:get(0)
                "#,
            )
            .unwrap_err();
        assert!(err.to_string().contains("index") || err.to_string().contains("lua runtime"));
    }
}
