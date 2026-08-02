//! Lua C functions implementing the MatLua module and Array methods.

#![allow(non_snake_case)]

use std::os::raw::c_int;
use std::ptr;

use crate::array::Array;
use crate::linalg;

use super::ffi::*;
use super::ud::{
    array_from_table, check_array, indices_1_based, push_array, push_shape_table, shape_from_args,
    test_array, ARRAY_MT,
};

macro_rules! lua_try {
    ($L:expr, $expr:expr) => {
        match $expr {
            Ok(v) => v,
            Err(e) => {
                let msg = e.to_string();
                return super::ud::lua_error_msg($L, &msg);
            }
        }
    };
}

// ----- module functions -----

pub unsafe extern "C" fn l_zeros(L: *mut lua_State) -> c_int {
    let shape = lua_try!(L, unsafe { shape_from_args(L, 1) });
    let a = lua_try!(L, Array::zeros(shape));
    unsafe { push_array(L, a) };
    1
}

pub unsafe extern "C" fn l_ones(L: *mut lua_State) -> c_int {
    let shape = lua_try!(L, unsafe { shape_from_args(L, 1) });
    let a = lua_try!(L, Array::ones(shape));
    unsafe { push_array(L, a) };
    1
}

pub unsafe extern "C" fn l_full(L: *mut lua_State) -> c_int {
    let top = unsafe { lua_gettop(L) };
    if top < 2 {
        return super::ud::lua_error_msg(L, "full(shape..., value) requires a value");
    }
    let value = unsafe { luaL_checknumber(L, top) };
    let shape = if top == 2 && unsafe { lua_type(L, 1) } == LUA_TTABLE {
        lua_try!(L, unsafe { super::ud::shape_from_table(L, 1) })
    } else {
        let mut dims = Vec::new();
        for i in 1..top {
            if !unsafe { lua_isnumber(L, i) } {
                return super::ud::lua_error_msg(L, "full shape dims must be numbers");
            }
            let n = unsafe { luaL_checkinteger(L, i) };
            if n < 0 {
                return super::ud::lua_error_msg(L, "shape dims must be non-negative");
            }
            dims.push(n as usize);
        }
        dims
    };
    let a = lua_try!(L, Array::full(shape, value));
    unsafe { push_array(L, a) };
    1
}

pub unsafe extern "C" fn l_arange(L: *mut lua_State) -> c_int {
    let start = unsafe { luaL_checknumber(L, 1) };
    let stop = unsafe { luaL_checknumber(L, 2) };
    let step = unsafe { luaL_optnumber(L, 3, 1.0) };
    let a = lua_try!(L, Array::arange_step(start, stop, step));
    unsafe { push_array(L, a) };
    1
}

pub unsafe extern "C" fn l_array(L: *mut lua_State) -> c_int {
    let a = lua_try!(L, unsafe { array_from_table(L, 1) });
    unsafe { push_array(L, a) };
    1
}

pub unsafe extern "C" fn l_eye(L: *mut lua_State) -> c_int {
    let n = unsafe { luaL_checkinteger(L, 1) };
    if n < 0 {
        return super::ud::lua_error_msg(L, "eye(n) requires n >= 0");
    }
    let a = lua_try!(L, linalg::eye(n as usize));
    unsafe { push_array(L, a) };
    1
}

pub unsafe extern "C" fn l_matmul(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let b = unsafe { &*check_array(L, 2) };
    let c = lua_try!(L, linalg::matmul(&a.array, &b.array));
    unsafe { push_array(L, c) };
    1
}

pub unsafe extern "C" fn l_matmul_at(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let b = unsafe { &*check_array(L, 2) };
    let c = lua_try!(L, linalg::matmul_at(&a.array, &b.array));
    unsafe { push_array(L, c) };
    1
}

pub unsafe extern "C" fn l_normal_eq(L: *mut lua_State) -> c_int {
    let x = unsafe { &*check_array(L, 1) };
    let y = unsafe { &*check_array(L, 2) };
    let b = lua_try!(L, linalg::normal_eq(&x.array, &y.array));
    unsafe { push_array(L, b) };
    1
}

pub unsafe extern "C" fn l_solve(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let b = unsafe { &*check_array(L, 2) };
    let x = lua_try!(L, linalg::solve(&a.array, &b.array));
    unsafe { push_array(L, x) };
    1
}

pub unsafe extern "C" fn l_lstsq(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let b = unsafe { &*check_array(L, 2) };
    let x = lua_try!(L, linalg::lstsq(&a.array, &b.array));
    unsafe { push_array(L, x) };
    1
}

pub unsafe extern "C" fn l_eigh(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let (w, v) = lua_try!(L, linalg::eigh(&a.array));
    unsafe {
        push_array(L, w);
        push_array(L, v);
    }
    2
}

pub unsafe extern "C" fn l_pinv(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let p = lua_try!(L, linalg::pinv(&a.array));
    unsafe { push_array(L, p) };
    1
}

pub unsafe extern "C" fn l_transpose(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let t = lua_try!(L, linalg::transpose(&a.array));
    unsafe { push_array(L, t) };
    1
}

pub unsafe extern "C" fn l_dot(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let b = unsafe { &*check_array(L, 2) };
    let d = lua_try!(L, linalg::dot(&a.array, &b.array));
    unsafe { lua_pushnumber(L, d) };
    1
}

pub unsafe extern "C" fn l_norm(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let n = lua_try!(L, linalg::norm(&a.array));
    unsafe { lua_pushnumber(L, n) };
    1
}

pub unsafe extern "C" fn l_cholesky(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let l = lua_try!(L, linalg::cholesky(&a.array));
    unsafe { push_array(L, l) };
    1
}

pub unsafe extern "C" fn l_qr(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let (q, r) = lua_try!(L, linalg::qr(&a.array));
    unsafe {
        push_array(L, q);
        push_array(L, r);
    }
    2
}

pub unsafe extern "C" fn l_svd(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let (u, s, v) = lua_try!(L, linalg::svd(&a.array));
    unsafe {
        push_array(L, u);
        push_array(L, s);
        push_array(L, v);
    }
    3
}

// ----- array methods / metamethods -----

pub unsafe extern "C" fn a_gc(L: *mut lua_State) -> c_int {
    let ud = unsafe { check_array(L, 1) };
    unsafe { ptr::drop_in_place(ud) };
    0
}

pub unsafe extern "C" fn a_tostring(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let s = format!(
        "matlua.Array shape={} rank={} len={}",
        a.array.shape(),
        a.array.rank(),
        a.array.len()
    );
    unsafe {
        lua_pushlstring(L, s.as_ptr() as *const _, s.len());
    }
    1
}

pub unsafe extern "C" fn a_len(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { lua_pushinteger(L, a.array.len() as lua_Integer) };
    1
}

pub unsafe extern "C" fn a_shape(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_shape_table(L, a.array.dims()) };
    1
}

pub unsafe extern "C" fn a_rank(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { lua_pushinteger(L, a.array.rank() as lua_Integer) };
    1
}

pub unsafe extern "C" fn a_get(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let top = unsafe { lua_gettop(L) };
    let rank = a.array.rank();
    // Fast path: rank-1 single index (no multi-index buffer walk).
    if rank == 1 && top == 2 {
        let v = unsafe { luaL_checkinteger(L, 2) };
        if v < 1 {
            return super::ud::lua_error_msg(L, "index must be >= 1");
        }
        let v = lua_try!(L, a.array.get(&[(v as usize) - 1]));
        unsafe { lua_pushnumber(L, v) };
        return 1;
    }
    let idx = lua_try!(L, unsafe { indices_1_based(L, 2, top, rank) });
    let v = lua_try!(L, a.array.get(&idx));
    unsafe { lua_pushnumber(L, v) };
    1
}

pub unsafe extern "C" fn a_set(L: *mut lua_State) -> c_int {
    let a = unsafe { &mut *check_array(L, 1) };
    let top = unsafe { lua_gettop(L) };
    if top < 3 {
        return super::ud::lua_error_msg(L, "set(i..., value) needs indices and a value");
    }
    let value = unsafe { luaL_checknumber(L, top) };
    let rank = a.array.rank();
    if rank == 1 && top == 3 {
        let v = unsafe { luaL_checkinteger(L, 2) };
        if v < 1 {
            return super::ud::lua_error_msg(L, "index must be >= 1");
        }
        lua_try!(L, a.array.set(&[(v as usize) - 1], value));
        return 0;
    }
    let idx = lua_try!(L, unsafe { indices_1_based(L, 2, top - 1, rank) });
    lua_try!(L, a.array.set(&idx, value));
    0
}

pub unsafe extern "C" fn a_sum(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { lua_pushnumber(L, a.array.sum()) };
    1
}

pub unsafe extern "C" fn a_mean(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let m = lua_try!(L, a.array.mean());
    unsafe { lua_pushnumber(L, m) };
    1
}

pub unsafe extern "C" fn a_min(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let m = lua_try!(L, a.array.min());
    unsafe { lua_pushnumber(L, m) };
    1
}

pub unsafe extern "C" fn a_max(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let m = lua_try!(L, a.array.max());
    unsafe { lua_pushnumber(L, m) };
    1
}

pub unsafe extern "C" fn a_copy(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.clone()) };
    1
}

pub unsafe extern "C" fn a_reshape(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let shape = lua_try!(L, unsafe { shape_from_args(L, 2) });
    let b = lua_try!(L, a.array.reshape(shape));
    unsafe { push_array(L, b) };
    1
}

pub unsafe extern "C" fn a_transpose(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let t = lua_try!(L, linalg::transpose(&a.array));
    unsafe { push_array(L, t) };
    1
}

pub unsafe extern "C" fn a_fill(L: *mut lua_State) -> c_int {
    let a = unsafe { &mut *check_array(L, 1) };
    let v = unsafe { luaL_checknumber(L, 2) };
    a.array.fill(v);
    0
}

pub unsafe extern "C" fn a_add_op(L: *mut lua_State) -> c_int {
    let ta = unsafe { test_array(L, 1) };
    let tb = unsafe { test_array(L, 2) };
    if !ta.is_null() && !tb.is_null() {
        let c = lua_try!(L, Array::add(&(*ta).array, &(*tb).array));
        unsafe { push_array(L, c) };
        return 1;
    }
    if !ta.is_null() && unsafe { lua_isnumber(L, 2) } {
        let s = unsafe { luaL_checknumber(L, 2) };
        unsafe { push_array(L, (*ta).array.add_scalar(s)) };
        return 1;
    }
    if unsafe { lua_isnumber(L, 1) } && !tb.is_null() {
        let s = unsafe { luaL_checknumber(L, 1) };
        unsafe { push_array(L, (*tb).array.add_scalar(s)) };
        return 1;
    }
    super::ud::lua_error_msg(L, "add expects arrays or array and number")
}

pub unsafe extern "C" fn a_sub_op(L: *mut lua_State) -> c_int {
    let ta = unsafe { test_array(L, 1) };
    let tb = unsafe { test_array(L, 2) };
    if !ta.is_null() && !tb.is_null() {
        let c = lua_try!(L, Array::sub(&(*ta).array, &(*tb).array));
        unsafe { push_array(L, c) };
        return 1;
    }
    if !ta.is_null() && unsafe { lua_isnumber(L, 2) } {
        let s = unsafe { luaL_checknumber(L, 2) };
        unsafe { push_array(L, (*ta).array.sub_scalar(s)) };
        return 1;
    }
    if unsafe { lua_isnumber(L, 1) } && !tb.is_null() {
        let s = unsafe { luaL_checknumber(L, 1) };
        unsafe { push_array(L, (*tb).array.scalar_sub(s)) };
        return 1;
    }
    super::ud::lua_error_msg(L, "sub expects arrays or array and number")
}

pub unsafe extern "C" fn a_mul_op(L: *mut lua_State) -> c_int {
    let ta = unsafe { test_array(L, 1) };
    let tb = unsafe { test_array(L, 2) };
    if !ta.is_null() && !tb.is_null() {
        let c = lua_try!(L, Array::mul(&(*ta).array, &(*tb).array));
        unsafe { push_array(L, c) };
        return 1;
    }
    if !ta.is_null() && unsafe { lua_isnumber(L, 2) } {
        let s = unsafe { luaL_checknumber(L, 2) };
        unsafe { push_array(L, (*ta).array.mul_scalar(s)) };
        return 1;
    }
    if unsafe { lua_isnumber(L, 1) } && !tb.is_null() {
        let s = unsafe { luaL_checknumber(L, 1) };
        unsafe { push_array(L, (*tb).array.mul_scalar(s)) };
        return 1;
    }
    super::ud::lua_error_msg(L, "mul expects arrays or array and number")
}

pub unsafe extern "C" fn a_div_op(L: *mut lua_State) -> c_int {
    let ta = unsafe { test_array(L, 1) };
    let tb = unsafe { test_array(L, 2) };
    if !ta.is_null() && !tb.is_null() {
        let c = lua_try!(L, Array::div(&(*ta).array, &(*tb).array));
        unsafe { push_array(L, c) };
        return 1;
    }
    if !ta.is_null() && unsafe { lua_isnumber(L, 2) } {
        let s = unsafe { luaL_checknumber(L, 2) };
        unsafe { push_array(L, (*ta).array.div_scalar(s)) };
        return 1;
    }
    if unsafe { lua_isnumber(L, 1) } && !tb.is_null() {
        let s = unsafe { luaL_checknumber(L, 1) };
        unsafe { push_array(L, (*tb).array.scalar_div(s)) };
        return 1;
    }
    super::ud::lua_error_msg(L, "div expects arrays or array and number")
}

pub unsafe extern "C" fn a_unm(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.neg()) };
    1
}


pub unsafe extern "C" fn a_abs(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.abs()) };
    1
}
pub unsafe extern "C" fn a_sqrt(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.sqrt()) };
    1
}
pub unsafe extern "C" fn a_exp(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.exp()) };
    1
}
pub unsafe extern "C" fn a_log(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.log()) };
    1
}
pub unsafe extern "C" fn a_log1p(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.log1p()) };
    1
}
pub unsafe extern "C" fn a_sign(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.sign()) };
    1
}
pub unsafe extern "C" fn a_power(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    if unsafe { lua_isnumber(L, 2) } {
        let p = unsafe { luaL_checknumber(L, 2) };
        unsafe { push_array(L, a.array.power_scalar(p)) };
        return 1;
    }
    let b = unsafe { &*check_array(L, 2) };
    let c = lua_try!(L, a.array.power(&b.array));
    unsafe { push_array(L, c) };
    1
}
pub unsafe extern "C" fn a_clip(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let lo = unsafe { luaL_checknumber(L, 2) };
    let hi = unsafe { luaL_checknumber(L, 3) };
    let c = lua_try!(L, a.array.clip(lo, hi));
    unsafe { push_array(L, c) };
    1
}
pub unsafe extern "C" fn a_isnan(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.isnan()) };
    1
}
pub unsafe extern "C" fn a_isfinite(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.isfinite()) };
    1
}
pub unsafe extern "C" fn a_cumsum(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    unsafe { push_array(L, a.array.cumsum()) };
    1
}
pub unsafe extern "C" fn a_argmin(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let i = lua_try!(L, a.array.argmin());
    unsafe { lua_pushinteger(L, (i + 1) as _) }; // 1-based
    1
}
pub unsafe extern "C" fn a_argmax(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let i = lua_try!(L, a.array.argmax());
    unsafe { lua_pushinteger(L, (i + 1) as _) };
    1
}
pub unsafe extern "C" fn a_var(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let ddof = if unsafe { lua_gettop(L) } >= 2 {
        let d = unsafe { luaL_checkinteger(L, 2) };
        if d < 0 {
            return super::ud::lua_error_msg(L, "var ddof must be >= 0");
        }
        d as usize
    } else {
        0
    };
    let v = lua_try!(L, a.array.var(ddof));
    unsafe { lua_pushnumber(L, v) };
    1
}
pub unsafe extern "C" fn a_std(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array(L, 1) };
    let ddof = if unsafe { lua_gettop(L) } >= 2 {
        let d = unsafe { luaL_checkinteger(L, 2) };
        if d < 0 {
            return super::ud::lua_error_msg(L, "std ddof must be >= 0");
        }
        d as usize
    } else {
        0
    };
    let v = lua_try!(L, a.array.std(ddof));
    unsafe { lua_pushnumber(L, v) };
    1
}
pub unsafe extern "C" fn l_where(L: *mut lua_State) -> c_int {
    let c = unsafe { &*check_array(L, 1) };
    let x = unsafe { &*check_array(L, 2) };
    let y = unsafe { &*check_array(L, 3) };
    let o = lua_try!(L, Array::where_cond(&c.array, &x.array, &y.array));
    unsafe { push_array(L, o) };
    1
}


pub unsafe extern "C" fn l_concatenate(L: *mut lua_State) -> c_int {
    let axis = unsafe { luaL_checkinteger(L, 1) };
    if axis < 0 {
        return super::ud::lua_error_msg(L, "concatenate axis must be >= 0");
    }
    let top = unsafe { lua_gettop(L) };
    if top < 3 {
        return super::ud::lua_error_msg(L, "concatenate(axis, a, b, ...) needs arrays");
    }
    let mut owned = Vec::new();
    for i in 2..=top {
        let a = unsafe { &*check_array(L, i) };
        owned.push(&a.array as *const _);
    }
    // Safety: arrays are on stack, alive for call
    let refs: Vec<&Array> = owned.iter().map(|p| unsafe { &**p }).collect();
    let out = lua_try!(L, Array::concatenate(axis as usize, &refs));
    unsafe { push_array(L, out) };
    1
}
pub unsafe extern "C" fn l_stack(L: *mut lua_State) -> c_int {
    let axis = unsafe { luaL_checkinteger(L, 1) };
    if axis < 0 {
        return super::ud::lua_error_msg(L, "stack axis must be >= 0");
    }
    let top = unsafe { lua_gettop(L) };
    if top < 3 {
        return super::ud::lua_error_msg(L, "stack(axis, a, b, ...) needs arrays");
    }
    let mut owned = Vec::new();
    for i in 2..=top {
        let a = unsafe { &*check_array(L, i) };
        owned.push(&a.array as *const _);
    }
    let refs: Vec<&Array> = owned.iter().map(|p| unsafe { &**p }).collect();
    let out = lua_try!(L, Array::stack(axis as usize, &refs));
    unsafe { push_array(L, out) };
    1
}
/// Module open: push library table.
pub unsafe extern "C" fn luaopen_matlua(L: *mut lua_State) -> c_int {
    unsafe {
        if luaL_newmetatable(L, ARRAY_MT.as_ptr()) != 0 {
            lua_newtable(L);
            let methods: [(&std::ffi::CStr, unsafe extern "C" fn(*mut lua_State) -> c_int); 27] = [
                (c"shape", a_shape),
                (c"rank", a_rank),
                (c"get", a_get),
                (c"set", a_set),
                (c"sum", a_sum),
                (c"mean", a_mean),
                (c"min", a_min),
                (c"max", a_max),
                (c"copy", a_copy),
                (c"reshape", a_reshape),
                (c"transpose", a_transpose),
                (c"fill", a_fill),
                (c"abs", a_abs),
                (c"sqrt", a_sqrt),
                (c"exp", a_exp),
                (c"log", a_log),
                (c"log1p", a_log1p),
                (c"sign", a_sign),
                (c"power", a_power),
                (c"clip", a_clip),
                (c"isnan", a_isnan),
                (c"isfinite", a_isfinite),
                (c"cumsum", a_cumsum),
                (c"argmin", a_argmin),
                (c"argmax", a_argmax),
                (c"var", a_var),
                (c"std", a_std),
            ];
            for (name, f) in methods {
                lua_pushcfunction(L, Some(f));
                lua_setfield(L, -2, name.as_ptr());
            }
            lua_setfield(L, -2, c"__index".as_ptr());

            lua_pushcfunction(L, Some(a_gc));
            lua_setfield(L, -2, c"__gc".as_ptr());
            lua_pushcfunction(L, Some(a_tostring));
            lua_setfield(L, -2, c"__tostring".as_ptr());
            lua_pushcfunction(L, Some(a_len));
            lua_setfield(L, -2, c"__len".as_ptr());
            lua_pushcfunction(L, Some(a_add_op));
            lua_setfield(L, -2, c"__add".as_ptr());
            lua_pushcfunction(L, Some(a_sub_op));
            lua_setfield(L, -2, c"__sub".as_ptr());
            lua_pushcfunction(L, Some(a_mul_op));
            lua_setfield(L, -2, c"__mul".as_ptr());
            lua_pushcfunction(L, Some(a_div_op));
            lua_setfield(L, -2, c"__div".as_ptr());
            lua_pushcfunction(L, Some(a_unm));
            lua_setfield(L, -2, c"__unm".as_ptr());
        }
        lua_pop(L, 1);

        lua_newtable(L);
        let funcs: [(&std::ffi::CStr, unsafe extern "C" fn(*mut lua_State) -> c_int); 22] = [
            (c"zeros", l_zeros),
            (c"ones", l_ones),
            (c"full", l_full),
            (c"arange", l_arange),
            (c"array", l_array),
            (c"eye", l_eye),
            (c"where", l_where),
            (c"concatenate", l_concatenate),
            (c"stack", l_stack),
            (c"matmul", l_matmul),
            (c"matmul_at", l_matmul_at),
            (c"normal_eq", l_normal_eq),
            (c"solve", l_solve),
            (c"lstsq", l_lstsq),
            (c"eigh", l_eigh),
            (c"pinv", l_pinv),
            (c"transpose", l_transpose),
            (c"dot", l_dot),
            (c"norm", l_norm),
            (c"cholesky", l_cholesky),
            (c"qr", l_qr),
            (c"svd", l_svd),
        ];
        for (name, f) in funcs {
            lua_pushcfunction(L, Some(f));
            lua_setfield(L, -2, name.as_ptr());
        }
    }
    1
}
