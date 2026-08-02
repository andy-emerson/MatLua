//! Lua face for [`ArrayI64`](crate::array::ArrayI64).

#![allow(non_snake_case)]

use std::os::raw::c_int;
use std::ptr;

use crate::array::ArrayI64;

use super::ffi::*;
use super::ud::{indices_1_based, push_array, push_shape_table, shape_from_args, shape_from_table};
use super::ud_i64::{
    array_i64_from_table, check_array_i64, push_array_i64, test_array_i64, ARRAY_I64_MT,
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

pub unsafe extern "C" fn l_zeros_i64(L: *mut lua_State) -> c_int {
    let shape = lua_try!(L, unsafe { shape_from_args(L, 1) });
    let a = lua_try!(L, ArrayI64::zeros(shape));
    unsafe { push_array_i64(L, a) };
    1
}
pub unsafe extern "C" fn l_ones_i64(L: *mut lua_State) -> c_int {
    let shape = lua_try!(L, unsafe { shape_from_args(L, 1) });
    let a = lua_try!(L, ArrayI64::ones(shape));
    unsafe { push_array_i64(L, a) };
    1
}
pub unsafe extern "C" fn l_full_i64(L: *mut lua_State) -> c_int {
    let top = unsafe { lua_gettop(L) };
    if top < 2 {
        return super::ud::lua_error_msg(L, "full_i64(shape..., value) requires a value");
    }
    let value = unsafe { luaL_checkinteger(L, top) };
    let dims = if top == 2 && unsafe { lua_type(L, 1) } == LUA_TTABLE {
        lua_try!(L, unsafe { shape_from_table(L, 1) })
    } else {
        let mut dims = Vec::new();
        for i in 1..top {
            let v = unsafe { luaL_checkinteger(L, i) };
            if v < 0 {
                return super::ud::lua_error_msg(L, "shape dims must be non-negative");
            }
            dims.push(v as usize);
        }
        dims
    };
    let a = lua_try!(L, ArrayI64::full(dims, value));
    unsafe { push_array_i64(L, a) };
    1
}
pub unsafe extern "C" fn l_arange_i64(L: *mut lua_State) -> c_int {
    let top = unsafe { lua_gettop(L) };
    let (start, stop, step) = if top >= 3 {
        (
            unsafe { luaL_checkinteger(L, 1) },
            unsafe { luaL_checkinteger(L, 2) },
            unsafe { luaL_checkinteger(L, 3) },
        )
    } else if top == 2 {
        (
            unsafe { luaL_checkinteger(L, 1) },
            unsafe { luaL_checkinteger(L, 2) },
            1,
        )
    } else {
        (0, unsafe { luaL_checkinteger(L, 1) }, 1)
    };
    let a = lua_try!(L, ArrayI64::arange_step(start, stop, step));
    unsafe { push_array_i64(L, a) };
    1
}
pub unsafe extern "C" fn l_array_i64(L: *mut lua_State) -> c_int {
    let a = lua_try!(L, unsafe { array_i64_from_table(L, 1) });
    unsafe { push_array_i64(L, a) };
    1
}
pub unsafe extern "C" fn l_eye_i64(L: *mut lua_State) -> c_int {
    let n = unsafe { luaL_checkinteger(L, 1) };
    if n < 0 {
        return super::ud::lua_error_msg(L, "eye_i64(n) requires n >= 0");
    }
    let a = lua_try!(L, ArrayI64::eye(n as usize));
    unsafe { push_array_i64(L, a) };
    1
}
pub unsafe extern "C" fn l_diag_i64(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let d = lua_try!(L, ArrayI64::diag(&a.array));
    unsafe { push_array_i64(L, d) };
    1
}
pub unsafe extern "C" fn l_outer_i64(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, ArrayI64::outer(&a.array, &b.array));
    unsafe { push_array_i64(L, o) };
    1
}

pub unsafe extern "C" fn a_i64_gc(L: *mut lua_State) -> c_int {
    let p = unsafe { check_array_i64(L, 1) };
    unsafe { ptr::drop_in_place(p) };
    0
}
pub unsafe extern "C" fn a_i64_len(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { lua_pushinteger(L, a.array.len() as lua_Integer) };
    1
}
pub unsafe extern "C" fn a_i64_tostring(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let s = format!(
        "ArrayI64(dtype=i64, shape={}, len={})",
        a.array.shape(),
        a.array.len()
    );
    unsafe { lua_pushlstring(L, s.as_ptr() as *const _, s.len()) };
    1
}
pub unsafe extern "C" fn a_i64_dtype(L: *mut lua_State) -> c_int {
    unsafe {
        lua_pushstring(L, c"i64".as_ptr());
    }
    1
}
pub unsafe extern "C" fn a_i64_shape(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_shape_table(L, a.array.dims()) };
    1
}
pub unsafe extern "C" fn a_i64_rank(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { lua_pushinteger(L, a.array.rank() as lua_Integer) };
    1
}
pub unsafe extern "C" fn a_i64_get(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let top = unsafe { lua_gettop(L) };
    let rank = a.array.rank();
    if rank == 1 && top == 2 {
        let v = unsafe { luaL_checkinteger(L, 2) };
        if v < 1 {
            return super::ud::lua_error_msg(L, "index must be >= 1");
        }
        let val = lua_try!(L, a.array.get(&[(v as usize) - 1]));
        unsafe { lua_pushinteger(L, val) };
        return 1;
    }
    let idx = lua_try!(L, unsafe { indices_1_based(L, 2, top, rank) });
    let val = lua_try!(L, a.array.get(&idx));
    unsafe { lua_pushinteger(L, val) };
    1
}
pub unsafe extern "C" fn a_i64_set(L: *mut lua_State) -> c_int {
    let a = unsafe { &mut *check_array_i64(L, 1) };
    let top = unsafe { lua_gettop(L) };
    if top < 3 {
        return super::ud::lua_error_msg(L, "set(i..., value) needs indices and a value");
    }
    let value = unsafe { luaL_checkinteger(L, top) };
    let rank = a.array.rank();
    let idx = lua_try!(L, unsafe { indices_1_based(L, 2, top - 1, rank) });
    lua_try!(L, a.array.set(&idx, value));
    0
}
pub unsafe extern "C" fn a_i64_sum(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_gettop(L) } >= 2 {
        let axis = (unsafe { luaL_checkinteger(L, 2) }) as usize;
        let s = lua_try!(L, a.array.sum_axis(axis));
        unsafe { push_array_i64(L, s) };
    } else {
        unsafe { lua_pushinteger(L, a.array.sum()) };
    }
    1
}
pub unsafe extern "C" fn a_i64_mean(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_gettop(L) } >= 2 {
        let axis = (unsafe { luaL_checkinteger(L, 2) }) as usize;
        let m = lua_try!(L, a.array.mean_axis(axis));
        unsafe { push_array(L, m) };
    } else {
        let m = lua_try!(L, a.array.mean());
        unsafe { lua_pushnumber(L, m) };
    }
    1
}
pub unsafe extern "C" fn a_i64_min(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_gettop(L) } >= 2 {
        let axis = (unsafe { luaL_checkinteger(L, 2) }) as usize;
        let s = lua_try!(L, a.array.min_axis(axis));
        unsafe { push_array_i64(L, s) };
    } else {
        let v = lua_try!(L, a.array.min());
        unsafe { lua_pushinteger(L, v) };
    }
    1
}
pub unsafe extern "C" fn a_i64_max(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_gettop(L) } >= 2 {
        let axis = (unsafe { luaL_checkinteger(L, 2) }) as usize;
        let s = lua_try!(L, a.array.max_axis(axis));
        unsafe { push_array_i64(L, s) };
    } else {
        let v = lua_try!(L, a.array.max());
        unsafe { lua_pushinteger(L, v) };
    }
    1
}
pub unsafe extern "C" fn a_i64_copy(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_array_i64(L, a.array.copy()) };
    1
}
pub unsafe extern "C" fn a_i64_reshape(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let shape = lua_try!(L, unsafe { shape_from_args(L, 2) });
    let r = lua_try!(L, a.array.reshape(shape));
    unsafe { push_array_i64(L, r) };
    1
}
pub unsafe extern "C" fn a_i64_fill(L: *mut lua_State) -> c_int {
    let a = unsafe { &mut *check_array_i64(L, 1) };
    let v = unsafe { luaL_checkinteger(L, 2) };
    a.array.fill(v);
    0
}
pub unsafe extern "C" fn a_i64_abs(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_array_i64(L, a.array.abs()) };
    1
}
pub unsafe extern "C" fn a_i64_transpose(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let t = lua_try!(L, a.array.transpose());
    unsafe { push_array_i64(L, t) };
    1
}
pub unsafe extern "C" fn a_i64_to_f64(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_array(L, a.array.to_f64()) };
    1
}
pub unsafe extern "C" fn a_i64_eq(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_type(L, 2) } == LUA_TNUMBER {
        let s = unsafe { luaL_checkinteger(L, 2) };
        unsafe { push_array_i64(L, a.array.eq_scalar(s)) };
        return 1;
    }
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.eq(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_lt(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_type(L, 2) } == LUA_TNUMBER {
        let s = unsafe { luaL_checkinteger(L, 2) };
        unsafe { push_array_i64(L, a.array.lt_scalar(s)) };
        return 1;
    }
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.lt(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_argsort(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let desc = if unsafe { lua_gettop(L) } >= 2 {
        unsafe { lua_toboolean(L, 2) != 0 }
    } else {
        false
    };
    let idx = lua_try!(L, a.array.argsort(desc));
    let mut data = idx.as_slice().to_vec();
    for x in &mut data {
        *x += 1;
    }
    let out = lua_try!(L, ArrayI64::from_shape_vec(vec![data.len()], data));
    unsafe { push_array_i64(L, out) };
    1
}
pub unsafe extern "C" fn a_i64_take(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let idx = unsafe { &*check_array_i64(L, 2) };
    let mut zero = idx.array.as_slice().to_vec();
    for x in &mut zero {
        *x -= 1;
    }
    let z = lua_try!(L, ArrayI64::from_shape_vec(vec![zero.len()], zero));
    let out = lua_try!(L, a.array.take(&z));
    unsafe { push_array_i64(L, out) };
    1
}

pub unsafe extern "C" fn a_i64_add(L: *mut lua_State) -> c_int {
    bin_op(L, |a, b| a.add(b), |a, s| a.add_scalar(s), true)
}
pub unsafe extern "C" fn a_i64_sub(L: *mut lua_State) -> c_int {
    bin_op(L, |a, b| a.sub(b), |a, s| a.sub_scalar(s), false)
}
pub unsafe extern "C" fn a_i64_mul(L: *mut lua_State) -> c_int {
    bin_op(L, |a, b| a.mul(b), |a, s| a.mul_scalar(s), true)
}
pub unsafe extern "C" fn a_i64_div(L: *mut lua_State) -> c_int {
    bin_op(L, |a, b| a.div(b), |a, s| a.div_scalar(s), false)
}
pub unsafe extern "C" fn a_i64_unm(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_array_i64(L, a.array.neg()) };
    1
}

unsafe fn bin_op(
    L: *mut lua_State,
    arr_arr: impl Fn(&ArrayI64, &ArrayI64) -> crate::Result<ArrayI64>,
    arr_scalar: impl Fn(&ArrayI64, i64) -> ArrayI64,
    commute_scalar: bool,
) -> c_int {
    let a_ud = unsafe { test_array_i64(L, 1) };
    let b_ud = unsafe { test_array_i64(L, 2) };
    if !a_ud.is_null() && !b_ud.is_null() {
        let o = lua_try!(L, arr_arr(&unsafe { &*a_ud }.array, &unsafe { &*b_ud }.array));
        unsafe { push_array_i64(L, o) };
        return 1;
    }
    if !a_ud.is_null() && unsafe { lua_type(L, 2) } == LUA_TNUMBER {
        let s = unsafe { luaL_checkinteger(L, 2) };
        unsafe { push_array_i64(L, arr_scalar(&unsafe { &*a_ud }.array, s)) };
        return 1;
    }
    if commute_scalar && !b_ud.is_null() && unsafe { lua_type(L, 1) } == LUA_TNUMBER {
        let s = unsafe { luaL_checkinteger(L, 1) };
        unsafe { push_array_i64(L, arr_scalar(&unsafe { &*b_ud }.array, s)) };
        return 1;
    }
    super::ud::lua_error_msg(L, "i64 op expects ArrayI64 and ArrayI64 or number")
}


pub unsafe extern "C" fn l_where_i64(L: *mut lua_State) -> c_int {
    let c = unsafe { &*check_array_i64(L, 1) };
    let x = unsafe { &*check_array_i64(L, 2) };
    let y = unsafe { &*check_array_i64(L, 3) };
    let o = lua_try!(L, ArrayI64::where_cond(&c.array, &x.array, &y.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn l_concatenate_i64(L: *mut lua_State) -> c_int {
    let axis = unsafe { luaL_checkinteger(L, 1) };
    if axis < 0 {
        return super::ud::lua_error_msg(L, "concatenate_i64 axis must be >= 0");
    }
    let top = unsafe { lua_gettop(L) };
    if top < 3 {
        return super::ud::lua_error_msg(L, "concatenate_i64(axis, a, b, ...) needs arrays");
    }
    let mut owned = Vec::new();
    for i in 2..=top {
        let a = unsafe { &*check_array_i64(L, i) };
        owned.push(&a.array as *const _);
    }
    let refs: Vec<&ArrayI64> = owned.iter().map(|p| unsafe { &**p }).collect();
    let out = lua_try!(L, ArrayI64::concatenate(axis as usize, &refs));
    unsafe { push_array_i64(L, out) };
    1
}
pub unsafe extern "C" fn l_stack_i64(L: *mut lua_State) -> c_int {
    let axis = unsafe { luaL_checkinteger(L, 1) };
    if axis < 0 {
        return super::ud::lua_error_msg(L, "stack_i64 axis must be >= 0");
    }
    let top = unsafe { lua_gettop(L) };
    if top < 3 {
        return super::ud::lua_error_msg(L, "stack_i64(axis, a, b, ...) needs arrays");
    }
    let mut owned = Vec::new();
    for i in 2..=top {
        let a = unsafe { &*check_array_i64(L, i) };
        owned.push(&a.array as *const _);
    }
    let refs: Vec<&ArrayI64> = owned.iter().map(|p| unsafe { &**p }).collect();
    let out = lua_try!(L, ArrayI64::stack(axis as usize, &refs));
    unsafe { push_array_i64(L, out) };
    1
}
pub unsafe extern "C" fn l_broadcast_to_i64(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let shape = lua_try!(L, unsafe { shape_from_args(L, 2) });
    let o = lua_try!(L, a.array.broadcast_to(shape));
    unsafe { push_array_i64(L, o) };
    1
}

pub unsafe extern "C" fn a_i64_ne(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_type(L, 2) } == LUA_TNUMBER {
        let s = unsafe { luaL_checkinteger(L, 2) };
        unsafe { push_array_i64(L, a.array.ne_scalar(s)) };
        return 1;
    }
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.ne(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_le(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_type(L, 2) } == LUA_TNUMBER {
        let s = unsafe { luaL_checkinteger(L, 2) };
        unsafe { push_array_i64(L, a.array.le_scalar(s)) };
        return 1;
    }
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.le(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_gt(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_type(L, 2) } == LUA_TNUMBER {
        let s = unsafe { luaL_checkinteger(L, 2) };
        unsafe { push_array_i64(L, a.array.gt_scalar(s)) };
        return 1;
    }
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.gt(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_ge(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_type(L, 2) } == LUA_TNUMBER {
        let s = unsafe { luaL_checkinteger(L, 2) };
        unsafe { push_array_i64(L, a.array.ge_scalar(s)) };
        return 1;
    }
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.ge(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_sign(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_array_i64(L, a.array.sign()) };
    1
}
pub unsafe extern "C" fn a_i64_clip(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let lo = unsafe { luaL_checkinteger(L, 2) };
    let hi = unsafe { luaL_checkinteger(L, 3) };
    let o = lua_try!(L, a.array.clip(lo, hi));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_cumsum(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_array_i64(L, a.array.cumsum()) };
    1
}
pub unsafe extern "C" fn a_i64_argmin(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let i = lua_try!(L, a.array.argmin());
    unsafe { lua_pushinteger(L, (i + 1) as lua_Integer) }; // 1-based
    1
}
pub unsafe extern "C" fn a_i64_argmax(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let i = lua_try!(L, a.array.argmax());
    unsafe { lua_pushinteger(L, (i + 1) as lua_Integer) };
    1
}
pub unsafe extern "C" fn a_i64_any(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_gettop(L) } >= 2 {
        let axis = (unsafe { luaL_checkinteger(L, 2) }) as usize;
        let o = lua_try!(L, a.array.any_axis(axis));
        unsafe { push_array_i64(L, o) };
    } else {
        unsafe { lua_pushboolean(L, a.array.any() as i32) };
    }
    1
}
pub unsafe extern "C" fn a_i64_all(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_gettop(L) } >= 2 {
        let axis = (unsafe { luaL_checkinteger(L, 2) }) as usize;
        let o = lua_try!(L, a.array.all_axis(axis));
        unsafe { push_array_i64(L, o) };
    } else {
        unsafe { lua_pushboolean(L, a.array.all() as i32) };
    }
    1
}
pub unsafe extern "C" fn a_i64_var(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let ddof = if unsafe { lua_gettop(L) } >= 2 {
        (unsafe { luaL_checkinteger(L, 2) }) as usize
    } else {
        0
    };
    let v = lua_try!(L, a.array.var(ddof));
    unsafe { lua_pushnumber(L, v) };
    1
}
pub unsafe extern "C" fn a_i64_std(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let ddof = if unsafe { lua_gettop(L) } >= 2 {
        (unsafe { luaL_checkinteger(L, 2) }) as usize
    } else {
        0
    };
    let v = lua_try!(L, a.array.std(ddof));
    unsafe { lua_pushnumber(L, v) };
    1
}
pub unsafe extern "C" fn a_i64_slice(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let start = unsafe { luaL_checkinteger(L, 2) };
    let stop = unsafe { luaL_checkinteger(L, 3) };
    if start < 1 || stop < start {
        return super::ud::lua_error_msg(L, "slice uses 1-based half-open [start, stop)");
    }
    // Lua half-open 1-based → 0-based [start-1, stop-1)
    let o = lua_try!(L, a.array.slice((start as usize) - 1, (stop as usize) - 1));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_rows(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let start = unsafe { luaL_checkinteger(L, 2) };
    let stop = unsafe { luaL_checkinteger(L, 3) };
    if start < 1 || stop < start {
        return super::ud::lua_error_msg(L, "rows uses 1-based half-open [start, stop)");
    }
    let o = lua_try!(L, a.array.rows((start as usize) - 1, (stop as usize) - 1));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_row(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let i = unsafe { luaL_checkinteger(L, 2) };
    if i < 1 {
        return super::ud::lua_error_msg(L, "row index must be >= 1");
    }
    let o = lua_try!(L, a.array.row((i as usize) - 1));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_col(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let j = unsafe { luaL_checkinteger(L, 2) };
    if j < 1 {
        return super::ud::lua_error_msg(L, "col index must be >= 1");
    }
    let o = lua_try!(L, a.array.col((j as usize) - 1));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_diagonal(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let o = lua_try!(L, a.array.diagonal());
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_trace(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let v = lua_try!(L, a.array.trace());
    unsafe { lua_pushinteger(L, v) };
    1
}
pub unsafe extern "C" fn a_i64_broadcast_to(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let shape = lua_try!(L, unsafe { shape_from_args(L, 2) });
    let o = lua_try!(L, a.array.broadcast_to(shape));
    unsafe { push_array_i64(L, o) };
    1
}


pub unsafe extern "C" fn l_matmul_i64(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let c = lua_try!(L, crate::linalg::i64_ops::matmul(&a.array, &b.array));
    unsafe { push_array_i64(L, c) };
    1
}
pub unsafe extern "C" fn l_matmul_at_i64(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let c = lua_try!(L, crate::linalg::i64_ops::matmul_at(&a.array, &b.array));
    unsafe { push_array_i64(L, c) };
    1
}
pub unsafe extern "C" fn l_matmul_bt_i64(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let c = lua_try!(L, crate::linalg::i64_ops::matmul_bt(&a.array, &b.array));
    unsafe { push_array_i64(L, c) };
    1
}
pub unsafe extern "C" fn l_dot_i64(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let d = lua_try!(L, crate::linalg::i64_ops::dot(&a.array, &b.array));
    unsafe { lua_pushinteger(L, d) };
    1
}
pub unsafe extern "C" fn l_norm_i64(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let n = lua_try!(L, crate::linalg::i64_ops::norm(&a.array));
    unsafe { lua_pushnumber(L, n) };
    1
}
pub unsafe extern "C" fn l_transpose_i64(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let t = lua_try!(L, crate::linalg::i64_ops::transpose(&a.array));
    unsafe { push_array_i64(L, t) };
    1
}


pub unsafe extern "C" fn a_i64_bitand(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.bitand(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_bitor(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.bitor(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_bitxor(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.bitxor(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_bitnot(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_array_i64(L, a.array.bitnot()) };
    1
}
pub unsafe extern "C" fn a_i64_shift_left(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { luaL_checkinteger(L, 2) };
    if b < 0 { return super::ud::lua_error_msg(L, "shift_left bits must be >= 0"); }
    unsafe { push_array_i64(L, a.array.shift_left(b as u32)) };
    1
}
pub unsafe extern "C" fn a_i64_shift_right(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { luaL_checkinteger(L, 2) };
    if b < 0 { return super::ud::lua_error_msg(L, "shift_right bits must be >= 0"); }
    unsafe { push_array_i64(L, a.array.shift_right(b as u32)) };
    1
}
pub unsafe extern "C" fn a_i64_rem(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_type(L, 2) } == LUA_TNUMBER {
        let s = unsafe { luaL_checkinteger(L, 2) };
        unsafe { push_array_i64(L, a.array.rem_scalar(s)) };
        return 1;
    }
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.rem(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_unique(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let o = lua_try!(L, a.array.unique());
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_unique_counts(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let (u, c) = lua_try!(L, a.array.unique_counts());
    unsafe { push_array_i64(L, u) };
    unsafe { push_array_i64(L, c) };
    2
}
pub unsafe extern "C" fn a_i64_isin(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let t = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.isin(&t.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_bincount(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let minlength = if unsafe { lua_gettop(L) } >= 2 {
        let v = unsafe { luaL_checkinteger(L, 2) };
        if v < 0 { return super::ud::lua_error_msg(L, "bincount minlength >= 0"); }
        v as usize
    } else { 0 };
    let o = lua_try!(L, a.array.bincount(minlength));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_searchsorted(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let side_right = if unsafe { lua_gettop(L) } >= 3 {
        unsafe { lua_toboolean(L, 3) != 0 }
    } else {
        false
    };
    if unsafe { test_array_i64(L, 2) }.is_null() {
        let v = unsafe { luaL_checkinteger(L, 2) };
        let i = lua_try!(L, a.array.searchsorted(v, side_right));
        unsafe { lua_pushinteger(L, (i + 1) as lua_Integer) };
        1
    } else {
        let vals = unsafe { &*check_array_i64(L, 2) };
        let o = lua_try!(L, a.array.searchsorted_array(&vals.array, side_right));
        let mut d = o.as_slice().to_vec();
        for x in &mut d {
            *x += 1;
        }
        let out = lua_try!(L, ArrayI64::from_shape_vec(vec![d.len()], d));
        unsafe { push_array_i64(L, out) };
        1
    }
}
pub unsafe extern "C" fn a_i64_sort(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let desc = if unsafe { lua_gettop(L) } >= 2 {
        unsafe { lua_toboolean(L, 2) != 0 }
    } else { false };
    let o = lua_try!(L, a.array.sort(desc));
    unsafe { push_array_i64(L, o) };
    1
}


pub unsafe extern "C" fn a_i64_power(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    if unsafe { lua_type(L, 2) } == LUA_TNUMBER {
        let e = unsafe { luaL_checkinteger(L, 2) };
        if e < 0 {
            return super::ud::lua_error_msg(L, "power exponent must be >= 0");
        }
        unsafe { push_array_i64(L, a.array.power_scalar(e as u32)) };
        return 1;
    }
    let e = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.power(&e.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_divmod(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let (q, r) = lua_try!(L, a.array.divmod(&b.array));
    unsafe { push_array_i64(L, q) };
    unsafe { push_array_i64(L, r) };
    2
}
pub unsafe extern "C" fn a_i64_gcd(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.gcd(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_lcm(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let b = unsafe { &*check_array_i64(L, 2) };
    let o = lua_try!(L, a.array.lcm(&b.array));
    unsafe { push_array_i64(L, o) };
    1
}
pub unsafe extern "C" fn a_i64_count_ones(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_array_i64(L, a.array.count_ones()) };
    1
}
pub unsafe extern "C" fn a_i64_leading_zeros(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_array_i64(L, a.array.leading_zeros()) };
    1
}
pub unsafe extern "C" fn a_i64_trailing_zeros(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    unsafe { push_array_i64(L, a.array.trailing_zeros()) };
    1
}


pub unsafe extern "C" fn a_i64_median(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let m = lua_try!(L, a.array.median());
    unsafe { lua_pushnumber(L, m) };
    1
}
pub unsafe extern "C" fn a_i64_quantile(L: *mut lua_State) -> c_int {
    let a = unsafe { &*check_array_i64(L, 1) };
    let q = unsafe { luaL_checknumber(L, 2) };
    let v = lua_try!(L, a.array.quantile(q));
    unsafe { lua_pushnumber(L, v) };
    1
}

/// Install `ArrayI64` metatable.
pub unsafe fn install_metatable(L: *mut lua_State) {
    unsafe {
        luaL_newmetatable(L, ARRAY_I64_MT.as_ptr());
        lua_pushvalue(L, -1);
        lua_setfield(L, -2, c"__index".as_ptr());
        let methods: [(&std::ffi::CStr, unsafe extern "C" fn(*mut lua_State) -> c_int); 69] = [
            (c"__gc", a_i64_gc),
            (c"__len", a_i64_len),
            (c"__tostring", a_i64_tostring),
            (c"__add", a_i64_add),
            (c"__sub", a_i64_sub),
            (c"__mul", a_i64_mul),
            (c"__div", a_i64_div),
            (c"__unm", a_i64_unm),
            (c"dtype", a_i64_dtype),
            (c"shape", a_i64_shape),
            (c"rank", a_i64_rank),
            (c"get", a_i64_get),
            (c"set", a_i64_set),
            (c"sum", a_i64_sum),
            (c"mean", a_i64_mean),
            (c"min", a_i64_min),
            (c"max", a_i64_max),
            (c"copy", a_i64_copy),
            (c"reshape", a_i64_reshape),
            (c"fill", a_i64_fill),
            (c"abs", a_i64_abs),
            (c"sign", a_i64_sign),
            (c"clip", a_i64_clip),
            (c"cumsum", a_i64_cumsum),
            (c"argmin", a_i64_argmin),
            (c"argmax", a_i64_argmax),
            (c"any", a_i64_any),
            (c"all", a_i64_all),
            (c"var", a_i64_var),
            (c"std", a_i64_std),
            (c"transpose", a_i64_transpose),
            (c"to_f64", a_i64_to_f64),
            (c"eq", a_i64_eq),
            (c"ne", a_i64_ne),
            (c"lt", a_i64_lt),
            (c"le", a_i64_le),
            (c"gt", a_i64_gt),
            (c"ge", a_i64_ge),
            (c"argsort", a_i64_argsort),
            (c"take", a_i64_take),
            (c"slice", a_i64_slice),
            (c"rows", a_i64_rows),
            (c"row", a_i64_row),
            (c"col", a_i64_col),
            (c"diagonal", a_i64_diagonal),
            (c"trace", a_i64_trace),
            (c"broadcast_to", a_i64_broadcast_to),
            (c"bitand", a_i64_bitand),
            (c"bitor", a_i64_bitor),
            (c"bitxor", a_i64_bitxor),
            (c"bitnot", a_i64_bitnot),
            (c"shift_left", a_i64_shift_left),
            (c"shift_right", a_i64_shift_right),
            (c"rem", a_i64_rem),
            (c"unique", a_i64_unique),
            (c"unique_counts", a_i64_unique_counts),
            (c"isin", a_i64_isin),
            (c"bincount", a_i64_bincount),
            (c"searchsorted", a_i64_searchsorted),
            (c"sort", a_i64_sort),
            (c"power", a_i64_power),
            (c"divmod", a_i64_divmod),
            (c"gcd", a_i64_gcd),
            (c"lcm", a_i64_lcm),
            (c"count_ones", a_i64_count_ones),
            (c"leading_zeros", a_i64_leading_zeros),
            (c"trailing_zeros", a_i64_trailing_zeros),
            (c"median", a_i64_median),
            (c"quantile", a_i64_quantile),
        ];
        for (name, f) in methods {
            lua_pushcfunction(L, Some(f));
            lua_setfield(L, -2, name.as_ptr());
        }
        lua_pop(L, 1);
    }
}

/// Register i64 constructors on the module table (top of stack).
pub unsafe fn register_module_funcs(L: *mut lua_State) {
    unsafe {
        let funcs: [(&std::ffi::CStr, unsafe extern "C" fn(*mut lua_State) -> c_int); 18] = [
            (c"zeros_i64", l_zeros_i64),
            (c"ones_i64", l_ones_i64),
            (c"full_i64", l_full_i64),
            (c"arange_i64", l_arange_i64),
            (c"array_i64", l_array_i64),
            (c"eye_i64", l_eye_i64),
            (c"diag_i64", l_diag_i64),
            (c"outer_i64", l_outer_i64),
            (c"where_i64", l_where_i64),
            (c"concatenate_i64", l_concatenate_i64),
            (c"stack_i64", l_stack_i64),
            (c"broadcast_to_i64", l_broadcast_to_i64),
            (c"matmul_i64", l_matmul_i64),
            (c"matmul_at_i64", l_matmul_at_i64),
            (c"matmul_bt_i64", l_matmul_bt_i64),
            (c"dot_i64", l_dot_i64),
            (c"norm_i64", l_norm_i64),
            (c"transpose_i64", l_transpose_i64),
        ];
        for (name, f) in funcs {
            lua_pushcfunction(L, Some(f));
            lua_setfield(L, -2, name.as_ptr());
        }
    }
}
