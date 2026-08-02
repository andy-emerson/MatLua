//! Integration correctness for the Lua face (`lua` feature).

#![cfg(feature = "lua")]

use matlua::lua::Lua;

#[test]
fn require_solve_and_matmul() {
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local A = ml.array({{3, 1}, {1, 2}})
local b = ml.array({9, 8})
local x = ml.solve(A, b)
assert(math.abs(x:get(1) - 2) < 1e-9)
assert(math.abs(x:get(2) - 3) < 1e-9)
local M = ml.array({{1, 2}, {3, 4}})
local v = ml.array({1, 1})
local y = ml.matmul(M, v)
assert(y:rank() == 1)
assert(math.abs(y:get(1) - 3) < 1e-12)
assert(math.abs(y:get(2) - 7) < 1e-12)
"#,
    )
    .unwrap();
}

#[test]
fn one_based_get_set() {
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local a = ml.zeros(2, 2)
a:set(1, 1, 5)
a:set(2, 2, 7)
assert(a:get(1, 1) == 5)
assert(a:get(2, 2) == 7)
local ok, err = pcall(function() a:get(0, 1) end)
assert(not ok)
"#,
    )
    .unwrap();
}

#[test]
fn matmul_at_and_normal_eq_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local X = ml.array({{1,0},{1,1},{1,2},{1,3}})
local y = ml.array({0,1,2,3})
local long = ml.matmul(X:transpose(), y)
local short = ml.matmul_at(X, y)
assert(#long == #short)
for i = 1, #long do
  assert(math.abs(long:get(i) - short:get(i)) < 1e-9)
end
local b1 = ml.normal_eq(X, y)
local b2 = ml.solve(ml.matmul(X:transpose(), X), ml.matmul(X:transpose(), y))
for i = 1, #b1 do
  assert(math.abs(b1:get(i) - b2:get(i)) < 1e-8)
end
"#,
    )
    .unwrap();
}

#[test]
fn factorizations_and_empty_shape_error() {
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local A = ml.array({{2, 0.5}, {0.5, 1}})
local L = ml.cholesky(A)
assert(L:rank() == 2)
local Q, R = ml.qr(A)
assert(Q:rank() == 2 and R:rank() == 2)
local U, s, V = ml.svd(A)
assert(s:rank() == 1)
local ok, err = pcall(function() return ml.zeros({}) end)
assert(not ok)
"#,
    )
    .unwrap();
}

#[test]
fn reshape_copy_and_high_rank_index() {
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local a = ml.arange(0, 6)
local b = a:reshape(2, 3)
assert(b:get(1, 1) == 0)
assert(b:get(2, 3) == 5)
-- rank-3 get/set
local c = ml.zeros(2, 2, 2)
c:set(1, 1, 1, 3.5)
assert(c:get(1, 1, 1) == 3.5)
local d = a:copy()
d:fill(9)
assert(a:get(1) == 0)
assert(d:get(1) == 9)
"#,
    )
    .unwrap();
}

#[test]
fn lstsq_eigh_pinv_face() {
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local X = ml.array({{1,0},{1,1},{1,2},{1,3}})
local y = ml.array({1,3,5,7})
local b = ml.lstsq(X, y)
assert(b:rank() == 1 and #b == 2)
local A = ml.array({{2,0.5},{0.5,1}})
local w, v = ml.eigh(A)
assert(w:rank() == 1 and v:rank() == 2)
local P = ml.pinv(X)
assert(P:shape()[1] == 2 and P:shape()[2] == 4)
"#,
    )
    .unwrap();
}

#[test]
fn tier1_ufuncs_face() {
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local a = ml.array({-1, 4, 9})
assert(a:abs():get(1) == 1)
assert(math.abs(a:sqrt():get(2) - 2) < 1e-12)
assert(a:argmin() == 1)
assert(a:argmax() == 3)
local b = ml.array({1, 2, 3})
assert(math.abs(b:var(0) - 2/3) < 1e-12)
local c = ml.array({1, 0, 1})
local x = ml.array({10, 20, 30})
local y = ml.array({1, 2, 3})
local w = ml.where(c, x, y)
assert(w:get(1) == 10 and w:get(2) == 2 and w:get(3) == 30)
"#,
    )
    .unwrap();
}
