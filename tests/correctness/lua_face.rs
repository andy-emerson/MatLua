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

#[test]
fn m5_remaining_face() {
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local m = ml.array({{1,2,3},{4,5,6}})
local row = ml.array({10,20,30})
local s = m + row
assert(s:get(1,1) == 11 and s:get(2,3) == 36)
local mask = m:lt(4)
assert(mask:get(1,1) == 1 and mask:get(2,1) == 0)
local a = ml.array({1, 0/0, 3, 5})
assert(math.abs(a:nansum() - 9) < 1e-12)
assert(a:nanmin() == 1 and a:nanmax() == 5)
local v = ml.array({1,2,3,4,5})
local sl = v:slice(2, 5)  -- 1-based half-open → 2,3,4
assert(#sl == 3 and sl:get(1) == 2 and sl:get(3) == 4)
assert(m:row(2):get(1) == 4)
assert(m:col(1):get(2) == 4)
local b = ml.broadcast_to(ml.array({1,2}), 2, 2)
assert(b:shape()[1] == 2 and b:get(2,2) == 2)
"#,
    )
    .unwrap();
}

#[test]
fn m6_tier2_face() {
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local m = ml.array({{1,2,3},{4,5,6}})
local s = m:sum(0)
assert(s:get(1) == 5 and s:get(3) == 9)
local x = ml.array({{1,2,3},{2,4,6}})
local c = ml.cov(x)
assert(math.abs(c:get(1,2) - 2) < 1e-9)
local r = ml.corrcoef(x)
assert(math.abs(r:get(1,2) - 1) < 1e-9)
local v = ml.array({3,1,4,2})
local idx = v:argsort()
assert(idx:get(1) == 2) -- 1-based
local t = v:take(idx)
assert(t:get(1) == 1 and t:get(4) == 4)
local d = ml.diag(ml.array({1,2}))
assert(d:shape()[1] == 2 and d:get(1,1) == 1 and d:get(1,2) == 0)
assert(d:trace() == 3)
local o = ml.outer(ml.array({1,2}), ml.array({3,4}))
assert(o:get(2,2) == 8)
local mask = ml.array({{0,1},{0,0}})
assert(mask:any() and not mask:all())
"#,
    )
    .unwrap();
}
