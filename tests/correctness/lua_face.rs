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
