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
