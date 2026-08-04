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
local xc = ml.cholesky_solve(A, b)
assert(math.abs(xc:get(1) - 2) < 1e-9)
assert(math.abs(xc:get(2) - 3) < 1e-9)
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
-- matmul_bt: A @ Bᵀ
local A = ml.array({{1,2,3},{4,5,6}})
local B = ml.array({{0.5,1.5,2.5},{3.5,4.5,5.5}})
local bt = ml.matmul_bt(A, B)
local long_bt = ml.matmul(A, B:transpose())
assert(bt:rank() == 2)
for i = 1, 2 do
  for j = 1, 2 do
    assert(math.abs(bt:get(i,j) - long_bt:get(i,j)) < 1e-12)
  end
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
local bm = ml.array({1,2}):broadcast_to(2, 2)  -- method form, parity with i64
assert(bm:shape()[1] == 2 and bm:get(2,2) == 2)
assert(ml.array({1,2}):dtype() == "f64")
-- i64 var_axis/std_axis: 1-based axis, f64 results (parity with f64 face)
local mi = ml.array_i64({{1,2},{3,4}})
local vi = mi:var_axis(1)          -- reduce down rows, ddof 0 → {1, 1}
assert(math.abs(vi:get(1) - 1) < 1e-12 and math.abs(vi:get(2) - 1) < 1e-12)
local si = mi:std_axis(2)          -- across columns → {0.5, 0.5}
assert(math.abs(si:get(1) - 0.5) < 1e-12 and math.abs(si:get(2) - 0.5) < 1e-12)
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
-- Lua-face axes are 1-based: axis 1 reduces over rows (NumPy axis 0).
local s = m:sum(1)
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

#[test]
fn m7_i64_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local a = ml.array_i64({{1,2,3},{4,5,6}})
assert(a:dtype() == "i64")
assert(a:rank() == 2)
assert(a:sum() == 21)
assert(a:get(1,1) == 1 and a:get(2,3) == 6)
local b = ml.arange_i64(0, 5)
assert(#b == 5 and b:get(1) == 0 and b:get(5) == 4)
local c = a + ml.full_i64({2,3}, 1)
assert(c:get(1,1) == 2)
local f = a:to_f64()
assert(type(f:sum()) == "number")
local z = ml.zeros_i64(3)
assert(z:sum() == 0)
-- Lua-face axes are 1-based: axis 1 reduces over rows (NumPy axis 0).
local s = a:sum(1)
assert(s:get(1) == 5 and s:get(3) == 9)
local idx = ml.array_i64({3,1,2}):argsort()
assert(idx:get(1) == 2 and idx:get(2) == 3 and idx:get(3) == 1)
local t = ml.array_i64({10,20,30}):take(idx)
assert(t:get(1) == 20 and t:get(2) == 30 and t:get(3) == 10)
"#,
    )
    .unwrap();
}


#[test]
fn m7_i64_extended_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local a = ml.array_i64({1,2,3})
local b = ml.array_i64({4,5,6})
local c = ml.concatenate_i64(1, a, b)
assert(#c == 6 and c:get(6) == 6)
local s = ml.stack_i64(1, a, b)
assert(s:rank() == 2 and s:get(2,3) == 6)
local cond = ml.array_i64({1,0,1})
local w = ml.where_i64(cond, a, b)
assert(w:get(1) == 1 and w:get(2) == 5 and w:get(3) == 3)
assert(a:sign():get(1) == 1)
assert(ml.array_i64({-5,0,7}):clip(0, 5):get(1) == 0)
assert(a:cumsum():get(3) == 6)
assert(a:argmin() == 1 and a:argmax() == 3)
assert(a:any() and not ml.zeros_i64(3):any())
local m = ml.array_i64({{1,0},{0,1}})
assert(not m:all()) -- has zeros
assert(ml.ones_i64(2,2):all())
assert(m:diagonal():get(1) == 1 and m:trace() == 2)
local row = m:row(1)
assert(#row == 2 and row:get(1) == 1)
local col = m:col(2)
assert(col:get(2) == 1)
assert(a:var(0) == 2/3)
local br = ml.array_i64({1,2}):broadcast_to(2, 2)
assert(br:rank() == 2)
assert(a:eq(a):all())
assert(a:lt(b):all())
"#,
    )
    .unwrap();
}

#[test]
fn m7_i64_matmul_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local A = ml.array_i64({{1,2},{3,4}})
local B = ml.array_i64({{5,6},{7,8}})
local C = ml.matmul_i64(A, B)
assert(C:get(1,1) == 19 and C:get(2,2) == 50)
local v = ml.array_i64({1,1})
local Av = ml.matmul_i64(A, v)
assert(#Av == 2 and Av:get(1) == 3 and Av:get(2) == 7)
assert(ml.dot_i64(ml.array_i64({1,2,3}), ml.array_i64({4,5,6})) == 32)
local X = ml.array_i64({{1,0},{1,1},{1,2}})
local y = ml.array_i64({1,2,3})
local xty = ml.matmul_at_i64(X, y)
assert(xty:get(1) == 6 and xty:get(2) == 8)
assert(A:eq(1):get(1,1) == 1 and A:eq(1):get(1,2) == 0)
"#,
    )
    .unwrap();
}

#[test]
fn m7_i64_unique_bits_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local a = ml.array_i64({3,1,2,1,3})
local u = a:unique()
assert(#u == 3 and u:get(1) == 1)
local vals, counts = a:unique_counts()
assert(counts:sum() == 5)
assert(a:isin(ml.array_i64({1,9})):sum() == 2)
assert(ml.array_i64({0,1,1,2}):bincount():get(2) == 2)
local s = ml.array_i64({1,3,5,7})
assert(s:searchsorted(4) == 3)
assert(a:sort():get(1) == 1)
local x = ml.array_i64({12, 10})
assert(x:rem(5):get(1) == 2)
assert(x:bitand(ml.array_i64({10,12})):get(1) == 8)
"#,
    )
    .unwrap();
}

#[test]
fn m7_i64_finish_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local a = ml.array_i64({2, 3, 4})
assert(a:power(2):get(1) == 4)
local q, r = a:divmod(ml.array_i64({2, 2, 3}))
assert(q:get(1) == 1 and r:get(3) == 1)
assert(ml.array_i64({12, 8}):gcd(ml.array_i64({8, 12})):get(1) == 4)
assert(ml.array_i64({4}):lcm(ml.array_i64({6})):get(1) == 12)
assert(ml.array_i64({7}):count_ones():get(1) == 3)
"#,
    )
    .unwrap();
}


#[test]
fn m7_solve_accepts_i64_returns_f64() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local A = ml.array_i64({{2,0},{0,2}})
local b = ml.array_i64({2,4})
local x = ml.solve(A, b)
-- f64 array: dtype via sum being number path; check values
assert(math.abs(x:get(1) - 1) < 1e-9)
assert(math.abs(x:get(2) - 2) < 1e-9)
local w, v = ml.eigh(ml.array_i64({{2,0},{0,3}}))
assert(w:rank() == 1 and v:rank() == 2)
-- integer matmul still returns i64
local C = ml.matmul(ml.array_i64({{1,2},{3,4}}), ml.array_i64({{1,0},{0,1}}))
assert(C:dtype() == "i64")
assert(C:get(2,1) == 3)
"#,
    )
    .unwrap();
}

#[test]
fn m7b_diagnostics_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local A = ml.array({{1,2},{3,4}})
assert(math.abs(ml.det(A) + 2) < 1e-9)
local sign, logabs = ml.slogdet(A)
assert(sign < 0)
assert(ml.matrix_rank(A) == 2)
assert(ml.cond(ml.eye(2)) < 1.01)
local wr, wi = ml.eigvals(ml.eye(3))
assert(#wr == 3)
-- i64 promote
assert(math.abs(ml.det(ml.array_i64({{1,2},{3,4}})) + 2) < 1e-9)
"#,
    )
    .unwrap();
}

#[test]
fn m7b_median_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local a = ml.array({1, 3, 2, 5, 4})
assert(math.abs(a:median() - 3) < 1e-12)
assert(math.abs(a:quantile(0) - 1) < 1e-12)
local m = ml.array({{1,2,3},{4,5,6}})
local row_med = m:median(2) -- axis 2 = columns direction (1-based axis=2 is axis 1 in 0-based)
assert(math.abs(row_med:get(1) - 2) < 1e-12)
assert(math.abs(ml.array_i64({10,20,30}):median() - 20) < 1e-12)
"#,
    )
    .unwrap();
}

#[test]
fn m7b_random_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
ml.seed(123)
local a = ml.random(4)
ml.seed(123)
local b = ml.random(4)
for i=1,4 do assert(a:get(i) == b:get(i)) end
local u = ml.uniform(2, 0, 10)
assert(u:get(1) >= 0 and u:get(1) < 10)
local n = ml.randn(3)
assert(#n == 3)
local ints = ml.integers(5, 0, 3)
assert(ints:dtype() == "i64")
local ch = ml.choice(ml.array({1,2,3}), 4)
assert(#ch == 4)
"#,
    )
    .unwrap();
}

#[test]
fn m7b_indexing_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local a = ml.array({0, 1, 0, 3, 4})
local nz = a:nonzero()
assert(nz:dtype() == "i64" and nz:get(1) == 2)
local c = a:compress(ml.array({0,1,0,1,0}))
assert(#c == 2 and c:get(1) == 1 and c:get(2) == 3)
local b = ml.zeros(4)
b:put(ml.array_i64({2, 4}), ml.array({10, 20}))
assert(b:get(2) == 10 and b:get(4) == 20)
b:put_mask(ml.array({1,0,1,0}), 7)
assert(b:get(1) == 7 and b:get(3) == 7)
local t = a:take(ml.array_i64({2, 4}))
assert(t:get(1) == 1 and t:get(2) == 3)
"#,
    )
    .unwrap();
}

#[test]
fn m7b_out_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local a = ml.array({1,2,3})
local b = ml.array({4,5,6})
local out = ml.zeros(3)
a:add_out(b, out)
assert(out:get(1) == 5 and out:get(3) == 9)
local A = ml.array({{1,2},{3,4}})
local B = ml.array({{5,6},{7,8}})
local C = ml.zeros(2,2)
ml.matmul_out(A, B, C)
assert(C:get(1,1) == 19 and C:get(2,2) == 50)
"#,
    )
    .unwrap();
}

#[test]
fn m7b_i64_quant_parity_face() {
    use matlua::lua::Lua;
    let lua = Lua::new().unwrap();
    lua.do_string(
        r#"
local ml = require "matlua"
local A = ml.array_i64({{1,2,3},{4,5,6}})
local mr = A:median_axis(2)
assert(mr:get(1) == 2 and mr:get(2) == 5)
local a = ml.array_i64({1,-2,3})
local b = ml.array_i64({4,5,6})
local out = ml.zeros_i64(3)
a:sub_out(b, out)
assert(out:get(1) == -3)
a:abs_out(out)
assert(out:get(2) == 2)
local M = ml.array_i64({{1,2},{3,4}})
local N = ml.array_i64({{5,6},{7,8}})
local C = ml.zeros_i64(2,2)
ml.matmul_out(M, N, C)
assert(C:get(1,1) == 19 and C:get(2,2) == 50)
"#,
    )
    .unwrap();
}
