//! i64 → f64 promote-out microbench (Rust + Lua). NumPy twin: numpy_i64_promote.py
//!
//! Ops: mean, std, median, quantile, norm, solve, cholesky, qr (from_i64 / faer).

use std::env;
use std::hint::black_box;
use std::time::Instant;

use matlua::array::ArrayI64;
use matlua::linalg::{self, i64_ops};
use matlua::lua::Lua;

fn median_t(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ms(iters: usize, warm: usize, mut body: impl FnMut()) -> f64 {
    for _ in 0..warm {
        body();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        body();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    median_t(&samples)
}

fn time_lua(lua: &Lua, setup: &str, body: &str, iters: usize, warm: usize) -> f64 {
    lua.do_string(&format!("{setup}\nfunction __bench_op()\n{body}\nend\n"))
        .unwrap();
    for _ in 0..warm {
        lua.call_global("__bench_op").unwrap();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        lua.call_global("__bench_op").unwrap();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    median_t(&samples)
}

fn dense(n: usize) -> ArrayI64 {
    let mut data = Vec::with_capacity(n * n);
    let mut x: i64 = 1;
    for _ in 0..n * n {
        data.push(x);
        x = x.wrapping_add(17);
    }
    ArrayI64::from_shape_vec(vec![n, n], data).unwrap()
}

fn spd_i64(n: usize) -> ArrayI64 {
    // Small-entry integer Gram matrix so AᵀA does **not** wrap at n=1024
    // (large `dense()` products wrap → non-SPD after f64 promote → cholesky fails).
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            data.push(((i + 2 * j) % 7) as i64);
        }
    }
    let a = ArrayI64::from_shape_vec(vec![n, n], data).unwrap();
    let at = a.transpose().unwrap();
    let mut s = i64_ops::matmul(&at, &a).unwrap();
    // s + (n+1)·I  → strictly diagonally dominant / SPD over reals
    let bump = (n as i64) + 1;
    for i in 0..n {
        let v = s.get(&[i, i]).unwrap().wrapping_add(bump);
        s.set(&[i, i], v).unwrap();
    }
    s
}

fn vec_n(n: usize) -> ArrayI64 {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push((i as i64).wrapping_mul(3).wrapping_add(1));
    }
    ArrayI64::from_shape_vec(vec![n], data).unwrap()
}

fn budget(n: usize, heavy: bool) -> (usize, usize) {
    if heavy {
        if n >= 1024 {
            (2, 1)
        } else if n >= 256 {
            (4, 1)
        } else {
            (10, 2)
        }
    } else if n >= 1024 {
        (6, 2)
    } else if n >= 256 {
        (12, 3)
    } else {
        (30, 5)
    }
}

fn emit(face: &str, op: &str, n: usize, ms: f64) {
    println!("{face}\t{op}\t{n}\t{ms:.6}");
}

fn lua_build(n: usize) -> String {
    format!(
        r#"
local n = {n}
A = ml.zeros_i64(n, n)
local x = 1
for i = 1, n do
  for j = 1, n do
    A:set(i, j, x)
    x = x + 17
  end
end
V = ml.zeros_i64(n)
for i = 1, n do V:set(i, (i-1)*3+1) end
-- Small-entry Gram + diagonal (match Rust spd_i64); avoid wrap → non-SPD
local G = ml.zeros_i64(n, n)
for i = 1, n do
  for j = 1, n do
    G:set(i, j, (i - 1 + 2 * (j - 1)) % 7)
  end
end
local Gt = G:transpose()
S = ml.matmul(Gt, G)
for i = 1, n do
  S:set(i, i, S:get(i, i) + n + 1)
end
rhs = V
"#
    )
}

fn bench_rust(sizes: &[usize]) {
    for &n in sizes {
        let a = dense(n);
        let s = spd_i64(n);
        let v = vec_n(n);
        let (it, wrm) = budget(n, false);
        let (ith, wrmh) = budget(n, true);

        emit("rust", "mean", n, time_ms(it, wrm, || {
            black_box(a.mean().unwrap());
        }));
        emit("rust", "std", n, time_ms(it, wrm, || {
            black_box(a.std(0).unwrap());
        }));
        emit("rust", "median", n, time_ms(it, wrm, || {
            black_box(a.median().unwrap());
        }));
        emit("rust", "quantile", n, time_ms(it, wrm, || {
            black_box(a.quantile(0.75).unwrap());
        }));
        emit("rust", "norm", n, time_ms(it, wrm, || {
            black_box(i64_ops::norm(&a).unwrap());
        }));
        emit("rust", "solve", n, time_ms(ith, wrmh, || {
            black_box(linalg::from_i64::solve(&s, &v).unwrap());
        }));
        emit("rust", "cholesky", n, time_ms(ith, wrmh, || {
            black_box(linalg::from_i64::cholesky(&s).unwrap());
        }));
        emit("rust", "qr", n, time_ms(ith, wrmh, || {
            black_box(linalg::from_i64::qr(&a).unwrap());
        }));
    }
}

fn bench_lua(sizes: &[usize]) {
    let lua = Lua::new().unwrap();
    lua.do_string(r#"ml = require "matlua""#).unwrap();
    for &n in sizes {
        let build = lua_build(n);
        let (it, wrm) = budget(n, false);
        let (ith, wrmh) = budget(n, true);
        emit("lua", "mean", n, time_lua(&lua, &build, "return A:mean()", it, wrm));
        emit("lua", "std", n, time_lua(&lua, &build, "return A:std()", it, wrm));
        emit("lua", "median", n, time_lua(&lua, &build, "return A:median()", it, wrm));
        emit("lua", "quantile", n, time_lua(&lua, &build, "return A:quantile(0.75)", it, wrm));
        emit("lua", "norm", n, time_lua(&lua, &build, "return ml.norm(A)", it, wrm));
        emit("lua", "solve", n, time_lua(&lua, &build, "return ml.solve(S, rhs)", ith, wrmh));
        emit("lua", "cholesky", n, time_lua(&lua, &build, "return ml.cholesky(S)", ith, wrmh));
        emit("lua", "qr", n, time_lua(&lua, &build, "return ml.qr(A)", ith, wrmh));
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if !args.iter().any(|a| a == "--run") {
        eprintln!("i64_promote: skipped (pass --run)");
        return;
    }
    let sizes: Vec<usize> = if let Some(i) = args.iter().position(|a| a == "--sizes") {
        args.get(i + 1)
            .map(|s| s.split(',').filter_map(|p| p.parse().ok()).collect())
            .unwrap_or_else(|| vec![64, 256, 1024])
    } else {
        vec![64, 256, 1024]
    };
    eprintln!("# i64→f64 promote-out. sizes={sizes:?}");
    println!("face\top\tn\tms");
    bench_rust(&sizes);
    eprintln!("# Lua face");
    bench_lua(&sizes);
}
