//! End-to-end composed-path bench: long vs short normal equations / AᵀB.
//!
//! ```text
//! cargo test --release --features lua --test compose_chain -- --run --sizes 64,256,1024
//! python3 tests/bench/numpy_compose.py --sizes 64,256,1024
//! python3 tests/bench/compare_compose.py
//! ```

use std::env;
use std::hint::black_box;
use std::time::Instant;

use matlua::array::Array;
use matlua::linalg::{matmul, matmul_at, normal_eq, solve, transpose};
use matlua::lua::Lua;

fn median(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // True median: average the middle pair on even counts.
    let m = v.len() / 2;
    if v.len() % 2 == 0 && v.len() >= 2 {
        (v[m - 1] + v[m]) / 2.0
    } else {
        v[m]
    }
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
    median(&samples)
}

/// Tall design matrix m×k with m = 4k (least-squares style).
fn design(k: usize) -> (Array, Array) {
    let m = 4 * k;
    let mut data = Vec::with_capacity(m * k);
    let mut x = 0.001_f64;
    for _ in 0..m * k {
        data.push(x);
        x += 0.000017;
    }
    let x = Array::from_shape_vec(vec![m, k], data).unwrap();
    let mut y = Vec::with_capacity(m);
    for i in 0..m {
        y.push(0.1 + i as f64 * 0.01);
    }
    let y = Array::from_shape_vec(vec![m], y).unwrap();
    (x, y)
}

fn emit(face: &str, op: &str, n: usize, ms: f64) {
    println!("{face}\t{op}\t{n}\t{ms:.6}");
}

fn budget(n: usize) -> (usize, usize) {
    match n {
        0..=64 => (40, 8),
        65..=256 => (12, 3),
        _ => (5, 2),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if !args.iter().any(|a| a == "--run") {
        eprintln!("compose_chain: skipped (pass --run)");
        return;
    }
    let sizes = args
        .windows(2)
        .find(|w| w[0] == "--sizes")
        .map(|w| {
            w[1]
                .split(',')
                .filter_map(|s| s.parse().ok())
                .collect::<Vec<usize>>()
        })
        .unwrap_or_else(|| vec![64, 256, 1024]);

    println!("face\top\tn\tms");

    for &k in &sizes {
        let (x, y) = design(k);
        let (iters, warm) = budget(k);

        // matmul_at vs transpose+matmul on XᵀX (square k×k result work dominates with tall m)
        let ms = time_ms(iters, warm, || {
            black_box(matmul(&transpose(&x).unwrap(), &x).unwrap());
        });
        emit("rust", "xtx_long", k, ms);
        let ms = time_ms(iters, warm, || {
            black_box(matmul_at(&x, &x).unwrap());
        });
        emit("rust", "xtx_short", k, ms);

        let ms = time_ms(iters, warm, || {
            let _ = solve(
                &matmul(&transpose(&x).unwrap(), &x).unwrap(),
                &matmul(&transpose(&x).unwrap(), &y).unwrap(),
            )
            .unwrap();
        });
        emit("rust", "normal_eq_long", k, ms);
        let ms = time_ms(iters, warm, || {
            black_box(normal_eq(&x, &y).unwrap());
        });
        emit("rust", "normal_eq_short", k, ms);
    }

    // Lua face
    let lua = Lua::new().unwrap();
    lua.do_string(r#"ml = require "matlua""#).unwrap();
    for &k in &sizes {
        let m = 4 * k;
        let (iters, warm) = budget(k);
        // build X,y once in Lua
        lua.do_string(&format!(
            r#"
local m, k = {m}, {k}
X = ml.zeros(m, k)
y = ml.zeros(m)
local v = 0.001
for i = 1, m do
  for j = 1, k do
    X:set(i, j, v)
    v = v + 0.000017
  end
  y:set(i, 0.1 + (i-1) * 0.01)
end
function __long_xtx() return ml.matmul(X:transpose(), X) end
function __short_xtx() return ml.matmul_at(X, X) end
function __long_ne()
  return ml.solve(ml.matmul(X:transpose(), X), ml.matmul(X:transpose(), y))
end
function __short_ne() return ml.normal_eq(X, y) end
"#
        ))
        .unwrap();

        for (op, g) in [
            ("xtx_long", "__long_xtx"),
            ("xtx_short", "__short_xtx"),
            ("normal_eq_long", "__long_ne"),
            ("normal_eq_short", "__short_ne"),
        ] {
            for _ in 0..warm {
                lua.call_global(g).unwrap();
            }
            let mut samples = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                lua.call_global(g).unwrap();
                samples.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            emit("lua", op, k, median(&samples));
        }
    }
}
