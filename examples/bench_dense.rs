//! Dense desk-math microbench for MatLua (P5).
//!
//! ```text
//! cargo run --release --example bench_dense
//! cargo run --release --features lua --example bench_dense
//! python3 benches/compare.py
//! ```
//!
//! Emits TSV lines: `face\top\tn\tms` (median wall time in milliseconds).

use std::env;
use std::time::Instant;

use matlua::array::Array;
use matlua::linalg;

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn time_ms(iters: usize, warmup: usize, mut body: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        body();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        body();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    median_ms(samples)
}

fn dense(n: usize) -> Array {
    let mut data = Vec::with_capacity(n * n);
    let mut x = 0.001_f64;
    for _ in 0..n * n {
        data.push(x);
        x += 0.000017;
    }
    Array::from_shape_vec(vec![n, n], data).unwrap()
}

fn spd(n: usize) -> Array {
    let a = dense(n);
    let at = linalg::transpose(&a).unwrap();
    let mut s = linalg::matmul(&at, &a).unwrap();
    let e = Array::eye(n).unwrap();
    s = Array::add(&s, &e).unwrap();
    s
}

fn vec_n(n: usize) -> Array {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(0.5 + i as f64 * 0.01);
    }
    Array::from_shape_vec(vec![n], data).unwrap()
}

fn emit(face: &str, op: &str, n: usize, ms: f64) {
    println!("{face}\t{op}\t{n}\t{ms:.6}");
}

fn bench_rust(sizes: &[usize]) {
    for &n in sizes {
        let a = dense(n);
        let b = dense(n);
        let (iters, warm) = if n >= 1024 {
            (5, 2)
        } else if n >= 256 {
            (15, 3)
        } else {
            (40, 5)
        };
        let ms = time_ms(iters, warm, || {
            let _ = linalg::matmul(&a, &b).unwrap();
        });
        emit("rust", "matmul", n, ms);

        let a = spd(n);
        let rhs = vec_n(n);
        let (iters, warm) = if n >= 1024 {
            (5, 2)
        } else if n >= 256 {
            (12, 3)
        } else {
            (30, 5)
        };
        let ms = time_ms(iters, warm, || {
            let _ = linalg::solve(&a, &rhs).unwrap();
        });
        emit("rust", "solve", n, ms);

        let x = dense(n);
        let y = dense(n);
        let (iters, warm) = if n >= 1024 {
            (20, 3)
        } else {
            (50, 5)
        };
        let ms = time_ms(iters, warm, || {
            let _ = Array::add(&x, &y).unwrap();
        });
        emit("rust", "elem_add", n, ms);
    }
}

#[cfg(feature = "lua")]
fn bench_lua(sizes: &[usize]) {
    use matlua::lua::Lua;

    let lua = Lua::new().unwrap();
    for &n in sizes {
        let (iters, warm) = if n >= 1024 {
            (3, 1)
        } else if n >= 256 {
            (8, 2)
        } else {
            (20, 3)
        };

        // Tabs/newlines as real Lua escapes: single backslash in raw string.
        let chunk = format!(
            r#"
local ml = require "matlua"
local n = {n}
local function median(t)
  table.sort(t)
  return t[math.floor(#t/2)+1]
end
local function time_op(iters, warm, fn)
  for i=1,warm do fn() end
  local samples = {{}}
  for i=1,iters do
    local t0 = os.clock()
    fn()
    local t1 = os.clock()
    samples[i] = (t1-t0)*1000.0
  end
  return median(samples)
end
local A = ml.full(n, n, 1.000017)
local B = ml.full(n, n, 1.000013)
local ms = time_op({iters}, {warm}, function() ml.matmul(A, B) end)
io.write(string.format("lua\tmatmul\t%d\t%.6f\n", n, ms))

local S = ml.eye(n) + ml.full(n, n, 0.01)
local St = S:transpose()
S = ml.matmul(St, S) + ml.eye(n)
local rhs = ml.full(n, 0.5)
ms = time_op({iters}, {warm}, function() ml.solve(S, rhs) end)
io.write(string.format("lua\tsolve\t%d\t%.6f\n", n, ms))

local X = ml.full(n, n, 1.1)
local Y = ml.full(n, n, 2.2)
ms = time_op({iters}, {warm}, function() local _ = X + Y end)
io.write(string.format("lua\telem_add\t%d\t%.6f\n", n, ms))
"#,
            n = n,
            iters = iters,
            warm = warm
        );
        lua.do_string(&chunk).unwrap();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let sizes: Vec<usize> = if let Some(i) = args.iter().position(|a| a == "--sizes") {
        args.get(i + 1)
            .map(|s| s.split(',').filter_map(|p| p.parse().ok()).collect())
            .unwrap_or_else(|| vec![64, 256, 1024])
    } else {
        vec![64, 256, 1024]
    };

    eprintln!("# MatLua dense bench (release). sizes={sizes:?}");
    println!("face\top\tn\tms");
    bench_rust(&sizes);

    #[cfg(feature = "lua")]
    {
        eprintln!("# Lua face");
        bench_lua(&sizes);
    }
    #[cfg(not(feature = "lua"))]
    {
        eprintln!("# Lua face skipped (build with --features lua)");
    }
}
