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
    // True median: average the middle pair on even counts (v[len/2] alone
    // returns the WORSE of 2 samples — one contention stall became the cell).
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
    median_t(&samples)
}

fn time_lua(lua: &Lua, setup: &str, body: &str, iters: usize, warm: usize) -> f64 {
    lua.do_string(&format!(
        "{setup}\nfunction __bench_op()\n{body}\nend\nfunction __bench_gc()\ncollectgarbage(\"collect\")\nend\n"
    ))
    .unwrap();
    for _ in 0..warm {
        lua.call_global("__bench_op").unwrap();
        let _ = lua.call_global("__bench_gc");
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        lua.call_global("__bench_op").unwrap();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
        let _ = lua.call_global("__bench_gc");
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
    // Diagonally dominant integer SPD (cheap at large n; Gram would be O(n³) setup).
    let mut data = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = ((i + 2 * j) % 7) as i64;
        }
        data[i * n + i] += (n as i64) + 1;
    }
    ArrayI64::from_shape_vec(vec![n, n], data).unwrap()
}

fn vec_n(n: usize) -> ArrayI64 {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push((i as i64).wrapping_mul(3).wrapping_add(1));
    }
    ArrayI64::from_shape_vec(vec![n], data).unwrap()
}

fn budget(n: usize, heavy: bool) -> (usize, usize) {
    // >=5 odd samples for heavy cells: a real median, robust to shared-host
    // stalls (matches fair_all/i64_surface; the old (1,0) published single
    // cold calls at n=4096).
    if heavy {
        if n >= 4096 {
            (5, 1)
        } else if n >= 1024 {
            (5, 1)
        } else if n >= 256 {
            (5, 2)
        } else {
            (11, 2)
        }
    } else if n >= 4096 {
        (5, 2)
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

fn bench_rust(sizes: &[usize]) {
    for &n in sizes {
        let a = dense(n);
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
        let s = spd_i64(n);
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
        let (it, wrm) = budget(n, false);
        let (ith, wrmh) = budget(n, true);
        // Identical inputs by construction: globals are copies of the exact
        // arrays the Rust face benches (previously the Lua face used constant
        // matrices and a different SPD system).
        lua.set_global_array_i64("A", &dense(n)).unwrap();
        lua.set_global_array_i64("S", &spd_i64(n)).unwrap();
        lua.set_global_array_i64("rhs", &vec_n(n)).unwrap();
        lua.do_string("collectgarbage(\"collect\")").unwrap();
        emit("lua", "mean", n, time_lua(&lua, "", "return A:mean()", it, wrm));
        emit("lua", "std", n, time_lua(&lua, "", "return A:std()", it, wrm));
        emit("lua", "median", n, time_lua(&lua, "", "return A:median()", it, wrm));
        emit("lua", "quantile", n, time_lua(&lua, "", "return A:quantile(0.75)", it, wrm));
        emit("lua", "norm", n, time_lua(&lua, "", "return ml.norm(A)", it, wrm));
        emit("lua", "solve", n, time_lua(&lua, "", "return ml.solve(S, rhs)", ith, wrmh));
        emit("lua", "cholesky", n, time_lua(&lua, "", "return ml.cholesky(S)", ith, wrmh));
        emit("lua", "qr", n, time_lua(&lua, "", "return ml.qr(A)", ith, wrmh));

        // Free this size's globals before the next.
        let _ = lua.do_string("A=nil;S=nil;rhs=nil;collectgarbage(\"collect\")");
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
            .unwrap_or_else(|| vec![64, 256, 1024, 4096])
    } else {
        vec![64, 256, 1024, 4096]
    };
    eprintln!("# i64→f64 promote-out. sizes={sizes:?}");
    println!("face\top\tn\tms");
    bench_rust(&sizes);
    eprintln!("# Lua face");
    bench_lua(&sizes);
}
