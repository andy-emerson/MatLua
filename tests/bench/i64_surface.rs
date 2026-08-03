//! i64 three-face microbench: NumPy / MatLua Rust / MatLua Lua.
//! Expanded surface (ctors + elementwise siblings + reductions + LA-ish).
//!
//! ```text
//! cargo test --release --features lua --test i64_surface -- --run --sizes 64,256,1024,4096
//! python3 tests/bench/numpy_i64_fair.py --sizes 64,256,1024,4096
//! ```

use std::env;
use std::hint::black_box;
use std::time::Instant;

use matlua::array::ArrayI64;
use matlua::linalg::i64_ops;
use matlua::lua::Lua;

fn median(samples: &[f64]) -> f64 {
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
    median(&samples)
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
    median(&samples)
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

fn dense2(n: usize) -> ArrayI64 {
    let mut data = Vec::with_capacity(n * n);
    let mut x: i64 = 2;
    for _ in 0..n * n {
        data.push(x);
        x = x.wrapping_add(13);
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
    // 4096 cells were single/double-sample and hostage to shared-host
    // stalls; >=5 odd samples give a real median (empirical noise floor
    // +/-10-20%, see tests/README Provenance).
    if heavy {
        if n >= 4096 {
            (5, 1)
        } else if n >= 1024 {
            (5, 1)
        } else if n >= 256 {
            (6, 2)
        } else {
            (15, 3)
        }
    } else if n >= 4096 {
        (5, 2)
    } else if n >= 1024 {
        (8, 2)
    } else if n >= 256 {
        (20, 4)
    } else {
        (50, 8)
    }
}

fn emit(face: &str, op: &str, n: usize, ms: f64) {
    println!("{face}\t{op}\t{n}\t{ms:.6}");
}

fn bench_rust(sizes: &[usize]) {
    for &n in sizes {
        let a = dense(n);
        let b = dense2(n);
        let v = vec_n(n);
        let (it, wrm) = budget(n, false);
        let (ith, wrmh) = budget(n, true);

        emit("rust", "zeros", n, time_ms(it, wrm, || {
            black_box(ArrayI64::zeros(vec![n, n]).unwrap());
        }));
        emit("rust", "ones", n, time_ms(it, wrm, || {
            black_box(ArrayI64::ones(vec![n, n]).unwrap());
        }));
        emit("rust", "full", n, time_ms(it, wrm, || {
            black_box(ArrayI64::full(vec![n, n], 7).unwrap());
        }));
        emit("rust", "eye", n, time_ms(it, wrm, || {
            black_box(ArrayI64::eye(n).unwrap());
        }));
        emit("rust", "arange", n, time_ms(it, wrm, || {
            black_box(ArrayI64::arange(0, n as i64).unwrap());
        }));
        emit("rust", "copy", n, time_ms(it, wrm, || {
            black_box(a.copy());
        }));
        if n % 2 == 0 {
            emit("rust", "reshape", n, time_ms(it, wrm, || {
                black_box(a.reshape(vec![n / 2, n * 2]).unwrap());
            }));
        }
        {
            let mut t = a.copy();
            emit("rust", "fill", n, time_ms(it, wrm, || {
                black_box(t.fill(3));
            }));
        }
        emit("rust", "elem_add", n, time_ms(it, wrm, || {
            black_box(a.add(&b).unwrap());
        }));
        emit("rust", "elem_sub", n, time_ms(it, wrm, || {
            black_box(a.sub(&b).unwrap());
        }));
        emit("rust", "elem_mul", n, time_ms(it, wrm, || {
            black_box(a.mul(&b).unwrap());
        }));
        emit("rust", "elem_div", n, time_ms(it, wrm, || {
            black_box(a.div(&b).unwrap());
        }));
        emit("rust", "sum", n, time_ms(it, wrm, || {
            black_box(a.sum());
        }));
        emit("rust", "min", n, time_ms(it, wrm, || {
            black_box(a.min().unwrap());
        }));
        emit("rust", "max", n, time_ms(it, wrm, || {
            black_box(a.max().unwrap());
        }));
        emit("rust", "transpose", n, time_ms(it, wrm, || {
            black_box(a.transpose().unwrap());
        }));
        emit("rust", "dot", n, time_ms(it, wrm, || {
            black_box(i64_ops::dot(&v, &v).unwrap());
        }));
        emit("rust", "matmul", n, time_ms(ith, wrmh, || {
            black_box(i64_ops::matmul(&a, &b).unwrap());
        }));
        emit("rust", "matmul_at", n, time_ms(ith, wrmh, || {
            black_box(i64_ops::matmul_at(&a, &b).unwrap());
        }));
        emit("rust", "matmul_bt", n, time_ms(ith, wrmh, || {
            black_box(i64_ops::matmul_bt(&a, &b).unwrap());
        }));
        emit("rust", "unique", n, time_ms(it, wrm, || {
            black_box(v.unique().unwrap());
        }));
        emit("rust", "isin", n, time_ms(it, wrm, || {
            black_box(a.isin(&v).unwrap());
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
        // arrays the Rust face benches (previously constant matrices made
        // e.g. isin degenerate to a single hot key on the Lua face).
        lua.set_global_array_i64("A", &dense(n)).unwrap();
        lua.set_global_array_i64("B", &dense2(n)).unwrap();
        lua.set_global_array_i64("V", &vec_n(n)).unwrap();
        lua.do_string("T = A:copy(); collectgarbage(\"collect\")").unwrap();
        let ab = "";
        let a_only = "";
        let v_only = "";

        emit("lua", "zeros", n, time_lua(&lua, "", &format!("return ml.zeros_i64({n},{n})"), it, wrm));
        emit("lua", "ones", n, time_lua(&lua, "", &format!("return ml.ones_i64({n},{n})"), it, wrm));
        emit("lua", "full", n, time_lua(&lua, "", &format!("return ml.full_i64({n},{n},7)"), it, wrm));
        emit("lua", "eye", n, time_lua(&lua, "", &format!("return ml.eye_i64({n})"), it, wrm));
        emit("lua", "arange", n, time_lua(&lua, "", &format!("return ml.arange_i64(0,{n})"), it, wrm));
        emit("lua", "copy", n, time_lua(&lua, a_only, "return A:copy()", it, wrm));
        if n % 2 == 0 {
            emit(
                "lua",
                "reshape",
                n,
                time_lua(
                    &lua,
                    &a_only,
                    &format!("return A:reshape({},{})", n / 2, n * 2),
                    it,
                    wrm,
                ),
            );
        }
        emit("lua", "fill", n, time_lua(&lua, "", "T:fill(3)", it, wrm));
        emit("lua", "elem_add", n, time_lua(&lua, ab, "return A + B", it, wrm));
        emit("lua", "elem_sub", n, time_lua(&lua, ab, "return A - B", it, wrm));
        emit("lua", "elem_mul", n, time_lua(&lua, ab, "return A * B", it, wrm));
        emit("lua", "elem_div", n, time_lua(&lua, ab, "return A / B", it, wrm));
        emit("lua", "sum", n, time_lua(&lua, a_only, "return A:sum()", it, wrm));
        emit("lua", "min", n, time_lua(&lua, a_only, "return A:min()", it, wrm));
        emit("lua", "max", n, time_lua(&lua, a_only, "return A:max()", it, wrm));
        emit("lua", "transpose", n, time_lua(&lua, a_only, "return A:transpose()", it, wrm));
        emit("lua", "dot", n, time_lua(&lua, v_only, "return ml.dot(V, V)", it, wrm));
        emit("lua", "matmul", n, time_lua(&lua, ab, "return ml.matmul(A, B)", ith, wrmh));
        emit("lua", "matmul_at", n, time_lua(&lua, ab, "return ml.matmul_at(A, B)", ith, wrmh));
        emit("lua", "matmul_bt", n, time_lua(&lua, ab, "return ml.matmul_bt(A, B)", ith, wrmh));
        emit("lua", "unique", n, time_lua(&lua, v_only, "return V:unique()", it, wrm));
        emit("lua", "isin", n, time_lua(&lua, "", "return A:isin(V)", it, wrm));

        // Free this size's globals before the next.
        let _ = lua.do_string(
            "A=nil;B=nil;V=nil;T=nil;collectgarbage(\"collect\")",
        );
    }
}


fn main() {
    let args: Vec<String> = env::args().collect();
    if !args.iter().any(|a| a == "--run") {
        eprintln!("i64_surface: skipped (pass --run)");
        return;
    }
    let sizes: Vec<usize> = if let Some(i) = args.iter().position(|a| a == "--sizes") {
        args.get(i + 1)
            .map(|s| s.split(',').filter_map(|p| p.parse().ok()).collect())
            .unwrap_or_else(|| vec![64, 256, 1024, 4096])
    } else {
        vec![64, 256, 1024, 4096]
    };
    eprintln!("# i64 three-face expanded. sizes={sizes:?}");
    println!("face\top\tn\tms");
    bench_rust(&sizes);
    eprintln!("# Lua face");
    bench_lua(&sizes);
}
