//! i64 three-face microbench: NumPy / MatLua Rust / MatLua Lua.
//! Expanded surface (ctors + elementwise siblings + reductions + LA-ish).
//!
//! ```text
//! cargo test --release --features lua --test i64_surface -- --run --sizes 64,256,1024
//! python3 tests/bench/numpy_i64_fair.py --sizes 64,256,1024
//! ```

use std::env;
use std::time::Instant;

use matlua::array::ArrayI64;
use matlua::linalg::i64_ops;
use matlua::lua::Lua;

fn median(samples: &[f64]) -> f64 {
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
    median(&samples)
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
    if heavy {
        if n >= 1024 {
            (3, 1)
        } else if n >= 256 {
            (6, 2)
        } else {
            (15, 3)
        }
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

fn lua_build_inputs(n: usize) -> String {
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
B = ml.zeros_i64(n, n)
x = 2
for i = 1, n do
  for j = 1, n do
    B:set(i, j, x)
    x = x + 13
  end
end
V = ml.zeros_i64(n)
for i = 1, n do
  V:set(i, (i - 1) * 3 + 1)
end
"#
    )
}

fn bench_rust(sizes: &[usize]) {
    for &n in sizes {
        let a = dense(n);
        let b = dense2(n);
        let v = vec_n(n);
        let (it, wrm) = budget(n, false);
        let (ith, wrmh) = budget(n, true);

        emit("rust", "zeros", n, time_ms(it, wrm, || {
            let _ = ArrayI64::zeros(vec![n, n]).unwrap();
        }));
        emit("rust", "ones", n, time_ms(it, wrm, || {
            let _ = ArrayI64::ones(vec![n, n]).unwrap();
        }));
        emit("rust", "full", n, time_ms(it, wrm, || {
            let _ = ArrayI64::full(vec![n, n], 7).unwrap();
        }));
        emit("rust", "eye", n, time_ms(it, wrm, || {
            let _ = ArrayI64::eye(n).unwrap();
        }));
        emit("rust", "arange", n, time_ms(it, wrm, || {
            let _ = ArrayI64::arange(0, n as i64).unwrap();
        }));
        emit("rust", "copy", n, time_ms(it, wrm, || {
            let _ = a.copy();
        }));
        if n % 2 == 0 {
            emit("rust", "reshape", n, time_ms(it, wrm, || {
                let _ = a.reshape(vec![n / 2, n * 2]).unwrap();
            }));
        }
        {
            let mut t = a.copy();
            emit("rust", "fill", n, time_ms(it, wrm, || {
                t.fill(3);
            }));
        }
        emit("rust", "elem_add", n, time_ms(it, wrm, || {
            let _ = a.add(&b).unwrap();
        }));
        emit("rust", "elem_sub", n, time_ms(it, wrm, || {
            let _ = a.sub(&b).unwrap();
        }));
        emit("rust", "elem_mul", n, time_ms(it, wrm, || {
            let _ = a.mul(&b).unwrap();
        }));
        emit("rust", "elem_div", n, time_ms(it, wrm, || {
            let _ = a.div(&b).unwrap();
        }));
        emit("rust", "sum", n, time_ms(it, wrm, || {
            let _ = a.sum();
        }));
        emit("rust", "min", n, time_ms(it, wrm, || {
            let _ = a.min().unwrap();
        }));
        emit("rust", "max", n, time_ms(it, wrm, || {
            let _ = a.max().unwrap();
        }));
        emit("rust", "transpose", n, time_ms(it, wrm, || {
            let _ = a.transpose().unwrap();
        }));
        emit("rust", "dot", n, time_ms(it, wrm, || {
            let _ = i64_ops::dot(&v, &v).unwrap();
        }));
        emit("rust", "matmul", n, time_ms(ith, wrmh, || {
            let _ = i64_ops::matmul(&a, &b).unwrap();
        }));
        emit("rust", "unique", n, time_ms(it, wrm, || {
            let _ = v.unique().unwrap();
        }));
        emit("rust", "isin", n, time_ms(it, wrm, || {
            let _ = a.isin(&v).unwrap();
        }));
    }
}

fn bench_lua(sizes: &[usize]) {
    let lua = Lua::new().unwrap();
    lua.do_string(r#"ml = require "matlua""#).unwrap();

    for &n in sizes {
        let build = lua_build_inputs(n);
        let (it, wrm) = budget(n, false);
        let (ith, wrmh) = budget(n, true);

        emit("lua", "zeros", n, time_lua(&lua, "", &format!("return ml.zeros_i64({n},{n})"), it, wrm));
        emit("lua", "ones", n, time_lua(&lua, "", &format!("return ml.ones_i64({n},{n})"), it, wrm));
        emit("lua", "full", n, time_lua(&lua, "", &format!("return ml.full_i64({n},{n},7)"), it, wrm));
        emit("lua", "eye", n, time_lua(&lua, "", &format!("return ml.eye_i64({n})"), it, wrm));
        emit("lua", "arange", n, time_lua(&lua, "", &format!("return ml.arange_i64(0,{n})"), it, wrm));
        emit("lua", "copy", n, time_lua(&lua, &build, "return A:copy()", it, wrm));
        if n % 2 == 0 {
            emit(
                "lua",
                "reshape",
                n,
                time_lua(
                    &lua,
                    &build,
                    &format!("return A:reshape({hr},{wr})", hr = n / 2, wr = n * 2),
                    it,
                    wrm,
                ),
            );
        }
        emit(
            "lua",
            "fill",
            n,
            time_lua(&lua, &(build.clone() + "T=A:copy()\n"), "T:fill(3)", it, wrm),
        );
        emit("lua", "elem_add", n, time_lua(&lua, &build, "return A + B", it, wrm));
        emit("lua", "elem_sub", n, time_lua(&lua, &build, "return A - B", it, wrm));
        emit("lua", "elem_mul", n, time_lua(&lua, &build, "return A * B", it, wrm));
        emit("lua", "elem_div", n, time_lua(&lua, &build, "return A / B", it, wrm));
        emit("lua", "sum", n, time_lua(&lua, &build, "return A:sum()", it, wrm));
        emit("lua", "min", n, time_lua(&lua, &build, "return A:min()", it, wrm));
        emit("lua", "max", n, time_lua(&lua, &build, "return A:max()", it, wrm));
        emit("lua", "transpose", n, time_lua(&lua, &build, "return A:transpose()", it, wrm));
        emit("lua", "dot", n, time_lua(&lua, &build, "return ml.dot(V, V)", it, wrm));
        emit("lua", "matmul", n, time_lua(&lua, &build, "return ml.matmul(A, B)", ith, wrmh));
        emit("lua", "unique", n, time_lua(&lua, &build, "return V:unique()", it, wrm));
        emit("lua", "isin", n, time_lua(&lua, &build, "return A:isin(V)", it, wrm));
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
            .unwrap_or_else(|| vec![64, 256, 1024])
    } else {
        vec![64, 256, 1024]
    };
    eprintln!("# i64 three-face expanded. sizes={sizes:?}");
    println!("face\top\tn\tms");
    bench_rust(&sizes);
    eprintln!("# Lua face");
    bench_lua(&sizes);
}
