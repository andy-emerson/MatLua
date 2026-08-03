//! Fair three-face microbench (NumPy / MatLua Rust / MatLua Lua).
//!
//! Same generation rule for inputs; wall-clock median; setup outside timer.
//!
//! ```text
//! cargo test --release --features lua --test fair_all -- --run --sizes 64,256,1024,4096
//! python3 tests/bench/numpy_fair.py --sizes 64,256,1024,4096
//! python3 tests/bench/compare_fair.py
//! ```

use std::env;
use std::time::Instant;

use matlua::array::Array;
use matlua::linalg;
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

/// Contiguous row-major fill matching numpy_bench_all.py
fn dense_n(n: usize) -> Array {
    let mut data = Vec::with_capacity(n * n);
    let mut x = 0.001_f64;
    for _ in 0..n * n {
        data.push(x);
        x += 0.000017;
    }
    Array::from_shape_vec(vec![n, n], data).unwrap()
}

fn dense2_n(n: usize) -> Array {
    let mut data = Vec::with_capacity(n * n);
    let mut x = 0.002_f64;
    for _ in 0..n * n {
        data.push(x);
        x += 0.000013;
    }
    Array::from_shape_vec(vec![n, n], data).unwrap()
}

fn vec_n(n: usize) -> Array {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(0.5 + i as f64 * 0.01);
    }
    Array::from_shape_vec(vec![n], data).unwrap()
}

fn vec2_n(n: usize) -> Array {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(0.25 + i as f64 * 0.007);
    }
    Array::from_shape_vec(vec![n], data).unwrap()
}

fn spd_n(n: usize) -> Array {
    // Avoid O(n³) Gram setup at huge n: diagonally dominant SPD is enough for solve/chol.
    if n >= 1024 {
        let mut data = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = 0.01 * ((i + 2 * j) % 7) as f64;
                data[i * n + j] = v;
            }
            data[i * n + i] += (n as f64) + 1.0;
        }
        return Array::from_shape_vec(vec![n, n], data).unwrap();
    }
    let a = dense_n(n);
    let at = linalg::transpose(&a).unwrap();
    Array::add(&linalg::matmul(&at, &a).unwrap(), &Array::eye(n).unwrap()).unwrap()
}

fn emit(face: &str, op: &str, n: usize, ms: f64) {
    println!("{face}\t{op}\t{n}\t{ms:.6}");
}

fn budget(n: usize, heavy: bool) -> (usize, usize) {
    if heavy {
        if n >= 4096 {
            (1, 0)
        } else if n >= 1024 {
            (3, 1)
        } else if n >= 256 {
            (6, 2)
        } else {
            (15, 3)
        }
    } else if n >= 4096 {
        (2, 1)
    } else if n >= 1024 {
        (8, 2)
    } else if n >= 256 {
        (15, 3)
    } else {
        (40, 5)
    }
}

fn bench_rust(sizes: &[usize]) {
    for &n in sizes {
        let a = dense_n(n);
        let b = dense2_n(n);
        let v = vec_n(n);
        let w = vec2_n(n);
        let s = spd_n(n);
        let rhs = vec_n(n);

        let (it, wrm) = budget(n, false);
        emit("rust", "zeros", n, time_ms(it, wrm, || {
            let _ = Array::zeros(vec![n, n]).unwrap();
        }));
        emit("rust", "ones", n, time_ms(it, wrm, || {
            let _ = Array::ones(vec![n, n]).unwrap();
        }));
        emit("rust", "full", n, time_ms(it, wrm, || {
            let _ = Array::full(vec![n, n], 1.5).unwrap();
        }));
        emit("rust", "eye", n, time_ms(it, wrm, || {
            let _ = Array::eye(n).unwrap();
        }));
        emit("rust", "arange", n, time_ms(it, wrm, || {
            let _ = Array::arange(0.0, n as f64).unwrap();
        }));
        emit("rust", "copy", n, time_ms(it, wrm, || {
            let _ = a.copy();
        }));
        // reshape n×n → (n/2)×(2n) when n even
        if n % 2 == 0 {
            let dims = vec![n / 2, n * 2];
            emit("rust", "reshape", n, time_ms(it, wrm, || {
                let _ = a.reshape(dims.clone()).unwrap();
            }));
        }
        emit("rust", "fill", n, {
            let mut t = a.copy();
            time_ms(it, wrm, || {
                t.fill(3.0);
            })
        });
        emit("rust", "elem_add", n, time_ms(it, wrm, || {
            let _ = Array::add(&a, &b).unwrap();
        }));
        emit("rust", "elem_sub", n, time_ms(it, wrm, || {
            let _ = Array::sub(&a, &b).unwrap();
        }));
        emit("rust", "elem_mul", n, time_ms(it, wrm, || {
            let _ = Array::mul(&a, &b).unwrap();
        }));
        emit("rust", "elem_div", n, time_ms(it, wrm, || {
            let _ = Array::div(&a, &b).unwrap();
        }));
        emit("rust", "elem_add_scalar", n, time_ms(it, wrm, || {
            let _ = &a + 2.5;
        }));
        emit("rust", "sum", n, time_ms(it, wrm, || {
            let _ = a.sum();
        }));
        emit("rust", "mean", n, time_ms(it, wrm, || {
            let _ = a.mean();
        }));
        emit("rust", "min", n, time_ms(it, wrm, || {
            let _ = a.min().unwrap();
        }));
        emit("rust", "max", n, time_ms(it, wrm, || {
            let _ = a.max().unwrap();
        }));
        emit("rust", "transpose", n, time_ms(it, wrm, || {
            let _ = linalg::transpose(&a).unwrap();
        }));
        emit("rust", "dot", n, time_ms(it, wrm, || {
            let _ = linalg::dot(&v, &w).unwrap();
        }));
        emit("rust", "norm", n, time_ms(it, wrm, || {
            let _ = linalg::norm(&a).unwrap();
        }));

        let (it, wrm) = budget(n, true);
        emit("rust", "matmul", n, time_ms(it, wrm, || {
            let _ = linalg::matmul(&a, &b).unwrap();
        }));
        emit("rust", "solve", n, time_ms(it, wrm, || {
            let _ = linalg::solve(&s, &rhs).unwrap();
        }));
        emit("rust", "cholesky", n, time_ms(it, wrm, || {
            let _ = linalg::cholesky(&s).unwrap();
        }));
        emit("rust", "qr", n, time_ms(it, wrm, || {
            let _ = linalg::qr(&a).unwrap();
        }));
        emit("rust", "svd", n, time_ms(it, wrm, || {
            let _ = linalg::svd(&a).unwrap();
        }));
    }
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
    // free setup globals
    let _ = lua.do_string("A=nil;B=nil;V=nil;W=nil;S=nil;rhs=nil;T=nil;collectgarbage(\"collect\")");
    median(&samples)
}

/// Build globals A,B,V,W,S,rhs with same numeric rules as Rust helpers.
fn lua_build_inputs(n: usize) -> String {
    // Bulk constructors only (no O(n²) :set loops) so n=4096 is feasible.
    format!(
        r#"
local n = {n}
A = ml.arange(0, n * n):reshape(n, n)
B = ml.full(n, n, 1.5)
V = ml.arange(0, n)
W = ml.arange(1, n + 1)
S = ml.full(n, n, 0.01)
for i = 1, n do
  S:set(i, i, n + 1)
end
rhs = V
"#
    )
}

fn bench_lua(sizes: &[usize]) {
    let lua = Lua::new().unwrap();
    lua.do_string(r#"ml = require "matlua""#).unwrap();

    for &n in sizes {
        let (it, wrm) = budget(n, false);
        let (ith, wrmh) = budget(n, true);
        let ab = format!(
            "A=ml.full({n},{n},1.0); B=ml.full({n},{n},1.5); collectgarbage(\"collect\")"
        );
        let a_only = format!("A=ml.full({n},{n},1.0); collectgarbage(\"collect\")");
        let v_only = format!("V=ml.arange(0,{n}); collectgarbage(\"collect\")");
        let vw = format!(
            "V=ml.arange(0,{n}); W=ml.arange(1,{n}+1); collectgarbage(\"collect\")"
        );
        let spd = format!(
            "S=ml.full({n},{n},0.01); for i=1,{n} do S:set(i,i,{n}+1) end; rhs=ml.arange(0,{n}); collectgarbage(\"collect\")"
        );
        let ab_mat = format!(
            "A=ml.full({n},{n},1.0); B=ml.full({n},{n},1.5); collectgarbage(\"collect\")"
        );

        emit("lua", "zeros", n, time_lua(&lua, "", &format!("return ml.zeros({n},{n})"), it, wrm));
        emit("lua", "ones", n, time_lua(&lua, "", &format!("return ml.ones({n},{n})"), it, wrm));
        emit("lua", "full", n, time_lua(&lua, "", &format!("return ml.full({n},{n},1.5)"), it, wrm));
        emit("lua", "eye", n, time_lua(&lua, "", &format!("return ml.eye({n})"), it, wrm));
        emit("lua", "arange", n, time_lua(&lua, "", &format!("return ml.arange(0,{n})"), it, wrm));
        emit("lua", "copy", n, time_lua(&lua, &a_only, "return A:copy()", it, wrm));
        if n % 2 == 0 {
            emit(
                "lua",
                "reshape",
                n,
                time_lua(
                    &lua,
                    &a_only,
                    &format!("return A:reshape({hr},{wr})", hr = n / 2, wr = n * 2),
                    it,
                    wrm,
                ),
            );
        }
        emit("lua", "fill", n, time_lua(&lua, &(a_only.clone() + "\nT=A:copy()\n"), "T:fill(3.0)", it, wrm));
        emit("lua", "elem_add", n, time_lua(&lua, &ab, "return A + B", it, wrm));
        emit("lua", "elem_sub", n, time_lua(&lua, &ab, "return A - B", it, wrm));
        emit("lua", "elem_mul", n, time_lua(&lua, &ab, "return A * B", it, wrm));
        emit("lua", "elem_div", n, time_lua(&lua, &ab, "return A / B", it, wrm));
        emit("lua", "elem_add_scalar", n, time_lua(&lua, &a_only, "return A + 2.5", it, wrm));
        emit("lua", "sum", n, time_lua(&lua, &a_only, "return A:sum()", it, wrm));
        emit("lua", "mean", n, time_lua(&lua, &a_only, "return A:mean()", it, wrm));
        emit("lua", "min", n, time_lua(&lua, &a_only, "return A:min()", it, wrm));
        emit("lua", "max", n, time_lua(&lua, &a_only, "return A:max()", it, wrm));
        emit("lua", "transpose", n, time_lua(&lua, &a_only, "return A:transpose()", it, wrm));
        emit("lua", "dot", n, time_lua(&lua, &vw, "return ml.dot(V, W)", it, wrm));
        emit("lua", "norm", n, time_lua(&lua, &a_only, "return ml.norm(A)", it, wrm));
        emit("lua", "matmul", n, time_lua(&lua, &ab_mat, "return ml.matmul(A, B)", ith, wrmh));
        emit("lua", "solve", n, time_lua(&lua, &spd, "return ml.solve(S, rhs)", ith, wrmh));
        emit("lua", "cholesky", n, time_lua(&lua, &spd, "return ml.cholesky(S)", ith, wrmh));
        emit("lua", "qr", n, time_lua(&lua, &a_only, "return ml.qr(A)", ith, wrmh));
        emit("lua", "svd", n, time_lua(&lua, &a_only, "return ml.svd(A)", ith, wrmh));
    }

}


fn main() {
    let args: Vec<String> = env::args().collect();
    // Not part of default `cargo test` wall time: require explicit --run.
    if !args.iter().any(|a| a == "--run") {
        eprintln!("fair_all: skipped (pass --run, e.g. cargo test --release --features lua --test fair_all -- --run)");
        return;
    }
    let sizes: Vec<usize> = if let Some(i) = args.iter().position(|a| a == "--sizes") {
        args.get(i + 1)
            .map(|s| s.split(',').filter_map(|p| p.parse().ok()).collect())
            .unwrap_or_else(|| vec![64, 256, 1024, 4096])
    } else {
        vec![64, 256, 1024, 4096]
    };

    eprintln!("# Fair full-surface bench (release, wall clock). sizes={sizes:?}");
    println!("face\top\tn\tms");
    bench_rust(&sizes);
    eprintln!("# Lua face");
    bench_lua(&sizes);
}
