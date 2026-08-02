//! i64 surface microbench (M7.c). Same spirit as fair_all, integer path only.
//!
//! ```text
//! cargo test --release --test i64_surface -- --run --sizes 64,256,1024
//! ```

use std::env;
use std::time::Instant;

use matlua::array::ArrayI64;
use matlua::linalg::i64_ops;

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

fn dense(n: usize) -> ArrayI64 {
    let mut data = Vec::with_capacity(n * n);
    let mut x: i64 = 1;
    for _ in 0..n * n {
        data.push(x);
        x = x.wrapping_add(17);
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

fn emit(op: &str, n: usize, ms: f64) {
    println!("i64\t{op}\t{n}\t{ms:.6}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if !args.iter().any(|a| a == "--run") {
        eprintln!("i64_surface: skipped (pass --run)");
        return;
    }
    let sizes: Vec<usize> = args
        .windows(2)
        .find(|w| w[0] == "--sizes")
        .map(|w| {
            w[1]
                .split(',')
                .filter_map(|s| s.parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![64, 256]);

    for n in sizes {
        let a = dense(n);
        let b = dense(n);
        let v = vec_n(n);
        let (it, wrm) = budget(n, false);
        let (ith, wrmh) = budget(n, true);

        emit("elem_add", n, time_ms(it, wrm, || {
            let _ = a.add(&b).unwrap();
        }));
        emit("elem_mul", n, time_ms(it, wrm, || {
            let _ = a.mul(&b).unwrap();
        }));
        emit("sum", n, time_ms(it, wrm, || {
            let _ = a.sum();
        }));
        emit("min", n, time_ms(it, wrm, || {
            let _ = a.min().unwrap();
        }));
        emit("transpose", n, time_ms(it, wrm, || {
            let _ = a.transpose().unwrap();
        }));
        emit("dot", n, time_ms(it, wrm, || {
            let _ = i64_ops::dot(&v, &v).unwrap();
        }));
        emit("matmul", n, time_ms(ith, wrmh, || {
            let _ = i64_ops::matmul(&a, &b).unwrap();
        }));
        let u = vec_n(n.min(4096).max(n)); // rank-1 of length n for unique
        emit("unique", n, time_ms(it, wrm, || {
            let _ = u.unique().unwrap();
        }));
        emit("isin", n, time_ms(it, wrm, || {
            let _ = a.isin(&v).unwrap();
        }));
    }
}
