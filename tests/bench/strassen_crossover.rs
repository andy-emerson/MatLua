//! GEBP vs forced one-level Strassen at many n; print recommended leaf.
//!
//! ```text
//! cargo test --release --test strassen_crossover -- --run
//! ```

use std::env;
use std::hint::black_box;
use std::time::Instant;

use matlua::array::ArrayI64;
use matlua::linalg::i64_ops;

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ms(iters: usize, warm: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warm {
        f();
    }
    let mut s = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        s.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    median(&s)
}

fn dense(n: usize) -> ArrayI64 {
    let mut d = Vec::with_capacity(n * n);
    let mut x = 1i64;
    for _ in 0..n * n {
        d.push(x);
        x = x.wrapping_add(17);
    }
    ArrayI64::from_shape_vec(vec![n, n], d).unwrap()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if !args.iter().any(|a| a == "--run") {
        eprintln!("strassen_crossover: skipped (pass --run)");
        return;
    }
    // Even sizes; include above current production leaf.
    let sizes = [
        256usize, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048,
    ];
    println!(
        "n\tgebp_ms\tstrassen_ms\tratio_s/g\twinner\tprod_leaf={}",
        i64_ops::STRASSEN_LEAF
    );
    let mut first_win: Option<usize> = None;
    for &n in &sizes {
        let a = dense(n);
        let b = dense(n);
        let (it, w) = if n >= 1536 {
            (2, 1)
        } else if n >= 1024 {
            (3, 1)
        } else if n >= 512 {
            (5, 2)
        } else {
            (8, 2)
        };
        let g = time_ms(it, w, || {
            black_box(i64_ops::matmul_gebp_only(&a, &b).unwrap());
        });
        // Force Strassen path: temporarily use matmul which only Strassens if n>=leaf.
        // For n < leaf, call matmul_strassen_force if exported.
        let s = time_ms(it, w, || {
            black_box(i64_ops::matmul_strassen_force(&a, &b).unwrap());
        });
        let ratio = s / g;
        let winner = if ratio < 0.97 {
            "strassen"
        } else if ratio > 1.03 {
            "gebp"
        } else {
            "tie"
        };
        if first_win.is_none() && ratio < 0.97 {
            first_win = Some(n);
        }
        println!("{n}\t{g:.3}\t{s:.3}\t{ratio:.3}\t{winner}");
    }
    if let Some(n) = first_win {
        eprintln!(
            "# recommend STRASSEN_LEAF ≈ {n} (first n with S/G < 0.97 on this host)"
        );
    } else {
        eprintln!(
            "# Strassen never clearly beat GEBP in this range; keep high leaf (e.g. 2048+)"
        );
    }
}
