//! Path-length A/B microbench: short vs long for the four audit options.
//! cargo test --release --test path_length -- --run

use std::hint::black_box;
use std::time::Instant;

use matlua::array::Array;
use matlua::linalg;

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ms(iters: usize, warmup: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(11);
    for _ in 0..11 {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        samples.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
    }
    median_ms(samples)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if !args.iter().any(|a| a == "--run") {
        eprintln!("path_length: skipped (pass --run)");
        return;
    }

    println!("op\tsize\tlong_ms\tshort_ms\tratio_short_over_long");

    // --- cov: long = transpose+matmul, short = matmul_bt (via Array::cov) ---
    for &(d, n) in &[(8usize, 256usize), (32, 1024), (64, 1024)] {
        let mut data = Vec::with_capacity(d * n);
        for i in 0..d * n {
            data.push((i as f64 * 0.001).sin());
        }
        let x = Array::from_shape_vec(vec![d, n], data).unwrap();
        // precompute centered for gram-only compare
        let means = x.mean_axis(1).unwrap();
        let mu = means.as_slice();
        let src = x.as_slice();
        let mut centered = vec![0.0; d * n];
        for i in 0..d {
            for j in 0..n {
                centered[i * n + j] = src[i * n + j] - mu[i];
            }
        }
        let xc = Array::from_shape_vec(vec![d, n], centered).unwrap();

        let iters = if d * n > 20_000 { 20 } else { 80 };
        let long_ms = time_ms(iters, 3, || {
            let xt = linalg::transpose(black_box(&xc)).unwrap();
            let g = linalg::matmul(black_box(&xc), &xt).unwrap();
            black_box(g);
        });
        let short_ms = time_ms(iters, 3, || {
            let g = linalg::matmul_bt(black_box(&xc), black_box(&xc)).unwrap();
            black_box(g);
        });
        let cov_ms = time_ms(iters, 3, || {
            black_box(Array::cov(black_box(&x), 1).unwrap());
        });
        println!(
            "gram_xxT\t{d}x{n}\t{long_ms:.4}\t{short_ms:.4}\t{:.3}",
            short_ms / long_ms
        );
        println!("cov_full\t{d}x{n}\t-\t{cov_ms:.4}\t-");
    }

    // --- mean_axis / var_axis ---
    for &(m, n) in &[(256usize, 256usize), (1024, 64), (64, 1024)] {
        let data: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.01).sin()).collect();
        let a = Array::from_shape_vec(vec![m, n], data).unwrap();
        let iters = 50;
        // long mean = sum_axis + scale
        let mean_long = time_ms(iters, 3, || {
            let s = a.sum_axis(1).unwrap();
            let inv = 1.0 / n as f64;
            let scaled: Vec<f64> = s.as_slice().iter().map(|x| x * inv).collect();
            black_box(scaled);
        });
        let mean_short = time_ms(iters, 3, || {
            black_box(a.mean_axis(1).unwrap());
        });
        println!(
            "mean_axis1\t{m}x{n}\t{mean_long:.4}\t{mean_short:.4}\t{:.3}",
            mean_short / mean_long
        );

        let var_long = time_ms(iters, 3, || {
            let mean = a.mean_axis(1).unwrap();
            let mu = mean.as_slice();
            let src = a.as_slice();
            let mut out = vec![0.0; m];
            for i in 0..m {
                let mut ss = 0.0;
                for j in 0..n {
                    let d = src[i * n + j] - mu[i];
                    ss += d * d;
                }
                out[i] = ss / (n - 1) as f64;
            }
            black_box(out);
        });
        let var_short = time_ms(iters, 3, || {
            black_box(a.var_axis(1, 1).unwrap());
        });
        println!(
            "var_axis1\t{m}x{n}\t{var_long:.4}\t{var_short:.4}\t{:.3}",
            var_short / var_long
        );
    }

    // --- broadcast matrix + row ---
    for &(m, n) in &[(256usize, 256usize), (1024, 1024)] {
        let a = Array::from_shape_vec(
            vec![m, n],
            (0..m * n).map(|i| i as f64 * 0.001).collect(),
        )
        .unwrap();
        let row = Array::from_shape_vec(vec![n], (0..n).map(|i| i as f64).collect()).unwrap();
        let col = Array::from_shape_vec(vec![m, 1], (0..m).map(|i| i as f64).collect()).unwrap();
        let iters = if m * n > 500_000 { 15 } else { 40 };

        // long: force materialize path by using a method that still goes through
        // owned_from_kernel fallback — compare broadcast_to + add_slices manually
        let long_ms = time_ms(iters, 3, || {
            let left = a.broadcast_to(vec![m, n]).unwrap();
            let right = row.broadcast_to(vec![m, n]).unwrap();
            black_box(left.add(&right).unwrap()); // same shape after materialize
        });
        // Wait: after both materialize, add is same-shape. Cost includes two broadcasts.
        // short is a.add(&row) fused
        let short_ms = time_ms(iters, 3, || {
            black_box(a.add(black_box(&row)).unwrap());
        });
        // fairer long: only materialize row then same-shape add
        let long2 = time_ms(iters, 3, || {
            let right = row.broadcast_to(vec![m, n]).unwrap();
            black_box(a.add(&right).unwrap());
        });
        println!(
            "elem_add_row\t{m}x{n}\t{long2:.4}\t{short_ms:.4}\t{:.3}",
            short_ms / long2
        );
        let short_col = time_ms(iters, 3, || {
            black_box(a.add(black_box(&col)).unwrap());
        });
        let long_col = time_ms(iters, 3, || {
            let right = col.broadcast_to(vec![m, n]).unwrap();
            black_box(a.add(&right).unwrap());
        });
        println!(
            "elem_add_col\t{m}x{n}\t{long_col:.4}\t{short_col:.4}\t{:.3}",
            short_col / long_col
        );
        let _ = long_ms;
    }

    // --- corrcoef inherits cov; spot time ---
    let x = Array::from_shape_vec(
        vec![16, 512],
        (0..16 * 512).map(|i| (i as f64 * 0.01).sin()).collect(),
    )
    .unwrap();
    let corr_ms = time_ms(40, 3, || {
        black_box(Array::corrcoef(black_box(&x)).unwrap());
    });
    println!("corrcoef\t16x512\t-\t{corr_ms:.4}\t-");
}
