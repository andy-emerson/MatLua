//! Machine roofline for exact wrapping `i64` multiply-add (DESIGN §7.1.2).
//!
//! Answers: how fast can THIS host possibly run wrapping i64 MACs, so that
//! i64 GEMM results can be judged as **% of achievable** instead of only as a
//! ratio to NumPy f64 BLAS (which conflates kernel quality with ISA physics —
//! there is no 64-bit vector multiply below AVX-512DQ, so some multiple of
//! the f64 time is unavoidable for exact i64).
//!
//! Kernels (all L1-resident, compute-bound by construction):
//! - `scalar_chain`   — one dependent MAC chain (latency floor, context only)
//! - `scalar_ilp8`    — 8 independent scalar accumulators (scalar issue rate)
//! - `vec_mac_i64`    — `c[j] += a[j]*b[j]` slice loop (what LLVM auto-vectorizes
//!                      at this build's target features)
//! - `tile_4x8_i64`   — GEBP-shaped rank-1 update into a 4×8 register tile
//!                      (the exact inner shape of the packed GEMM micro-kernel)
//! - `vec_mac_f64`    — same slice loop in f64 (ISA-physics context: the i64/f64
//!                      roofline ratio bounds any exact-i64 vs BLAS comparison)
//! - `*_par`          — aggregate over all cores (`available_parallelism`)
//! - `gemm_1024`      — shipped `i64_ops::matmul` at n=1024, reported as
//!                      achieved Gops and % of the aggregate i64 roofline
//!
//! Ops convention: 1 MAC = 2 ops (mul + add), matching GEMM's `2·m·n·k`.
//!
//! ```text
//! cargo test --release --test i64_roofline -- --run
//! ```
//!
//! Output: `roofline\t<name>\t<gops>\t<detail>` plus provenance lines.

use std::env;
use std::hint::black_box;
use std::time::Instant;

use matlua::array::ArrayI64;
use matlua::linalg::i64_ops;

/// Elements per working buffer: 1024 × 8 B = 8 KB each; a+b+c ≤ 24 KB, inside
/// any 32 KB L1d. Compute-bound by construction (analyzed, DESIGN §3.26).
const N: usize = 1024;
/// Depth of the packed tile k-loop; 256 × (4+8) × 8 B = 24 KB, L1-resident.
const KDEPTH: usize = 256;

fn fill_i64(len: usize, seed: i64) -> Vec<i64> {
    let mut v = Vec::with_capacity(len);
    let mut x = seed;
    for _ in 0..len {
        // LCG; values are irrelevant, only that they are opaque to the optimizer.
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push(x);
    }
    v
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// Time `body` (which performs `ops_per_call` MAC-ops) with enough repetitions
/// per sample for ~100 ms, median of 5 samples. Returns Gops (2 ops per MAC).
fn gops(ops_per_call: u64, mut body: impl FnMut()) -> f64 {
    // Calibrate repetitions to ~100 ms per sample.
    let t0 = Instant::now();
    body();
    let one = t0.elapsed().as_secs_f64().max(1e-9);
    let reps = ((0.1 / one).ceil() as u64).clamp(1, 100_000_000);
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            body();
        }
        let dt = t.elapsed().as_secs_f64();
        samples.push((reps * ops_per_call * 2) as f64 / dt / 1e9);
    }
    median(samples)
}

fn scalar_chain(a: &[i64]) -> i64 {
    let mut s = 1i64;
    for &x in a {
        s = s.wrapping_mul(x).wrapping_add(x);
    }
    s
}

fn scalar_ilp8(a: &[i64], b: &[i64]) -> i64 {
    let mut s = [0i64; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    for (x, y) in ca.zip(cb) {
        for j in 0..8 {
            s[j] = s[j].wrapping_add(x[j].wrapping_mul(y[j]));
        }
    }
    s.iter().fold(0i64, |acc, &v| acc.wrapping_add(v))
}

fn vec_mac_i64(a: &[i64], b: &[i64], c: &mut [i64]) {
    for j in 0..c.len() {
        c[j] = c[j].wrapping_add(a[j].wrapping_mul(b[j]));
    }
}

fn vec_mac_f64(a: &[f64], b: &[f64], c: &mut [f64]) {
    for j in 0..c.len() {
        c[j] += a[j] * b[j];
    }
}

/// GEBP-shaped inner loop: rank-1 updates into a 4×8 accumulator tile from
/// packed A columns (4 per k) and packed B rows (8 per k).
macro_rules! tile_4x8_body {
    ($a_pack:expr, $b_pack:expr, $out:expr) => {{
        let (a_pack, b_pack) = ($a_pack, $b_pack);
        let out: &mut [i64; 32] = $out;
        let mut acc = [[0i64; 8]; 4];
        for p in 0..KDEPTH {
            let av = &a_pack[p * 4..p * 4 + 4];
            let bv = &b_pack[p * 8..p * 8 + 8];
            for i in 0..4 {
                let ai = av[i];
                for j in 0..8 {
                    acc[i][j] = acc[i][j].wrapping_add(ai.wrapping_mul(bv[j]));
                }
            }
        }
        for i in 0..4 {
            for j in 0..8 {
                out[i * 8 + j] = out[i * 8 + j].wrapping_add(acc[i][j]);
            }
        }
    }};
}

fn tile_4x8_i64(a_pack: &[i64], b_pack: &[i64], out: &mut [i64; 32]) {
    tile_4x8_body!(a_pack, b_pack, out);
}

/// Same tile compiled with AVX-512DQ enabled (native 64-bit lane multiply) —
/// the ceiling proxy for the shipped kernel's runtime-dispatched ISA path.
///
/// # Safety
/// Call only after `is_x86_feature_detected!` confirms avx512f+dq+bw+vl.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn tile_4x8_i64_avx512(a_pack: &[i64], b_pack: &[i64], out: &mut [i64; 32]) {
    tile_4x8_body!(a_pack, b_pack, out);
}

#[cfg(target_arch = "x86_64")]
fn avx512_ok() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}
#[cfg(not(target_arch = "x86_64"))]
fn avx512_ok() -> bool {
    false
}

fn emit(name: &str, gops: f64, detail: &str) {
    println!("roofline\t{name}\t{gops:.3}\t{detail}");
}

/// Run the tile kernel on every core at once (per-thread private buffers,
/// allocated once outside the timed region); returns aggregate Gops.
fn par_tile_gops(nthreads: usize, reps: u64, isa: bool) -> f64 {
    // Per-thread buffers really are allocated outside the timed region now:
    // a review found the previous version allocated and LCG-filled them
    // inside the clock, biasing the calibration-gate baselines low.
    let bufs: Vec<(Vec<i64>, Vec<i64>)> = (0..nthreads)
        .map(|tid| {
            (
                fill_i64(KDEPTH * 4, 17 + tid as i64),
                fill_i64(KDEPTH * 8, 19 + tid as i64),
            )
        })
        .collect();
    let t = Instant::now();
    std::thread::scope(|s| {
        for (ap, bp) in &bufs {
            s.spawn(move || {
                let mut out = [0i64; 32];
                for _ in 0..reps {
                    #[cfg(target_arch = "x86_64")]
                    if isa {
                        // SAFETY: caller gates on avx512_ok().
                        unsafe {
                            tile_4x8_i64_avx512(black_box(ap), black_box(bp), black_box(&mut out))
                        };
                        continue;
                    }
                    let _ = isa;
                    tile_4x8_i64(black_box(ap), black_box(bp), black_box(&mut out));
                }
            });
        }
    });
    let dt = t.elapsed().as_secs_f64();
    (nthreads as u64 * reps * (KDEPTH as u64 * 32) * 2) as f64 / dt / 1e9
}

fn dense(n: usize, seed: i64) -> ArrayI64 {
    ArrayI64::from_shape_vec(vec![n, n], fill_i64(n * n, seed)).unwrap()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if !args.iter().any(|a| a == "--run") {
        println!("i64_roofline: skipped (pass --run)");
        return;
    }
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("provenance\tthreads\t{nthreads}");
    println!("provenance\tops_convention\t1 MAC = 2 ops (matches GEMM 2mnk)");

    let a = black_box(fill_i64(N, 3));
    let b = black_box(fill_i64(N, 5));
    let mut c = black_box(fill_i64(N, 7));
    let af: Vec<f64> = a.iter().map(|&x| (x % 1024) as f64).collect();
    let bf: Vec<f64> = b.iter().map(|&x| (x % 1024) as f64).collect();
    let mut cf = vec![0.0f64; N];
    let a_pack = black_box(fill_i64(KDEPTH * 4, 11));
    let b_pack = black_box(fill_i64(KDEPTH * 8, 13));

    let g = gops(N as u64, || {
        black_box(scalar_chain(black_box(&a)));
    });
    emit("scalar_chain", g, "dependent MAC latency floor (context)");

    let g = gops(N as u64, || {
        black_box(scalar_ilp8(black_box(&a), black_box(&b)));
    });
    emit("scalar_ilp8", g, "8 independent scalar accumulators");

    let g = gops(N as u64, || {
        vec_mac_i64(black_box(&a), black_box(&b), black_box(&mut c));
    });
    emit("vec_mac_i64", g, "c[j]+=a[j]*b[j] slice loop, single thread");

    let g = gops(N as u64, || {
        vec_mac_f64(black_box(&af), black_box(&bf), black_box(&mut cf));
    });
    emit("vec_mac_f64", g, "f64 context: ISA physics ratio vs vec_mac_i64");

    let mut tile_out = [0i64; 32];
    let g_tile = gops((KDEPTH * 32) as u64, || {
        tile_4x8_i64(black_box(&a_pack), black_box(&b_pack), black_box(&mut tile_out));
    });
    emit("tile_4x8_i64", g_tile, "GEBP rank-1 update, 4x8 tile, single thread");

    // Aggregate (all cores). Repetitions sized from the single-thread rate for
    // a ~0.5 s window; setup is outside the timed loop.
    let per_call_ops = (KDEPTH * 32 * 2) as f64;
    let reps = ((0.5 * g_tile * 1e9) / per_call_ops).ceil() as u64;
    let g_par = par_tile_gops(nthreads, reps.max(1), false);
    emit("tile_4x8_i64_par", g_par, "aggregate over all cores");

    // ISA-dispatched tile (matches the shipped kernel's runtime AVX-512 path).
    let mut ceiling = g_par;
    let mut ceiling_name = "tile_par";
    if avx512_ok() {
        #[cfg(target_arch = "x86_64")]
        {
            let mut tile_out = [0i64; 32];
            let g_isa = gops((KDEPTH * 32) as u64, || {
                // SAFETY: gated on avx512_ok() above.
                unsafe {
                    tile_4x8_i64_avx512(
                        black_box(&a_pack),
                        black_box(&b_pack),
                        black_box(&mut tile_out),
                    )
                };
            });
            emit("tile_4x8_i64_isa", g_isa, "same tile, AVX-512DQ codegen, 1 thread");
            let reps = ((0.5 * g_isa * 1e9) / per_call_ops).ceil() as u64;
            let g_isa_par = par_tile_gops(nthreads, reps.max(1), true);
            emit("tile_4x8_i64_isa_par", g_isa_par, "aggregate over all cores");
            if g_isa_par > ceiling {
                ceiling = g_isa_par;
                ceiling_name = "isa_tile_par";
            }
        }
    }

    // Shipped GEMM, achieved Gops and % of the applicable tile roofline.
    let n = 1024usize;
    let am = dense(n, 23);
    let bm = dense(n, 29);
    // One untimed warm call (first-touch pages, rayon pool spin-up), then 5
    // samples like every other row.
    black_box(i64_ops::matmul(black_box(&am), black_box(&bm)).unwrap());
    let mut samples = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        black_box(i64_ops::matmul(black_box(&am), black_box(&bm)).unwrap());
        samples.push(t.elapsed().as_secs_f64());
    }
    let dt = median(samples);
    let achieved = (2 * n as u64 * n as u64 * n as u64) as f64 / dt / 1e9;
    let pct = 100.0 * achieved / ceiling;
    println!(
        "roofline\tgemm_1024\t{achieved:.3}\tachieved Gops ({:.1} ms); {pct:.0}% of {ceiling_name}",
        dt * 1e3
    );
}
