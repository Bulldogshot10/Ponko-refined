use ponko_cpu_kernels::quant::{matmul_q4_0_f32, matmul_q4_0_f32_reference, BlockQ4_0};
use ponko_cpu_kernels::{matmul_f32, matmul_f32_reference};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn inputs(len: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| ((i * 17 % modulus) as f32 - (modulus as f32 / 2.0)) * scale).collect()
}

fn time_loop(iterations: usize, mut f: impl FnMut()) -> Duration {
    f();
    let start = Instant::now();
    for _ in 0..iterations { f(); }
    start.elapsed()
}

fn time_case(m: usize, n: usize, k: usize, iterations: usize) {
    let a = inputs(m * k, 37, 0.03125);
    let b = inputs(k * n, 41, 0.015625);
    let mut c = vec![0.0_f32; m * n];
    let reference = time_loop(iterations, || {
        matmul_f32_reference(black_box(&a), black_box(&b), black_box(&mut c), m, n, k).unwrap();
    });
    let dispatched = time_loop(iterations, || {
        matmul_f32(black_box(&a), black_box(&b), black_box(&mut c), m, n, k).unwrap();
    });
    println!(
        "FP32 M={m} N={n} K={k}: reference={:.3} ms/op, dispatch={:.3} ms/op, speedup={:.2}x",
        reference.as_secs_f64() * 1000.0 / iterations as f64,
        dispatched.as_secs_f64() * 1000.0 / iterations as f64,
        reference.as_secs_f64() / dispatched.as_secs_f64(),
    );
}

fn time_q4_case(m: usize, n: usize, k: usize, iterations: usize) {
    assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let q4 = vec![BlockQ4_0 { d_bits: 0x3c00, qs: [0x89; 16] }; m * blocks_per_row];
    let activations = inputs(n * k, 43, 0.01);
    let mut output = vec![0.0_f32; m * n];
    let reference = time_loop(iterations, || {
        matmul_q4_0_f32_reference(
            black_box(&q4), black_box(&activations), black_box(&mut output), m, n, k,
        ).unwrap();
    });
    let dispatched = time_loop(iterations, || {
        matmul_q4_0_f32(
            black_box(&q4), black_box(&activations), black_box(&mut output), m, n, k,
        ).unwrap();
    });
    println!(
        "Q4_0xF32 M={m} N={n} K={k}: reference={:.3} ms/op, Exynos dispatch={:.3} ms/op, speedup={:.2}x",
        reference.as_secs_f64() * 1000.0 / iterations as f64,
        dispatched.as_secs_f64() * 1000.0 / iterations as f64,
        reference.as_secs_f64() / dispatched.as_secs_f64(),
    );
}

fn main() {
    println!("Ponko CPU kernel microbench; run on Exynos 2400E for target-device timings.");
    time_case(32, 48, 64, 40);
    time_case(64, 96, 256, 8);
    time_case(128, 128, 384, 2);
    time_q4_case(256, 16, 1024, 4);
}
