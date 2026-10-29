use ponko_cpu_kernels::{matmul_f32, matmul_f32_reference};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn inputs(len: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 17 % modulus) as f32 - (modulus as f32 / 2.0)) * scale)
        .collect()
}

fn time_case(m: usize, n: usize, k: usize, iterations: usize) {
    let a = inputs(m * k, 37, 0.03125);
    let b = inputs(k * n, 41, 0.015625);
    let mut c = vec![0.0_f32; m * n];

    let reference = time_loop(iterations, || {
        matmul_f32_reference(black_box(&a), black_box(&b), black_box(&mut c), m, n, k)
            .expect("reference kernel");
    });
    let dispatched = time_loop(iterations, || {
        matmul_f32(black_box(&a), black_box(&b), black_box(&mut c), m, n, k)
            .expect("dispatched kernel");
    });

    println!(
        "shape M={m} N={n} K={k} iterations={iterations}: reference={:.3} ms/op, dispatch={:.3} ms/op, speedup={:.2}x",
        reference.as_secs_f64() * 1_000.0 / iterations as f64,
        dispatched.as_secs_f64() * 1_000.0 / iterations as f64,
        reference.as_secs_f64() / dispatched.as_secs_f64(),
    );
}

fn time_loop(iterations: usize, mut f: impl FnMut()) -> Duration {
    // A warm-up run keeps one-time runtime effects out of the timed sample.
    f();
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed()
}

fn main() {
    println!("Ponko CPU kernel microbenchmark; timings are local measurements, not CI assertions.");
    time_case(32, 48, 64, 40);
    time_case(64, 96, 256, 8);
    time_case(128, 128, 384, 2);
}
