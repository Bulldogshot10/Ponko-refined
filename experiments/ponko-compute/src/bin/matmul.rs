use anyhow::{ensure, Context, Result};
use bytemuck::{cast_slice, try_cast_slice};
use ponko_compute::{benchmark::summarize, ComputeDevice};
use std::{env, path::PathBuf, time::Instant};

const DEFAULT_DIMENSION: usize = 64;
const MAX_DIMENSION: usize = 4096;
const MAX_MULTIPLIES: usize = 64_000_000;
const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_ITERATIONS: usize = 10;
const MAX_WARMUPS: usize = 100;
const MAX_ITERATIONS: usize = 1_000;

fn parse_usize(value: Option<String>, label: &str, default: usize) -> Result<usize> {
    value
        .map(|value| value.parse::<usize>())
        .transpose()
        .with_context(|| format!("parse {label}"))
        .map(|value| value.unwrap_or(default))
}

fn verify_output(bytes: &[u8], expected: &[f32]) -> Result<f32> {
    let actual: &[f32] = try_cast_slice(bytes).context("cast matrix output bytes to f32")?;
    ensure!(
        actual.len() == expected.len(),
        "matrix output length mismatch"
    );
    let mut max_abs_error = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        max_abs_error = max_abs_error.max(error);
        if !actual.is_finite() || error > 1.0e-3 {
            anyhow::bail!(
                "matrix mismatch at index {index}: GPU={actual}, CPU={expected}, abs_error={error}"
            );
        }
    }
    Ok(max_abs_error)
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let shader_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "shaders/matmul.spv".to_owned()),
    );
    let m = parse_usize(args.next(), "M dimension", DEFAULT_DIMENSION)?;
    let n = parse_usize(args.next(), "N dimension", DEFAULT_DIMENSION)?;
    let k = parse_usize(args.next(), "K dimension", DEFAULT_DIMENSION)?;
    let warmups = parse_usize(args.next(), "warm-up iteration count", DEFAULT_WARMUPS)?;
    let iterations = parse_usize(args.next(), "timed iteration count", DEFAULT_ITERATIONS)?;

    ensure!(
        (1..=MAX_DIMENSION).contains(&m)
            && (1..=MAX_DIMENSION).contains(&n)
            && (1..=MAX_DIMENSION).contains(&k),
        "M, N, and K must each be in 1..={MAX_DIMENSION}"
    );
    let multiply_count = m
        .checked_mul(n)
        .and_then(|value| value.checked_mul(k))
        .context("matrix dimensions overflow")?;
    ensure!(
        multiply_count <= MAX_MULTIPLIES,
        "M*N*K must be <= {MAX_MULTIPLIES} to keep the CPU reference bounded"
    );
    ensure!(
        warmups <= MAX_WARMUPS,
        "warm-up iterations must be <= {MAX_WARMUPS}"
    );
    ensure!(
        (1..=MAX_ITERATIONS).contains(&iterations),
        "timed iterations must be in 1..={MAX_ITERATIONS}"
    );
    let a_count = m.checked_mul(k).context("A matrix size overflow")?;
    let b_count = k.checked_mul(n).context("B matrix size overflow")?;
    let c_count = m.checked_mul(n).context("C matrix size overflow")?;

    println!("operation=matmul shape={m}x{k} * {k}x{n} -> {m}x{n} warmups={warmups} iterations={iterations}");

    let a: Vec<f32> = (0..a_count)
        .map(|index| ((index % 23) as f32 - 11.0) / 11.0)
        .collect();
    let b: Vec<f32> = (0..b_count)
        .map(|index| ((index % 17) as f32 - 8.0) / 8.0)
        .collect();

    let mut cpu_output = vec![0.0_f32; c_count];
    let cpu_start = Instant::now();
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0_f32;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            cpu_output[row * n + col] = sum;
        }
    }
    let cpu_elapsed = cpu_start.elapsed();

    let device = ComputeDevice::open()?;
    println!("device={}", device.device_name());
    println!("shader={}", shader_path.display());

    let a_buffer = device.create_buffer(a_count * std::mem::size_of::<f32>())?;
    let b_buffer = device.create_buffer(b_count * std::mem::size_of::<f32>())?;
    let c_buffer = device.create_buffer(c_count * std::mem::size_of::<f32>())?;
    device.upload(&a_buffer, cast_slice(&a))?;
    device.upload(&b_buffer, cast_slice(&b))?;

    let kernel = device.load_kernel(&shader_path, 3, 3 * std::mem::size_of::<u32>() as u32)?;
    let push_constants = [m as u32, n as u32, k as u32];
    let workgroups = [((n as u32) + 7) / 8, ((m as u32) + 7) / 8, 1];

    let execute_and_verify = || -> Result<(std::time::Duration, f32)> {
        let elapsed = device.dispatch(
            &kernel,
            &[&a_buffer, &b_buffer, &c_buffer],
            cast_slice(&push_constants),
            workgroups,
        )?;
        let bytes = device.readback(&c_buffer, c_count * std::mem::size_of::<f32>())?;
        let max_error = verify_output(&bytes, &cpu_output)?;
        Ok((elapsed, max_error))
    };

    for _ in 0..warmups {
        execute_and_verify()?;
    }
    let mut samples = Vec::with_capacity(iterations);
    let mut max_abs_error = 0.0_f32;
    for _ in 0..iterations {
        let (elapsed, error) = execute_and_verify()?;
        samples.push(elapsed);
        max_abs_error = max_abs_error.max(error);
    }

    let summary = summarize(&samples)?;
    let ms = |duration: std::time::Duration| duration.as_secs_f64() * 1000.0;
    let median_seconds = summary.median.as_secs_f64();
    let effective_gflops = if median_seconds == 0.0 {
        0.0
    } else {
        (2.0 * multiply_count as f64) / median_seconds / 1.0e9
    };
    println!("cpu_reference_ms={:.3}", ms(cpu_elapsed));
    println!("dispatch_ms_min={:.3}", ms(summary.min));
    println!("dispatch_ms_median={:.3}", ms(summary.median));
    println!("dispatch_ms_p95={:.3}", ms(summary.p95));
    println!("dispatch_ms_max={:.3}", ms(summary.max));
    println!("dispatch_ms_mean={:.3}", summary.mean_ms);
    println!("effective_gflops_at_median={effective_gflops:.4}");
    println!("max_abs_error={max_abs_error:.8}");
    println!("PASS: matrix-multiply output matches CPU reference for {warmups} warm-ups and {iterations} timed runs");
    Ok(())
}
