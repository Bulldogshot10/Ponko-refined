use anyhow::{ensure, Context, Result};
use bytemuck::{cast_slice, try_cast_slice};
use ponko_compute::ComputeDevice;
use std::{env, path::PathBuf, time::Instant};

const DEFAULT_DIMENSION: usize = 64;
const MAX_DIMENSION: usize = 4096;
const MAX_MULTIPLIES: usize = 64_000_000;

fn parse_dimension(value: Option<String>, label: &str) -> Result<usize> {
    value
        .map(|value| value.parse::<usize>())
        .transpose()
        .with_context(|| format!("parse {label} dimension"))
        .map(|value| value.unwrap_or(DEFAULT_DIMENSION))
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let shader_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "shaders/matmul.spv".to_owned()),
    );
    let m = parse_dimension(args.next(), "M")?;
    let n = parse_dimension(args.next(), "N")?;
    let k = parse_dimension(args.next(), "K")?;

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
    let a_count = m.checked_mul(k).context("A matrix size overflow")?;
    let b_count = k.checked_mul(n).context("B matrix size overflow")?;
    let c_count = m.checked_mul(n).context("C matrix size overflow")?;

    println!("matrix= A[{m}x{k}] * B[{k}x{n}] -> C[{m}x{n}]");

    // Deterministic, bounded values make this run reproducible and keep the
    // numerical comparison useful even on small test matrices.
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

    // The shader maps the x dimension to output columns and y to output rows.
    let kernel = device.load_kernel(&shader_path, 3, 3 * std::mem::size_of::<u32>() as u32)?;
    let push_constants = [m as u32, n as u32, k as u32];
    let workgroups = [
        ((n as u32) + 7) / 8,
        ((m as u32) + 7) / 8,
        1,
    ];
    let gpu_elapsed = device.dispatch(
        &kernel,
        &[&a_buffer, &b_buffer, &c_buffer],
        cast_slice(&push_constants),
        workgroups,
    )?;

    let bytes = device.readback(&c_buffer, c_count * std::mem::size_of::<f32>())?;
    let gpu_output: &[f32] = try_cast_slice(&bytes).context("cast output bytes to f32")?;
    let mut max_abs_error = 0.0_f32;
    for (index, (&actual, &expected)) in gpu_output.iter().zip(&cpu_output).enumerate() {
        let error = (actual - expected).abs();
        max_abs_error = max_abs_error.max(error);
        if !actual.is_finite() || error > 1.0e-3 {
            anyhow::bail!(
                "mismatch at index {index}: GPU={actual}, CPU={expected}, abs_error={error}"
            );
        }
    }

    let cpu_ms = cpu_elapsed.as_secs_f64() * 1000.0;
    let gpu_ms = gpu_elapsed.as_secs_f64() * 1000.0;
    let effective_gflops = if gpu_elapsed.is_zero() {
        0.0
    } else {
        (2.0 * multiply_count as f64) / gpu_elapsed.as_secs_f64() / 1.0e9
    };
    println!("cpu_reference_ms={cpu_ms:.3}");
    println!("vulkan_dispatch_wait_ms={gpu_ms:.3}");
    println!("effective_gflops={effective_gflops:.4}");
    println!("max_abs_error={max_abs_error:.8}");
    println!("PASS: matrix-multiply output matches CPU reference");
    Ok(())
}
