use anyhow::{ensure, Context, Result};
use bytemuck::{cast_slice, try_cast_slice};
use ponko_compute::{benchmark::summarize, ComputeDevice};
use std::{env, path::PathBuf, time::Instant};

const DEFAULT_COUNT: usize = 65_536;
const MAX_COUNT: usize = 16_777_216;
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
    let actual: &[f32] = try_cast_slice(bytes).map_err(|error| anyhow::anyhow!("cast vector-add bytes to f32: {error:?}"))?;
    ensure!(
        actual.len() == expected.len(),
        "vector-add output length mismatch"
    );
    let mut max_abs_error = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        max_abs_error = max_abs_error.max(error);
        if !actual.is_finite() || error > 1.0e-5 {
            anyhow::bail!(
                "vector-add mismatch at index {index}: GPU={actual}, CPU={expected}, abs_error={error}"
            );
        }
    }
    Ok(max_abs_error)
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let shader_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "shaders/vector_add.spv".to_owned()),
    );
    let count = parse_usize(args.next(), "element count", DEFAULT_COUNT)?;
    let warmups = parse_usize(args.next(), "warm-up iteration count", DEFAULT_WARMUPS)?;
    let iterations = parse_usize(args.next(), "timed iteration count", DEFAULT_ITERATIONS)?;

    ensure!(
        (1..=MAX_COUNT).contains(&count),
        "element count must be in 1..={MAX_COUNT}"
    );
    ensure!(
        warmups <= MAX_WARMUPS,
        "warm-up iterations must be <= {MAX_WARMUPS}"
    );
    ensure!(
        (1..=MAX_ITERATIONS).contains(&iterations),
        "timed iterations must be in 1..={MAX_ITERATIONS}"
    );

    let a: Vec<f32> = (0..count).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..count).map(|i| i as f32 * 0.25 + 1.0).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
    let cpu_start = Instant::now();
    let mut cpu_check = vec![0.0_f32; count];
    for (out, (&a, &b)) in cpu_check.iter_mut().zip(a.iter().zip(&b)) {
        *out = a + b;
    }
    let cpu_elapsed = cpu_start.elapsed();
    ensure!(cpu_check == expected, "CPU reference self-check failed");

    let device = ComputeDevice::open()?;
    println!("device={}", device.device_name());
    println!("operation=vector_add elements={count} warmups={warmups} iterations={iterations}");
    println!("shader={}", shader_path.display());

    let a_buffer = device.create_buffer(count * std::mem::size_of::<f32>())?;
    let b_buffer = device.create_buffer(count * std::mem::size_of::<f32>())?;
    let out_buffer = device.create_buffer(count * std::mem::size_of::<f32>())?;
    device.upload(&a_buffer, cast_slice(&a))?;
    device.upload(&b_buffer, cast_slice(&b))?;
    let kernel = device.load_kernel(&shader_path, 3, std::mem::size_of::<u32>() as u32)?;
    let count_u32 = count as u32;
    let groups_x = ((count_u32 + 63) / 64).max(1);

    let execute_and_verify = || -> Result<(std::time::Duration, f32)> {
        let elapsed = device.dispatch(
            &kernel,
            &[&a_buffer, &b_buffer, &out_buffer],
            &count_u32.to_ne_bytes(),
            [groups_x, 1, 1],
        )?;
        let bytes = device.readback(&out_buffer, count * std::mem::size_of::<f32>())?;
        let max_error = verify_output(&bytes, &expected)?;
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
    println!("cpu_reference_ms={:.3}", ms(cpu_elapsed));
    println!("dispatch_ms_min={:.3}", ms(summary.min));
    println!("dispatch_ms_median={:.3}", ms(summary.median));
    println!("dispatch_ms_p95={:.3}", ms(summary.p95));
    println!("dispatch_ms_max={:.3}", ms(summary.max));
    println!("dispatch_ms_mean={:.3}", summary.mean_ms);
    println!("max_abs_error={max_abs_error:.8}");
    println!("PASS: vector-add output matches CPU reference for {warmups} warm-ups and {iterations} timed runs");
    Ok(())
}
