use anyhow::{ensure, Context, Result};
use bytemuck::{cast_slice, try_cast_slice};
use ponko_compute::ComputeDevice;
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let shader_path = PathBuf::from(
        args.next().unwrap_or_else(|| "shaders/vector_add.spv".into()),
    );
    let count: usize = args
        .next()
        .map(|value| value.parse())
        .transpose()
        .context("parse element count")?
        .unwrap_or(65_536);
    ensure!(
        count > 0 && count <= 16_777_216,
        "element count must be in 1..=16,777,216"
    );

    let device = ComputeDevice::open()?;
    println!("device={}", device.device_name());
    println!("elements={count}; shader={}", shader_path.display());

    let a: Vec<f32> = (0..count).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..count).map(|i| i as f32 * 0.25 + 1.0).collect();
    let mut cpu_out = vec![0.0_f32; count];
    let cpu_start = Instant::now();
    for i in 0..count {
        cpu_out[i] = a[i] + b[i];
    }
    let cpu_elapsed = cpu_start.elapsed();

    let a_buffer = device.create_buffer(count * std::mem::size_of::<f32>())?;
    let b_buffer = device.create_buffer(count * std::mem::size_of::<f32>())?;
    let out_buffer = device.create_buffer(count * std::mem::size_of::<f32>())?;
    device.upload(&a_buffer, cast_slice(&a))?;
    device.upload(&b_buffer, cast_slice(&b))?;
    let kernel = device.load_kernel(&shader_path, 3, std::mem::size_of::<u32>() as u32)?;
    let count_u32 = count as u32;
    let groups_x = ((count_u32 + 63) / 64).max(1);
    let gpu_elapsed = device.dispatch(
        &kernel,
        &[&a_buffer, &b_buffer, &out_buffer],
        &count_u32.to_ne_bytes(),
        [groups_x, 1, 1],
    )?;

    let bytes = device.readback(&out_buffer, count * std::mem::size_of::<f32>())?;
    let output: &[f32] = try_cast_slice(&bytes).context("cast output bytes to f32")?;
    let mut max_abs_error = 0.0_f32;
    for ((&actual, &expected), index) in output.iter().zip(&cpu_out).zip(0..) {
        let error = (actual - expected).abs();
        max_abs_error = max_abs_error.max(error);
        if !actual.is_finite() || error > 1.0e-5 {
            anyhow::bail!(
                "mismatch at index {index}: GPU={actual}, CPU={expected}, abs_error={error}"
            );
        }
    }

    println!("cpu_vector_add_ms={:.3}", cpu_elapsed.as_secs_f64() * 1000.0);
    println!("vulkan_dispatch_wait_ms={:.3}", gpu_elapsed.as_secs_f64() * 1000.0);
    println!("max_abs_error={max_abs_error:.8}");
    println!("PASS: vector-add output matches CPU reference");
    Ok(())
}
