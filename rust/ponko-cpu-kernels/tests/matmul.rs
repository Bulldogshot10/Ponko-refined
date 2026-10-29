use ponko_cpu_kernels::{
    matmul_f32, matmul_f32_reference, matmul_f32_rhs_transposed,
    matmul_f32_rhs_transposed_reference, selected_f32_kernel, KernelError, KernelKind,
};
use ponko_cpu_kernels::quant::{
    matmul_q4_0_f32, matmul_q4_0_f32_reference, matmul_q4_0_q8_0_reference,
    matmul_q4_0_q8_0_reference_range,
    BlockQ4_0, BlockQ8_0,
};

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        let difference = (a - e).abs();
        assert!(difference <= tolerance,
            "element {index}: actual={a}, expected={e}, difference={difference}");
    }
}

#[test]
fn reference_computes_non_square_product() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = [0.0; 4];
    matmul_f32_reference(&a, &b, &mut c, 2, 2, 3).unwrap();
    assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn dispatcher_matches_reference_for_odd_dimensions_and_tail_columns() {
    let (m, n, k) = (3, 7, 5);
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.125).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i * 11 % 29) as f32 - 14.0) * 0.0625).collect();
    let mut expected = vec![0.0; m * n];
    let mut actual = vec![0.0; m * n];
    matmul_f32_reference(&a, &b, &mut expected, m, n, k).unwrap();
    matmul_f32(&a, &b, &mut actual, m, n, k).unwrap();
    assert_close(&actual, &expected, 1.0e-4);
}

#[test]
fn ggml_rhs_transposed_layout_matches_dot_product_reference() {
    let (m, n, k) = (2, 2, 3);
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = [50.0, 122.0, 68.0, 167.0];
    let mut reference = [0.0; 4];
    let mut dispatched = [0.0; 4];
    matmul_f32_rhs_transposed_reference(&a, &b, &mut reference, m, n, k).unwrap();
    matmul_f32_rhs_transposed(&a, &b, &mut dispatched, m, n, k).unwrap();
    assert_eq!(reference, expected);
    assert_close(&dispatched, &expected, 1.0e-4);
}

#[test]
fn q4_0_q8_0_reference_matches_packed_blocks_and_row_partition() {
    let q4 = vec![BlockQ4_0 { d_bits: 0x3c00, qs: [0; 16] }; 2];
    let q8 = vec![BlockQ8_0 { d_bits: 0x3c00, qs: [1; 32] }; 3];
    let mut full = [0.0; 6];
    matmul_q4_0_q8_0_reference(&q4, &q8, &mut full, 2, 3, 32).unwrap();
    assert_eq!(full, [-256.0; 6]);

    let mut partition = [0.0; 2];
    matmul_q4_0_q8_0_reference_range(&q4, &q8, &mut partition, 2, 3, 32, 1, 2).unwrap();
    assert_eq!(partition, [-256.0; 2]);
}

#[test]
fn q4_0_f32_reference_uses_q8_0_activation_quantization() {
    let q4 = vec![BlockQ4_0 { d_bits: 0x3c00, qs: [0; 16] }; 2];
    let b = [1.0_f32; 96];
    let mut c = [0.0; 6];
    matmul_q4_0_f32_reference(&q4, &b, &mut c, 2, 3, 32).unwrap();
    assert_close(&c, &[-255.984375; 6], 1.0e-4);
}

#[test]
fn q4_0_requires_block_aligned_k() {
    let err = matmul_q4_0_f32_reference(&[], &[], &mut [], 1, 1, 31).unwrap_err();
    assert_eq!(err, KernelError::InvalidQuantizedDimension {
        name: "K", value: 31, multiple: 32,
    });
}

#[test]
fn supports_empty_output_dimensions() {
    let mut c = [];
    assert_eq!(matmul_f32(&[], &[], &mut c, 0, 4, 3), Ok(()));
    assert_eq!(matmul_f32(&[], &[], &mut c, 2, 0, 3), Ok(()));
}

#[test]
fn rejects_short_input_buffers() {
    let mut c = [0.0; 4];
    let err = matmul_f32(&[1.0, 2.0], &[1.0, 2.0, 3.0, 4.0], &mut c, 2, 2, 2).unwrap_err();
    assert_eq!(err, KernelError::InvalidInputLength { name: "A", expected: 4, actual: 2 });
}

#[test]
fn rejects_dimension_overflow_before_indexing() {
    assert_eq!(
        matmul_f32_reference(&[], &[], &mut [], usize::MAX, 2, 2).unwrap_err(),
        KernelError::DimensionOverflow
    );
}

#[test]
fn reports_the_selected_implementation() {
    #[cfg(target_arch = "aarch64")]
    assert_eq!(selected_f32_kernel(), KernelKind::Aarch64NeonF32);
    #[cfg(not(target_arch = "aarch64"))]
    assert_eq!(selected_f32_kernel(), KernelKind::Reference);
    assert!(!selected_f32_kernel().name().is_empty());
}


#[test]
fn q4_0_dispatch_matches_scalar_reference() {
    let q4 = vec![BlockQ4_0 { d_bits: 0x3c00, qs: [0x89; 16] }; 4];
    let b: Vec<f32> = (0..3 * 64).map(|i| ((i % 19) as f32 - 8.0) * 0.125).collect();
    let mut reference = vec![0.0; 2 * 3];
    let mut dispatched = vec![0.0; 2 * 3];
    matmul_q4_0_f32_reference(&q4, &b, &mut reference, 2, 3, 64).unwrap();
    matmul_q4_0_f32(&q4, &b, &mut dispatched, 2, 3, 64).unwrap();
    assert_close(&dispatched, &reference, 1.0e-3);
}
