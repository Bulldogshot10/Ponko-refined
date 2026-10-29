use ponko_cpu_kernels::{
    matmul_f32, matmul_f32_reference, selected_f32_kernel, KernelError, KernelKind,
};

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        let difference = (a - e).abs();
        assert!(
            difference <= tolerance,
            "element {index}: actual={a}, expected={e}, difference={difference}"
        );
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
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.125)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 11 % 29) as f32 - 14.0) * 0.0625)
        .collect();
    let mut expected = vec![0.0; m * n];
    let mut actual = vec![0.0; m * n];

    matmul_f32_reference(&a, &b, &mut expected, m, n, k).unwrap();
    matmul_f32(&a, &b, &mut actual, m, n, k).unwrap();

    assert_close(&actual, &expected, 1.0e-4);
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

    let err = matmul_f32(&[1.0, 2.0], &[1.0, 2.0, 3.0, 4.0], &mut c, 2, 2, 2)
        .unwrap_err();

    assert_eq!(
        err,
        KernelError::InvalidInputLength {
            name: "A",
            expected: 4,
            actual: 2,
        }
    );
}

#[test]
fn rejects_dimension_overflow_before_indexing() {
    let err = matmul_f32_reference(&[], &[], &mut [], usize::MAX, 2, 2).unwrap_err();
    assert_eq!(err, KernelError::DimensionOverflow);
}

#[test]
fn reports_the_selected_implementation() {
    #[cfg(target_arch = "aarch64")]
    assert_eq!(selected_f32_kernel(), KernelKind::Aarch64NeonF32);

    #[cfg(not(target_arch = "aarch64"))]
    assert_eq!(selected_f32_kernel(), KernelKind::Reference);

    assert!(!selected_f32_kernel().name().is_empty());
}
