use core::fmt;

/// Errors returned when a matrix multiplication input does not match its declared shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// Computing one of the required buffer lengths overflowed usize.
    DimensionOverflow,
    /// One of the input/output slices is shorter than the shape requires.
    InvalidInputLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOverflow => write!(f, "matrix dimensions overflow usize"),
            Self::InvalidInputLength {
                name,
                expected,
                actual,
            } => write!(
                f,
                "{name} buffer is too short: expected at least {expected} elements, got {actual}"
            ),
        }
    }
}

impl std::error::Error for KernelError {}

/// Validate row-major A[M,K], B[K,N], and C[M,N] buffers.
///
/// Extra elements are permitted so callers can pass a sub-buffer without copying. The current
/// API assumes contiguous row-major inputs; explicit strides can be added when a real workload
/// needs them.
pub(crate) fn validate_lengths(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(usize, usize, usize), KernelError> {
    let a_len = m.checked_mul(k).ok_or(KernelError::DimensionOverflow)?;
    let b_len = k.checked_mul(n).ok_or(KernelError::DimensionOverflow)?;
    let c_len = m.checked_mul(n).ok_or(KernelError::DimensionOverflow)?;

    // An empty output is a valid no-op; none of A or B will be read.
    if c_len == 0 {
        return Ok((a_len, b_len, c_len));
    }

    for (name, actual, expected) in [
        ("A", a.len(), a_len),
        ("B", b.len(), b_len),
        ("C", c.len(), c_len),
    ] {
        if actual < expected {
            return Err(KernelError::InvalidInputLength {
                name,
                expected,
                actual,
            });
        }
    }

    Ok((a_len, b_len, c_len))
}

/// Scalar reference implementation for C[M,N] = A[M,K] × B[K,N].
///
/// All matrices are contiguous and row-major. The loop order walks each B row and C row
/// contiguously, making this useful both as a reference implementation and a modest scalar
/// baseline. It intentionally uses ordinary multiply-plus-add rather than an explicit fused
/// multiply-add so optimized implementations can be compared with a clear reference path.
pub fn matmul_f32_reference(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), KernelError> {
    let (_, _, c_len) = validate_lengths(a, b, c, m, n, k)?;
    matmul_f32_reference_validated(a, b, c, m, n, k, c_len);
    Ok(())
}

pub(crate) fn matmul_f32_reference_validated(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    c_len: usize,
) {
    if c_len == 0 {
        return;
    }

    c[..c_len].fill(0.0);

    for row in 0..m {
        let a_row = row * k;
        let c_row_start = row * n;

        for p in 0..k {
            let a_value = a[a_row + p];
            let b_row = p * n;
            let c_row = &mut c[c_row_start..c_row_start + n];
            let b_values = &b[b_row..b_row + n];

            for col in 0..n {
                c_row[col] += a_value * b_values[col];
            }
        }
    }
}
