use core::fmt;

/// Errors returned when a kernel input or shape does not match its declared contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    DimensionOverflow,
    InvalidInputLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidRowRange {
        start: usize,
        end: usize,
        total: usize,
    },
    InvalidQuantizedDimension {
        name: &'static str,
        value: usize,
        multiple: usize,
    },
    AllocationFailure,
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOverflow => write!(f, "matrix dimensions overflow usize"),
            Self::InvalidInputLength { name, expected, actual } => write!(
                f,
                "{name} buffer is too short: expected at least {expected} elements, got {actual}"
            ),
            Self::InvalidRowRange { start, end, total } => {
                write!(f, "invalid output row range {start}..{end} for {total} rows")
            }
            Self::InvalidQuantizedDimension { name, value, multiple } => write!(
                f,
                "{name} dimension {value} must be a multiple of {multiple}"
            ),
            Self::AllocationFailure => write!(f, "kernel scratch allocation failed"),
        }
    }
}

impl std::error::Error for KernelError {}

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

    if c_len == 0 {
        return Ok((a_len, b_len, c_len));
    }

    for (name, actual, expected) in [
        ("A", a.len(), a_len),
        ("B", b.len(), b_len),
        ("C", c.len(), c_len),
    ] {
        if actual < expected {
            return Err(KernelError::InvalidInputLength { name, expected, actual });
        }
    }

    Ok((a_len, b_len, c_len))
}

pub(crate) fn validate_rhs_transposed_lengths(
    a: &[f32],
    b: &[f32],
    c_rows: &[f32],
    m: usize,
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
) -> Result<(usize, usize, usize), KernelError> {
    if row_start > row_end || row_end > n {
        return Err(KernelError::InvalidRowRange { start: row_start, end: row_end, total: n });
    }

    let a_len = m.checked_mul(k).ok_or(KernelError::DimensionOverflow)?;
    let b_len = n.checked_mul(k).ok_or(KernelError::DimensionOverflow)?;
    let c_len = (row_end - row_start).checked_mul(m).ok_or(KernelError::DimensionOverflow)?;

    if c_len == 0 {
        return Ok((a_len, b_len, c_len));
    }
    for (name, actual, expected) in [
        ("A", a.len(), a_len),
        ("B", b.len(), b_len),
        ("C row partition", c_rows.len(), c_len),
    ] {
        if actual < expected {
            return Err(KernelError::InvalidInputLength { name, expected, actual });
        }
    }
    Ok((a_len, b_len, c_len))
}

/// Scalar reference for C[M,N] = A[M,K] × B[K,N] using row-major contiguous matrices.
pub fn matmul_f32_reference(
    a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize,
) -> Result<(), KernelError> {
    let (_, _, c_len) = validate_lengths(a, b, c, m, n, k)?;
    if c_len == 0 { return Ok(()); }
    c[..c_len].fill(0.0);
    for row in 0..m {
        let a_row = row * k;
        let c_row = row * n;
        for p in 0..k {
            let av = a[a_row + p];
            let b_row = p * n;
            for col in 0..n {
                c[c_row + col] += av * b[b_row + col];
            }
        }
    }
    Ok(())
}

/// Scalar reference matching GGML's contiguous F32 MUL_MAT layout.
///
/// A has M rows of K floats, B has N rows of K floats, and output has N rows of M floats.
/// In other words, C[n,m] = dot(A[m,:], B[n,:]).
pub fn matmul_f32_rhs_transposed_reference(
    a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize,
) -> Result<(), KernelError> {
    let (_, _, c_len) = validate_rhs_transposed_lengths(a, b, c, m, n, k, 0, n)?;
    if c_len == 0 { return Ok(()); }
    for row_n in 0..n {
        for row_m in 0..m {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[row_m * k + p] * b[row_n * k + p];
            }
            c[row_n * m + row_m] = sum;
        }
    }
    Ok(())
}
