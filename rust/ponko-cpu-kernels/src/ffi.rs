//! Minimal C ABI used by the optional GGML CPU integration path.

use core::slice;

use crate::dispatch::matmul_f32_rhs_transposed_range;
use crate::matmul::KernelError;
use crate::quant::q4_0::{matmul_q4_0_f32_range, BlockQ4_0, QK4_0};

fn result_code(result: Result<(), KernelError>) -> i32 {
    if result.is_ok() { 1 } else { 0 }
}

/// C passes pointers to full contiguous matrices; this call owns output rows row_start..row_end.
#[no_mangle]
pub unsafe extern "C" fn ponko_cpu_matmul_f32_rhs_transposed(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: u64,
    n: u64,
    k: u64,
    row_start: u64,
    row_end: u64,
) -> i32 {
    let (Ok(m), Ok(n), Ok(k), Ok(row_start), Ok(row_end)) = (
        usize::try_from(m), usize::try_from(n), usize::try_from(k),
        usize::try_from(row_start), usize::try_from(row_end),
    ) else { return 0; };

    if row_start > row_end || row_end > n { return 0; }
    let Some(a_len) = m.checked_mul(k) else { return 0; };
    let Some(b_len) = n.checked_mul(k) else { return 0; };
    let Some(c_len) = (row_end - row_start).checked_mul(m) else { return 0; };
    let Some(c_offset) = row_start.checked_mul(m) else { return 0; };
    let Some(_total_c_len) = n.checked_mul(m) else { return 0; };
    if c_len == 0 { return 1; }
    if c.is_null() || (a_len != 0 && a.is_null()) || (b_len != 0 && b.is_null()) { return 0; }

    let a_slice = if a_len == 0 { &[] } else {
        // SAFETY: the patched GGML call passes validated tensor buffers.
        unsafe { slice::from_raw_parts(a, a_len) }
    };
    let b_slice = if b_len == 0 { &[] } else {
        // SAFETY: the patched GGML call passes validated tensor buffers.
        unsafe { slice::from_raw_parts(b, b_len) }
    };
    // SAFETY: each GGML worker owns a disjoint range of rows in the full output matrix.
    let c_slice = unsafe { slice::from_raw_parts_mut(c.add(c_offset), c_len) };
    result_code(matmul_f32_rhs_transposed_range(
        a_slice, b_slice, c_slice, m, n, k, row_start, row_end,
    ))
}

/// Compute a partition of GGML Q4_0 × F32 MUL_MAT using Q8_0 quantization internally.
///
/// The Exynos build dispatches paired rows to I8MM and odd edges to DOTPROD; host/reference
/// builds use the scalar implementation for correctness comparison.
#[no_mangle]
pub unsafe extern "C" fn ponko_cpu_matmul_q4_0_f32(
    a: *const core::ffi::c_void,
    b: *const f32,
    c: *mut f32,
    m: u64,
    n: u64,
    k: u64,
    row_start: u64,
    row_end: u64,
) -> i32 {
    let (Ok(m), Ok(n), Ok(k), Ok(row_start), Ok(row_end)) = (
        usize::try_from(m), usize::try_from(n), usize::try_from(k),
        usize::try_from(row_start), usize::try_from(row_end),
    ) else { return 0; };

    if k % QK4_0 != 0 || row_start > row_end || row_end > n { return 0; }
    let blocks = k / QK4_0;
    let Some(a_len) = m.checked_mul(blocks) else { return 0; };
    let Some(b_len) = n.checked_mul(k) else { return 0; };
    let Some(c_len) = (row_end - row_start).checked_mul(m) else { return 0; };
    if c_len == 0 { return 1; }
    if c.is_null() || (a_len != 0 && a.is_null()) || (b_len != 0 && b.is_null()) { return 0; }

    let a_blocks: &[BlockQ4_0] = if a_len == 0 { &[] } else {
        // SAFETY: GGML Q4_0 buffers contain a_len complete, naturally aligned blocks.
        unsafe { slice::from_raw_parts(a.cast::<BlockQ4_0>(), a_len) }
    };
    let b_slice: &[f32] = if b_len == 0 { &[] } else {
        // SAFETY: GGML passed a contiguous F32 activation tensor of n*k elements.
        unsafe { slice::from_raw_parts(b, b_len) }
    };
    // SAFETY: workers use disjoint output row ranges.
    let c_slice = unsafe { slice::from_raw_parts_mut(c.add(row_start * m), c_len) };

    result_code(matmul_q4_0_f32_range(
        a_blocks, b_slice, c_slice, m, n, k, row_start, row_end, false,
    ))
}
