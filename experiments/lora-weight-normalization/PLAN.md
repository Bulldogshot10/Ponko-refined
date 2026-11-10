# LoRA Weight Normalization — Work Slot

**Status:** Not implemented. Design and code-path audit pending.
**Scope:** LoRA weight normalization only. Keep this independent from the Rust CPU-kernel dispatch work.

## Objective

Design and implement an explicit normalization stage for LoRA weights or their effective update. The exact mathematical operation and insertion point are intentionally undecided until the current adapter loading and fusion paths have been inspected.

## First pass: trace the existing path

- Trace LoRA file loading, tensor-name matching, factor registration, alpha/rank scaling, fusion/application, and any export/conversion path.
- Identify the exact tensor representation available at each stage (individual factors, reconstructed delta, or fused base weight).
- Identify supported base model families, LoRA formats, tensor dtypes, and any quantized paths.

## Decisions required before implementation

- **What is normalized:** each factor independently, the effective low-rank update, or another representation.
- **Norm/statistic:** choose and document the mathematical norm or statistic; do not assume unit-norm normalization is correct by default.
- **Granularity/axis:** per tensor, per output channel, per input channel, per rank component, or another clearly defined grouping.
- **Target scale:** define whether the target is fixed, derived from the base weight, derived from the adapter, or user-configurable.
- **Scaling interaction:** preserve the intended semantics of LoRA rank and alpha, adapter multipliers, and multiple simultaneously applied adapters.
- **Application point:** decide whether normalization belongs at load time, immediately before fusion/application, or in an explicit preprocessing/export operation.
- **Numerics:** define behavior for zero and near-zero norms, non-finite values, FP16/BF16/FP32, and quantized tensors.
- **Compatibility:** determine whether the feature is opt-in and how existing unnormalized LoRAs retain their current behavior.

## Validation plan

- Unit tests for formula, dimensions, axis/granularity, zero and near-zero inputs, NaN/Inf handling, and dtype conversion.
- Regression tests covering rank/alpha scaling, adapter multipliers, multiple LoRAs, and representative SD model families.
- Compare effective weight deltas and generated outputs before/after normalization with fixed seeds and settings.
- Measure runtime and peak memory at the selected insertion point.
- Run GitHub Actions CI; record build/test validation separately from real-device validation.

## Boundaries

- Do not modify the user's original adapter files in place.
- Do not silently alter existing LoRA scale/alpha semantics.
- Do not select a normalization formula from a benchmark convenience alone; justify it against the intended behavior and current code path.
- No implementation is included in this planning slot yet.
