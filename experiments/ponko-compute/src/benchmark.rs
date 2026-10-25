//! Shared timing statistics for repeatable compute-kernel smoke tests.

use anyhow::{ensure, Result};
use std::time::Duration;

/// Summary of dispatch timings. Percentiles use nearest-rank semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingSummary {
    pub min: Duration,
    pub median: Duration,
    pub p95: Duration,
    pub max: Duration,
    pub mean_ms: f64,
}

fn duration_from_nanos(nanos: u128) -> Duration {
    Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

/// Summarize non-empty samples without silently accepting an empty benchmark.
pub fn summarize(samples: &[Duration]) -> Result<TimingSummary> {
    ensure!(!samples.is_empty(), "at least one timing sample is required");
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();

    let middle = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        duration_from_nanos(
            (sorted[middle - 1].as_nanos() + sorted[middle].as_nanos()) / 2,
        )
    } else {
        sorted[middle]
    };
    let p95_index = ((sorted.len() * 95).div_ceil(100) - 1).min(sorted.len() - 1);
    let mean_ms = sorted
        .iter()
        .map(Duration::as_secs_f64)
        .sum::<f64>()
        * 1000.0
        / sorted.len() as f64;

    Ok(TimingSummary {
        min: sorted[0],
        median,
        p95: sorted[p95_index],
        max: *sorted.last().expect("non-empty samples checked above"),
        mean_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(values: &[u64]) -> Vec<Duration> {
        values.iter().copied().map(Duration::from_millis).collect()
    }

    #[test]
    fn rejects_empty_samples() {
        assert!(summarize(&[]).is_err());
    }

    #[test]
    fn summarizes_unsorted_odd_sample_set() {
        let summary = summarize(&ms(&[5, 1, 4, 2, 3])).unwrap();
        assert_eq!(summary.min, Duration::from_millis(1));
        assert_eq!(summary.median, Duration::from_millis(3));
        assert_eq!(summary.p95, Duration::from_millis(5));
        assert_eq!(summary.max, Duration::from_millis(5));
        assert!((summary.mean_ms - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn averages_middle_pair_for_even_median() {
        let summary = summarize(&ms(&[4, 1, 3, 2])).unwrap();
        assert_eq!(summary.median, Duration::from_millis(2500));
        assert_eq!(summary.p95, Duration::from_millis(4));
    }

    #[test]
    fn p95_uses_nearest_rank() {
        let samples = (1..=100).collect::<Vec<u64>>();
        let summary = summarize(&ms(&samples)).unwrap();
        assert_eq!(summary.p95, Duration::from_millis(95));
    }
}
