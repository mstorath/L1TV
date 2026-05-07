// Ported from Auxiliary/distTransformL1.m (Storath & Weinmann)
// by Claude Opus coding agent, Anthropic, 2026.
//
// Forward + backward L1 distance transform on a sorted 1-D candidate grid.
//
// Given an initial cost vector b[k] anchored at sorted positions r[k] and a
// non-negative slope alpha, computes the lower envelope
//
//     b[k] := min_j ( b_initial[j] + alpha * |r[k] - r[j]| )
//
// in O(K) by two sweeps over the array (forward, then backward). When the
// caller also wants to know which source index j won at each k, use the
// `_with_argmin` variant: it carries argmin pointers along the same sweeps.

/// In-place L1 distance transform on a sorted grid.
///
/// `r` must be sorted ascending; `b.len() == r.len()`; `alpha >= 0`.
///
/// Both DP solvers in this crate use `l1_dt_with_argmin` instead — they
/// always need argmin pointers for backtracking. The argmin-free variant is
/// retained for testing and for downstream Rust users who only need the
/// envelope.
#[allow(dead_code)]
pub fn l1_dt(b: &mut [f64], r: &[f64], alpha: f64) {
    let k = b.len();
    debug_assert_eq!(r.len(), k);
    if k <= 1 {
        return;
    }
    for i in 1..k {
        let cand = b[i - 1] + alpha * (r[i] - r[i - 1]);
        if cand < b[i] {
            b[i] = cand;
        }
    }
    for i in (0..k - 1).rev() {
        let cand = b[i + 1] + alpha * (r[i + 1] - r[i]);
        if cand < b[i] {
            b[i] = cand;
        }
    }
}

/// In-place L1 distance transform with argmin tracking.
///
/// On return, `argmin[k]` holds the index j ∈ {0..b.len()} of the source value
/// that minimised `b_initial[j] + alpha * |r[k] - r[j]|`. The caller does not
/// need to pre-fill `argmin`; this function initialises it.
pub fn l1_dt_with_argmin(b: &mut [f64], r: &[f64], alpha: f64, argmin: &mut [u32]) {
    let k = b.len();
    debug_assert_eq!(r.len(), k);
    debug_assert_eq!(argmin.len(), k);
    if k == 0 {
        return;
    }
    for i in 0..k {
        argmin[i] = i as u32;
    }
    if k == 1 {
        return;
    }
    for i in 1..k {
        let cand = b[i - 1] + alpha * (r[i] - r[i - 1]);
        if cand < b[i] {
            b[i] = cand;
            argmin[i] = argmin[i - 1];
        }
    }
    for i in (0..k - 1).rev() {
        let cand = b[i + 1] + alpha * (r[i + 1] - r[i]);
        if cand < b[i] {
            b[i] = cand;
            argmin[i] = argmin[i + 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// O(K²) reference: brute-force minimum over all source indices.
    fn naive_dt(b: &[f64], r: &[f64], alpha: f64) -> Vec<f64> {
        let k = b.len();
        let mut out = vec![f64::INFINITY; k];
        for i in 0..k {
            for j in 0..k {
                let cand = b[j] + alpha * (r[i] - r[j]).abs();
                if cand < out[i] {
                    out[i] = cand;
                }
            }
        }
        out
    }

    #[test]
    fn matches_naive_on_symmetric_v() {
        let b_init = vec![10.0, 5.0, 0.0, 5.0, 10.0];
        let r = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        for &alpha in &[0.0, 0.1, 1.0, 10.0] {
            let mut b = b_init.clone();
            l1_dt(&mut b, &r, alpha);
            let expected = naive_dt(&b_init, &r, alpha);
            for i in 0..b.len() {
                assert!(
                    (b[i] - expected[i]).abs() < 1e-12,
                    "mismatch at i={i}, alpha={alpha}: got {}, expected {}",
                    b[i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn matches_naive_on_random() {
        // Deterministic LCG for reproducibility without rand-crate.
        let mut state: u64 = 0xDEADBEEF;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0
        };
        for _trial in 0..20 {
            let k = 8 + (next().abs() * 50.0) as usize;
            let mut b: Vec<f64> = (0..k).map(|_| next() * 10.0).collect();
            let mut r: Vec<f64> = (0..k).map(|_| next() * 5.0).collect();
            r.sort_by(|a, b| a.partial_cmp(b).unwrap());
            r.dedup();
            let k = r.len();
            b.truncate(k);
            let alpha = next().abs() * 2.0 + 0.1;
            let b_init = b.clone();
            l1_dt(&mut b, &r, alpha);
            let expected = naive_dt(&b_init, &r, alpha);
            for i in 0..k {
                assert!(
                    (b[i] - expected[i]).abs() < 1e-9,
                    "trial: k={k}, alpha={alpha}, i={i}, got {}, expected {}",
                    b[i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn argmin_recovers_min() {
        let b_init = vec![10.0, 5.0, 0.0, 5.0, 10.0];
        let r = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let alpha = 1.0;
        let mut b = b_init.clone();
        let mut am = vec![0u32; b.len()];
        l1_dt_with_argmin(&mut b, &r, alpha, &mut am);
        for i in 0..b.len() {
            let j = am[i] as usize;
            let recomputed = b_init[j] + alpha * (r[i] - r[j]).abs();
            assert!(
                (b[i] - recomputed).abs() < 1e-12,
                "argmin[{i}]={j} did not produce the recorded min"
            );
        }
    }

    #[test]
    fn zero_alpha_yields_global_min_everywhere() {
        // With alpha=0 the cost function is independent of position, so the
        // lower envelope at every k equals min(b_init).
        let b_init = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let r = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut b = b_init.clone();
        l1_dt(&mut b, &r, 0.0);
        let global_min = b_init.iter().cloned().fold(f64::INFINITY, f64::min);
        for &v in &b {
            assert_eq!(v, global_min);
        }
    }

    #[test]
    fn singleton_unchanged() {
        let mut b = vec![42.0];
        let r = vec![0.0];
        l1_dt(&mut b, &r, 1.0);
        assert_eq!(b, vec![42.0]);
    }

    #[test]
    fn empty_does_not_panic() {
        let mut b: Vec<f64> = vec![];
        let r: Vec<f64> = vec![];
        l1_dt(&mut b, &r, 1.0);
        let mut am: Vec<u32> = vec![];
        l1_dt_with_argmin(&mut b, &r, 1.0, &mut am);
    }
}
