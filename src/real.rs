// Ported from L1TV_Real.m (Storath, Weinmann, Unser 2016)
// by Claude Opus coding agent, Anthropic, 2026.
//
// Exact L1-TV solver for real-valued 1-D signals.
//
// Solves
//   min_{x ∈ R^N}  α · ∑_{i=1..N-1} |x[i] - x[i+1]|
//                + ∑_{i=1..N}     w[i] · |x[i] - y[i]|
//
// Reference: Storath, Weinmann, Unser. "Exact algorithms for L^1-TV
// regularization of real-valued or circle-valued signals." SIAM J. Sci.
// Comput. 38(1), A614-A630, 2016.
//
// Algorithm: dynamic programming over the candidate set V = sorted unique(y).
// The candidate-set theorem (loc. cit. §3, weighted-median argument) guarantees
// the optimum lies in V even with per-sample weights. The DP recurrence
//
//   B[i, n] = w[n]·|V[i] - y[n]|  +  min_j ( B[j, n-1] + α·|V[i] - V[j]| )
//
// is computed in O(NK) per step using the L1 distance transform; the naive
// O(NK²) path is retained behind `use_dist_transform=false` for testing parity.
//
// Memory: rather than MATLAB's full O(NK)·8-byte cost table, we store only a
// rolling pair of K-cost columns plus an O(NK)·4-byte argmin-pointer table for
// O(N) backtracking. Net memory ~half MATLAB's at typical sizes; can drop to
// a quarter by switching argmin pointers to u16 when K < 65536.

use crate::dist_transform::l1_dt_with_argmin;

/// Solve the real-valued L1-TV problem.
///
/// `y`        — input signal (any length, including 0 / 1).
/// `alpha`    — TV regularisation strength (must be ≥ 0).
/// `weights`  — optional per-sample data-fidelity weights; `None` → all-ones.
/// `use_dist_transform` — `true` for the O(NK) DT path (production), `false`
/// for the O(NK²) naive path used in parity tests.
///
/// Panics if `weights.len() != y.len()`, or if any value (in `y` or `weights`)
/// is non-finite. Caller is responsible for input validation at the API edge;
/// inside the algorithm we trust the inputs.
pub fn solve_l1_tv_real(
    y: &[f64],
    alpha: f64,
    weights: Option<&[f64]>,
    use_dist_transform: bool,
) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![y[0]];
    }

    let default_w: Vec<f64>;
    let w: &[f64] = match weights {
        Some(ww) => {
            assert_eq!(ww.len(), n, "weights length must equal y.len()");
            ww
        }
        None => {
            default_w = vec![1.0; n];
            &default_w
        }
    };

    let v = sorted_unique(y);
    let k = v.len();

    // Degenerate case: every y[i] is identical, so the only candidate is y[0].
    if k == 1 {
        return vec![v[0]; n];
    }

    let mut b_prev = vec![0.0_f64; k];
    let mut b_curr = vec![0.0_f64; k];
    // prev[col * k + i] = the index j ∈ {0..k} of the V[j] from which V[i]'s
    // best path into column `col` came. Column 0 is unused (initial column).
    let mut prev: Vec<u32> = vec![0u32; k * n];

    // Initialise column 0: B[i, 0] = w[0] · |V[i] - y[0]|.
    for i in 0..k {
        b_prev[i] = w[0] * (v[i] - y[0]).abs();
    }

    // Forward DP.
    for col in 1..n {
        let prev_slot = &mut prev[col * k..(col + 1) * k];

        if use_dist_transform {
            b_curr.copy_from_slice(&b_prev);
            l1_dt_with_argmin(&mut b_curr, &v, alpha, prev_slot);
        } else {
            // Naive O(K²) path with explicit argmin tracking.
            for i in 0..k {
                let mut best = f64::INFINITY;
                let mut best_j: u32 = 0;
                for j in 0..k {
                    let cand = b_prev[j] + alpha * (v[i] - v[j]).abs();
                    if cand < best {
                        best = cand;
                        best_j = j as u32;
                    }
                }
                b_curr[i] = best;
                prev_slot[i] = best_j;
            }
        }

        // Add the data-fidelity term at column `col`.
        let wn = w[col];
        let yn = y[col];
        for i in 0..k {
            b_curr[i] += wn * (v[i] - yn).abs();
        }

        std::mem::swap(&mut b_prev, &mut b_curr);
    }

    // Backtracking: l_n = argmin B[:, N-1] (0-indexed); l_{c-1} = prev[c, l_c].
    let mut x = vec![0.0_f64; n];
    let mut l = argmin(&b_prev);
    x[n - 1] = v[l];
    for col in (1..n).rev() {
        l = prev[col * k + l] as usize;
        x[col - 1] = v[l];
    }

    x
}

/// Sorted unique values of `y` (matches MATLAB `unique(y)`).
fn sorted_unique(y: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = y.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("L1TV: NaN in input y"));
    v.dedup();
    v
}

/// Index of the smallest entry (first wins on ties, matching MATLAB `min`).
fn argmin(b: &[f64]) -> usize {
    let mut best = b[0];
    let mut idx = 0;
    for i in 1..b.len() {
        if b[i] < best {
            best = b[i];
            idx = i;
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L1-TV objective value (no weights → all-ones).
    fn cost(x: &[f64], y: &[f64], alpha: f64, w: Option<&[f64]>) -> f64 {
        let mut c = 0.0;
        for i in 1..x.len() {
            c += alpha * (x[i] - x[i - 1]).abs();
        }
        for i in 0..x.len() {
            let wi = w.map_or(1.0, |w| w[i]);
            c += wi * (x[i] - y[i]).abs();
        }
        c
    }

    #[test]
    fn dt_and_naive_agree_on_objective() {
        // 10-point input; both paths should reach the same cost (tied minima
        // permit different x — see feedback memory on objective parity).
        let y = vec![1.0, 3.0, 2.0, 5.0, 4.0, 1.0, 3.0, 2.0, 6.0, 5.0];
        let alpha = 0.5;
        let x_dt = solve_l1_tv_real(&y, alpha, None, true);
        let x_nv = solve_l1_tv_real(&y, alpha, None, false);
        let c_dt = cost(&x_dt, &y, alpha, None);
        let c_nv = cost(&x_nv, &y, alpha, None);
        let relerr = (c_dt - c_nv).abs() / c_dt.abs().max(1.0);
        assert!(
            relerr < 1e-12,
            "DT vs naive cost mismatch: dt={c_dt}, naive={c_nv}"
        );
    }

    #[test]
    fn dt_and_naive_agree_on_objective_with_weights() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let w = vec![0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0, 2.0];
        let alpha = 0.7;
        let x_dt = solve_l1_tv_real(&y, alpha, Some(&w), true);
        let x_nv = solve_l1_tv_real(&y, alpha, Some(&w), false);
        let c_dt = cost(&x_dt, &y, alpha, Some(&w));
        let c_nv = cost(&x_nv, &y, alpha, Some(&w));
        assert!((c_dt - c_nv).abs() / c_dt.abs().max(1.0) < 1e-12);
    }

    #[test]
    fn alpha_zero_recovers_input() {
        let y = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let x = solve_l1_tv_real(&y, 0.0, None, true);
        for i in 0..y.len() {
            assert!((x[i] - y[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn huge_alpha_collapses_to_constant() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = solve_l1_tv_real(&y, 1e10, None, true);
        for i in 1..x.len() {
            assert!(
                (x[i] - x[0]).abs() < 1e-9,
                "huge alpha should collapse to constant; got x={x:?}"
            );
        }
    }

    #[test]
    fn empty_input() {
        let x = solve_l1_tv_real(&[], 1.0, None, true);
        assert!(x.is_empty());
    }

    #[test]
    fn singleton_returns_input() {
        let x = solve_l1_tv_real(&[42.0], 1.0, None, true);
        assert_eq!(x, vec![42.0]);
    }

    #[test]
    fn output_lies_in_unique_y() {
        // Candidate-set theorem: every x[i] must equal some y[j].
        let y = vec![1.5, 2.5, 1.5, 3.5, 2.5, 1.5, 4.0];
        let alpha = 0.3;
        let x = solve_l1_tv_real(&y, alpha, None, true);
        let v: std::collections::BTreeSet<u64> = y.iter().map(|&v| v.to_bits()).collect();
        for &xi in &x {
            assert!(
                v.contains(&xi.to_bits()),
                "candidate-set theorem violated: x value {xi} not in unique(y)"
            );
        }
    }

    #[test]
    fn all_equal_y_returns_constant() {
        let y = vec![3.14; 7];
        let x = solve_l1_tv_real(&y, 0.5, None, true);
        for &xi in &x {
            assert_eq!(xi, 3.14);
        }
    }

    #[test]
    fn cost_monotone_in_alpha() {
        // As alpha grows, cost(x_alpha, y, alpha) at the OPTIMUM is
        // non-decreasing in alpha (by definition: it's a max over alpha-affine
        // functions of x... actually it's a min of an affine-in-alpha objective
        // → optimal cost is concave in alpha and finite, so the derivative is
        // bounded and the data-fidelity part is non-increasing in alpha; the TV
        // part is non-decreasing). We test the weaker fact that the recovered
        // x[i] becomes more "constant-like" as alpha grows: number of distinct
        // values is non-increasing.
        let y = vec![1.0, 5.0, 2.0, 8.0, 3.0, 6.0, 4.0, 7.0];
        let mut prev_distinct = usize::MAX;
        for &alpha in &[0.0, 0.1, 0.5, 1.0, 5.0, 1e6] {
            let x = solve_l1_tv_real(&y, alpha, None, true);
            let distinct: std::collections::BTreeSet<u64> =
                x.iter().map(|v| v.to_bits()).collect();
            assert!(
                distinct.len() <= prev_distinct,
                "distinct levels grew with alpha: alpha={alpha}, levels={}, prev={prev_distinct}",
                distinct.len()
            );
            prev_distinct = distinct.len();
        }
    }
}
