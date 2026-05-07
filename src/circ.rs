// Ported from L1TV_Circ.m + Auxiliary/{distAngle,wrapAngle}.m
// (Storath, Weinmann, Unser 2016)
// by Claude Opus coding agent, Anthropic, 2026.
//
// Exact L1-TV solver for circle-valued (phase / orientation) 1-D signals.
//
// Solves
//   min_{x ∈ T^N}  α · ∑ d_arc(x[i], x[i+1])
//                + ∑ w[i] · d_arc(x[i], y[i])
// where T is the unit circle and d_arc is the shortest-arc distance.
//
// Algorithm: same DP scaffolding as `real.rs`, with two changes.
//   1. Candidate set V = unique(y_wrapped) ∪ antipodes — the optimum on a
//      circular segment may sit at an antipodal point of a data sample, not
//      just at a sample itself (Storath-Weinmann-Unser 2016 §4).
//   2. The L1 distance transform is applied to the 3K-replicated grid
//      [V-2π; V; V+2π] so that linear distance on the replicated grid equals
//      shortest-arc distance for the central K positions. The MATLAB ref
//      then takes a defensive min across the three K-blocks; we do the same.

use crate::dist_transform::l1_dt_with_argmin;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Solve the circle-valued L1-TV problem.
///
/// Inputs `y` may lie outside `[-π, π]`; they are wrapped before processing.
/// Output values are in `(-π, π]`.
pub fn solve_l1_tv_circ(
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
        return vec![wrap_angle(y[0])];
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

    // Wrap y to (-π, π]; matches MATLAB `yAng = angle(exp(1i*y))`.
    let y_ang: Vec<f64> = y.iter().map(|&yi| wrap_angle(yi)).collect();

    // Candidate set: sorted (unique(y_ang) ∪ antipodes(unique(y_ang))).
    // MATLAB does not dedup after concat; we match that for byte-for-byte
    // parity (duplicates just give equal-cost columns; correctness unchanged).
    let v = build_candidates(&y_ang);
    let k = v.len();
    if k == 1 {
        return vec![v[0]; n];
    }

    let mut b_prev = vec![0.0_f64; k];
    let mut b_curr = vec![0.0_f64; k];
    let mut prev: Vec<u32> = vec![0u32; k * n];

    // Replicated grid for the DT path. Built once and reused per column.
    // v_rep is sorted ascending because v is sorted and the three blocks
    // V-2π < V < V+2π are non-overlapping (V ⊂ [-π, π]).
    let mut v_rep: Vec<f64> = Vec::with_capacity(3 * k);
    for &vi in &v {
        v_rep.push(vi - TWO_PI);
    }
    for &vi in &v {
        v_rep.push(vi);
    }
    for &vi in &v {
        v_rep.push(vi + TWO_PI);
    }
    let mut b_rep = vec![0.0_f64; 3 * k];
    let mut argmin_rep = vec![0u32; 3 * k];

    // Initialise column 0: B[i, 0] = w[0] · d_arc(V[i], y_ang[0]).
    for i in 0..k {
        b_prev[i] = w[0] * arc_distance(v[i], y_ang[0]);
    }

    for col in 1..n {
        let prev_slot = &mut prev[col * k..(col + 1) * k];

        if use_dist_transform {
            // Replicate b_prev three times.
            for i in 0..k {
                b_rep[i] = b_prev[i];
                b_rep[k + i] = b_prev[i];
                b_rep[2 * k + i] = b_prev[i];
            }
            l1_dt_with_argmin(&mut b_rep, &v_rep, alpha, &mut argmin_rep);
            // For each central position, take the min across the 3 K-blocks
            // (defensive — the central block already captures wraparound, but
            // mirroring the MATLAB reference avoids any boundary-effect
            // surprises at positions 0 and 3K-1).
            for i in 0..k {
                let mut best = b_rep[i];
                let mut win_pos = i;
                if b_rep[k + i] < best {
                    best = b_rep[k + i];
                    win_pos = k + i;
                }
                if b_rep[2 * k + i] < best {
                    best = b_rep[2 * k + i];
                    win_pos = 2 * k + i;
                }
                b_curr[i] = best;
                // argmin in v_rep is a 3K position; the original V index is
                // the position modulo k (Brep[j] = B[j mod k] by replication).
                prev_slot[i] = (argmin_rep[win_pos] as usize % k) as u32;
            }
        } else {
            // Naive O(K²) path.
            for i in 0..k {
                let mut best = f64::INFINITY;
                let mut best_j: u32 = 0;
                for j in 0..k {
                    let cand = b_prev[j] + alpha * arc_distance(v[i], v[j]);
                    if cand < best {
                        best = cand;
                        best_j = j as u32;
                    }
                }
                b_curr[i] = best;
                prev_slot[i] = best_j;
            }
        }

        // Add data-fidelity term.
        let wn = w[col];
        let yn = y_ang[col];
        for i in 0..k {
            b_curr[i] += wn * arc_distance(v[i], yn);
        }

        std::mem::swap(&mut b_prev, &mut b_curr);
    }

    let mut x = vec![0.0_f64; n];
    let mut l = argmin(&b_prev);
    x[n - 1] = v[l];
    for col in (1..n).rev() {
        l = prev[col * k + l] as usize;
        x[col - 1] = v[l];
    }
    x
}

/// Wrap an angle to `(-π, π]` (matches MATLAB `angle(exp(1i*a))`).
pub(crate) fn wrap_angle(a: f64) -> f64 {
    let r = a.rem_euclid(TWO_PI);
    if r > std::f64::consts::PI {
        r - TWO_PI
    } else {
        r
    }
}

/// Antipodal point on the unit circle (matches MATLAB `angle(-exp(1i*a))`).
fn antipode(a: f64) -> f64 {
    wrap_angle(a + std::f64::consts::PI)
}

/// Shortest-arc distance assuming inputs in `(-π, π]`.
fn arc_distance(phi: f64, psi: f64) -> f64 {
    let d = (phi - psi).abs();
    d.min(TWO_PI - d)
}

fn build_candidates(y_ang: &[f64]) -> Vec<f64> {
    let mut unique: Vec<f64> = y_ang.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).expect("L1TV_Circ: NaN in input y"));
    unique.dedup();
    let n_unique = unique.len();
    let mut v: Vec<f64> = Vec::with_capacity(2 * n_unique);
    for &u in &unique {
        v.push(u);
    }
    for &u in &unique {
        v.push(antipode(u));
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // No dedup post-concat — match MATLAB exactly. Duplicates yield equal-cost
    // columns; the optimum is unchanged.
    v
}

fn argmin(b: &[f64]) -> usize {
    let mut best = b[0];
    let mut idx = 0;
    for (i, &v) in b.iter().enumerate().skip(1) {
        if v < best {
            best = v;
            idx = i;
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn cost(x: &[f64], y: &[f64], alpha: f64, w: Option<&[f64]>) -> f64 {
        let mut c = 0.0;
        for i in 1..x.len() {
            c += alpha * arc_distance(x[i], x[i - 1]);
        }
        for i in 0..x.len() {
            let wi = w.map_or(1.0, |w| w[i]);
            c += wi * arc_distance(x[i], y[i]);
        }
        c
    }

    #[test]
    fn wrap_angle_corners() {
        // Match MATLAB's angle(exp(1i*x)) which returns values in (-π, π].
        let eps = 1e-12;
        assert!((wrap_angle(0.0) - 0.0).abs() < eps);
        assert!((wrap_angle(PI) - PI).abs() < eps);
        assert!((wrap_angle(-PI) - PI).abs() < eps); // angle(exp(-iπ)) = angle(-1) = π
        assert!((wrap_angle(2.0 * PI) - 0.0).abs() < eps);
        assert!((wrap_angle(3.0 * PI) - PI).abs() < eps);
        assert!((wrap_angle(-PI / 2.0) - (-PI / 2.0)).abs() < eps);
        assert!((wrap_angle(3.0 * PI / 2.0) - (-PI / 2.0)).abs() < eps);
    }

    #[test]
    fn antipode_corners() {
        let eps = 1e-12;
        assert!((antipode(0.0) - PI).abs() < eps);
        // antipode(π) = wrap(2π) = 0
        assert!(antipode(PI).abs() < eps);
        // antipode(-π/2) = wrap(π/2) = π/2
        assert!((antipode(-PI / 2.0) - PI / 2.0).abs() < eps);
    }

    #[test]
    fn arc_distance_symmetric_and_bounded() {
        for &(a, b) in &[(0.0, 0.0), (0.0, PI), (-PI / 2.0, PI / 2.0), (0.1, -0.1)] {
            let d1 = arc_distance(a, b);
            let d2 = arc_distance(b, a);
            assert!((d1 - d2).abs() < 1e-15);
            assert!((0.0..=PI + 1e-12).contains(&d1));
        }
    }

    #[test]
    fn dt_and_naive_agree_on_objective() {
        let y: Vec<f64> = (0..20).map(|i| (i as f64 * 0.7).sin() * PI).collect();
        let alpha = 0.4;
        let x_dt = solve_l1_tv_circ(&y, alpha, None, true);
        let x_nv = solve_l1_tv_circ(&y, alpha, None, false);
        let c_dt = cost(&x_dt, &y, alpha, None);
        let c_nv = cost(&x_nv, &y, alpha, None);
        let relerr = (c_dt - c_nv).abs() / c_dt.abs().max(1.0);
        assert!(
            relerr < 1e-12,
            "DT vs naive cost mismatch: dt={c_dt}, naive={c_nv}"
        );
    }

    #[test]
    fn dt_and_naive_agree_with_weights_and_branch_cut() {
        // Inputs straddling the ±π branch cut.
        let y: Vec<f64> = vec![
            -PI + 0.05,
            PI - 0.05,
            -PI + 0.1,
            PI - 0.1,
            0.0,
            PI / 2.0,
            -PI / 2.0,
            -PI + 0.02,
            PI,
        ];
        let w: Vec<f64> = (0..y.len()).map(|i| 0.5 + (i as f64) * 0.2).collect();
        let alpha = 0.6;
        let x_dt = solve_l1_tv_circ(&y, alpha, Some(&w), true);
        let x_nv = solve_l1_tv_circ(&y, alpha, Some(&w), false);
        let c_dt = cost(&x_dt, &y, alpha, Some(&w));
        let c_nv = cost(&x_nv, &y, alpha, Some(&w));
        let relerr = (c_dt - c_nv).abs() / c_dt.abs().max(1.0);
        assert!(
            relerr < 1e-12,
            "branch-cut DT vs naive: dt={c_dt}, nv={c_nv}"
        );
    }

    #[test]
    fn alpha_zero_recovers_input_wrapped() {
        // Input outside [-π, π]; output should be the wrapped version.
        let y = vec![0.5, PI + 0.3, -PI - 0.2, 2.5 * PI];
        let x = solve_l1_tv_circ(&y, 0.0, None, true);
        for i in 0..y.len() {
            let expected = wrap_angle(y[i]);
            assert!(
                arc_distance(x[i], expected) < 1e-12,
                "x[{i}]={} expected wrap_angle={}",
                x[i],
                expected
            );
        }
    }

    #[test]
    fn huge_alpha_collapses_to_constant() {
        let y = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];
        let x = solve_l1_tv_circ(&y, 1e10, None, true);
        for i in 1..x.len() {
            let d = arc_distance(x[i], x[0]);
            assert!(
                d < 1e-9,
                "huge alpha should collapse to constant arc, got {d}"
            );
        }
    }

    #[test]
    fn empty_input() {
        let x = solve_l1_tv_circ(&[], 1.0, None, true);
        assert!(x.is_empty());
    }

    #[test]
    fn singleton_returns_wrapped_input() {
        let x = solve_l1_tv_circ(&[3.0 * PI], 1.0, None, true);
        assert_eq!(x.len(), 1);
        assert!(arc_distance(x[0], PI).abs() < 1e-12);
    }

    #[test]
    fn output_in_candidate_set() {
        // x[i] should lie in unique(wrap(y)) ∪ antipodes(unique(wrap(y))).
        let y = vec![0.5, -0.3, 1.2, -0.3, 2.5, 0.5];
        let x = solve_l1_tv_circ(&y, 0.4, None, true);
        let v = build_candidates(&y.iter().map(|&yi| wrap_angle(yi)).collect::<Vec<_>>());
        let v_bits: std::collections::BTreeSet<u64> = v.iter().map(|v| v.to_bits()).collect();
        for &xi in &x {
            assert!(
                v_bits.contains(&xi.to_bits()),
                "x value {xi} not in candidate set"
            );
        }
    }
}
