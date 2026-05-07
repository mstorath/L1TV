"""API tests for l1tv.min_l1_tv_circ (circle-valued solver)."""

from __future__ import annotations

import numpy as np
import pytest

import l1tv

from conftest import arc_distance, cost_circ, wrap


def test_basic_smoke():
    rng = np.random.default_rng(0)
    y = wrap(rng.standard_normal(50))
    x = l1tv.min_l1_tv_circ(y, alpha=0.3)
    assert x.dtype == np.float64
    assert x.shape == y.shape


def test_output_in_minus_pi_pi():
    y = np.array([3.5, -3.5, 0.1, -0.1, 7.0])  # values outside (-pi, pi]
    x = l1tv.min_l1_tv_circ(y, alpha=0.5)
    assert np.all(x <= np.pi + 1e-12)
    assert np.all(x > -np.pi - 1e-12)


def test_alpha_zero_recovers_wrapped_input():
    rng = np.random.default_rng(1)
    y = rng.uniform(-2 * np.pi, 2 * np.pi, size=20)
    x = l1tv.min_l1_tv_circ(y, alpha=0.0)
    expected = wrap(y)
    # Compare via arc distance (handles ±π aliasing).
    np.testing.assert_allclose(arc_distance(x, expected), 0.0, atol=1e-12)


def test_huge_alpha_collapses_to_constant_arc():
    y = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    x = l1tv.min_l1_tv_circ(y, alpha=1e10)
    for i in range(1, len(x)):
        assert arc_distance(x[i], x[0]) < 1e-9


def test_dt_naive_objective_parity_with_branch_cut():
    """Inputs deliberately straddle the ±π branch cut."""
    y = wrap(
        np.array(
            [-np.pi + 0.05, np.pi - 0.05, -np.pi + 0.1, np.pi - 0.1,
             0.0, np.pi / 2, -np.pi / 2, -np.pi + 0.02, np.pi]
        )
    )
    alpha = 0.6
    x_dt = l1tv.min_l1_tv_circ(y, alpha, use_dist_transform=True)
    x_nv = l1tv.min_l1_tv_circ(y, alpha, use_dist_transform=False)
    c_dt = cost_circ(x_dt, y, alpha)
    c_nv = cost_circ(x_nv, y, alpha)
    rel = abs(c_dt - c_nv) / max(1.0, abs(c_dt))
    assert rel < 1e-12, f"DT vs naive: dt={c_dt}, naive={c_nv}"


def test_weights_row_vector_accepted():
    """Mirror of the MATLAB row-vector silent-garbage bug — should now work."""
    y = np.array([0.5, -0.5, 1.0, -1.0])
    w_row = np.array([1.0, 2.0, 1.5, 0.5]).reshape(1, -1)
    x_row = l1tv.min_l1_tv_circ(y, alpha=0.4, weights=w_row)
    x_col = l1tv.min_l1_tv_circ(y, alpha=0.4, weights=w_row.ravel())
    # cost must match (not necessarily x — but with non-tied minima they should)
    c_row = cost_circ(x_row, y, alpha=0.4, weights=w_row.ravel())
    c_col = cost_circ(x_col, y, alpha=0.4, weights=w_row.ravel())
    assert abs(c_row - c_col) < 1e-12


def test_singleton_returns_wrapped():
    x = l1tv.min_l1_tv_circ([3.0 * np.pi], alpha=1.0)
    assert x.shape == (1,)
    assert arc_distance(x[0], np.pi) < 1e-12


def test_empty_input():
    x = l1tv.min_l1_tv_circ([], alpha=1.0)
    assert x.shape == (0,)


# --- Input validation ---------------------------------------------------------


def test_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        l1tv.min_l1_tv_circ([0.0, np.nan, 1.0], alpha=1.0)


def test_rejects_negative_alpha():
    with pytest.raises(ValueError, match="non-negative"):
        l1tv.min_l1_tv_circ([0.0, 1.0], alpha=-0.1)


def test_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        l1tv.min_l1_tv_circ([0.0, 1.0], alpha=1.0, weights=[1.0, -1.0])
