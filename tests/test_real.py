"""API tests for l1tv.min_l1_tv (real-valued solver)."""

from __future__ import annotations

import numpy as np
import pytest

import l1tv

from conftest import cost_real


def test_basic_smoke():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50)
    x = l1tv.min_l1_tv(y, alpha=0.3)
    assert x.dtype == np.float64
    assert x.shape == y.shape


def test_accepts_list_input():
    y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    x = l1tv.min_l1_tv(y, alpha=0.5)
    assert isinstance(x, np.ndarray)
    assert x.shape == (len(y),)


def test_accepts_row_vector():
    # Row-vector input — silently wrong in raw MATLAB, fine here.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(1, -1)
    x = l1tv.min_l1_tv(y, alpha=0.5)
    assert x.shape == (5,)


def test_accepts_column_vector():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
    x = l1tv.min_l1_tv(y, alpha=0.5)
    assert x.shape == (5,)


def test_alpha_zero_recovers_input():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(30)
    x = l1tv.min_l1_tv(y, alpha=0.0)
    np.testing.assert_allclose(x, y, atol=1e-12)


def test_huge_alpha_collapses_to_constant():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = l1tv.min_l1_tv(y, alpha=1e10)
    np.testing.assert_allclose(x, x[0], atol=1e-9)


def test_dt_naive_objective_parity():
    """At tied minima the two paths can return different x; they must agree
    on the L1-TV objective."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(100)
    alpha = 0.4
    x_dt = l1tv.min_l1_tv(y, alpha, use_dist_transform=True)
    x_nv = l1tv.min_l1_tv(y, alpha, use_dist_transform=False)
    c_dt = cost_real(x_dt, y, alpha)
    c_nv = cost_real(x_nv, y, alpha)
    rel = abs(c_dt - c_nv) / max(1.0, abs(c_dt))
    assert rel < 1e-12, f"DT vs naive cost mismatch: dt={c_dt}, naive={c_nv}"


def test_weights_default_equivalent_to_ones():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(40)
    alpha = 0.7
    x_default = l1tv.min_l1_tv(y, alpha)
    x_explicit = l1tv.min_l1_tv(y, alpha, weights=np.ones_like(y))
    # x can differ at tied minima, but cost must match.
    c_default = cost_real(x_default, y, alpha)
    c_explicit = cost_real(x_explicit, y, alpha, weights=np.ones_like(y))
    assert abs(c_default - c_explicit) < 1e-12


def test_zero_weight_floats_freely():
    """A zero data-fidelity weight should not constrain that sample."""
    y = np.array([1.0, 2.0, 3.0])
    w = np.array([0.0, 1.0, 0.0])
    x = l1tv.min_l1_tv(y, alpha=1.0, weights=w)
    # Optimum: x = [2, 2, 2] with cost 0. (Any constant equal to 2 works; tied
    # x values around 2 are also optimal.)
    c = cost_real(x, y, alpha=1.0, weights=w)
    assert c < 1e-12


def test_empty_input_returns_empty():
    x = l1tv.min_l1_tv([], alpha=1.0)
    assert x.shape == (0,)


def test_singleton_returns_input():
    x = l1tv.min_l1_tv([42.0], alpha=1.0)
    np.testing.assert_array_equal(x, [42.0])


def test_output_lies_in_unique_y():
    """Candidate-set theorem: every x[i] is some y[j]."""
    rng = np.random.default_rng(3)
    y = rng.choice([1.0, 2.5, 3.7, 5.0], size=30)
    x = l1tv.min_l1_tv(y, alpha=0.4)
    unique_y = set(y.tolist())
    for xi in x:
        assert xi in unique_y, f"x value {xi} not in unique(y) = {unique_y}"


# --- Input validation ---------------------------------------------------------


def test_rejects_nan():
    y = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="finite"):
        l1tv.min_l1_tv(y, alpha=1.0)


def test_rejects_inf():
    y = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="finite"):
        l1tv.min_l1_tv(y, alpha=1.0)


def test_rejects_negative_alpha():
    with pytest.raises(ValueError, match="non-negative"):
        l1tv.min_l1_tv([1.0, 2.0], alpha=-0.1)


def test_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="shape"):
        l1tv.min_l1_tv([1.0, 2.0, 3.0], alpha=1.0, weights=[1.0, 1.0])


def test_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        l1tv.min_l1_tv([1.0, 2.0, 3.0], alpha=1.0, weights=[1.0, -1.0, 1.0])
