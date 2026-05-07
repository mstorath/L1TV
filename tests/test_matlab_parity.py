"""MATLAB-vs-Rust objective parity test for the L1TV port.

Each fixture in tests/matlab_fixtures/ stores y, alpha, weights, the MATLAB
output x, and the MATLAB-side L1-TV objective cost. This test runs the Rust
port on the same inputs and asserts that the L1-TV objective values agree to
relative error < 1e-10.

We compare *objectives*, not pointwise x: L1-TV minima are non-unique under
tied configurations, so MATLAB's `min` and Rust's `min` can pick different
argmins of equal cost. See feedback memory `feedback_parity_test_objective.md`
and Pottslab's tests/test_matlab_parity.py:24 for the same lesson on L1-Potts.

Skipped if scipy is not installed (loadmat dependency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import l1tv

scipy_io = pytest.importorskip("scipy.io", reason="scipy required to load .mat fixtures")

FIXTURES_DIR = Path(__file__).parent / "matlab_fixtures"
TWO_PI = 2.0 * np.pi


def _load(name: str) -> dict:
    raw = scipy_io.loadmat(FIXTURES_DIR / f"{name}.mat")
    y = np.ascontiguousarray(raw["y"].ravel().astype(np.float64))
    alpha = float(raw["alpha"].ravel()[0])
    w_raw = raw["w"]
    weights = None if w_raw.size == 0 else np.ascontiguousarray(w_raw.ravel().astype(np.float64))
    cost_matlab = float(raw["cost"].ravel()[0])
    kind = str(raw["kind"][0]) if hasattr(raw["kind"], "shape") else str(raw["kind"])
    return {"y": y, "alpha": alpha, "weights": weights, "cost_matlab": cost_matlab, "kind": kind}


def _cost_real(x, y, alpha, weights):
    w = weights if weights is not None else np.ones_like(y)
    return float(alpha * np.sum(np.abs(np.diff(x))) + np.sum(w * np.abs(x - y)))


def _arc(a, b):
    d = np.abs(a - b)
    return np.minimum(d, TWO_PI - d)


def _cost_circ(x, y, alpha, weights):
    w = weights if weights is not None else np.ones_like(y)
    return float(alpha * np.sum(_arc(x[1:], x[:-1])) + np.sum(w * _arc(x, y)))


REAL_CASES = [
    "real_tiny",
    "real_small",
    "real_n100",
    "real_n100_w",
    "real_n2000",
    "real_alpha0",
    "real_huge",
    "real_int",
]

CIRC_CASES = [
    "circ_tiny",
    "circ_branchcut",
    "circ_n100",
    "circ_n100_w",
    "circ_n2000",
    "circ_alpha0",
    "circ_huge",
    "circ_corners",
]


@pytest.mark.parametrize("name", REAL_CASES)
def test_matlab_parity_real(name: str):
    f = _load(name)
    assert f["kind"] == "real"
    x = l1tv.min_l1_tv(f["y"], f["alpha"], weights=f["weights"])
    cost_rust = _cost_real(x, f["y"], f["alpha"], f["weights"])
    cost_matlab = f["cost_matlab"]
    relerr = abs(cost_rust - cost_matlab) / max(1.0, abs(cost_matlab))
    assert relerr < 1e-10, (
        f"{name}: cost mismatch — matlab={cost_matlab:.10g} rust={cost_rust:.10g} "
        f"relerr={relerr:.3e}"
    )


@pytest.mark.parametrize("name", CIRC_CASES)
def test_matlab_parity_circ(name: str):
    f = _load(name)
    assert f["kind"] == "circ"
    x = l1tv.min_l1_tv_circ(f["y"], f["alpha"], weights=f["weights"])
    cost_rust = _cost_circ(x, f["y"], f["alpha"], f["weights"])
    cost_matlab = f["cost_matlab"]
    relerr = abs(cost_rust - cost_matlab) / max(1.0, abs(cost_matlab))
    assert relerr < 1e-10, (
        f"{name}: cost mismatch — matlab={cost_matlab:.10g} rust={cost_rust:.10g} "
        f"relerr={relerr:.3e}"
    )
