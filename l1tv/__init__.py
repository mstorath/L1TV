# Ported from MATLAB l1tv (Storath, Weinmann, Unser 2016) by Claude Opus coding
# agent, Anthropic, 2026.
"""
l1tv — Exact L1-TV regularisation of real- or circle-valued 1-D signals.

This package is an **automated port** of the original MATLAB ``l1tv`` library
(Martin Storath, Andreas Weinmann), performed by Claude Opus coding agent
(Anthropic, 2026). The performance-critical dynamic-programming solvers are
implemented in Rust (via PyO3) and compiled as a C extension; the public API is
pure Python with NumPy.

Public API
----------
1-D solvers (exact, non-iterative):
  ``min_l1_tv(y, alpha)``       — real-valued signals.
  ``min_l1_tv_circ(y, alpha)``  — circle-valued (phase / orientation) signals
                                  with shortest-arc distance.

References
----------
M. Storath, A. Weinmann, M. Unser. *Exact algorithms for L1-TV regularization
of real-valued or circle-valued signals.* SIAM Journal on Scientific Computing,
38(1), A614-A630, 2016. https://doi.org/10.1137/15M101796X
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from l1tv import _core
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "l1tv._core (Rust extension) failed to load. If you installed from "
        "source, ensure `maturin develop` (or `pip install .`) completed "
        "without errors and that the wheel matches your Python interpreter."
    ) from e

__version__: str = _core.__version__
__original_authors__ = "Martin Storath, Andreas Weinmann"
__ported_by__ = (
    "Claude Opus coding agent (Anthropic, 2026) — "
    "automated port from MATLAB l1tv"
)


def _validate_inputs(
    y: ArrayLike,
    alpha: float,
    weights: Optional[ArrayLike],
    *,
    func_name: str,
) -> tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
    """Coerce y/weights to contiguous float64 1-D and validate at the boundary."""
    y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    if y_arr.ndim != 1:  # ravel guarantees ndim==1, but assert for clarity
        raise ValueError(f"{func_name}: y must be 1-D after .ravel().")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError(f"{func_name}: y must contain only finite values.")
    if not np.isfinite(alpha):
        raise ValueError(f"{func_name}: alpha must be finite.")
    if alpha < 0.0:
        raise ValueError(f"{func_name}: alpha must be non-negative (got {alpha}).")

    if weights is None:
        return y_arr, None
    w_arr = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).ravel())
    if w_arr.shape != y_arr.shape:
        raise ValueError(
            f"{func_name}: weights shape {w_arr.shape} does not match y shape "
            f"{y_arr.shape}."
        )
    if not np.all(np.isfinite(w_arr)):
        raise ValueError(f"{func_name}: weights must contain only finite values.")
    if np.any(w_arr < 0.0):
        raise ValueError(f"{func_name}: weights must be non-negative.")
    return y_arr, w_arr


def min_l1_tv(
    y: ArrayLike,
    alpha: float,
    weights: Optional[ArrayLike] = None,
    *,
    use_dist_transform: bool = True,
) -> NDArray[np.float64]:
    """Exact L1-TV regularisation of a real-valued 1-D signal.

    Solves
        min_x  alpha * sum_{i=1..N-1} |x[i] - x[i+1]|
             + sum_{i=1..N}  w[i] * |x[i] - y[i]|
    by dynamic programming over the candidate set ``unique(y)``, in O(N*K)
    when ``K = |unique(y)|`` and the L1 distance-transform path is used
    (the default).

    Parameters
    ----------
    y : array_like, 1-D
        Input signal. Lists, tuples, or NumPy arrays of any 1-D shape (row,
        column, or generic) are accepted; ``np.asarray(y).ravel()`` is applied
        internally.
    alpha : float
        TV regularisation strength; must be non-negative.
    weights : array_like, 1-D, optional
        Per-sample data-fidelity weights; must be non-negative and the same
        length as ``y``. Default: all ones.
    use_dist_transform : bool, keyword-only, default True
        If ``False``, runs the O(N*K^2) brute-force inner loop. Provided for
        parity testing; production callers should leave this at the default.

    Returns
    -------
    x : ndarray of float64, shape (N,)
        The exact minimiser of the L1-TV functional.

    References
    ----------
    Storath, Weinmann, Unser. "Exact algorithms for L1-TV regularization of
    real-valued or circle-valued signals." SIAM J. Sci. Comput. 38(1), 2016.
    """
    y_arr, w_arr = _validate_inputs(y, alpha, weights, func_name="min_l1_tv")
    return _core.solve_l1_tv_real(y_arr, alpha, w_arr, use_dist_transform)


def min_l1_tv_circ(
    y: ArrayLike,
    alpha: float,
    weights: Optional[ArrayLike] = None,
    *,
    use_dist_transform: bool = True,
) -> NDArray[np.float64]:
    """Exact L1-TV regularisation of a circle-valued 1-D signal.

    Same as :func:`min_l1_tv` but with shortest-arc distance on the unit
    circle replacing absolute value. Inputs may lie outside ``(-pi, pi]``;
    they are wrapped before processing. Outputs lie in ``(-pi, pi]``.

    Parameters
    ----------
    y, alpha, weights, use_dist_transform
        Same semantics as :func:`min_l1_tv`.

    Returns
    -------
    x : ndarray of float64, shape (N,)
        The exact minimiser, with values in ``(-pi, pi]``.
    """
    y_arr, w_arr = _validate_inputs(y, alpha, weights, func_name="min_l1_tv_circ")
    return _core.solve_l1_tv_circ(y_arr, alpha, w_arr, use_dist_transform)


__all__ = [
    "min_l1_tv",
    "min_l1_tv_circ",
    "__version__",
    "__original_authors__",
    "__ported_by__",
]
