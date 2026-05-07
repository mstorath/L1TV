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
  ``min_l1_tv(y, alpha)``      — real-valued signals.
  ``min_l1_tv_circ(y, alpha)`` — circle-valued (phase / orientation) signals
                                 with shortest-arc distance.

References
----------
M. Storath, A. Weinmann, M. Unser. *Exact algorithms for L1-TV regularization
of real-valued or circle-valued signals.* SIAM Journal on Scientific Computing,
38(1), A614-A630, 2016. https://doi.org/10.1137/15M101796X
"""

from __future__ import annotations

try:
    from l1tv import _core
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "l1tv._core (Rust extension) failed to load. If you installed from "
        "source, ensure `maturin develop` (or `pip install .`) completed "
        "without errors and that the wheel matches your Python interpreter."
    ) from e

__version__ = _core.__version__
__original_authors__ = "Martin Storath, Andreas Weinmann"
__ported_by__ = (
    "Claude Opus coding agent (Anthropic, 2026) — "
    "automated port from MATLAB l1tv"
)

__all__ = [
    # Solvers will be re-exported here as Day 2 / Day 3 land.
    "__version__",
    "__original_authors__",
    "__ported_by__",
]
