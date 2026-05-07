# Test helpers shared across the suite.

from __future__ import annotations

import numpy as np


TWO_PI = 2.0 * np.pi


def cost_real(x, y, alpha, weights=None):
    """L1-TV objective value for the real-valued problem."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=np.float64)
    return alpha * np.sum(np.abs(np.diff(x))) + np.sum(w * np.abs(x - y))


def arc_distance(phi, psi):
    """Shortest-arc distance for inputs in (-pi, pi]."""
    d = np.abs(phi - psi)
    return np.minimum(d, TWO_PI - d)


def cost_circ(x, y, alpha, weights=None):
    """L1-TV objective for the circle-valued problem (assumes x, y wrapped)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=np.float64)
    tv = alpha * np.sum(arc_distance(x[1:], x[:-1]))
    fid = np.sum(w * arc_distance(x, y))
    return tv + fid


def wrap(a):
    """Wrap to (-pi, pi] (mirrors the Rust core's wrap_angle)."""
    a = np.asarray(a, dtype=np.float64)
    r = np.mod(a, TWO_PI)
    return np.where(r > np.pi, r - TWO_PI, r)
