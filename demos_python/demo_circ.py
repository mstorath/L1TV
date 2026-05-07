"""Demo: L1-TV denoising of a circle-valued (phase) signal.

Mirrors Demos/demo_L1TV_Circ.m. Run as:
    python demos_python/demo_circ.py
"""

from __future__ import annotations

import sys

import numpy as np

import l1tv


def rand_cp(rng, r, lam):
    """Bernoulli-thinning of an innovation vector (matches randCP.m)."""
    r = np.asarray(r, dtype=np.float64).copy()
    mask = rng.random(r.shape) <= np.exp(-lam)
    r[mask] = 0.0
    return r


def randl(rng, shape):
    x = rng.random(shape) - 0.5
    return -np.sign(x) * np.log1p(-2.0 * np.abs(x)) / np.sqrt(2.0)


def gauss_kernel_1d(width, sigma):
    halfwin = (int(width) - 1) // 2
    t = np.arange(-halfwin, halfwin + 1, dtype=np.float64)
    h = np.exp(-(t**2) / (2.0 * sigma**2))
    return h / h.sum()


def wrap(a):
    return np.mod(a + np.pi, 2.0 * np.pi) - np.pi


def arc_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 2.0 * np.pi - d)


def delta_snr_circ(ground_truth, data, estimate):
    return 10.0 * np.log10(
        np.sum(arc_dist(ground_truth, data) ** 2)
        / np.sum(arc_dist(ground_truth, estimate) ** 2)
    )


def main():
    rng = np.random.default_rng(12345)
    n = 2000
    t = 2.0 * np.pi
    lam = 20.0 / n
    sigma = 0.3

    # Match MATLAB exactly: innovation drawn uniformly in [-pi, pi], then thinned.
    raw = (rng.random((n,)) - 0.5) * t
    innovation = rand_cp(rng, raw, lam)
    unwrapped = np.cumsum(innovation)
    h = gauss_kernel_1d(n // 10, 10.0)
    smoothed = np.convolve(unwrapped, h, mode="same")
    ground_truth = wrap(smoothed)

    y = wrap(ground_truth + sigma * randl(rng, (n,)))

    alpha = np.sqrt(n) * sigma
    x = l1tv.min_l1_tv_circ(y, alpha)

    print(f"l1tv {l1tv.__version__}: circle-valued L1-TV demo")
    print(f"  N = {n}, alpha = {alpha:.4f}")
    print(f"  SNR improvement: {delta_snr_circ(ground_truth, y, x):.2f} dB")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(y, ".", markersize=2)
        axs[0].set_ylim(-np.pi, np.pi)
        axs[0].set_title("Circle-valued data")
        axs[1].plot(x, ".", markersize=2)
        axs[1].set_ylim(-np.pi, np.pi)
        axs[1].set_title(f"L1-TV restoration ({delta_snr_circ(ground_truth, y, x):.2f} dB)")
        from pathlib import Path

        out = Path(__file__).with_name("demo_circ.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"  Plot saved: {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
