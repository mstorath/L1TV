"""Demo: L1-TV denoising of a real-valued piecewise-constant signal.

Mirrors Demos/demo_L1TV_Real.m. Run as:
    python demos_python/demo_real.py

Plots are saved alongside this script; matplotlib is optional — if absent,
the demo runs without plotting and only prints the SNR improvement.
"""

from __future__ import annotations

import sys

import numpy as np

import l1tv


def rand_cp(rng, r, lam):
    """Bernoulli-thinning of an innovation vector (matches randCP.m).

    Each entry of r is independently zeroed with probability exp(-lam);
    otherwise it is left at the supplied value. This is a Bernoulli-zeroed
    surrogate for a compound Poisson, accurate for small lam (cf. the
    docstring fix in the MATLAB bugfix branch).
    """
    r = np.asarray(r, dtype=np.float64).copy()
    mask = rng.random(r.shape) <= np.exp(-lam)
    r[mask] = 0.0
    return r


def randl(rng, shape):
    """Laplace random samples with mean 0, variance 1 (matches randl.m)."""
    x = rng.random(shape) - 0.5
    return -np.sign(x) * np.log1p(-2.0 * np.abs(x)) / np.sqrt(2.0)


def gauss_kernel_1d(width, sigma):
    halfwin = (int(width) - 1) // 2
    t = np.arange(-halfwin, halfwin + 1, dtype=np.float64)
    h = np.exp(-(t**2) / (2.0 * sigma**2))
    return h / h.sum()


def delta_snr(ground_truth, data, estimate):
    return 10.0 * np.log10(
        np.sum((ground_truth - data) ** 2) / np.sum((ground_truth - estimate) ** 2)
    )


def main():
    rng = np.random.default_rng(12345)
    n = 2000
    lam = 20.0 / n
    scale = 100.0
    sigma = scale * 0.3

    innovation = rand_cp(rng, rng.standard_normal((n,)), lam)
    signal = scale * np.cumsum(innovation)
    h = gauss_kernel_1d(n // 10, 10.0)
    ground_truth = np.convolve(signal, h, mode="same")
    y = ground_truth + sigma * randl(rng, (n,))

    alpha = np.sqrt(n) * sigma / scale
    x = l1tv.min_l1_tv(y, alpha)

    print(f"l1tv {l1tv.__version__}: real-valued L1-TV demo")
    print(f"  N = {n}, alpha = {alpha:.4f}")
    print(f"  SNR improvement: {delta_snr(ground_truth, y, x):.2f} dB")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(y, ".", markersize=2)
        axs[0].set_title("Real-valued data")
        ylim = axs[0].get_ylim()
        axs[1].plot(x, ".", markersize=2)
        axs[1].set_ylim(ylim)
        axs[1].set_title(f"L1-TV restoration ({delta_snr(ground_truth, y, x):.2f} dB)")
        from pathlib import Path

        out = Path(__file__).with_name("demo_real.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"  Plot saved: {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
