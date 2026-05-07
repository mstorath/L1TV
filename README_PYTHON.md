# l1tv

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Exact L1-TV regularisation of real-valued or circle-valued 1-D signals — a
Python/Rust port of the original MATLAB
[L1TV](https://github.com/mstorath/L1TV) by Martin Storath and Andreas Weinmann.

## What it does

Solves the L1 total-variation problem in **one dimension** to global optimum:

- **Real-valued** —
  ```
  argmin_x  α · ∑ |x[i] - x[i+1]|  +  ∑ w[i] · |x[i] - y[i]|
  ```
- **Circle-valued** (phase / orientation data) —
  ```
  argmin_x  α · ∑ d_arc(x[i], x[i+1])  +  ∑ w[i] · d_arc(x[i], y[i])
  ```
  where `d_arc` is the shortest-arc distance on the unit circle.

Unlike iterative TV solvers (PDHG, ADMM, FISTA), the algorithm is **exact and
non-iterative** — it returns a global minimiser in `O(N · K)` time, where
`K = |unique(y)|`, via dynamic programming on the candidate set plus an L1
distance transform speedup.

## Why use it

- **Exactness.** No tolerance/maxiter knobs — the returned `x` is a global
  optimum to floating-point precision.
- **Robustness.** L1 data fidelity tolerates impulse / heavy-tail noise that
  derails L2 methods.
- **Phase data.** The circular variant handles wrap-around without needing
  unwrapping, useful for phase-unwrapping / orientation-field denoising.

## Installation

From PyPI (once published):

```bash
pip install l1tv
```

From source (Rust toolchain required):

```bash
git clone https://github.com/mstorath/L1TV
cd L1TV
pip install maturin
maturin develop --release
```

Optional test/dev dependencies:

```bash
pip install l1tv[test]   # adds scipy + pytest + matplotlib
```

## Quick start

```python
import numpy as np
import l1tv

# --- real-valued: piecewise-constant denoising ---
rng = np.random.default_rng(0)
y = np.repeat(rng.standard_normal(20), 50) + 0.3 * rng.standard_normal(1000)
x = l1tv.min_l1_tv(y, alpha=0.5)            # exact L1-TV solution

# --- circle-valued: phase denoising ---
phase = np.repeat(rng.uniform(-np.pi, np.pi, 20), 50)
y_phase = (phase + 0.3 * rng.standard_normal(1000) + np.pi) % (2*np.pi) - np.pi
x_phase = l1tv.min_l1_tv_circ(y_phase, alpha=0.5)
```

Per-sample data weights:

```python
weights = np.where(np.isnan(y), 0.0, 1.0)   # zero-weight masked samples
x = l1tv.min_l1_tv(np.nan_to_num(y), alpha=0.5, weights=weights)
```

## Reference

M. Storath, A. Weinmann, M. Unser. *Exact algorithms for L1-TV regularization
of real-valued or circle-valued signals.* SIAM Journal on Scientific Computing,
38(1), A614–A630, 2016. https://doi.org/10.1137/15M101796X

If you use this package in academic work, please cite both the paper above
and the package via `CITATION.cff` (GitHub renders a "Cite this repository"
button on the repo page).

## Provenance and license

This Python/Rust package is an **automated port** of the original MATLAB
implementation by Storath and Weinmann, performed by Claude Opus coding agent
(Anthropic, 2026). See [PORTED_BY.md](PORTED_BY.md) for the file-by-file
mapping. Algorithmic semantics are preserved — only input handling, memory
layout, and tooling surface changed. License: MIT (unchanged from the
original).

## Related packages in the same research family

- [pottslab](https://github.com/mstorath/Pottslab) — Potts /
  piecewise-constant Mumford-Shah segmentation (1-D and 2-D, L1 and L2).
- [pycirclemedianfilter](https://pypi.org/project/pycirclemedianfilter/) —
  fast median filter for circle-valued data.
- [CSSD](https://github.com/mstorath/CSSD) — cubic smoothing splines with
  discontinuities.

These packages all originate from the
[Lab for Mathematical Methods in Computer Vision and Machine Learning, THWS](https://www.thws.de).
