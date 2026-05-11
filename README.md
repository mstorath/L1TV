# L1TV — Exact L1-TV regularisation of real- or circle-valued signals

[![PyPI](https://img.shields.io/pypi/v/l1tv.svg)](https://pypi.org/project/l1tv/)
[![Python](https://img.shields.io/pypi/pyversions/l1tv.svg)](https://pypi.org/project/l1tv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/mstorath/L1TV/actions/workflows/ci.yml/badge.svg)](https://github.com/mstorath/L1TV/actions/workflows/ci.yml)
[![MATLAB](https://img.shields.io/badge/MATLAB-supported-orange.svg)](#matlab)

Exact, non-iterative L1-TV regularisation for noisy 1D signals — for both real-valued and circle-valued (phase / orientation) data.

![L1 TV denoising of a real-valued signal](/Docs/L1TV_Real.png)

## Paper

> M. Storath, A. Weinmann, M. Unser.
> [*Exact algorithms for L¹-TV regularization of real-valued or circle-valued signals.*](https://arxiv.org/pdf/1504.00499.pdf)
> SIAM Journal on Scientific Computing, 38(1), A614–A630, 2016.

## Quickstart

### Python (Rust core)

```bash
pip install l1tv
```

```python
import numpy as np
from l1tv import min_l1_tv

# noisy real-valued signal
y = np.array([0.0, 0.1, -0.05, 1.05, 0.95, 1.1])
x = min_l1_tv(y, alpha=0.5)
```

The Python package wraps a Rust extension built with [PyO3](https://pyo3.rs)
and [maturin](https://maturin.rs); algorithm crate lives under `src/`,
demos under [`demos_python/`](demos_python/).
See [`README_PYTHON.md`](README_PYTHON.md) for the full Python API,
including the circle-valued variant `min_l1_tv_circ`.

### MATLAB

The original MATLAB reference implementation is in this same repository:

1. Set the MATLAB path by running `setPath.m`.
2. Run a demo from the `Demos/` folder.

## Application examples

The hero image above shows the real-valued case. The circle-valued variant handles phase / orientation data with wrap-around, without any unwrapping pre-step:

![L1 TV denoising of circle-valued signal](/Docs/L1TV_Circ.png)

## How to cite

If you use this software, please cite the paper above. GitHub's "Cite this repository" button on the repo page reads the `version` and `date-released` fields from [`CITATION.cff`](CITATION.cff) and renders BibTeX/APA.

## See also

Sibling projects from the same research program on variational methods for signal and image processing:

- [Pottslab](https://github.com/mstorath/Pottslab) — multilabel image segmentation via the Potts / piecewise-constant Mumford-Shah model
- [CSSD](https://github.com/mstorath/CSSD) — cubic smoothing splines for signals with discontinuities
- [MumfordShah2D](https://github.com/mstorath/MumfordShah2D) — edge-preserving image restoration via the Mumford-Shah model
- [CircleMedianFilter](https://github.com/mstorath/CircleMedianFilter) — fast median filtering for phase or orientation data
- [DCEBE](https://github.com/mstorath/DCEBE) — bolus arrival time estimation for DCE-MRI signals

## License

Released under the MIT License. See [LICENSE](LICENSE).

---

### Project history

The Python / Rust port of this codebase was generated from the original MATLAB reference by a Claude coding agent in 2026.
See [`PORTED_BY.md`](PORTED_BY.md) for full attribution and the porting plan.
