# Changelog

All notable changes to the `l1tv` Python package are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The MATLAB reference implementation is tracked separately in this same repo's
git history; algorithmic semantics are unchanged across the port.

## [1.0.0] — 2026-05-11

Promoted from 0.1.0 to 1.0.0 as part of the lab-wide maintenance cycle. The
Python/Rust port has been stable since its initial commit and the public API
(`min_l1_tv`, `min_l1_tv_circ`) is unchanged; 1.0.0 signals API stability.

### Changed

- Released as 1.0.0 (was 0.1.0 unreleased).
- README rewritten and harmonised with the lab-repo family (badge block,
  Quickstart-first ordering, "See also" section linking the five sibling
  repos, License footer).
- Auto-create GitHub Release on tag push (alongside PyPI publish).
- `CITATION.cff` populated with `version` and `date-released` fields so
  citation tooling reads from the file directly rather than falling back to
  GitHub Releases.

## [0.1.0] — Initial port (Python/Rust, pre-release)

Initial Python/Rust port of the MATLAB L1TV library by Storath, Weinmann
(2016).

### Added

- Rust core (`_l1tv_core` cdylib, pyo3 0.24.1, numpy 0.24, ndarray 0.16):
  - `solve_l1_tv_real(y, alpha, weights, use_dist_transform)` — exact
    L1-TV solver for real-valued 1-D signals.
  - `solve_l1_tv_circ(y, alpha, weights, use_dist_transform)` — exact
    L1-TV solver for circle-valued 1-D signals with shortest-arc distance.
  - `dist_transform.l1_dt_with_argmin` — forward + backward L1 distance
    transform with argmin tracking (used by both solvers).
- Python facade (`l1tv` package):
  - `min_l1_tv` and `min_l1_tv_circ` public functions.
  - Boundary handling: any 1-D shape (list, tuple, row, column) is
    accepted via `np.ascontiguousarray + .ravel()`. Validates finite y,
    non-negative alpha, shape-matched non-negative weights.
- Tests:
  - 25 Rust unit tests covering DT correctness, candidate-set theorem,
    edge cases, and naive-vs-DT objective parity for both solvers.
  - 28 Python tests covering the public API, input validation, and shape
    handling.
  - 16 MATLAB-vs-Rust objective parity tests (parametrised) using
    pre-generated `.mat` fixtures; relerr < 1e-10 on every fixture.
- Demos: `demos_python/demo_real.py` and `demo_circ.py` mirror the
  original MATLAB demos and run end-to-end with optional matplotlib
  plotting.

### Memory layout

The Rust port uses a rolling pair of K-cost columns plus a `u32` argmin-
pointer table for O(N) backtracking, replacing MATLAB's full `(K, N)`
double-precision cost table. Net memory roughly halves at typical sizes
and can drop further if argmin pointers shrink to `u16` (when `K < 65536`).

### Differences from the MATLAB reference

- Input shape: silently produced wrong results for row-vector input in
  `L1TV_Circ.m` (the antipodal vertical concat became a 2×K matrix). The
  Python facade rules this out by construction.
- Numerical type: `f64` only. Cast at the boundary if you need `f32`.
- `setPath.m`-style invasive path setup is not relevant; just
  `pip install l1tv`.

These differences do not affect algorithmic semantics for column-vector
MATLAB inputs, which is the only shape the MATLAB demos used.

[0.1.0]: https://github.com/mstorath/L1TV/releases/tag/v0.1.0
