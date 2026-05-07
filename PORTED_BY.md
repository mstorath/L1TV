# Port Attribution

This Python/Rust package (`l1tv`) is an **automated port** of the original
MATLAB *L1TV* library:

| | |
|---|---|
| **Original authors** | Martin Storath, Andreas Weinmann |
| **Original license** | MIT |
| **Original language** | MATLAB |
| **Original repository** | https://github.com/mstorath/L1TV |
| **Reference paper** | M. Storath, A. Weinmann, M. Unser. *Exact algorithms for L1-TV regularization of real-valued or circle-valued signals.* SIAM J. Sci. Comput. 38(1), A614–A630, 2016. https://doi.org/10.1137/15M101796X |

## Port details

| | |
|---|---|
| **Port author** | Claude Opus coding agent (Anthropic, 2026) |
| **Port language** | Python (API layer) + Rust/PyO3 (performance core) |
| **Port license** | MIT (unchanged) |
| **Port repository** | same repo, `claude/l1tv-port-2026-05` branch and successors |

## What was ported

All algorithms are faithful translations of the original MATLAB implementation:

- **`src/dist_transform.rs`** — direct port of `Auxiliary/distTransformL1.m`:
  forward + backward L1 distance transform on a sorted candidate grid.
- **`src/real.rs`** — direct port of `L1TV_Real.m`:
  exact dynamic-programming solver for the real-valued L1-TV problem with
  weighted L1 data fidelity, candidate set `unique(y)`.
- **`src/circ.rs`** — direct port of `L1TV_Circ.m`:
  exact DP solver for circle-valued data, candidate set
  `unique(y) ∪ antipodes`, distance transform over the 3K-replicated grid for
  shortest-arc handling.
- **`l1tv/__init__.py`** — Python facade with input validation, NumPy boundary,
  and pure-Python fallback for environments where the Rust extension is not
  available.

## What changed

The port preserves algorithmic semantics. Notable non-algorithmic differences:

- **Input handling.** The Python facade accepts any 1-D shape (list, tuple,
  row/column NumPy array) and coerces internally. The original MATLAB silently
  produced wrong results on row-vector input for `L1TV_Circ.m`; the Rust port
  is shape-agnostic by construction.
- **Memory layout.** MATLAB allocates a full O(NK) cost table for backtracking.
  The Rust port stores argmin pointers (uint32) plus a rolling K-cost buffer,
  reducing peak memory by 2–8× at typical problem sizes.
- **Parity policy.** Comparisons against the MATLAB reference are made on the
  L1-TV objective value, not on `x` pointwise — the L1-TV problem has tied
  minima for which different argmin tie-breaking is permitted (see
  `tests/test_matlab_parity.py` for the rationale).
- **Numerical type.** `f64` only. Users wanting `f32` cast at the boundary.

The original MATLAB sources remain in this repository under `L1TV_Real.m`,
`L1TV_Circ.m`, `Auxiliary/`, `Demos/`, and `setPath.m`.
