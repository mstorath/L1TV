# l1tv

Exact L1-TV regularisation of real- or circle-valued 1-D signals — Python/Rust
port of the original MATLAB
[L1TV](https://github.com/mstorath/L1TV) by Martin Storath and Andreas Weinmann.

> **Status: Day 1 scaffold (v0.1.0).** The package is being ported from MATLAB
> in stages. Solver functions arrive on subsequent days; the current build
> exposes only metadata. See [reports/07-l1tv-port-plan.md](https://github.com/mstorath/L1TV/blob/claude/l1tv-port-2026-05/reports/07-l1tv-port-plan.md)
> for the full plan.

## Installation

Once published to PyPI:

```bash
pip install l1tv
```

From source (Rust toolchain required):

```bash
pip install maturin
maturin develop --release
```

## Usage

```python
import l1tv

print(l1tv.__version__)
# Real- and circle-valued solvers will be exposed in v0.2 / v0.3.
```

## Reference

M. Storath, A. Weinmann, M. Unser. *Exact algorithms for L1-TV regularization of
real-valued or circle-valued signals.* SIAM Journal on Scientific Computing,
38(1), A614–A630, 2016. https://doi.org/10.1137/15M101796X

## License

MIT — see `LICENSE`.
