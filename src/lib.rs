// L1TV Rust core — automated port of MATLAB l1tv (Storath, Weinmann, Unser 2016)
// by Claude Opus coding agent, Anthropic, 2026.
//
// PyO3 module: exposes Rust implementations to Python.
//
// This is the Day 1 scaffold; the module currently exports only a version
// constant. The real- and circle-valued solvers land on subsequent days.

use pyo3::prelude::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pymodule]
#[pyo3(name = "_core")]
fn l1tv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    Ok(())
}
