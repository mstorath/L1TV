// L1TV Rust core — automated port of MATLAB l1tv (Storath, Weinmann, Unser 2016)
// by Claude Opus coding agent, Anthropic, 2026.
//
// PyO3 module: exposes Rust implementations to Python.

mod dist_transform;
mod real;

use pyo3::prelude::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pymodule]
#[pyo3(name = "_core")]
fn l1tv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    Ok(())
}
