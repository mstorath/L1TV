// L1TV Rust core — automated port of MATLAB l1tv (Storath, Weinmann, Unser 2016)
// by Claude Opus coding agent, Anthropic, 2026.
//
// PyO3 module: exposes Rust implementations to Python.

mod circ;
mod dist_transform;
mod real;

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Real-valued L1-TV solver — Python entry point.
///
/// Caller-side input validation is the Python facade's responsibility; this
/// wrapper trusts the caller to pass a contiguous float64 1-D array and a
/// matching contiguous weights array (or None).
#[pyfunction]
#[pyo3(signature = (y, alpha, weights=None, use_dist_transform=true))]
fn solve_l1_tv_real<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    use_dist_transform: bool,
) -> Bound<'py, PyArray1<f64>> {
    let y_slice = y.as_slice().expect("y must be C-contiguous (caller's job)");
    let w_vec: Option<Vec<f64>> =
        weights.map(|w| w.as_slice().expect("weights must be C-contiguous").to_vec());
    let result = real::solve_l1_tv_real(y_slice, alpha, w_vec.as_deref(), use_dist_transform);
    Array1::from_vec(result).into_pyarray(py)
}

/// Circle-valued L1-TV solver — Python entry point.
#[pyfunction]
#[pyo3(signature = (y, alpha, weights=None, use_dist_transform=true))]
fn solve_l1_tv_circ<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    use_dist_transform: bool,
) -> Bound<'py, PyArray1<f64>> {
    let y_slice = y.as_slice().expect("y must be C-contiguous (caller's job)");
    let w_vec: Option<Vec<f64>> =
        weights.map(|w| w.as_slice().expect("weights must be C-contiguous").to_vec());
    let result = circ::solve_l1_tv_circ(y_slice, alpha, w_vec.as_deref(), use_dist_transform);
    Array1::from_vec(result).into_pyarray(py)
}

#[pymodule]
#[pyo3(name = "_core")]
fn l1tv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(solve_l1_tv_real, m)?)?;
    m.add_function(wrap_pyfunction!(solve_l1_tv_circ, m)?)?;
    Ok(())
}
