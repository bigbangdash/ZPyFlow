//! ZPyFlow — Rust core for high-performance lazy Python query pipelines.
//!
//! This crate is compiled as a Python extension module (`cdylib`) via PyO3.
//! It exposes:
//!
//!   - `Query`     — lazy pipeline builder
//!   - `col`       — expression proxy for the expression DSL
//!   - `Expr`      — expression type (result of `col > 5`, `col * 2`, etc.)
//!
//! All other types (`ZStream`, `NumericPipeline`, etc.) are Rust-internal.

pub mod core;
pub mod io;
pub mod python;

use pyo3::prelude::*;
use python::{_convert_to_columnar, _hash_join_by_field, _infer_schema, field, PyAggSpec, PyColProxy, PyExpr, PyFieldExpr, PyQuery};

#[pymodule]
fn _zpyflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQuery>()?;
    m.add_class::<PyExpr>()?;
    m.add_class::<PyColProxy>()?;
    m.add_class::<PyAggSpec>()?;
    m.add_class::<PyFieldExpr>()?;
    m.add_function(wrap_pyfunction!(field, m)?)?;
    m.add_function(wrap_pyfunction!(_infer_schema, m)?)?;
    m.add_function(wrap_pyfunction!(_convert_to_columnar, m)?)?;
    m.add_function(wrap_pyfunction!(_hash_join_by_field, m)?)?;

    m.add("col", PyColProxy {})?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
