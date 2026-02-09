//! Python bindings for core COORDS operations.

use pyo3::prelude::*;

use crate::types::coords::{deserialize, serialize};
use crate::adapters::pdb;

#[pyfunction]
pub fn deserialize_coords_py(coords_bytes: Vec<u8>) -> PyResult<Vec<u8>> {
    deserialize(&coords_bytes)
        .and_then(|coords| serialize(&coords))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
pub fn pdb_to_coords(pdb_str: String) -> PyResult<Vec<u8>> {
    pdb::pdb_to_coords(&pdb_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
pub fn mmcif_to_coords(cif_str: String) -> PyResult<Vec<u8>> {
    pdb::mmcif_to_coords(&cif_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
pub fn coords_to_pdb(coords_bytes: Vec<u8>) -> PyResult<String> {
    pdb::coords_to_pdb(&coords_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
pub fn coords_to_atom_array(py: Python, coords_bytes: Vec<u8>) -> PyResult<Py<PyAny>> {
    let coords = deserialize(&coords_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let num_atoms = coords.num_atoms;

    let coords_list: Vec<Vec<f32>> = coords.atoms.iter().map(|a| vec![a.x, a.y, a.z]).collect();
    let chain_ids: Vec<String> = coords
        .chain_ids
        .iter()
        .map(|&c| (c as char).to_string())
        .collect();
    let res_nums: Vec<i32> = coords.res_nums.clone();
    let res_names: Vec<String> = coords
        .res_names
        .iter()
        .map(|b| std::str::from_utf8(b).unwrap_or("UNK").trim().to_string())
        .collect();
    let atom_names: Vec<String> = coords
        .atom_names
        .iter()
        .map(|b| std::str::from_utf8(b).unwrap_or("X").trim().to_string())
        .collect();

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("coord", coords_list)?;
    dict.set_item("chain_id", chain_ids)?;
    dict.set_item("res_id", res_nums)?;
    dict.set_item("res_name", res_names)?;
    dict.set_item("atom_name", atom_names)?;
    dict.set_item("array_length", num_atoms)?;

    Ok(dict.unbind().into_any())
}
