pub mod adapters;
pub mod cif;
pub mod ffi;
pub mod ops;
pub mod render;
pub mod secondary_structure;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule(name = "foldit_conv")]
fn foldit_conv(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Core COORDS functions
    m.add_function(wrap_pyfunction!(python::pdb_to_coords, m)?)?;
    m.add_function(wrap_pyfunction!(python::mmcif_to_coords, m)?)?;
    m.add_function(wrap_pyfunction!(python::coords_to_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(python::deserialize_coords_py, m)?)?;
    // AtomArray / AtomArrayPlus converters (entity-aware, preserves molecule types and bonds)
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::entities_to_atom_array,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::entities_to_atom_array_plus,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::atom_array_to_entities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::coords_to_atom_array,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::coords_to_atom_array_plus,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::atom_array_to_coords,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::entities_to_atom_array_parsed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::parse_file_to_entities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(adapters::atomworks::parse_file_full, m)?)?;

    Ok(())
}
