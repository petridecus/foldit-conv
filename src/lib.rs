pub mod adapters;
pub mod coords;
pub mod ffi;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule(name = "foldit_conv")]
fn foldit_conv(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Core COORDS functions
    m.add_function(wrap_pyfunction!(coords::pdb_to_coords, m)?)?;
    m.add_function(wrap_pyfunction!(coords::mmcif_to_coords, m)?)?;
    m.add_function(wrap_pyfunction!(coords::coords_to_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(coords::deserialize_coords_py, m)?)?;
    m.add_function(wrap_pyfunction!(coords::coords_to_atom_array, m)?)?;

    // Biotite adapter (used by RC-Foundry models: RFDiffusion3, RosettaFold3, LigandMPNN)
    m.add_function(wrap_pyfunction!(
        adapters::biotite::coords_to_biotite_atom_array,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(adapters::biotite::atom_array_to_coords, m)?)?;

    // ESMFold adapter
    m.add_function(wrap_pyfunction!(adapters::esmfold::esmfold_to_coords, m)?)?;
    m.add_function(wrap_pyfunction!(
        adapters::esmfold::esmfold_atom14_to_coords,
        m
    )?)?;

    // SimpleFold adapter (Boltz)
    m.add_function(wrap_pyfunction!(
        adapters::simplefold::simplefold_to_coords,
        m
    )?)?;

    Ok(())
}
