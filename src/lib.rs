pub mod adapters;
pub mod cif;
pub mod ffi;
pub mod ops;
pub mod render;
pub mod secondary_structure;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

// Backwards-compatible re-exports: `foldit_conv::coords::*` still works
// by aliasing `coords` to a module that re-exports from the new locations.
pub mod coords {
    // Types
    pub use crate::types::coords as types;
    pub use crate::types::coords::{
        AtomMetadata, Coords, CoordsAtom, CoordsError, Element, ResidueAtoms, ValidationResult,
    };

    // Binary serialization — also available as `coords::binary::*`
    pub use crate::types::coords as binary;
    pub use crate::types::coords::{
        atom_count as coords_atom_count, deserialize as deserialize_coords_internal,
        serialize as serialize_coords,
    };

    // Entity
    pub use crate::types::entity;
    pub use crate::types::entity::{
        classify_residue, extract_by_type, merge_entities, split_into_entities, MoleculeEntity,
        MoleculeType, NucleotideRing,
    };

    // Adapters
    pub use crate::adapters::bcif;
    pub use crate::adapters::bcif::{bcif_file_to_coords, bcif_to_coords};
    pub use crate::adapters::dcd;
    pub use crate::adapters::dcd::{dcd_file_to_frames, DcdFrame, DcdHeader, DcdReader};
    pub use crate::adapters::mrc;
    pub use crate::adapters::mrc::{mrc_file_to_density, mrc_to_density};
    pub use crate::adapters::pdb;
    pub use crate::adapters::pdb::{
        coords_to_pdb as coords_bytes_to_pdb, mmcif_file_to_coords, mmcif_str_to_coords,
        mmcif_to_coords as mmcif_to_coords_internal, pdb_file_to_coords, pdb_str_to_coords,
        pdb_to_coords as pdb_to_coords_internal, structure_file_to_coords,
    };

    // Density
    pub use crate::types::density::{DensityMap, DensityError};

    // GPU
    pub use crate::render::gpu;
    pub use crate::render::gpu::{
        to_atom_metadata as coords_to_atom_metadata, to_positions_f32 as coords_to_positions_f32,
        to_positions_flat as coords_to_positions_flat,
    };

    // Render — also available as `coords::render::*`
    pub use crate::render;
    pub use crate::render::{
        extract_sequences, RenderBackboneResidue, RenderCoords, RenderSidechainAtom,
    };

    // Ops - transform
    pub use crate::ops::transform;
    pub use crate::ops::transform::{
        align_coords_bytes, align_to_reference, backbone_only, build_ca_position_map, centroid,
        extract_backbone_chains, extract_ca_from_chains, extract_ca_positions, filter_atoms,
        filter_residues, get_atom_by_name, get_atom_position, get_backbone_atoms_from_chains,
        get_ca_for_residue, get_ca_position_from_chains, get_closest_atom_for_residue,
        get_closest_atom_with_name, get_closest_backbone_atom, heavy_atoms_only,
        interpolate_coords, interpolate_coords_collapse, kabsch_alignment,
        kabsch_alignment_with_scale, protein_only, set_atom_position, transform_coords,
        transform_coords_with_scale, PROTEIN_RESIDUES,
    };

    // Ops - validation
    pub use crate::ops::validation;
    pub use crate::ops::validation::{
        atom_counts, atoms_by_residue, backbone_atoms, completeness_report, expected_heavy_atoms,
        has_complete_backbone, validate_completeness, AtomCounts,
    };

    // Ops - bond inference
    pub use crate::ops::bond_inference;
    pub use crate::ops::bond_inference::{infer_bonds, BondOrder, InferredBond, DEFAULT_TOLERANCE};

    // Python bindings
    #[cfg(feature = "python")]
    pub use crate::python::{
        coords_to_atom_array, coords_to_pdb, deserialize_coords_py, mmcif_to_coords,
        pdb_to_coords,
    };
}

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
    m.add_function(wrap_pyfunction!(adapters::atomworks::entities_to_atom_array, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::atomworks::entities_to_atom_array_plus, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::atomworks::atom_array_to_entities, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::atomworks::coords_to_atom_array, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::atomworks::coords_to_atom_array_plus, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::atomworks::atom_array_to_coords, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::atomworks::entities_to_atom_array_parsed, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::atomworks::parse_file_to_entities, m)?)?;
    m.add_function(wrap_pyfunction!(adapters::atomworks::parse_file_full, m)?)?;

    Ok(())
}
