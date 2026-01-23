//! COORDS format handling - a compact binary format for protein coordinates.
//!
//! This module provides:
//! - Core types: `Coords`, `CoordsAtom`, `CoordsError`, `AtomMetadata`
//! - Binary serialization: `binary::serialize`, `binary::deserialize`
//! - PDB/mmCIF conversion: `pdb::pdb_to_coords`, `pdb::mmcif_to_coords`, `pdb::coords_to_pdb`
//! - GPU extraction: `gpu::to_positions_f32`, `gpu::to_atom_metadata`
//! - Python bindings (feature-gated)

pub mod binary;
pub mod gpu;
pub mod pdb;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

// Re-export commonly used items at the module level
pub use binary::{
    atom_count as coords_atom_count, deserialize as deserialize_coords_internal,
    serialize as serialize_coords,
};
pub use gpu::{
    to_atom_metadata as coords_to_atom_metadata, to_positions_f32 as coords_to_positions_f32,
    to_positions_flat as coords_to_positions_flat,
};
pub use pdb::{
    coords_to_pdb as coords_bytes_to_pdb, mmcif_to_coords as mmcif_to_coords_internal,
    pdb_to_coords as pdb_to_coords_internal,
};
pub use types::{AtomMetadata, Coords, CoordsAtom, CoordsError};

// Re-export Python functions at module level for lib.rs registration
#[cfg(feature = "python")]
pub use python::{
    coords_to_atom_array, coords_to_pdb, deserialize_coords_py, mmcif_to_coords, pdb_to_coords,
};
