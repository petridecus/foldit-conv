//! Format adapters for converting various formats to/from COORDS.

pub mod bcif;
pub mod dcd;
pub mod mrc;
pub mod pdb;

#[cfg(feature = "python")]
pub mod atomworks;

// Re-export commonly used items
pub use bcif::{bcif_file_to_coords, bcif_file_to_entities, bcif_to_coords, bcif_to_entities};
pub use dcd::{dcd_file_to_frames, DcdFrame, DcdHeader, DcdReader};
pub use mrc::{mrc_file_to_density, mrc_to_density};
pub use pdb::{
    coords_to_pdb as coords_bytes_to_pdb, mmcif_file_to_coords, mmcif_file_to_entities,
    mmcif_str_to_coords, mmcif_str_to_entities, mmcif_to_coords as mmcif_to_coords_internal,
    pdb_file_to_coords, pdb_file_to_entities, pdb_str_to_coords, pdb_str_to_entities,
    pdb_to_coords as pdb_to_coords_internal, structure_file_to_coords, structure_file_to_entities,
};
