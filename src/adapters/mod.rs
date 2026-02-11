//! Format adapters for converting various formats to/from COORDS.

pub mod bcif;
pub mod dcd;
pub mod mrc;
pub mod pdb;

#[cfg(feature = "python")]
pub mod biotite;
#[cfg(feature = "python")]
pub mod esmfold;
#[cfg(feature = "python")]
pub mod simplefold;

// Re-export commonly used items
pub use bcif::{bcif_file_to_coords, bcif_to_coords};
pub use dcd::{dcd_file_to_frames, DcdFrame, DcdHeader, DcdReader};
pub use mrc::{mrc_file_to_density, mrc_to_density};
pub use pdb::{
    coords_to_pdb as coords_bytes_to_pdb, mmcif_file_to_coords, mmcif_str_to_coords,
    mmcif_to_coords as mmcif_to_coords_internal, pdb_file_to_coords, pdb_str_to_coords,
    pdb_to_coords as pdb_to_coords_internal, structure_file_to_coords,
};
