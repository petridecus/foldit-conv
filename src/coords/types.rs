//! Core data structures for COORDS format.

use thiserror::Error;

/// Errors that can occur during COORDS operations.
#[derive(Error, Debug)]
pub enum CoordsError {
    #[error("Invalid COORDS format: {0}")]
    InvalidFormat(String),
    #[error("Failed to parse PDB: {0}")]
    PdbParseError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Single atom with coordinates and crystallographic factors.
#[derive(Debug, Clone)]
pub struct CoordsAtom {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub occupancy: f32,
    pub b_factor: f32,
}

/// Complete coordinate structure with atom metadata.
#[derive(Debug, Clone)]
pub struct Coords {
    pub num_atoms: usize,
    pub atoms: Vec<CoordsAtom>,
    pub chain_ids: Vec<u8>,
    pub res_names: Vec<[u8; 3]>,
    pub res_nums: Vec<i32>,
    pub atom_names: Vec<[u8; 4]>,
}

/// Metadata about atoms for GPU uniform buffers (coloring, selection, etc.)
#[derive(Debug, Clone)]
pub struct AtomMetadata {
    /// Chain IDs as bytes (e.g., b'A', b'B')
    pub chain_ids: Vec<u8>,
    /// Residue indices
    pub residue_indices: Vec<i32>,
    /// Atom type indices (derived from atom name for coloring)
    pub atom_type_indices: Vec<u8>,
    /// B-factors (can drive sphere size/opacity)
    pub b_factors: Vec<f32>,
}

/// Atoms present in a single residue (for validation).
#[derive(Debug, Clone)]
pub struct ResidueAtoms {
    pub chain_id: u8,
    pub res_num: i32,
    pub res_name: [u8; 3],
    /// Atom names present in this residue
    pub atoms: Vec<[u8; 4]>,
}

/// Result of validating COORDS completeness.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether all expected atoms are present
    pub is_complete: bool,
    /// Missing atoms: (res_num, res_name, list of missing atom names)
    pub missing_atoms: Vec<(i32, String, Vec<String>)>,
    /// Unexpected atoms: (res_num, res_name, list of extra atom names)
    pub extra_atoms: Vec<(i32, String, Vec<String>)>,
    /// Total residues checked
    pub total_residues: usize,
    /// Residues with missing atoms
    pub incomplete_residues: usize,
}
