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
