//! Core data types for the foldit-conv crate.

pub mod assembly;
pub mod coords;
pub mod entity;

// Re-export commonly used items
pub use coords::{
    AtomMetadata, Coords, CoordsAtom, CoordsError, Element,
    ResidueAtoms, ValidationResult,
    // Binary serialization
    deserialize, serialize, atom_count, COORDS_MAGIC,
};
pub use entity::{
    classify_residue, extract_by_type, merge_entities, split_into_entities,
    MoleculeEntity, MoleculeType,
};
pub use assembly::{Assembly, CombinedSession, prepare_combined_session, split_combined_result};
