//! Core data types for the foldit-conv crate.

pub mod assembly;
pub mod coords;
pub mod density;
pub mod entity;

// Re-export commonly used items
pub use coords::{
    AtomMetadata, ChainIdMapper, Coords, CoordsAtom, CoordsError, Element,
    ResidueAtoms, ValidationResult,
    // Binary serialization
    deserialize, serialize, atom_count, COORDS_MAGIC,
    ASSEMBLY_MAGIC, serialize_assembly, deserialize_assembly,
};
pub use entity::{
    classify_residue, extract_by_type, merge_entities, split_into_entities,
    MoleculeEntity, MoleculeType,
};
pub use assembly::{
    CombinedSession, prepare_combined_session, split_combined_result,
    CombinedAssembly, prepare_combined_assembly,
    // Free functions (replace Assembly struct methods)
    protein_coords, protein_coords_bytes, assembly_bytes, ca_positions, residue_count,
    update_protein_entities, update_entities_from_backend,
};
pub use density::{DensityMap, DensityError};
