//! Core data types for the foldit-conv crate.

pub mod assembly;
pub mod coords;
pub mod density;
pub mod entity;

// Re-export commonly used items
pub use assembly::{
    assembly_bytes,
    ca_positions,
    prepare_combined_assembly,
    prepare_combined_session,
    // Free functions (replace Assembly struct methods)
    protein_coords,
    protein_coords_bytes,
    residue_count,
    split_combined_result,
    update_entities_from_backend,
    update_protein_entities,
    CombinedAssembly,
    CombinedSession,
};
pub use coords::{
    atom_count,
    // Binary serialization
    deserialize,
    deserialize_assembly,
    serialize,
    serialize_assembly,
    AtomMetadata,
    ChainIdMapper,
    Coords,
    CoordsAtom,
    CoordsError,
    Element,
    ResidueAtoms,
    ValidationResult,
    ASSEMBLY_MAGIC,
    COORDS_MAGIC,
};
pub use density::{DensityError, DensityMap};
pub use entity::{
    classify_residue, extract_by_type, merge_entities, split_into_entities, MoleculeEntity,
    MoleculeType,
};
