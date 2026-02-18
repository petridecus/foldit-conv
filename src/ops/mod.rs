//! Operations on coordinate types.

pub mod bond_inference;
pub mod transform;
pub mod validation;

// Re-export commonly used items
pub use bond_inference::{infer_bonds, BondOrder, InferredBond, DEFAULT_TOLERANCE};
pub use transform::{
    align_coords_bytes, align_to_reference, backbone_only, build_ca_position_map, centroid,
    extract_backbone_chains, extract_ca_from_chains, extract_ca_positions, filter_atoms,
    filter_residues, get_atom_by_name, get_atom_position, get_backbone_atoms_from_chains,
    get_ca_for_residue, get_ca_position_from_chains, get_closest_atom_for_residue,
    get_closest_atom_with_name, get_closest_backbone_atom, heavy_atoms_only, interpolate_coords,
    interpolate_coords_collapse, kabsch_alignment, kabsch_alignment_with_scale, protein_only,
    set_atom_position, transform_coords, transform_coords_with_scale, PROTEIN_RESIDUES,
};
pub use validation::{
    atom_counts, atoms_by_residue, backbone_atoms, completeness_report, expected_heavy_atoms,
    has_complete_backbone, validate_completeness, AtomCounts,
};
