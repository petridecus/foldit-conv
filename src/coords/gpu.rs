//! GPU-friendly extraction functions for foldit-render integration.

use super::binary;
use super::types::{AtomMetadata, CoordsError};

/// Map atom name to a type index for GPU coloring.
/// Returns indices suitable for a color lookup table.
fn atom_name_to_type_index(name: &[u8; 4]) -> u8 {
    // Get the first non-space alphabetic character (element symbol)
    let element = name
        .iter()
        .find(|&&b| b != b' ' && b.is_ascii_alphabetic())
        .copied()
        .unwrap_or(b'X');

    match element.to_ascii_uppercase() {
        b'C' => 0, // Carbon - gray
        b'N' => 1, // Nitrogen - blue
        b'O' => 2, // Oxygen - red
        b'S' => 3, // Sulfur - yellow
        b'H' => 4, // Hydrogen - white
        b'P' => 5, // Phosphorus - orange
        _ => 6,    // Other - pink/magenta
    }
}

/// Extract positions array suitable for GPU upload.
/// Returns Vec<[f32; 4]> with w=1.0 for homogeneous coordinates.
pub fn to_positions_f32(coords_bytes: &[u8]) -> Result<Vec<[f32; 4]>, CoordsError> {
    let coords = binary::deserialize(coords_bytes)?;
    Ok(coords
        .atoms
        .iter()
        .map(|a| [a.x, a.y, a.z, 1.0])
        .collect())
}

/// Extract positions as flat f32 array [x0, y0, z0, x1, y1, z1, ...].
/// Suitable for storage buffers without padding.
pub fn to_positions_flat(coords_bytes: &[u8]) -> Result<Vec<f32>, CoordsError> {
    let coords = binary::deserialize(coords_bytes)?;
    Ok(coords
        .atoms
        .iter()
        .flat_map(|a| [a.x, a.y, a.z])
        .collect())
}

/// Extract atom metadata for GPU uniform buffers.
/// Useful for coloring atoms by chain, element, or B-factor.
pub fn to_atom_metadata(coords_bytes: &[u8]) -> Result<AtomMetadata, CoordsError> {
    let coords = binary::deserialize(coords_bytes)?;

    Ok(AtomMetadata {
        chain_ids: coords.chain_ids,
        residue_indices: coords.res_nums,
        atom_type_indices: coords
            .atom_names
            .iter()
            .map(atom_name_to_type_index)
            .collect(),
        b_factors: coords.atoms.iter().map(|a| a.b_factor).collect(),
    })
}
