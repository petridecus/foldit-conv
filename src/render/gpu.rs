//! GPU-friendly extraction functions for foldit-render integration.

use crate::types::coords::{deserialize, AtomMetadata, CoordsError};

/// Map atom name to a type index for GPU coloring.
fn atom_name_to_type_index(name: &[u8; 4]) -> u8 {
    let element = name
        .iter()
        .find(|&&b| b != b' ' && b.is_ascii_alphabetic())
        .copied()
        .unwrap_or(b'X');

    match element.to_ascii_uppercase() {
        b'C' => 0,
        b'N' => 1,
        b'O' => 2,
        b'S' => 3,
        b'H' => 4,
        b'P' => 5,
        _ => 6,
    }
}

/// Extract positions array suitable for GPU upload.
pub fn to_positions_f32(coords_bytes: &[u8]) -> Result<Vec<[f32; 4]>, CoordsError> {
    let coords = deserialize(coords_bytes)?;
    Ok(coords
        .atoms
        .iter()
        .map(|a| [a.x, a.y, a.z, 1.0])
        .collect())
}

/// Extract positions as flat f32 array [x0, y0, z0, x1, y1, z1, ...].
pub fn to_positions_flat(coords_bytes: &[u8]) -> Result<Vec<f32>, CoordsError> {
    let coords = deserialize(coords_bytes)?;
    Ok(coords
        .atoms
        .iter()
        .flat_map(|a| [a.x, a.y, a.z])
        .collect())
}

/// Extract atom metadata for GPU uniform buffers.
pub fn to_atom_metadata(coords_bytes: &[u8]) -> Result<AtomMetadata, CoordsError> {
    let coords = deserialize(coords_bytes)?;

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
