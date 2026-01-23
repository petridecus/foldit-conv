//! COORDS binary format serialization and deserialization.
//!
//! Binary format (COORDS00):
//! - 8-byte magic header: "COORDS00"
//! - 4-byte big-endian u32: number of atoms
//! - Per-atom payload (24 bytes each):
//!   - 12 bytes: x, y, z (f32 each, big-endian)
//!   - 1 byte: chain_id (ASCII byte)
//!   - 3 bytes: residue name (3-character code)
//!   - 4 bytes: residue number (i32, big-endian)
//!   - 4 bytes: atom name (4-character code)

use super::types::{Coords, CoordsAtom, CoordsError};

pub const COORDS_MAGIC: &[u8; 8] = b"COORDS00";

/// Deserialize COORDS binary format to Coords struct.
pub fn deserialize(coords_bytes: &[u8]) -> Result<Coords, CoordsError> {
    if coords_bytes.len() < 8 {
        return Err(CoordsError::InvalidFormat(
            "Data too short to be valid COORDS".to_string(),
        ));
    }

    let magic = &coords_bytes[0..8];
    if magic != COORDS_MAGIC {
        return Err(CoordsError::InvalidFormat(
            "Invalid magic number in COORDS header".to_string(),
        ));
    }

    let cursor = &mut &coords_bytes[8..];

    let num_atoms = u32::from_be_bytes(
        cursor
            .get(0..4)
            .ok_or_else(|| CoordsError::InvalidFormat("Missing num_atoms field".to_string()))?
            .try_into()
            .map_err(|_| CoordsError::SerializationError("Invalid num_atoms size".to_string()))?,
    ) as usize;
    *cursor = &cursor[4..];

    if cursor.len() < num_atoms * (12 + 1 + 3 + 4 + 4) {
        return Err(CoordsError::InvalidFormat(
            "Data too short for declared number of atoms".to_string(),
        ));
    }

    let mut atoms = Vec::with_capacity(num_atoms);
    let mut chain_ids = Vec::with_capacity(num_atoms);
    let mut res_names = Vec::with_capacity(num_atoms);
    let mut res_nums = Vec::with_capacity(num_atoms);
    let mut atom_names = Vec::with_capacity(num_atoms);

    for _ in 0..num_atoms {
        let x = f32::from_be_bytes(cursor[0..4].try_into().map_err(|_| {
            CoordsError::SerializationError("Invalid x coordinate".to_string())
        })?);
        let y = f32::from_be_bytes(cursor[4..8].try_into().map_err(|_| {
            CoordsError::SerializationError("Invalid y coordinate".to_string())
        })?);
        let z = f32::from_be_bytes(cursor[8..12].try_into().map_err(|_| {
            CoordsError::SerializationError("Invalid z coordinate".to_string())
        })?);

        atoms.push(CoordsAtom {
            x,
            y,
            z,
            occupancy: 1.0,
            b_factor: 0.0,
        });
        *cursor = &cursor[12..];

        let chain_id = cursor[0];
        chain_ids.push(chain_id);
        *cursor = &cursor[1..];

        let mut res_name = [0u8; 3];
        res_name.copy_from_slice(&cursor[0..3]);
        res_names.push(res_name);
        *cursor = &cursor[3..];

        let res_num = i32::from_be_bytes(cursor[0..4].try_into().map_err(|_| {
            CoordsError::SerializationError("Invalid residue number".to_string())
        })?);
        res_nums.push(res_num);
        *cursor = &cursor[4..];

        let mut atom_name = [0u8; 4];
        atom_name.copy_from_slice(&cursor[0..4]);
        atom_names.push(atom_name);
        *cursor = &cursor[4..];
    }

    Ok(Coords {
        num_atoms,
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
    })
}

/// Serialize Coords struct to COORDS binary format.
pub fn serialize(coords: &Coords) -> Result<Vec<u8>, CoordsError> {
    let mut buffer = Vec::with_capacity(8 + 4 + coords.num_atoms * (12 + 4 + 4 + 1 + 3 + 4 + 4));

    buffer.extend_from_slice(COORDS_MAGIC);

    let num_atoms_u32 = coords.num_atoms as u32;
    buffer.extend_from_slice(&num_atoms_u32.to_be_bytes());

    for i in 0..coords.num_atoms {
        let atom = &coords.atoms[i];
        buffer.extend_from_slice(&atom.x.to_be_bytes());
        buffer.extend_from_slice(&atom.y.to_be_bytes());
        buffer.extend_from_slice(&atom.z.to_be_bytes());

        buffer.push(coords.chain_ids[i]);

        buffer.extend_from_slice(&coords.res_names[i]);

        buffer.extend_from_slice(&coords.res_nums[i].to_be_bytes());

        buffer.extend_from_slice(&coords.atom_names[i]);
    }

    Ok(buffer)
}

/// Get the number of atoms in a COORDS binary without full deserialization.
/// Useful for pre-allocating GPU buffers.
pub fn atom_count(coords_bytes: &[u8]) -> Result<usize, CoordsError> {
    if coords_bytes.len() < 12 {
        return Err(CoordsError::InvalidFormat(
            "Data too short to read atom count".to_string(),
        ));
    }

    let magic = &coords_bytes[0..8];
    if magic != COORDS_MAGIC {
        return Err(CoordsError::InvalidFormat(
            "Invalid magic number".to_string(),
        ));
    }

    let num_atoms = u32::from_be_bytes(
        coords_bytes[8..12]
            .try_into()
            .map_err(|_| CoordsError::InvalidFormat("Invalid atom count bytes".to_string()))?,
    ) as usize;

    Ok(num_atoms)
}
