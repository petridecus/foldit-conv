//! Core data structures and binary serialization for COORDS format.
//!
//! Binary format (COORDS01, backward compatible with COORDS00):
//! - 8-byte magic header: "COORDS01" (or "COORDS00" for legacy)
//! - 4-byte big-endian u32: number of atoms
//! - Per-atom payload:
//!   - 12 bytes: x, y, z (f32 each, big-endian)
//!   - 1 byte: chain_id (ASCII byte)
//!   - 3 bytes: residue name (3-character code)
//!   - 4 bytes: residue number (i32, big-endian)
//!   - 4 bytes: atom name (4-character code)
//!   - [COORDS01 only] 2 bytes: element symbol (padded with 0)

use thiserror::Error;

/// Chemical element for atoms in a molecular structure.
///
/// Covers biologically-relevant elements found in proteins, nucleic acids,
/// ligands, ions, and waters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    H,
    C,
    N,
    O,
    S,
    P,
    Se,
    Fe,
    Zn,
    Mg,
    Ca,
    Na,
    Cl,
    K,
    Mn,
    Co,
    Ni,
    Cu,
    Br,
    I,
    F,
    Unknown,
}

impl Element {
    /// Parse element from a 1-2 character symbol string (case-insensitive).
    pub fn from_symbol(s: &str) -> Self {
        match s.trim().to_uppercase().as_str() {
            "H" => Element::H,
            "C" => Element::C,
            "N" => Element::N,
            "O" => Element::O,
            "S" => Element::S,
            "P" => Element::P,
            "SE" => Element::Se,
            "FE" => Element::Fe,
            "ZN" => Element::Zn,
            "MG" => Element::Mg,
            "CA" => Element::Ca,
            "NA" => Element::Na,
            "CL" => Element::Cl,
            "K" => Element::K,
            "MN" => Element::Mn,
            "CO" => Element::Co,
            "NI" => Element::Ni,
            "CU" => Element::Cu,
            "BR" => Element::Br,
            "I" => Element::I,
            "F" => Element::F,
            _ => Element::Unknown,
        }
    }

    /// Infer element from a standard protein atom name (e.g., "CA" -> C, "OG" -> O, "SD" -> S).
    ///
    /// For standard protein atoms, the first alphabetic character reliably identifies
    /// the element. This is the same heuristic used by `gpu.rs:atom_name_to_type_index`.
    pub fn from_atom_name(name: &str) -> Self {
        let name = name.trim();
        // Find first alphabetic character
        if let Some(ch) = name.chars().find(|c| c.is_alphabetic()) {
            match ch.to_ascii_uppercase() {
                'C' => Element::C,
                'N' => Element::N,
                'O' => Element::O,
                'S' => Element::S,
                'H' => Element::H,
                'P' => Element::P,
                _ => Element::Unknown,
            }
        } else {
            Element::Unknown
        }
    }

    /// Standard CPK coloring (Corey-Pauling-Koltun).
    pub fn cpk_color(&self) -> [f32; 3] {
        match self {
            Element::H => [1.0, 1.0, 1.0],       // White
            Element::C => [0.4, 0.4, 0.4],       // Dark gray
            Element::N => [0.2, 0.2, 1.0],       // Blue
            Element::O => [1.0, 0.2, 0.2],       // Red
            Element::S => [1.0, 0.85, 0.2],      // Yellow
            Element::P => [1.0, 0.5, 0.0],       // Orange
            Element::Se => [1.0, 0.63, 0.0],     // Orange-yellow
            Element::Fe => [0.56, 0.25, 0.08],   // Rust brown
            Element::Zn => [0.49, 0.50, 0.69],   // Slate blue
            Element::Mg => [0.0, 0.55, 0.0],     // Dark green
            Element::Ca => [0.0, 0.55, 0.0],     // Dark green
            Element::Na => [0.67, 0.36, 0.95],   // Purple
            Element::Cl => [0.12, 0.94, 0.12],   // Green
            Element::K => [0.56, 0.25, 0.83],    // Violet
            Element::Mn => [0.61, 0.48, 0.78],   // Purple-gray
            Element::Co => [0.94, 0.56, 0.63],   // Pink
            Element::Ni => [0.31, 0.82, 0.31],   // Green
            Element::Cu => [0.78, 0.50, 0.20],   // Copper
            Element::Br => [0.65, 0.16, 0.16],   // Dark red
            Element::I => [0.58, 0.0, 0.58],     // Purple
            Element::F => [0.56, 0.88, 0.31],    // Yellow-green
            Element::Unknown => [0.7, 0.7, 0.7], // Light gray
        }
    }

    /// Covalent radius in angstroms (Cambridge CSD values).
    pub fn covalent_radius(&self) -> f32 {
        match self {
            Element::H => 0.31,
            Element::C => 0.76,
            Element::N => 0.71,
            Element::O => 0.66,
            Element::S => 1.05,
            Element::P => 1.07,
            Element::Se => 1.20,
            Element::Fe => 1.32,
            Element::Zn => 1.22,
            Element::Mg => 1.41,
            Element::Ca => 1.76,
            Element::Na => 1.66,
            Element::Cl => 1.02,
            Element::K => 2.03,
            Element::Mn => 1.39,
            Element::Co => 1.26,
            Element::Ni => 1.24,
            Element::Cu => 1.32,
            Element::Br => 1.20,
            Element::I => 1.39,
            Element::F => 0.57,
            Element::Unknown => 0.77,
        }
    }

    /// Van der Waals radius in angstroms.
    pub fn vdw_radius(&self) -> f32 {
        match self {
            Element::H => 1.20,
            Element::C => 1.70,
            Element::N => 1.55,
            Element::O => 1.52,
            Element::S => 1.80,
            Element::P => 1.80,
            Element::Se => 1.90,
            Element::Fe => 2.00,
            Element::Zn => 1.39,
            Element::Mg => 1.73,
            Element::Ca => 2.31,
            Element::Na => 2.27,
            Element::Cl => 1.75,
            Element::K => 2.75,
            Element::Mn => 2.00,
            Element::Co => 2.00,
            Element::Ni => 1.63,
            Element::Cu => 1.40,
            Element::Br => 1.85,
            Element::I => 1.98,
            Element::F => 1.47,
            Element::Unknown => 1.70,
        }
    }

    /// Two-character symbol (padded with space if single char).
    pub fn symbol(&self) -> &'static str {
        match self {
            Element::H => "H",
            Element::C => "C",
            Element::N => "N",
            Element::O => "O",
            Element::S => "S",
            Element::P => "P",
            Element::Se => "Se",
            Element::Fe => "Fe",
            Element::Zn => "Zn",
            Element::Mg => "Mg",
            Element::Ca => "Ca",
            Element::Na => "Na",
            Element::Cl => "Cl",
            Element::K => "K",
            Element::Mn => "Mn",
            Element::Co => "Co",
            Element::Ni => "Ni",
            Element::Cu => "Cu",
            Element::Br => "Br",
            Element::I => "I",
            Element::F => "F",
            Element::Unknown => "X",
        }
    }
}

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

/// Maps multi-character chain ID strings to unique `u8` values.
///
/// Structures with >26 chains (ribosomes, virus capsids) use multi-character
/// chain IDs in mmCIF format (e.g., "AA", "AB"). Since `Coords.chain_ids` stores
/// a single `u8` per atom, this mapper assigns a unique byte to each distinct
/// chain string, preventing collisions that cause cross-chain rendering artifacts.
///
/// Assigns printable ASCII characters (A-Z, a-z, 0-9, then other printable chars)
/// so that PDB export produces valid chain ID columns.
pub struct ChainIdMapper {
    map: std::collections::HashMap<String, u8>,
    next_idx: usize,
}

/// Printable chain ID characters in conventional order: A-Z, a-z, 0-9, then
/// remaining printable ASCII. Covers up to 94 unique chains.
const CHAIN_CHARS: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,-./:;<=>?@[]^_`{|}~";

impl ChainIdMapper {
    pub fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            next_idx: 0,
        }
    }

    /// Get or assign a unique `u8` for the given chain ID string.
    pub fn get_or_assign(&mut self, chain_id: &str) -> u8 {
        if let Some(&id) = self.map.get(chain_id) {
            return id;
        }
        let byte = if self.next_idx < CHAIN_CHARS.len() {
            CHAIN_CHARS[self.next_idx]
        } else {
            // Fallback for >94 chains: use raw sequential bytes past printable range.
            // Extremely rare — only theoretical virus capsids approach this.
            (self.next_idx - CHAIN_CHARS.len()) as u8
        };
        self.next_idx += 1;
        self.map.insert(chain_id.to_string(), byte);
        byte
    }
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
    /// Chemical element per atom (for ball-and-stick rendering, bond inference).
    pub elements: Vec<Element>,
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

// ============================================================================
// Binary serialization (COORDS01 format)
// ============================================================================

pub const COORDS_MAGIC: &[u8; 8] = b"COORDS01";
const COORDS_MAGIC_V0: &[u8; 8] = b"COORDS00";
pub const ASSEMBLY_MAGIC: &[u8; 8] = b"ASSEM01\0";

/// Deserialize COORDS binary format to Coords struct.
/// Supports both COORDS00 (no element data) and COORDS01 (with element data).
pub fn deserialize(coords_bytes: &[u8]) -> Result<Coords, CoordsError> {
    if coords_bytes.len() < 8 {
        return Err(CoordsError::InvalidFormat(
            "Data too short to be valid COORDS".to_string(),
        ));
    }

    let magic = &coords_bytes[0..8];
    let has_elements = magic == COORDS_MAGIC;
    if magic != COORDS_MAGIC && magic != COORDS_MAGIC_V0 {
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

    let per_atom = if has_elements {
        12 + 1 + 3 + 4 + 4 + 2
    } else {
        12 + 1 + 3 + 4 + 4
    };
    if cursor.len() < num_atoms * per_atom {
        return Err(CoordsError::InvalidFormat(
            "Data too short for declared number of atoms".to_string(),
        ));
    }

    let mut atoms = Vec::with_capacity(num_atoms);
    let mut chain_ids = Vec::with_capacity(num_atoms);
    let mut res_names = Vec::with_capacity(num_atoms);
    let mut res_nums = Vec::with_capacity(num_atoms);
    let mut atom_names = Vec::with_capacity(num_atoms);
    let mut elements = Vec::with_capacity(num_atoms);

    for _ in 0..num_atoms {
        let x =
            f32::from_be_bytes(cursor[0..4].try_into().map_err(|_| {
                CoordsError::SerializationError("Invalid x coordinate".to_string())
            })?);
        let y =
            f32::from_be_bytes(cursor[4..8].try_into().map_err(|_| {
                CoordsError::SerializationError("Invalid y coordinate".to_string())
            })?);
        let z =
            f32::from_be_bytes(cursor[8..12].try_into().map_err(|_| {
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

        let res_num =
            i32::from_be_bytes(cursor[0..4].try_into().map_err(|_| {
                CoordsError::SerializationError("Invalid residue number".to_string())
            })?);
        res_nums.push(res_num);
        *cursor = &cursor[4..];

        let mut atom_name = [0u8; 4];
        atom_name.copy_from_slice(&cursor[0..4]);
        atom_names.push(atom_name);
        *cursor = &cursor[4..];

        if has_elements {
            let sym_bytes = &cursor[0..2];
            let sym_str = std::str::from_utf8(sym_bytes)
                .unwrap_or("")
                .trim_matches('\0')
                .trim();
            elements.push(Element::from_symbol(sym_str));
            *cursor = &cursor[2..];
        } else {
            // V0: infer from atom name
            let aname = std::str::from_utf8(&atom_name).unwrap_or("");
            elements.push(Element::from_atom_name(aname));
        }
    }

    Ok(Coords {
        num_atoms,
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    })
}

/// Serialize Coords struct to COORDS binary format (COORDS01).
pub fn serialize(coords: &Coords) -> Result<Vec<u8>, CoordsError> {
    let mut buffer = Vec::with_capacity(8 + 4 + coords.num_atoms * (12 + 1 + 3 + 4 + 4 + 2));

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

        // Element symbol: 2 bytes, null-padded
        let sym = coords.elements.get(i).map_or("X", |e| e.symbol());
        let sym_bytes = sym.as_bytes();
        buffer.push(sym_bytes.first().copied().unwrap_or(b'X'));
        buffer.push(sym_bytes.get(1).copied().unwrap_or(0));
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
    if magic != COORDS_MAGIC && magic != COORDS_MAGIC_V0 {
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

// ============================================================================
// ASSEM01 binary serialization (assembly format with entity metadata)
// ============================================================================

use super::entity::{MoleculeEntity, MoleculeType};

/// Serialize a list of entities to ASSEM01 binary format.
///
/// Format:
/// - 8 bytes: magic "ASSEM01\0"
/// - 4 bytes: entity_count (u32 BE)
/// - Per entity header (5 bytes each):
///   - 1 byte: molecule_type wire byte
///   - 4 bytes: atom_count (u32 BE)
/// - Per atom (26 bytes, same as COORDS01):
///   - 12 bytes: x,y,z (f32 BE)
///   - 1 byte: chain_id
///   - 3 bytes: res_name
///   - 4 bytes: res_num (i32 BE)
///   - 4 bytes: atom_name
///   - 2 bytes: element symbol
pub fn serialize_assembly(entities: &[MoleculeEntity]) -> Result<Vec<u8>, CoordsError> {
    let total_atoms: usize = entities.iter().map(|e| e.coords.num_atoms).sum();
    let header_size = 8 + 4 + entities.len() * 5;
    let atom_size = total_atoms * 26;
    let mut buffer = Vec::with_capacity(header_size + atom_size);

    // Magic
    buffer.extend_from_slice(ASSEMBLY_MAGIC);

    // Entity count
    buffer.extend_from_slice(&(entities.len() as u32).to_be_bytes());

    // Per-entity headers
    for entity in entities {
        buffer.push(entity.molecule_type.to_wire_byte());
        buffer.extend_from_slice(&(entity.coords.num_atoms as u32).to_be_bytes());
    }

    // Atom data (same layout as COORDS01)
    for entity in entities {
        let c = &entity.coords;
        for i in 0..c.num_atoms {
            let atom = &c.atoms[i];
            buffer.extend_from_slice(&atom.x.to_be_bytes());
            buffer.extend_from_slice(&atom.y.to_be_bytes());
            buffer.extend_from_slice(&atom.z.to_be_bytes());
            buffer.push(c.chain_ids[i]);
            buffer.extend_from_slice(&c.res_names[i]);
            buffer.extend_from_slice(&c.res_nums[i].to_be_bytes());
            buffer.extend_from_slice(&c.atom_names[i]);
            let sym = c.elements.get(i).map_or("X", |e| e.symbol());
            let sym_bytes = sym.as_bytes();
            buffer.push(sym_bytes.first().copied().unwrap_or(b'X'));
            buffer.push(sym_bytes.get(1).copied().unwrap_or(0));
        }
    }

    Ok(buffer)
}

/// Deserialize ASSEM01 binary format back to a list of entities.
pub fn deserialize_assembly(bytes: &[u8]) -> Result<Vec<MoleculeEntity>, CoordsError> {
    if bytes.len() < 12 {
        return Err(CoordsError::InvalidFormat(
            "Data too short for ASSEM01 header".to_string(),
        ));
    }

    let magic = &bytes[0..8];
    if magic != ASSEMBLY_MAGIC {
        return Err(CoordsError::InvalidFormat(
            "Invalid magic number for ASSEM01".to_string(),
        ));
    }

    let entity_count = u32::from_be_bytes(
        bytes[8..12]
            .try_into()
            .map_err(|_| CoordsError::InvalidFormat("Invalid entity count".to_string()))?,
    ) as usize;

    let headers_end = 12 + entity_count * 5;
    if bytes.len() < headers_end {
        return Err(CoordsError::InvalidFormat(
            "Data too short for entity headers".to_string(),
        ));
    }

    // Parse entity headers
    let mut entity_headers: Vec<(MoleculeType, usize)> = Vec::with_capacity(entity_count);
    let mut offset = 12;
    for _ in 0..entity_count {
        let mol_type = MoleculeType::from_wire_byte(bytes[offset]).ok_or_else(|| {
            CoordsError::InvalidFormat(format!("Unknown molecule type byte: {}", bytes[offset]))
        })?;
        offset += 1;
        let atom_count = u32::from_be_bytes(bytes[offset..offset + 4].try_into().map_err(|_| {
            CoordsError::InvalidFormat("Invalid atom count in entity header".to_string())
        })?) as usize;
        offset += 4;
        entity_headers.push((mol_type, atom_count));
    }

    let total_atoms: usize = entity_headers.iter().map(|(_, c)| c).sum();
    if bytes.len() < headers_end + total_atoms * 26 {
        return Err(CoordsError::InvalidFormat(
            "Data too short for atom data".to_string(),
        ));
    }

    // Parse atom data per entity
    let mut cursor = &bytes[headers_end..];
    let mut entities = Vec::with_capacity(entity_count);

    for (entity_id, (mol_type, atom_count)) in entity_headers.into_iter().enumerate() {
        let mut atoms = Vec::with_capacity(atom_count);
        let mut chain_ids = Vec::with_capacity(atom_count);
        let mut res_names = Vec::with_capacity(atom_count);
        let mut res_nums = Vec::with_capacity(atom_count);
        let mut atom_names = Vec::with_capacity(atom_count);
        let mut elements = Vec::with_capacity(atom_count);

        for _ in 0..atom_count {
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
            cursor = &cursor[12..];

            chain_ids.push(cursor[0]);
            cursor = &cursor[1..];

            let mut rn = [0u8; 3];
            rn.copy_from_slice(&cursor[0..3]);
            res_names.push(rn);
            cursor = &cursor[3..];

            let res_num = i32::from_be_bytes(cursor[0..4].try_into().map_err(|_| {
                CoordsError::SerializationError("Invalid residue number".to_string())
            })?);
            res_nums.push(res_num);
            cursor = &cursor[4..];

            let mut an = [0u8; 4];
            an.copy_from_slice(&cursor[0..4]);
            atom_names.push(an);
            cursor = &cursor[4..];

            let sym_str = std::str::from_utf8(&cursor[0..2])
                .unwrap_or("")
                .trim_matches('\0')
                .trim();
            elements.push(Element::from_symbol(sym_str));
            cursor = &cursor[2..];
        }

        entities.push(MoleculeEntity {
            entity_id: entity_id as u32,
            molecule_type: mol_type,
            coords: Coords {
                num_atoms: atom_count,
                atoms,
                chain_ids,
                res_names,
                res_nums,
                atom_names,
                elements,
            },
        });
    }

    Ok(entities)
}
