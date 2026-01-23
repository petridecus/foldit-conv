//! PDB and mmCIF format parsing and writing.

use pdbtbx::{
    ContainsAtomConformer, ContainsAtomConformerResidue, ContainsAtomConformerResidueChain,
    Format, ReadOptions, StrictnessLevel,
};
use std::io::BufReader;

use super::binary;
use super::types::{Coords, CoordsAtom, CoordsError};

/// Parse PDB format string to COORDS binary format.
pub fn pdb_to_coords(pdb_str: &str) -> Result<Vec<u8>, CoordsError> {
    parse_structure(pdb_str, Format::Pdb)
}

/// Parse mmCIF format string to COORDS binary format.
pub fn mmcif_to_coords(cif_str: &str) -> Result<Vec<u8>, CoordsError> {
    parse_structure(cif_str, Format::Mmcif)
}

/// Internal function to parse either PDB or mmCIF using pdbtbx.
fn parse_structure(input: &str, format: Format) -> Result<Vec<u8>, CoordsError> {
    let reader = BufReader::new(input.as_bytes());

    let (pdb, _errors) = ReadOptions::new()
        .set_format(format)
        .set_level(StrictnessLevel::Loose)
        .read_raw(reader)
        .map_err(|errs| {
            CoordsError::PdbParseError(
                errs.iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })?;

    let mut atoms = Vec::new();
    let mut chain_ids = Vec::new();
    let mut res_names = Vec::new();
    let mut res_nums = Vec::new();
    let mut atom_names = Vec::new();

    for hier in pdb.atoms_with_hierarchy() {
        let atom = hier.atom();
        let chain = hier.chain();
        let residue = hier.residue();
        let conformer = hier.conformer();

        atoms.push(CoordsAtom {
            x: atom.x() as f32,
            y: atom.y() as f32,
            z: atom.z() as f32,
            occupancy: atom.occupancy() as f32,
            b_factor: atom.b_factor() as f32,
        });

        // Chain ID (first char of chain id string, default to 'A')
        chain_ids.push(chain.id().bytes().next().unwrap_or(b'A'));

        // Residue name from conformer
        let name = conformer.name();
        let mut res_name_bytes = [b' '; 3];
        for (i, b) in name.bytes().take(3).enumerate() {
            res_name_bytes[i] = b;
        }
        res_names.push(res_name_bytes);

        // Residue number
        res_nums.push(residue.serial_number() as i32);

        // Atom name
        let aname = atom.name();
        let mut atom_name_bytes = [b' '; 4];
        for (i, b) in aname.bytes().take(4).enumerate() {
            atom_name_bytes[i] = b;
        }
        atom_names.push(atom_name_bytes);
    }

    if atoms.is_empty() {
        return Err(CoordsError::PdbParseError(
            "No atoms found in structure".to_string(),
        ));
    }

    let coords = Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
    };

    binary::serialize(&coords)
}

/// Convert COORDS binary to PDB format string.
pub fn coords_to_pdb(coords_bytes: &[u8]) -> Result<String, CoordsError> {
    let coords = binary::deserialize(coords_bytes)?;

    let mut pdb_string = String::new();

    for i in 0..coords.num_atoms {
        let atom = &coords.atoms[i];
        let chain_id = coords.chain_ids[i] as char;
        let res_num = coords.res_nums[i];

        // Convert byte arrays to strings, trimming whitespace
        let atom_name = std::str::from_utf8(&coords.atom_names[i]).unwrap_or("X   ");
        let res_name = std::str::from_utf8(&coords.res_names[i]).unwrap_or("UNK");

        // PDB format per spec v3.3
        pdb_string.push_str(&format!(
            "ATOM  {:>5} {:<4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}\n",
            i + 1,
            atom_name,
            res_name,
            chain_id,
            res_num,
            atom.x,
            atom.y,
            atom.z,
            atom.occupancy,
            atom.b_factor
        ));
    }

    pdb_string.push_str("END\n");

    Ok(pdb_string)
}
