//! Molecule-type classification and per-entity coordinate splitting.
//!
//! Provides:
//! - `MoleculeType` — classification of residues into protein, DNA, RNA, ligand, ion, water
//! - `MoleculeEntity` — a single entity (chain or group) with its own `Coords`
//! - `classify_residue()` — classify a residue name into a `MoleculeType`
//! - `split_into_entities()` — split flat `Coords` into per-entity groups
//! - `merge_entities()` — recombine entities back into flat `Coords`

use crate::ops::transform::PROTEIN_RESIDUES;
use super::coords::{Coords, Element};
use std::collections::BTreeMap;

/// Classification of molecule types found in structural biology files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoleculeType {
    Protein,
    DNA,
    RNA,
    Ligand,
    Ion,
    Water,
}

/// A single entity: one logical molecule (a protein chain, a ligand, waters, etc.)
/// with its own coordinate set.
#[derive(Debug, Clone)]
pub struct MoleculeEntity {
    pub entity_id: u32,
    pub molecule_type: MoleculeType,
    pub coords: Coords,
}

impl MoleculeEntity {
    /// All atom positions as Vec3.
    pub fn positions(&self) -> Vec<glam::Vec3> {
        self.coords
            .atoms
            .iter()
            .map(|a| glam::Vec3::new(a.x, a.y, a.z))
            .collect()
    }

    /// Human-readable label (e.g. "Protein Chain A", "Ligand (ATP)", "Zn²⁺ Ion").
    pub fn label(&self) -> String {
        match self.molecule_type {
            MoleculeType::Protein => {
                // Collect unique chain IDs
                let mut chains: Vec<u8> = self.coords.chain_ids.iter().copied().collect();
                chains.sort_unstable();
                chains.dedup();
                if chains.len() == 1 {
                    format!("Protein Chain {}", chains[0] as char)
                } else {
                    let chain_str: String = chains.iter().map(|&c| c as char).collect();
                    format!("Protein Chains {}", chain_str)
                }
            }
            MoleculeType::DNA => "DNA".to_string(),
            MoleculeType::RNA => "RNA".to_string(),
            MoleculeType::Ligand => {
                // Use the first residue name as the ligand identifier
                if let Some(rn) = self.coords.res_names.first() {
                    let name = std::str::from_utf8(rn).unwrap_or("???").trim();
                    format!("Ligand ({})", name)
                } else {
                    "Ligand".to_string()
                }
            }
            MoleculeType::Ion => {
                if let Some(rn) = self.coords.res_names.first() {
                    let name = std::str::from_utf8(rn).unwrap_or("???").trim();
                    format!("{} Ion", name)
                } else {
                    "Ion".to_string()
                }
            }
            MoleculeType::Water => format!("Water ({} molecules)", self.residue_count()),
        }
    }

    /// Whether this entity type participates in tab-cycling focus.
    /// Protein: no (focused at group level). Water: no (ambient).
    /// Ligand, Ion, DNA, RNA: yes.
    pub fn is_focusable(&self) -> bool {
        matches!(
            self.molecule_type,
            MoleculeType::Ligand | MoleculeType::Ion | MoleculeType::DNA | MoleculeType::RNA
        )
    }

    /// Number of residues (for protein/nucleic) or molecules (for small mol/ion/water).
    pub fn residue_count(&self) -> usize {
        if self.coords.num_atoms == 0 {
            return 0;
        }
        // Count unique (chain_id, res_num) pairs
        let mut seen = std::collections::HashSet::new();
        for i in 0..self.coords.num_atoms {
            seen.insert((self.coords.chain_ids[i], self.coords.res_nums[i]));
        }
        seen.len()
    }
}

/// Standard DNA residue names (mmCIF convention).
const DNA_RESIDUES: &[&str] = &["DA", "DC", "DG", "DT", "DU", "DI"];

/// Standard RNA residue names.
/// Single-letter names (A, C, G, U) are the mmCIF standard for RNA.
/// Three-letter variants are legacy PDB conventions.
const RNA_RESIDUES: &[&str] = &["A", "C", "G", "U", "ADE", "CYT", "GUA", "URA", "I"];

/// Water residue names.
const WATER_RESIDUES: &[&str] = &["HOH", "WAT", "H2O", "DOD"];

/// Known ion residue names. These are single-atom residues with well-known names.
const ION_RESIDUES: &[&str] = &[
    "ZN", "MG", "NA", "CL", "FE", "MN", "CO", "NI", "CU", "K", "CA", "BR", "I", "F", "LI",
    "CD", "SR", "BA", "CS", "RB", "PB", "HG", "PT", "AU", "AG",
];

/// Classify a residue name into a `MoleculeType`.
///
/// The name should be trimmed of whitespace before calling.
pub fn classify_residue(name: &str) -> MoleculeType {
    if PROTEIN_RESIDUES.contains(&name) {
        return MoleculeType::Protein;
    }
    if WATER_RESIDUES.contains(&name) {
        return MoleculeType::Water;
    }
    if DNA_RESIDUES.contains(&name) {
        return MoleculeType::DNA;
    }
    // RNA single-letter names overlap with element symbols, but in the context
    // of residue names (label_comp_id), single letters are nucleotides.
    if RNA_RESIDUES.contains(&name) {
        return MoleculeType::RNA;
    }
    if ION_RESIDUES.contains(&name) {
        return MoleculeType::Ion;
    }
    MoleculeType::Ligand
}

/// Key for grouping atoms into entities.
/// We use chain_id + molecule_type, except Water which is consolidated.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum EntityKey {
    /// A polymeric chain (protein, DNA, RNA) on a specific chain.
    Chain(u8, MoleculeTypeOrd),
    /// All water molecules consolidated into one entity.
    Water,
    /// A non-polymer residue (ligand, ion) keyed by chain + residue number.
    NonPolymer(u8, i32, MoleculeTypeOrd),
}

/// Wrapper for MoleculeType that implements Ord (for BTreeMap keys).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MoleculeTypeOrd(MoleculeType);

impl PartialOrd for MoleculeTypeOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MoleculeTypeOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0 as u8).cmp(&(other.0 as u8))
    }
}

/// Split a flat `Coords` into per-entity `MoleculeEntity` groups.
///
/// Grouping rules:
/// - Protein/DNA/RNA: grouped by (chain_id, molecule_type) — each polymer chain is one entity
/// - Water: all water residues are consolidated into a single entity
/// - Ligand/Ion: each unique (chain_id, res_num) is its own entity
///
/// Entity IDs are assigned sequentially starting from 0.
pub fn split_into_entities(coords: &Coords) -> Vec<MoleculeEntity> {
    // Group atom indices by entity key (BTreeMap for deterministic ordering)
    let mut groups: BTreeMap<EntityKey, Vec<usize>> = BTreeMap::new();

    for i in 0..coords.num_atoms {
        let res_name = std::str::from_utf8(&coords.res_names[i])
            .unwrap_or("")
            .trim();
        let mol_type = classify_residue(res_name);
        let chain_id = coords.chain_ids[i];

        let key = match mol_type {
            MoleculeType::Water => EntityKey::Water,
            MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA => {
                EntityKey::Chain(chain_id, MoleculeTypeOrd(mol_type))
            }
            MoleculeType::Ligand | MoleculeType::Ion => {
                let res_num = coords.res_nums[i];
                EntityKey::NonPolymer(chain_id, res_num, MoleculeTypeOrd(mol_type))
            }
        };

        groups.entry(key).or_default().push(i);
    }

    // Convert groups to entities
    groups
        .into_iter()
        .enumerate()
        .map(|(entity_id, (key, indices))| {
            let mol_type = match &key {
                EntityKey::Chain(_, mt) | EntityKey::NonPolymer(_, _, mt) => mt.0,
                EntityKey::Water => MoleculeType::Water,
            };

            let mut atoms = Vec::with_capacity(indices.len());
            let mut chain_ids = Vec::with_capacity(indices.len());
            let mut res_names = Vec::with_capacity(indices.len());
            let mut res_nums = Vec::with_capacity(indices.len());
            let mut atom_names = Vec::with_capacity(indices.len());
            let mut elements = Vec::with_capacity(indices.len());

            for &idx in &indices {
                atoms.push(coords.atoms[idx].clone());
                chain_ids.push(coords.chain_ids[idx]);
                res_names.push(coords.res_names[idx]);
                res_nums.push(coords.res_nums[idx]);
                atom_names.push(coords.atom_names[idx]);
                elements.push(coords.elements.get(idx).copied().unwrap_or(Element::Unknown));
            }

            MoleculeEntity {
                entity_id: entity_id as u32,
                molecule_type: mol_type,
                coords: Coords {
                    num_atoms: atoms.len(),
                    atoms,
                    chain_ids,
                    res_names,
                    res_nums,
                    atom_names,
                    elements,
                },
            }
        })
        .collect()
}

/// Merge multiple entities back into a single flat `Coords`.
///
/// Entities are concatenated in order. Useful for recombining before
/// sending to backends that expect a single coordinate set (e.g., Rosetta).
pub fn merge_entities(entities: &[MoleculeEntity]) -> Coords {
    let total_atoms: usize = entities.iter().map(|e| e.coords.num_atoms).sum();

    let mut atoms = Vec::with_capacity(total_atoms);
    let mut chain_ids = Vec::with_capacity(total_atoms);
    let mut res_names = Vec::with_capacity(total_atoms);
    let mut res_nums = Vec::with_capacity(total_atoms);
    let mut atom_names = Vec::with_capacity(total_atoms);
    let mut elements = Vec::with_capacity(total_atoms);

    for entity in entities {
        atoms.extend(entity.coords.atoms.iter().cloned());
        chain_ids.extend_from_slice(&entity.coords.chain_ids);
        res_names.extend_from_slice(&entity.coords.res_names);
        res_nums.extend_from_slice(&entity.coords.res_nums);
        atom_names.extend_from_slice(&entity.coords.atom_names);
        elements.extend_from_slice(&entity.coords.elements);
    }

    Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    }
}

/// Extract a merged `Coords` containing only entities of the given molecule type.
pub fn extract_by_type(entities: &[MoleculeEntity], mol_type: MoleculeType) -> Option<Coords> {
    let matching: Vec<&MoleculeEntity> = entities
        .iter()
        .filter(|e| e.molecule_type == mol_type)
        .collect();

    if matching.is_empty() {
        return None;
    }

    let total_atoms: usize = matching.iter().map(|e| e.coords.num_atoms).sum();
    let mut atoms = Vec::with_capacity(total_atoms);
    let mut chain_ids = Vec::with_capacity(total_atoms);
    let mut res_names = Vec::with_capacity(total_atoms);
    let mut res_nums = Vec::with_capacity(total_atoms);
    let mut atom_names = Vec::with_capacity(total_atoms);
    let mut elements = Vec::with_capacity(total_atoms);

    for entity in matching {
        atoms.extend(entity.coords.atoms.iter().cloned());
        chain_ids.extend_from_slice(&entity.coords.chain_ids);
        res_names.extend_from_slice(&entity.coords.res_names);
        res_nums.extend_from_slice(&entity.coords.res_nums);
        atom_names.extend_from_slice(&entity.coords.atom_names);
        elements.extend_from_slice(&entity.coords.elements);
    }

    Some(Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::coords::CoordsAtom;

    fn make_atom(x: f32) -> CoordsAtom {
        CoordsAtom {
            x,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            b_factor: 0.0,
        }
    }

    fn res_name(s: &str) -> [u8; 3] {
        let mut name = [b' '; 3];
        for (i, b) in s.bytes().take(3).enumerate() {
            name[i] = b;
        }
        name
    }

    fn atom_name(s: &str) -> [u8; 4] {
        let mut name = [b' '; 4];
        for (i, b) in s.bytes().take(4).enumerate() {
            name[i] = b;
        }
        name
    }

    #[test]
    fn test_classify_protein() {
        assert_eq!(classify_residue("ALA"), MoleculeType::Protein);
        assert_eq!(classify_residue("GLY"), MoleculeType::Protein);
        assert_eq!(classify_residue("MSE"), MoleculeType::Protein);
    }

    #[test]
    fn test_classify_nucleic() {
        assert_eq!(classify_residue("DA"), MoleculeType::DNA);
        assert_eq!(classify_residue("DT"), MoleculeType::DNA);
        assert_eq!(classify_residue("A"), MoleculeType::RNA);
        assert_eq!(classify_residue("U"), MoleculeType::RNA);
    }

    #[test]
    fn test_classify_water_ion_ligand() {
        assert_eq!(classify_residue("HOH"), MoleculeType::Water);
        assert_eq!(classify_residue("WAT"), MoleculeType::Water);
        assert_eq!(classify_residue("ZN"), MoleculeType::Ion);
        assert_eq!(classify_residue("MG"), MoleculeType::Ion);
        assert_eq!(classify_residue("ATP"), MoleculeType::Ligand);
        assert_eq!(classify_residue("HEM"), MoleculeType::Ligand);
    }

    #[test]
    fn test_split_protein_only() {
        let coords = Coords {
            num_atoms: 6,
            atoms: (0..6).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A'; 6],
            res_names: vec![
                res_name("ALA"), res_name("ALA"), res_name("ALA"),
                res_name("GLY"), res_name("GLY"), res_name("GLY"),
            ],
            res_nums: vec![1, 1, 1, 2, 2, 2],
            atom_names: vec![
                atom_name("N"), atom_name("CA"), atom_name("C"),
                atom_name("N"), atom_name("CA"), atom_name("C"),
            ],
            elements: vec![Element::Unknown; 6],
        };

        let entities = split_into_entities(&coords);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].molecule_type, MoleculeType::Protein);
        assert_eq!(entities[0].coords.num_atoms, 6);
    }

    #[test]
    fn test_split_mixed() {
        let coords = Coords {
            num_atoms: 5,
            atoms: (0..5).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'B', b'C'],
            res_names: vec![
                res_name("ALA"), res_name("ALA"), res_name("HOH"),
                res_name("ATP"), res_name("HOH"),
            ],
            res_nums: vec![1, 1, 100, 1, 200],
            atom_names: vec![
                atom_name("N"), atom_name("CA"), atom_name("O"),
                atom_name("C1"), atom_name("O"),
            ],
            elements: vec![Element::Unknown; 5],
        };

        let entities = split_into_entities(&coords);
        assert_eq!(entities.len(), 3);

        let protein = entities.iter().find(|e| e.molecule_type == MoleculeType::Protein).unwrap();
        assert_eq!(protein.coords.num_atoms, 2);

        let water = entities.iter().find(|e| e.molecule_type == MoleculeType::Water).unwrap();
        assert_eq!(water.coords.num_atoms, 2);

        let ligand = entities.iter().find(|e| e.molecule_type == MoleculeType::Ligand).unwrap();
        assert_eq!(ligand.coords.num_atoms, 1);
    }

    #[test]
    fn test_merge_roundtrip() {
        let coords = Coords {
            num_atoms: 4,
            atoms: (0..4).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'B', b'B'],
            res_names: vec![
                res_name("ALA"), res_name("ALA"),
                res_name("HOH"), res_name("HOH"),
            ],
            res_nums: vec![1, 1, 100, 101],
            atom_names: vec![
                atom_name("N"), atom_name("CA"),
                atom_name("O"), atom_name("O"),
            ],
            elements: vec![Element::Unknown; 4],
        };

        let entities = split_into_entities(&coords);
        let merged = merge_entities(&entities);
        assert_eq!(merged.num_atoms, coords.num_atoms);
    }

    #[test]
    fn test_extract_by_type() {
        let coords = Coords {
            num_atoms: 3,
            atoms: (0..3).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A'],
            res_names: vec![res_name("ALA"), res_name("HOH"), res_name("ZN")],
            res_nums: vec![1, 100, 200],
            atom_names: vec![atom_name("CA"), atom_name("O"), atom_name("ZN")],
            elements: vec![Element::Unknown; 3],
        };

        let entities = split_into_entities(&coords);

        let protein = extract_by_type(&entities, MoleculeType::Protein);
        assert!(protein.is_some());
        assert_eq!(protein.unwrap().num_atoms, 1);

        let dna = extract_by_type(&entities, MoleculeType::DNA);
        assert!(dna.is_none());
    }
}
