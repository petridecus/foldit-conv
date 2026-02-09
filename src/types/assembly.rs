//! Assembly: a loaded molecular structure with original atom order preserved.
//!
//! Assembly owns both the canonical Coords (original file order) and derived
//! entities (which may have different atom ordering due to BTreeMap grouping).
//! This separation ensures backends receive correct atom ordering while
//! renderers get per-entity access.

use super::coords::{Coords, CoordsError, Element, serialize, deserialize};
use super::entity::{MoleculeEntity, MoleculeType, split_into_entities, merge_entities};
use crate::ops::transform::{protein_only, extract_ca_positions};
use glam::Vec3;
use std::collections::HashMap;

/// A molecular assembly from a single source (file, prediction, design).
/// Owns canonical Coords (original atom order) alongside classified entities.
#[derive(Debug, Clone)]
pub struct Assembly {
    /// Original coords in file/source order — source of truth for backends.
    coords: Coords,
    /// Classified entities derived from coords (may have different atom ordering).
    entities: Vec<MoleculeEntity>,
    /// Human-readable name.
    pub name: String,
}

impl Assembly {
    /// Create an Assembly from Coords (the common path — adapters produce Coords).
    pub fn from_coords(coords: Coords, name: impl Into<String>) -> Self {
        let entities = split_into_entities(&coords);
        Self {
            coords,
            entities,
            name: name.into(),
        }
    }

    /// Create an Assembly from pre-split entities (e.g., when loading from a scene).
    pub fn from_entities(entities: Vec<MoleculeEntity>, name: impl Into<String>) -> Self {
        let coords = merge_entities(&entities);
        Self {
            coords,
            entities,
            name: name.into(),
        }
    }

    // -- Original-order access (for backends) --

    /// The original Coords in file/source order.
    pub fn coords(&self) -> &Coords {
        &self.coords
    }

    /// Protein-only Coords in original atom order (for backends like Rosetta).
    pub fn protein_coords(&self) -> Coords {
        protein_only(&self.coords)
    }

    /// Protein-only Coords serialized to COORDS01 bytes.
    pub fn protein_coords_bytes(&self) -> Result<Vec<u8>, CoordsError> {
        serialize(&protein_only(&self.coords))
    }

    // -- Entity access (for rendering, UI) --

    /// All classified entities.
    pub fn entities(&self) -> &[MoleculeEntity] {
        &self.entities
    }

    /// Protein entities only.
    pub fn protein_entities(&self) -> impl Iterator<Item = &MoleculeEntity> {
        self.entities.iter().filter(|e| e.molecule_type == MoleculeType::Protein)
    }

    /// Non-protein entities (ligands, ions, water, DNA, RNA).
    pub fn non_protein_entities(&self) -> impl Iterator<Item = &MoleculeEntity> {
        self.entities.iter().filter(|e| e.molecule_type != MoleculeType::Protein)
    }

    // -- Derived data --

    /// CA positions from the protein portion.
    pub fn ca_positions(&self) -> Vec<Vec3> {
        let protein = protein_only(&self.coords);
        extract_ca_positions(&protein)
    }

    /// Number of protein residues (CA count).
    pub fn residue_count(&self) -> usize {
        self.ca_positions().len()
    }

    // -- Mutation --

    /// Replace all coords and re-derive entities.
    pub fn update_coords(&mut self, coords: Coords) {
        self.entities = split_into_entities(&coords);
        self.coords = coords;
    }

    /// Replace protein entity coords (keeps non-protein entities).
    /// Splits the incoming combined protein coords by chain ID so each
    /// entity only receives its own chain's atoms, avoiding duplication.
    pub fn update_protein_coords(&mut self, protein: Coords) {
        // Re-split the incoming protein coords into per-chain entities
        let new_protein = split_into_entities(&protein);

        // Remove old protein entities, keep non-protein
        self.entities.retain(|e| e.molecule_type != MoleculeType::Protein);

        // Prepend new protein entities (proteins first, then non-proteins)
        let mut updated: Vec<MoleculeEntity> = new_protein
            .into_iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein)
            .collect();
        updated.append(&mut self.entities);

        // Re-assign sequential entity IDs
        for (i, entity) in updated.iter_mut().enumerate() {
            entity.entity_id = i as u32;
        }

        self.entities = updated;
        self.coords = merge_entities(&self.entities);
    }
}

// ============================================================================
// Combined session for multi-assembly Rosetta operations
// ============================================================================

/// Result of combining protein atoms from multiple assemblies for a Rosetta session.
#[derive(Debug, Clone)]
pub struct CombinedSession {
    /// COORDS01 bytes, protein-only, with unique chain IDs.
    pub bytes: Vec<u8>,
    /// Chain IDs assigned per input assembly (for splitting results back).
    pub chain_ids: Vec<Vec<u8>>,
    /// Residue ranges per input assembly: (start, end) — 1-indexed, inclusive.
    pub residue_ranges: Vec<(usize, usize)>,
    /// Total protein residues across all assemblies.
    pub total_residues: usize,
}

/// Combine protein atoms from multiple assemblies for a Rosetta session.
/// Uses original atom order from each assembly. Remaps chain IDs for uniqueness.
/// Serializes once at the end.
pub fn prepare_combined_session(assemblies: &[&Assembly]) -> Option<CombinedSession> {
    let mut combined_atoms = Vec::new();
    let mut combined_chain_ids = Vec::new();
    let mut combined_res_names = Vec::new();
    let mut combined_res_nums = Vec::new();
    let mut combined_atom_names = Vec::new();
    let mut combined_elements = Vec::new();

    let mut chain_ids_per_assembly: Vec<Vec<u8>> = Vec::new();
    let mut residue_ranges: Vec<(usize, usize)> = Vec::new();

    let mut next_chain_id = b'A';
    let mut global_residue_offset: usize = 0;

    for assembly in assemblies {
        let protein = assembly.protein_coords();
        if protein.num_atoms == 0 {
            chain_ids_per_assembly.push(Vec::new());
            continue;
        }

        // Count residues (CA atoms)
        let residue_count: usize = {
            let mut seen = std::collections::HashSet::new();
            for i in 0..protein.num_atoms {
                let atom_name = std::str::from_utf8(&protein.atom_names[i]).unwrap_or("").trim();
                if atom_name == "CA" {
                    seen.insert((protein.chain_ids[i], protein.res_nums[i]));
                }
            }
            seen.len()
        };

        // Map original chain IDs to new unique IDs
        let mut chain_id_map = HashMap::new();
        for i in 0..protein.num_atoms {
            let orig_chain = protein.chain_ids[i];
            let mapped_chain = *chain_id_map.entry(orig_chain).or_insert_with(|| {
                let id = next_chain_id;
                next_chain_id = if next_chain_id == b'Z' { b'a' } else { next_chain_id + 1 };
                id
            });

            combined_atoms.push(protein.atoms[i].clone());
            combined_chain_ids.push(mapped_chain);
            combined_res_names.push(protein.res_names[i]);
            combined_res_nums.push(protein.res_nums[i]);
            combined_atom_names.push(protein.atom_names[i]);
            combined_elements.push(protein.elements.get(i).copied().unwrap_or(Element::Unknown));
        }

        let assigned_chain_ids: Vec<u8> = chain_id_map.values().copied().collect();
        chain_ids_per_assembly.push(assigned_chain_ids);

        if residue_count > 0 {
            let start_residue = global_residue_offset + 1;
            let end_residue = global_residue_offset + residue_count;
            residue_ranges.push((start_residue, end_residue));
            global_residue_offset = end_residue;
        }
    }

    if combined_atoms.is_empty() {
        return None;
    }

    let num_atoms = combined_atoms.len();
    let combined = Coords {
        atoms: combined_atoms,
        chain_ids: combined_chain_ids,
        res_names: combined_res_names,
        res_nums: combined_res_nums,
        atom_names: combined_atom_names,
        elements: combined_elements,
        num_atoms,
    };

    serialize(&combined).ok().map(|bytes| CombinedSession {
        bytes,
        chain_ids: chain_ids_per_assembly,
        residue_ranges,
        total_residues: global_residue_offset,
    })
}

/// Split combined Rosetta output back to per-assembly protein coords.
pub fn split_combined_result(
    result_bytes: &[u8],
    chain_ids_per_assembly: &[Vec<u8>],
) -> Result<Vec<Coords>, String> {
    let coords = deserialize(result_bytes)
        .map_err(|e| format!("Failed to deserialize combined coords: {:?}", e))?;

    let mut results = Vec::with_capacity(chain_ids_per_assembly.len());

    for chain_ids in chain_ids_per_assembly {
        let mut atoms = Vec::new();
        let mut cids = Vec::new();
        let mut rnames = Vec::new();
        let mut rnums = Vec::new();
        let mut anames = Vec::new();
        let mut elems = Vec::new();

        for i in 0..coords.num_atoms {
            if chain_ids.contains(&coords.chain_ids[i]) {
                atoms.push(coords.atoms[i].clone());
                cids.push(coords.chain_ids[i]);
                rnames.push(coords.res_names[i]);
                rnums.push(coords.res_nums[i]);
                anames.push(coords.atom_names[i]);
                elems.push(coords.elements.get(i).copied().unwrap_or(Element::Unknown));
            }
        }

        results.push(Coords {
            num_atoms: atoms.len(),
            atoms,
            chain_ids: cids,
            res_names: rnames,
            res_nums: rnums,
            atom_names: anames,
            elements: elems,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::coords::{CoordsAtom, Element};
    use crate::types::entity::MoleculeType;

    fn make_atom(x: f32) -> CoordsAtom {
        CoordsAtom { x, y: 0.0, z: 0.0, occupancy: 1.0, b_factor: 0.0 }
    }

    fn res_name(s: &str) -> [u8; 3] {
        let mut name = [b' '; 3];
        for (i, b) in s.bytes().take(3).enumerate() { name[i] = b; }
        name
    }

    fn atom_name(s: &str) -> [u8; 4] {
        let mut name = [b' '; 4];
        for (i, b) in s.bytes().take(4).enumerate() { name[i] = b; }
        name
    }

    /// Regression test: update_protein_coords on a multi-chain assembly must NOT
    /// duplicate atoms. Previously, the full combined coords were cloned into each
    /// protein entity, causing exponential blowup (4×, 16×, ...).
    #[test]
    fn test_update_protein_coords_no_duplication() {
        // Build a 2-chain protein (A: 3 atoms, B: 3 atoms) + 1 water entity
        let coords = Coords {
            num_atoms: 7,
            atoms: (0..7).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'B', b'B', b'B', b'C'],
            res_names: vec![
                res_name("ALA"), res_name("ALA"), res_name("ALA"),
                res_name("GLY"), res_name("GLY"), res_name("GLY"),
                res_name("HOH"),
            ],
            res_nums: vec![1, 1, 1, 1, 1, 1, 100],
            atom_names: vec![
                atom_name("N"), atom_name("CA"), atom_name("C"),
                atom_name("N"), atom_name("CA"), atom_name("C"),
                atom_name("O"),
            ],
            elements: vec![Element::Unknown; 7],
        };

        let mut assembly = Assembly::from_coords(coords, "test");
        assert_eq!(assembly.entities().len(), 3); // chain A, chain B, water
        assert_eq!(assembly.coords().num_atoms, 7);

        // Simulate a Rosetta update: combined protein coords (both chains, 6 atoms)
        let updated_protein = Coords {
            num_atoms: 6,
            atoms: (10..16).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'B', b'B', b'B'],
            res_names: vec![
                res_name("ALA"), res_name("ALA"), res_name("ALA"),
                res_name("GLY"), res_name("GLY"), res_name("GLY"),
            ],
            res_nums: vec![1, 1, 1, 1, 1, 1],
            atom_names: vec![
                atom_name("N"), atom_name("CA"), atom_name("C"),
                atom_name("N"), atom_name("CA"), atom_name("C"),
            ],
            elements: vec![Element::Unknown; 6],
        };

        assembly.update_protein_coords(updated_protein);

        // Must still have 7 total atoms (6 protein + 1 water), NOT 13 (6+6+1)
        assert_eq!(assembly.coords().num_atoms, 7);
        assert_eq!(assembly.entities().len(), 3); // chain A, chain B, water

        // Protein entities should have 3 atoms each, not 6
        let protein_entities: Vec<_> = assembly
            .entities()
            .iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein)
            .collect();
        assert_eq!(protein_entities.len(), 2);
        assert_eq!(protein_entities[0].coords.num_atoms, 3);
        assert_eq!(protein_entities[1].coords.num_atoms, 3);

        // Verify protein coords were actually updated (x values should be 10+)
        let protein = assembly.protein_coords();
        assert_eq!(protein.num_atoms, 6);
        assert!(protein.atoms[0].x >= 10.0);

        // Second update should not grow either
        let updated_protein2 = Coords {
            num_atoms: 6,
            atoms: (20..26).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'B', b'B', b'B'],
            res_names: vec![
                res_name("ALA"), res_name("ALA"), res_name("ALA"),
                res_name("GLY"), res_name("GLY"), res_name("GLY"),
            ],
            res_nums: vec![1, 1, 1, 1, 1, 1],
            atom_names: vec![
                atom_name("N"), atom_name("CA"), atom_name("C"),
                atom_name("N"), atom_name("CA"), atom_name("C"),
            ],
            elements: vec![Element::Unknown; 6],
        };

        assembly.update_protein_coords(updated_protein2);
        assert_eq!(assembly.coords().num_atoms, 7); // Still 7, not growing
    }
}
