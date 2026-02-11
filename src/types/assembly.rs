//! Assembly: a loaded molecular structure with original atom order preserved.
//!
//! Assembly owns both the canonical Coords (original file order) and derived
//! entities (which may have different atom ordering due to BTreeMap grouping).
//! This separation ensures backends receive correct atom ordering while
//! renderers get per-entity access.

use super::coords::{Coords, CoordsError, Element, serialize, deserialize, serialize_assembly};
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

    /// All entities serialized to ASSEM01 bytes (includes molecule type metadata).
    pub fn assembly_bytes(&self) -> Result<Vec<u8>, CoordsError> {
        serialize_assembly(&self.entities)
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

    /// Update from a backend export (e.g., Rosetta COORDS output after wiggle/shake).
    /// Replaces entities whose molecule types appear in the export, preserves entities
    /// whose types don't appear (i.e., types the backend skipped/couldn't load).
    /// This handles the case where dualspace wiggle updates ligand positions while
    /// water/unknown ligands that Rosetta skipped remain at their original positions.
    pub fn update_from_backend(&mut self, backend_coords: Coords) {
        use std::collections::HashSet;

        eprintln!("[assembly::update_from_backend] incoming: {} atoms", backend_coords.num_atoms);

        // Split the backend export into entities to determine which types are present
        let new_entities = split_into_entities(&backend_coords);

        eprintln!("[assembly::update_from_backend] new entities:");
        for e in &new_entities {
            eprintln!("  {:?}: {} atoms (chain {:?})",
                e.molecule_type, e.coords.num_atoms,
                e.coords.chain_ids.first().map(|&c| c as char));
        }

        // Collect molecule types present in the export
        let exported_types: HashSet<MoleculeType> = new_entities.iter()
            .map(|e| e.molecule_type)
            .collect();

        eprintln!("[assembly::update_from_backend] exported types: {:?}", exported_types);

        // Keep entities whose type was NOT exported (backend skipped them)
        let mut kept: Vec<MoleculeEntity> = self.entities.drain(..)
            .filter(|e| !exported_types.contains(&e.molecule_type))
            .collect();

        eprintln!("[assembly::update_from_backend] kept {} entities from original", kept.len());

        // Combine: exported entities first, then kept entities
        let mut updated = new_entities;
        updated.append(&mut kept);

        // Re-assign sequential entity IDs
        for (i, entity) in updated.iter_mut().enumerate() {
            entity.entity_id = i as u32;
        }

        self.entities = updated;
        self.coords = merge_entities(&self.entities);

        eprintln!("[assembly::update_from_backend] final: {} entities, {} total atoms",
            self.entities.len(), self.coords.num_atoms);
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

// ============================================================================
// Combined assembly for multi-assembly Rosetta operations (all entity types)
// ============================================================================

/// Result of combining all entities from multiple assemblies for a Rosetta session.
/// Unlike `CombinedSession` (protein-only), this carries every entity type via ASSEM01.
#[derive(Debug, Clone)]
pub struct CombinedAssembly {
    /// ASSEM01 bytes — all entity types with molecule type metadata.
    pub bytes: Vec<u8>,
    /// Chain IDs assigned per input assembly (for splitting results back).
    pub chain_ids: Vec<Vec<u8>>,
    /// Residue ranges per input assembly: (start, end) — 1-indexed, inclusive.
    /// Counts only protein residues (for lock/selection mapping).
    pub residue_ranges: Vec<(usize, usize)>,
    /// Total protein residues across all assemblies.
    pub total_residues: usize,
}

/// Combine all entities from multiple assemblies for a Rosetta session.
/// Includes all molecule types (protein, ligand, ion, water, DNA, RNA).
/// Remaps chain IDs for uniqueness across assemblies.
pub fn prepare_combined_assembly(assemblies: &[&Assembly]) -> Option<CombinedAssembly> {
    let mut all_entities: Vec<MoleculeEntity> = Vec::new();
    let mut chain_ids_per_assembly: Vec<Vec<u8>> = Vec::new();
    let mut residue_ranges: Vec<(usize, usize)> = Vec::new();

    let mut next_chain_id = b'A';
    let mut global_residue_offset: usize = 0;

    for assembly in assemblies {
        let entities = assembly.entities();
        if entities.is_empty() {
            chain_ids_per_assembly.push(Vec::new());
            continue;
        }

        // Build chain ID remap for this assembly
        let mut chain_id_map: HashMap<u8, u8> = HashMap::new();
        for entity in entities {
            for &cid in &entity.coords.chain_ids {
                chain_id_map.entry(cid).or_insert_with(|| {
                    let id = next_chain_id;
                    next_chain_id = if next_chain_id == b'Z' { b'a' } else { next_chain_id + 1 };
                    id
                });
            }
        }

        let assigned_chain_ids: Vec<u8> = chain_id_map.values().copied().collect();
        chain_ids_per_assembly.push(assigned_chain_ids);

        // Count protein residues (CA atoms) for residue ranges
        let mut protein_residue_count: usize = 0;
        for entity in entities {
            if entity.molecule_type == MoleculeType::Protein {
                let mut seen = std::collections::HashSet::new();
                for i in 0..entity.coords.num_atoms {
                    let atom_name = std::str::from_utf8(&entity.coords.atom_names[i])
                        .unwrap_or("")
                        .trim();
                    if atom_name == "CA" {
                        seen.insert((entity.coords.chain_ids[i], entity.coords.res_nums[i]));
                    }
                }
                protein_residue_count += seen.len();
            }
        }

        if protein_residue_count > 0 {
            let start = global_residue_offset + 1;
            let end = global_residue_offset + protein_residue_count;
            residue_ranges.push((start, end));
            global_residue_offset = end;
        }

        // Clone and remap entities
        for entity in entities {
            let c = &entity.coords;
            let remapped_chain_ids: Vec<u8> = c.chain_ids.iter()
                .map(|&cid| *chain_id_map.get(&cid).unwrap_or(&cid))
                .collect();

            all_entities.push(MoleculeEntity {
                entity_id: entity.entity_id,
                molecule_type: entity.molecule_type,
                coords: Coords {
                    num_atoms: c.num_atoms,
                    atoms: c.atoms.clone(),
                    chain_ids: remapped_chain_ids,
                    res_names: c.res_names.clone(),
                    res_nums: c.res_nums.clone(),
                    atom_names: c.atom_names.clone(),
                    elements: c.elements.clone(),
                },
            });
        }
    }

    if all_entities.is_empty() {
        return None;
    }

    serialize_assembly(&all_entities).ok().map(|bytes| CombinedAssembly {
        bytes,
        chain_ids: chain_ids_per_assembly,
        residue_ranges,
        total_residues: global_residue_offset,
    })
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

    /// Test update_from_backend: Rosetta exports protein + known ion, skips water.
    /// Water entities should be preserved, protein + ion entities should be replaced.
    #[test]
    fn test_update_from_backend_preserves_skipped_types() {
        // Assembly: protein (chain A, 3 atoms) + ion (chain B, ZN) + water (chain C, HOH)
        let coords = Coords {
            num_atoms: 5,
            atoms: (0..5).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'B', b'C'],
            res_names: vec![
                res_name("ALA"), res_name("ALA"), res_name("ALA"),
                res_name("ZN"), res_name("HOH"),
            ],
            res_nums: vec![1, 1, 1, 100, 200],
            atom_names: vec![
                atom_name("N"), atom_name("CA"), atom_name("C"),
                atom_name("ZN"), atom_name("O"),
            ],
            elements: vec![Element::N, Element::C, Element::C, Element::Zn, Element::O],
        };

        let mut assembly = Assembly::from_coords(coords, "test");
        assert_eq!(assembly.entities().len(), 3); // protein, ion, water
        assert_eq!(assembly.coords().num_atoms, 5);

        // Rosetta export: protein with updated positions + ion with updated position
        // (Rosetta loaded both, but skipped water)
        let backend_export = Coords {
            num_atoms: 4,
            atoms: vec![make_atom(10.0), make_atom(11.0), make_atom(12.0), make_atom(50.0)],
            chain_ids: vec![b'A', b'A', b'A', b'B'],
            res_names: vec![
                res_name("ALA"), res_name("ALA"), res_name("ALA"),
                res_name("ZN"),
            ],
            res_nums: vec![1, 1, 1, 2],
            atom_names: vec![
                atom_name("N"), atom_name("CA"), atom_name("C"),
                atom_name("ZN"),
            ],
            elements: vec![Element::N, Element::C, Element::C, Element::Zn],
        };

        assembly.update_from_backend(backend_export);

        // Should have 5 atoms: 3 updated protein + 1 updated ion + 1 preserved water
        assert_eq!(assembly.coords().num_atoms, 5);

        // Water entity should be preserved (original position)
        let water: Vec<_> = assembly.entities().iter()
            .filter(|e| e.molecule_type == MoleculeType::Water)
            .collect();
        assert_eq!(water.len(), 1);
        assert_eq!(water[0].coords.num_atoms, 1);
        assert!((water[0].coords.atoms[0].x - 4.0).abs() < 1e-6); // original x=4.0

        // Protein should be updated
        let protein: Vec<_> = assembly.entities().iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein)
            .collect();
        assert_eq!(protein.len(), 1);
        assert!((protein[0].coords.atoms[0].x - 10.0).abs() < 1e-6); // updated x=10.0

        // Ion should be updated
        let ion: Vec<_> = assembly.entities().iter()
            .filter(|e| e.molecule_type == MoleculeType::Ion)
            .collect();
        assert_eq!(ion.len(), 1);
        assert!((ion[0].coords.atoms[0].x - 50.0).abs() < 1e-6); // updated x=50.0
    }

    /// Test update_from_backend: protein-only export preserves all non-protein.
    #[test]
    fn test_update_from_backend_protein_only_export() {
        // Assembly: protein + ligand + water
        let coords = Coords {
            num_atoms: 5,
            atoms: (0..5).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'B', b'C'],
            res_names: vec![
                res_name("ALA"), res_name("ALA"), res_name("ALA"),
                res_name("ATP"), res_name("HOH"),
            ],
            res_nums: vec![1, 1, 1, 100, 200],
            atom_names: vec![
                atom_name("N"), atom_name("CA"), atom_name("C"),
                atom_name("C1"), atom_name("O"),
            ],
            elements: vec![Element::Unknown; 5],
        };

        let mut assembly = Assembly::from_coords(coords, "test");

        // Rosetta only exports protein (skipped ATP and HOH)
        let backend_export = Coords {
            num_atoms: 3,
            atoms: vec![make_atom(20.0), make_atom(21.0), make_atom(22.0)],
            chain_ids: vec![b'A'; 3],
            res_names: vec![res_name("ALA"); 3],
            res_nums: vec![1; 3],
            atom_names: vec![atom_name("N"), atom_name("CA"), atom_name("C")],
            elements: vec![Element::Unknown; 3],
        };

        assembly.update_from_backend(backend_export);

        // Should have 5 atoms: 3 protein (updated) + 1 cofactor (preserved) + 1 water (preserved)
        assert_eq!(assembly.coords().num_atoms, 5);
        assert_eq!(assembly.entities().len(), 3);

        // Protein updated
        let protein: Vec<_> = assembly.entities().iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein)
            .collect();
        assert!((protein[0].coords.atoms[0].x - 20.0).abs() < 1e-6);

        // ATP is now classified as Cofactor, preserved from original
        let cofactor: Vec<_> = assembly.entities().iter()
            .filter(|e| e.molecule_type == MoleculeType::Cofactor)
            .collect();
        assert_eq!(cofactor.len(), 1);
        assert!((cofactor[0].coords.atoms[0].x - 3.0).abs() < 1e-6); // original x=3.0

        // Water preserved
        let water: Vec<_> = assembly.entities().iter()
            .filter(|e| e.molecule_type == MoleculeType::Water)
            .collect();
        assert_eq!(water.len(), 1);
    }

    #[test]
    fn test_assembly_bytes_roundtrip_mixed() {
        use crate::types::coords::deserialize_assembly;

        // Build assembly with protein + cofactor (ATP) + ion
        let coords = Coords {
            num_atoms: 5,
            atoms: vec![
                make_atom(1.0), make_atom(2.0), make_atom(3.0),
                make_atom(10.0), make_atom(20.0),
            ],
            chain_ids: vec![b'A', b'A', b'A', b'B', b'C'],
            res_names: vec![
                res_name("ALA"), res_name("ALA"), res_name("ALA"),
                res_name("ATP"), res_name("ZN"),
            ],
            res_nums: vec![1, 1, 1, 1, 1],
            atom_names: vec![
                atom_name("N"), atom_name("CA"), atom_name("C"),
                atom_name("C1"), atom_name("ZN"),
            ],
            elements: vec![Element::N, Element::C, Element::C, Element::C, Element::Zn],
        };

        let assembly = Assembly::from_coords(coords, "test");
        assert!(assembly.entities().len() >= 3); // protein, cofactor, ion

        let bytes = assembly.assembly_bytes().unwrap();
        assert_eq!(&bytes[0..8], b"ASSEM01\0");

        let roundtripped = deserialize_assembly(&bytes).unwrap();
        assert_eq!(roundtripped.len(), assembly.entities().len());

        // Verify molecule types preserved
        for (orig, rt) in assembly.entities().iter().zip(roundtripped.iter()) {
            assert_eq!(orig.molecule_type, rt.molecule_type);
            assert_eq!(orig.coords.num_atoms, rt.coords.num_atoms);
        }

        // Verify atom data preserved
        let orig_protein: Vec<_> = assembly.entities().iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein).collect();
        let rt_protein: Vec<_> = roundtripped.iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein).collect();
        assert_eq!(orig_protein.len(), rt_protein.len());
        for (o, r) in orig_protein.iter().zip(rt_protein.iter()) {
            for i in 0..o.coords.num_atoms {
                assert!((o.coords.atoms[i].x - r.coords.atoms[i].x).abs() < 1e-6);
                assert_eq!(o.coords.chain_ids[i], r.coords.chain_ids[i]);
                assert_eq!(o.coords.res_names[i], r.coords.res_names[i]);
                assert_eq!(o.coords.res_nums[i], r.coords.res_nums[i]);
                assert_eq!(o.coords.atom_names[i], r.coords.atom_names[i]);
            }
        }
    }

    #[test]
    fn test_assembly_bytes_protein_only() {
        use crate::types::coords::deserialize_assembly;

        let coords = Coords {
            num_atoms: 3,
            atoms: vec![make_atom(1.0), make_atom(2.0), make_atom(3.0)],
            chain_ids: vec![b'A'; 3],
            res_names: vec![res_name("ALA"); 3],
            res_nums: vec![1; 3],
            atom_names: vec![atom_name("N"), atom_name("CA"), atom_name("C")],
            elements: vec![Element::N, Element::C, Element::C],
        };

        let assembly = Assembly::from_coords(coords, "prot");
        let bytes = assembly.assembly_bytes().unwrap();
        let roundtripped = deserialize_assembly(&bytes).unwrap();
        assert_eq!(roundtripped.len(), 1);
        assert_eq!(roundtripped[0].molecule_type, MoleculeType::Protein);
        assert_eq!(roundtripped[0].coords.num_atoms, 3);
    }

    #[test]
    fn test_assembly_bytes_single_atom_ion() {
        use crate::types::coords::deserialize_assembly;

        let coords = Coords {
            num_atoms: 1,
            atoms: vec![make_atom(5.5)],
            chain_ids: vec![b'X'],
            res_names: vec![res_name("ZN")],
            res_nums: vec![99],
            atom_names: vec![atom_name("ZN")],
            elements: vec![Element::Zn],
        };

        let assembly = Assembly::from_coords(coords, "ion");
        let bytes = assembly.assembly_bytes().unwrap();
        let roundtripped = deserialize_assembly(&bytes).unwrap();
        assert_eq!(roundtripped.len(), 1);
        assert_eq!(roundtripped[0].molecule_type, MoleculeType::Ion);
        assert_eq!(roundtripped[0].coords.num_atoms, 1);
        assert!((roundtripped[0].coords.atoms[0].x - 5.5).abs() < 1e-6);
    }

    #[test]
    fn test_assembly_bytes_empty_entities() {
        use crate::types::coords::{serialize_assembly, deserialize_assembly};

        let entities: Vec<MoleculeEntity> = Vec::new();
        let bytes = serialize_assembly(&entities).unwrap();
        let roundtripped = deserialize_assembly(&bytes).unwrap();
        assert!(roundtripped.is_empty());
    }

    #[test]
    fn test_assembly_byte_layout() {
        use crate::types::coords::serialize_assembly;
        use crate::types::entity::MoleculeEntity;

        // Single protein entity with 1 atom
        let entities = vec![MoleculeEntity {
            entity_id: 0,
            molecule_type: MoleculeType::Protein,
            coords: Coords {
                num_atoms: 1,
                atoms: vec![CoordsAtom { x: 1.0, y: 2.0, z: 3.0, occupancy: 1.0, b_factor: 0.0 }],
                chain_ids: vec![b'A'],
                res_names: vec![res_name("ALA")],
                res_nums: vec![1],
                atom_names: vec![atom_name("CA")],
                elements: vec![Element::C],
            },
        }];

        let bytes = serialize_assembly(&entities).unwrap();

        // Magic: 8 bytes
        assert_eq!(&bytes[0..8], b"ASSEM01\0");
        // Entity count: 1
        assert_eq!(u32::from_be_bytes(bytes[8..12].try_into().unwrap()), 1);
        // Entity header: mol_type=0 (Protein), atom_count=1
        assert_eq!(bytes[12], 0); // Protein
        assert_eq!(u32::from_be_bytes(bytes[13..17].try_into().unwrap()), 1);
        // Atom data starts at offset 17
        // x=1.0
        assert_eq!(f32::from_be_bytes(bytes[17..21].try_into().unwrap()), 1.0);
        // y=2.0
        assert_eq!(f32::from_be_bytes(bytes[21..25].try_into().unwrap()), 2.0);
        // z=3.0
        assert_eq!(f32::from_be_bytes(bytes[25..29].try_into().unwrap()), 3.0);
        // chain_id = 'A'
        assert_eq!(bytes[29], b'A');
        // Total size: 8 + 4 + 5 + 26 = 43
        assert_eq!(bytes.len(), 43);
    }

    #[test]
    fn test_prepare_combined_assembly() {
        // Assembly 1: protein (chain A) + Zn ion
        let coords1 = Coords {
            num_atoms: 4,
            atoms: (0..4).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'A'],
            res_names: vec![
                res_name("ALA"), res_name("ALA"), res_name("ALA"),
                res_name("ZN"),
            ],
            res_nums: vec![1, 1, 1, 100],
            atom_names: vec![
                atom_name("N"), atom_name("CA"), atom_name("C"),
                atom_name("ZN"),
            ],
            elements: vec![Element::N, Element::C, Element::C, Element::Zn],
        };
        let asm1 = Assembly::from_coords(coords1, "asm1");

        // Assembly 2: protein (chain A) only
        let coords2 = Coords {
            num_atoms: 3,
            atoms: (10..13).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A'; 3],
            res_names: vec![res_name("GLY"); 3],
            res_nums: vec![1; 3],
            atom_names: vec![atom_name("N"), atom_name("CA"), atom_name("C")],
            elements: vec![Element::N, Element::C, Element::C],
        };
        let asm2 = Assembly::from_coords(coords2, "asm2");

        let combined = prepare_combined_assembly(&[&asm1, &asm2]).unwrap();

        // Should have bytes
        assert!(!combined.bytes.is_empty());
        assert_eq!(&combined.bytes[0..8], b"ASSEM01\0");

        // Both assemblies should have chain IDs assigned
        assert_eq!(combined.chain_ids.len(), 2);
        assert!(!combined.chain_ids[0].is_empty());
        assert!(!combined.chain_ids[1].is_empty());

        // Chain IDs should be disjoint
        for id in &combined.chain_ids[0] {
            assert!(!combined.chain_ids[1].contains(id));
        }

        // Total protein residues = 1 (from asm1) + 1 (from asm2)
        assert_eq!(combined.total_residues, 2);
    }
}
