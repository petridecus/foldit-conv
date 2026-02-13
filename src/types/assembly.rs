//! Assembly helpers: free functions for working with `Vec<MoleculeEntity>`.
//!
//! Previously this module housed the `Assembly` struct which wrapped
//! `Vec<MoleculeEntity>` + cached `Coords`.  The struct was a "smart cache"
//! that added indirection without real benefit — `merge_entities` already
//! derives `Coords` from entities.  Now every operation is a free function
//! that takes `&[MoleculeEntity]` or `&mut Vec<MoleculeEntity>`.

use super::coords::{Coords, CoordsError, Element, serialize, deserialize, serialize_assembly};
use super::entity::{MoleculeEntity, MoleculeType, split_into_entities, merge_entities};
use crate::ops::transform::{protein_only, extract_ca_positions};
use glam::Vec3;
use std::collections::HashMap;

// ============================================================================
// Derivation helpers (replace Assembly accessor methods)
// ============================================================================

/// Protein-only Coords derived from entities.
pub fn protein_coords(entities: &[MoleculeEntity]) -> Coords {
    let merged = merge_entities(entities);
    protein_only(&merged)
}

/// Protein-only Coords serialized to COORDS01 bytes.
pub fn protein_coords_bytes(entities: &[MoleculeEntity]) -> Result<Vec<u8>, CoordsError> {
    serialize(&protein_coords(entities))
}

/// All entities serialized to ASSEM01 bytes (includes molecule type metadata).
pub fn assembly_bytes(entities: &[MoleculeEntity]) -> Result<Vec<u8>, CoordsError> {
    serialize_assembly(entities)
}

/// CA positions from the protein portion.
pub fn ca_positions(entities: &[MoleculeEntity]) -> Vec<Vec3> {
    extract_ca_positions(&protein_coords(entities))
}

/// Number of protein residues (CA count).
pub fn residue_count(entities: &[MoleculeEntity]) -> usize {
    ca_positions(entities).len()
}

// ============================================================================
// Mutation helpers (replace Assembly mutation methods)
// ============================================================================

/// Replace protein entity coords (keeps non-protein entities).
/// Splits the incoming combined protein coords by chain ID so each
/// entity only receives its own chain's atoms, avoiding duplication.
pub fn update_protein_entities(
    entities: &mut Vec<MoleculeEntity>,
    protein: Coords,
) {
    // Re-split the incoming protein coords into per-chain entities
    let new_protein = split_into_entities(&protein);

    // Remove old protein entities, keep non-protein
    entities.retain(|e| e.molecule_type != MoleculeType::Protein);

    // Prepend new protein entities (proteins first, then non-proteins)
    let mut updated: Vec<MoleculeEntity> = new_protein
        .into_iter()
        .filter(|e| e.molecule_type == MoleculeType::Protein)
        .collect();
    updated.append(entities);

    // Re-assign sequential entity IDs
    for (i, entity) in updated.iter_mut().enumerate() {
        entity.entity_id = i as u32;
    }

    *entities = updated;
}

/// Update from a backend export (e.g., Rosetta COORDS output after wiggle/shake).
/// Replaces entities whose molecule types appear in the export, preserves entities
/// whose types don't appear (i.e., types the backend skipped/couldn't load).
pub fn update_entities_from_backend(
    entities: &mut Vec<MoleculeEntity>,
    backend_coords: Coords,
) {
    use std::collections::HashSet;

    eprintln!("[assembly::update_entities_from_backend] incoming: {} atoms", backend_coords.num_atoms);

    // Split the backend export into entities to determine which types are present
    let new_entities = split_into_entities(&backend_coords);

    eprintln!("[assembly::update_entities_from_backend] new entities:");
    for e in &new_entities {
        eprintln!("  {:?}: {} atoms (chain {:?})",
            e.molecule_type, e.coords.num_atoms,
            e.coords.chain_ids.first().map(|&c| c as char));
    }

    // Collect molecule types present in the export
    let exported_types: HashSet<MoleculeType> = new_entities.iter()
        .map(|e| e.molecule_type)
        .collect();

    eprintln!("[assembly::update_entities_from_backend] exported types: {:?}", exported_types);

    // Keep entities whose type was NOT exported (backend skipped them)
    let mut kept: Vec<MoleculeEntity> = entities.drain(..)
        .filter(|e| !exported_types.contains(&e.molecule_type))
        .collect();

    eprintln!("[assembly::update_entities_from_backend] kept {} entities from original", kept.len());

    // Combine: exported entities first, then kept entities
    let mut updated = new_entities;
    updated.append(&mut kept);

    // Re-assign sequential entity IDs
    for (i, entity) in updated.iter_mut().enumerate() {
        entity.entity_id = i as u32;
    }

    *entities = updated;

    let total_atoms: usize = entities.iter().map(|e| e.coords.num_atoms).sum();
    eprintln!("[assembly::update_entities_from_backend] final: {} entities, {} total atoms",
        entities.len(), total_atoms);
}

// ============================================================================
// Combined session for multi-assembly Rosetta operations
// ============================================================================

/// Result of combining protein atoms from multiple entity groups for a Rosetta session.
#[derive(Debug, Clone)]
pub struct CombinedSession {
    /// COORDS01 bytes, protein-only, with unique chain IDs.
    pub bytes: Vec<u8>,
    /// Chain IDs assigned per input group (for splitting results back).
    pub chain_ids: Vec<Vec<u8>>,
    /// Residue ranges per input group: (start, end) — 1-indexed, inclusive.
    pub residue_ranges: Vec<(usize, usize)>,
    /// Total protein residues across all groups.
    pub total_residues: usize,
}

/// Combine protein atoms from multiple entity groups for a Rosetta session.
/// Uses original atom order from each group. Remaps chain IDs for uniqueness.
/// Serializes once at the end.
pub fn prepare_combined_session(groups: &[&[MoleculeEntity]]) -> Option<CombinedSession> {
    let mut combined_atoms = Vec::new();
    let mut combined_chain_ids = Vec::new();
    let mut combined_res_names = Vec::new();
    let mut combined_res_nums = Vec::new();
    let mut combined_atom_names = Vec::new();
    let mut combined_elements = Vec::new();

    let mut chain_ids_per_group: Vec<Vec<u8>> = Vec::new();
    let mut residue_ranges: Vec<(usize, usize)> = Vec::new();

    let mut next_chain_id = b'A';
    let mut global_residue_offset: usize = 0;

    for entity_slice in groups {
        let protein = protein_coords(entity_slice);
        if protein.num_atoms == 0 {
            chain_ids_per_group.push(Vec::new());
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
        chain_ids_per_group.push(assigned_chain_ids);

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
        chain_ids: chain_ids_per_group,
        residue_ranges,
        total_residues: global_residue_offset,
    })
}

/// Split combined Rosetta output back to per-group protein coords.
pub fn split_combined_result(
    result_bytes: &[u8],
    chain_ids_per_group: &[Vec<u8>],
) -> Result<Vec<Coords>, String> {
    let coords = deserialize(result_bytes)
        .map_err(|e| format!("Failed to deserialize combined coords: {:?}", e))?;

    let mut results = Vec::with_capacity(chain_ids_per_group.len());

    for chain_ids in chain_ids_per_group {
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
// Combined assembly for multi-group Rosetta operations (all entity types)
// ============================================================================

/// Result of combining all entities from multiple groups for a Rosetta session.
/// Unlike `CombinedSession` (protein-only), this carries every entity type via ASSEM01.
#[derive(Debug, Clone)]
pub struct CombinedAssembly {
    /// ASSEM01 bytes — all entity types with molecule type metadata.
    pub bytes: Vec<u8>,
    /// Chain IDs assigned per input group (for splitting results back).
    pub chain_ids: Vec<Vec<u8>>,
    /// Residue ranges per input group: (start, end) — 1-indexed, inclusive.
    /// Counts only protein residues (for lock/selection mapping).
    pub residue_ranges: Vec<(usize, usize)>,
    /// Total protein residues across all groups.
    pub total_residues: usize,
}

/// Combine all entities from multiple groups for a Rosetta session.
/// Includes all molecule types (protein, ligand, ion, water, DNA, RNA).
/// Remaps chain IDs for uniqueness across groups.
pub fn prepare_combined_assembly(groups: &[&[MoleculeEntity]]) -> Option<CombinedAssembly> {
    let mut all_entities: Vec<MoleculeEntity> = Vec::new();
    let mut chain_ids_per_group: Vec<Vec<u8>> = Vec::new();
    let mut residue_ranges: Vec<(usize, usize)> = Vec::new();

    let mut next_chain_id = b'A';
    let mut global_residue_offset: usize = 0;

    for entity_slice in groups {
        if entity_slice.is_empty() {
            chain_ids_per_group.push(Vec::new());
            continue;
        }

        // Build chain ID remap for this group
        let mut chain_id_map: HashMap<u8, u8> = HashMap::new();
        for entity in *entity_slice {
            for &cid in &entity.coords.chain_ids {
                chain_id_map.entry(cid).or_insert_with(|| {
                    let id = next_chain_id;
                    next_chain_id = if next_chain_id == b'Z' { b'a' } else { next_chain_id + 1 };
                    id
                });
            }
        }

        let assigned_chain_ids: Vec<u8> = chain_id_map.values().copied().collect();
        chain_ids_per_group.push(assigned_chain_ids);

        // Count protein residues (CA atoms) for residue ranges
        let mut protein_residue_count: usize = 0;
        for entity in *entity_slice {
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
        for entity in *entity_slice {
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
        chain_ids: chain_ids_per_group,
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

    /// Regression test: update_protein_entities on a multi-chain set must NOT
    /// duplicate atoms.
    #[test]
    fn test_update_protein_entities_no_duplication() {
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

        let mut entities = split_into_entities(&coords);
        assert_eq!(entities.len(), 3); // chain A, chain B, water
        let total: usize = entities.iter().map(|e| e.coords.num_atoms).sum();
        assert_eq!(total, 7);

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

        update_protein_entities(&mut entities, updated_protein);

        // Must still have 7 total atoms (6 protein + 1 water), NOT 13
        let total: usize = entities.iter().map(|e| e.coords.num_atoms).sum();
        assert_eq!(total, 7);
        assert_eq!(entities.len(), 3); // chain A, chain B, water

        // Protein entities should have 3 atoms each, not 6
        let protein_entities: Vec<_> = entities.iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein)
            .collect();
        assert_eq!(protein_entities.len(), 2);
        assert_eq!(protein_entities[0].coords.num_atoms, 3);
        assert_eq!(protein_entities[1].coords.num_atoms, 3);

        // Verify protein coords were actually updated (x values should be 10+)
        let prot = protein_coords(&entities);
        assert_eq!(prot.num_atoms, 6);
        assert!(prot.atoms[0].x >= 10.0);

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

        update_protein_entities(&mut entities, updated_protein2);
        let total: usize = entities.iter().map(|e| e.coords.num_atoms).sum();
        assert_eq!(total, 7); // Still 7, not growing
    }

    /// Test update_entities_from_backend: Rosetta exports protein + known ion, skips water.
    #[test]
    fn test_update_from_backend_preserves_skipped_types() {
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

        let mut entities = split_into_entities(&coords);
        assert_eq!(entities.len(), 3);

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

        update_entities_from_backend(&mut entities, backend_export);

        let total: usize = entities.iter().map(|e| e.coords.num_atoms).sum();
        assert_eq!(total, 5);

        let water: Vec<_> = entities.iter()
            .filter(|e| e.molecule_type == MoleculeType::Water)
            .collect();
        assert_eq!(water.len(), 1);
        assert_eq!(water[0].coords.num_atoms, 1);
        assert!((water[0].coords.atoms[0].x - 4.0).abs() < 1e-6);

        let protein: Vec<_> = entities.iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein)
            .collect();
        assert_eq!(protein.len(), 1);
        assert!((protein[0].coords.atoms[0].x - 10.0).abs() < 1e-6);

        let ion: Vec<_> = entities.iter()
            .filter(|e| e.molecule_type == MoleculeType::Ion)
            .collect();
        assert_eq!(ion.len(), 1);
        assert!((ion[0].coords.atoms[0].x - 50.0).abs() < 1e-6);
    }

    /// Test update_entities_from_backend: protein-only export preserves all non-protein.
    #[test]
    fn test_update_from_backend_protein_only_export() {
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

        let mut entities = split_into_entities(&coords);

        let backend_export = Coords {
            num_atoms: 3,
            atoms: vec![make_atom(20.0), make_atom(21.0), make_atom(22.0)],
            chain_ids: vec![b'A'; 3],
            res_names: vec![res_name("ALA"); 3],
            res_nums: vec![1; 3],
            atom_names: vec![atom_name("N"), atom_name("CA"), atom_name("C")],
            elements: vec![Element::Unknown; 3],
        };

        update_entities_from_backend(&mut entities, backend_export);

        let total: usize = entities.iter().map(|e| e.coords.num_atoms).sum();
        assert_eq!(total, 5);
        assert_eq!(entities.len(), 3);

        let protein: Vec<_> = entities.iter()
            .filter(|e| e.molecule_type == MoleculeType::Protein)
            .collect();
        assert!((protein[0].coords.atoms[0].x - 20.0).abs() < 1e-6);

        let cofactor: Vec<_> = entities.iter()
            .filter(|e| e.molecule_type == MoleculeType::Cofactor)
            .collect();
        assert_eq!(cofactor.len(), 1);
        assert!((cofactor[0].coords.atoms[0].x - 3.0).abs() < 1e-6);

        let water: Vec<_> = entities.iter()
            .filter(|e| e.molecule_type == MoleculeType::Water)
            .collect();
        assert_eq!(water.len(), 1);
    }

    #[test]
    fn test_assembly_bytes_roundtrip_mixed() {
        use crate::types::coords::deserialize_assembly;

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

        let entities = split_into_entities(&coords);
        assert!(entities.len() >= 3);

        let bytes = assembly_bytes(&entities).unwrap();
        assert_eq!(&bytes[0..8], b"ASSEM01\0");

        let roundtripped = deserialize_assembly(&bytes).unwrap();
        assert_eq!(roundtripped.len(), entities.len());

        for (orig, rt) in entities.iter().zip(roundtripped.iter()) {
            assert_eq!(orig.molecule_type, rt.molecule_type);
            assert_eq!(orig.coords.num_atoms, rt.coords.num_atoms);
        }

        let orig_protein: Vec<_> = entities.iter()
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

        let entities = split_into_entities(&coords);
        let bytes = assembly_bytes(&entities).unwrap();
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

        let entities = split_into_entities(&coords);
        let bytes = assembly_bytes(&entities).unwrap();
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

        assert_eq!(&bytes[0..8], b"ASSEM01\0");
        assert_eq!(u32::from_be_bytes(bytes[8..12].try_into().unwrap()), 1);
        assert_eq!(bytes[12], 0); // Protein
        assert_eq!(u32::from_be_bytes(bytes[13..17].try_into().unwrap()), 1);
        assert_eq!(f32::from_be_bytes(bytes[17..21].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_be_bytes(bytes[21..25].try_into().unwrap()), 2.0);
        assert_eq!(f32::from_be_bytes(bytes[25..29].try_into().unwrap()), 3.0);
        assert_eq!(bytes[29], b'A');
        assert_eq!(bytes.len(), 43);
    }

    #[test]
    fn test_prepare_combined_assembly() {
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
        let entities1 = split_into_entities(&coords1);

        let coords2 = Coords {
            num_atoms: 3,
            atoms: (10..13).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A'; 3],
            res_names: vec![res_name("GLY"); 3],
            res_nums: vec![1; 3],
            atom_names: vec![atom_name("N"), atom_name("CA"), atom_name("C")],
            elements: vec![Element::N, Element::C, Element::C],
        };
        let entities2 = split_into_entities(&coords2);

        let groups: Vec<&[MoleculeEntity]> = vec![&entities1, &entities2];
        let combined = prepare_combined_assembly(&groups).unwrap();

        assert!(!combined.bytes.is_empty());
        assert_eq!(&combined.bytes[0..8], b"ASSEM01\0");

        assert_eq!(combined.chain_ids.len(), 2);
        assert!(!combined.chain_ids[0].is_empty());
        assert!(!combined.chain_ids[1].is_empty());

        for id in &combined.chain_ids[0] {
            assert!(!combined.chain_ids[1].contains(id));
        }

        assert_eq!(combined.total_residues, 2);
    }
}
