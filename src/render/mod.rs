//! Render-ready coordinate data extracted from Coords.
//!
//! RenderCoords is the bridge between the canonical Coords type and what
//! GPU renderers need. It separates backbone and sidechain data while
//! preserving atom identity for lookups.

pub mod gpu;

use crate::types::coords::Coords;
use glam::Vec3;
use std::collections::HashMap;

/// Full backbone atom positions for a single residue.
#[derive(Debug, Clone, Copy)]
pub struct RenderBackboneResidue {
    pub n_pos: Vec3,
    pub ca_pos: Vec3,
    pub c_pos: Vec3,
    pub o_pos: Vec3,
}

/// A sidechain atom with position and identity information.
#[derive(Debug, Clone)]
pub struct RenderSidechainAtom {
    pub position: Vec3,
    pub residue_idx: u32,
    pub atom_name: String,
    pub chain_id: u8,
    pub is_hydrophobic: bool,
}

/// Render-ready coordinate data extracted from Coords.
#[derive(Debug, Clone)]
pub struct RenderCoords {
    pub backbone_chains: Vec<Vec<Vec3>>,
    pub backbone_chain_ids: Vec<u8>,
    pub backbone_residue_chains: Vec<Vec<RenderBackboneResidue>>,
    pub sidechain_atoms: Vec<RenderSidechainAtom>,
    pub sidechain_bonds: Vec<(u32, u32)>,
    pub backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    pub all_positions: Vec<Vec3>,
    atom_lookup: HashMap<(u32, String), u32>,
}

impl RenderCoords {
    pub fn from_coords_with_topology<F, G>(
        coords: &Coords,
        is_hydrophobic_fn: F,
        get_bonds_fn: G,
    ) -> Self
    where
        F: Fn(&str) -> bool,
        G: Fn(&str) -> Option<Vec<(&'static str, &'static str)>>,
    {
        Self::from_coords_internal(coords, None, Some(&is_hydrophobic_fn), Some(&get_bonds_fn))
    }

    pub fn from_coords(coords: &Coords) -> Self {
        Self::from_coords_internal::<fn(&str) -> bool, fn(&str) -> Option<Vec<(&'static str, &'static str)>>>(
            coords, None, None, None
        )
    }

    pub fn from_coords_with_bonds(coords: &Coords, bonds: &[(u32, u32)]) -> Self {
        Self::from_coords_internal::<fn(&str) -> bool, fn(&str) -> Option<Vec<(&'static str, &'static str)>>>(
            coords, Some(bonds), None, None
        )
    }

    fn from_coords_internal<F, G>(
        coords: &Coords,
        explicit_bonds: Option<&[(u32, u32)]>,
        is_hydrophobic_fn: Option<&F>,
        get_bonds_fn: Option<&G>,
    ) -> Self
    where
        F: Fn(&str) -> bool,
        G: Fn(&str) -> Option<Vec<(&'static str, &'static str)>>,
    {
        let mut backbone_chains: Vec<Vec<Vec3>> = Vec::new();
        let mut backbone_chain_ids: Vec<u8> = Vec::new();
        let mut backbone_residue_chains: Vec<Vec<RenderBackboneResidue>> = Vec::new();
        let mut sidechain_atoms: Vec<RenderSidechainAtom> = Vec::new();
        let mut backbone_sidechain_bonds: Vec<(Vec3, u32)> = Vec::new();
        let mut all_positions: Vec<Vec3> = Vec::new();
        let mut atom_lookup: HashMap<(u32, String), u32> = HashMap::new();

        let mut atom_index_map: HashMap<(u8, i32, String), u32> = HashMap::new();

        let mut current_chain: Vec<Vec3> = Vec::new();
        let mut current_residues: Vec<RenderBackboneResidue> = Vec::new();
        let mut current_chain_id: Option<u8> = None;
        let mut last_chain_id: Option<u8> = None;
        let mut last_res_num: Option<i32> = None;


        let mut current_n: Option<Vec3> = None;
        let mut current_ca: Option<Vec3> = None;
        let mut current_c: Option<Vec3> = None;
        let mut current_o: Option<Vec3> = None;
        let mut current_res_key: Option<(u8, i32)> = None;

        let mut residue_idx_map: HashMap<(u8, i32), u32> = HashMap::new();
        let mut next_residue_idx: u32 = 0;

        let flush_residue = |current_n: &mut Option<Vec3>,
                             current_ca: &mut Option<Vec3>,
                             current_c: &mut Option<Vec3>,
                             current_o: &mut Option<Vec3>,
                             current_chain: &mut Vec<Vec3>,
                             current_residues: &mut Vec<RenderBackboneResidue>| {
            if let (Some(n), Some(ca), Some(c)) = (*current_n, *current_ca, *current_c) {
                current_chain.push(n);
                current_chain.push(ca);
                current_chain.push(c);

                if let Some(o) = *current_o {
                    current_residues.push(RenderBackboneResidue {
                        n_pos: n,
                        ca_pos: ca,
                        c_pos: c,
                        o_pos: o,
                    });
                }
            }
            *current_n = None;
            *current_ca = None;
            *current_c = None;
            *current_o = None;
        };

        for i in 0..coords.num_atoms {
            let atom_name = std::str::from_utf8(&coords.atom_names[i])
                .unwrap_or("")
                .trim()
                .to_string();
            let chain_id = coords.chain_ids[i];
            let res_num = coords.res_nums[i];
            let res_name = std::str::from_utf8(&coords.res_names[i])
                .unwrap_or("UNK")
                .trim();
            let pos = Vec3::new(coords.atoms[i].x, coords.atoms[i].y, coords.atoms[i].z);

            all_positions.push(pos);

            let res_key = (chain_id, res_num);

            let is_chain_break = last_chain_id.map_or(false, |c| c != chain_id);
            let is_sequence_gap = last_res_num.map_or(false, |r| (res_num - r).abs() > 1);
            let is_new_residue = current_res_key.map_or(true, |k| k != res_key);

            if is_new_residue && current_res_key.is_some() {
                flush_residue(
                    &mut current_n, &mut current_ca, &mut current_c, &mut current_o,
                    &mut current_chain, &mut current_residues,
                );
            }

            if (is_chain_break || is_sequence_gap) && !current_chain.is_empty() {
                backbone_chains.push(std::mem::take(&mut current_chain));
                backbone_residue_chains.push(std::mem::take(&mut current_residues));
                if let Some(cid) = current_chain_id {
                    backbone_chain_ids.push(cid);
                }
                current_chain_id = None;
            }

            current_res_key = Some(res_key);

            match atom_name.as_str() {
                "N" => {
                    current_n = Some(pos);

                }
                "CA" => {
                    current_ca = Some(pos);

                    if !residue_idx_map.contains_key(&res_key) {
                        residue_idx_map.insert(res_key, next_residue_idx);
                        next_residue_idx += 1;
                    }
                    if current_chain_id.is_none() {
                        current_chain_id = Some(chain_id);
                    }
                    last_res_num = Some(res_num);
                }
                "C" => {
                    current_c = Some(pos);

                }
                "O" => current_o = Some(pos),
                _ => {
                    let is_hydrogen = atom_name.starts_with('H')
                        || atom_name.starts_with("1H")
                        || atom_name.starts_with("2H")
                        || atom_name.starts_with("3H")
                        || (atom_name.len() >= 2
                            && atom_name.chars().next().unwrap().is_ascii_digit()
                            && atom_name.chars().nth(1) == Some('H'));

                    if !is_hydrogen {
                        let sidechain_idx = sidechain_atoms.len() as u32;
                        atom_index_map.insert((chain_id, res_num, atom_name.clone()), sidechain_idx);

                        let residue_idx = residue_idx_map.get(&res_key).copied().unwrap_or(0);

                        atom_lookup.insert((residue_idx, atom_name.clone()), sidechain_idx);

                        let hydrophobic = is_hydrophobic_fn.map_or(false, |f| f(res_name));
                        sidechain_atoms.push(RenderSidechainAtom {
                            position: pos,
                            residue_idx,
                            atom_name,
                            chain_id,
                            is_hydrophobic: hydrophobic,
                        });
                    }
                }
            }

            last_chain_id = Some(chain_id);
        }

        flush_residue(
            &mut current_n, &mut current_ca, &mut current_c, &mut current_o,
            &mut current_chain, &mut current_residues,
        );

        if !current_chain.is_empty() {
            backbone_chains.push(current_chain);
            backbone_residue_chains.push(current_residues);
            if let Some(cid) = current_chain_id {
                backbone_chain_ids.push(cid);
            }
        }

        // Second pass: generate CA-CB connections
        for i in 0..coords.num_atoms {
            let atom_name = std::str::from_utf8(&coords.atom_names[i])
                .unwrap_or("")
                .trim();
            let chain_id = coords.chain_ids[i];
            let res_num = coords.res_nums[i];

            if atom_name == "CA" {
                let ca_pos = Vec3::new(coords.atoms[i].x, coords.atoms[i].y, coords.atoms[i].z);
                let cb_key = (chain_id, res_num, "CB".to_string());
                if let Some(&cb_idx) = atom_index_map.get(&cb_key) {
                    backbone_sidechain_bonds.push((ca_pos, cb_idx));
                }
            }
        }

        let sidechain_bonds = if let Some(b) = explicit_bonds {
            b.to_vec()
        } else if let Some(get_bonds) = get_bonds_fn {
            Self::generate_sidechain_bonds(coords, &atom_index_map, get_bonds)
        } else {
            Vec::new()
        };

        Self {
            backbone_chains,
            backbone_chain_ids,
            backbone_residue_chains,
            sidechain_atoms,
            sidechain_bonds,
            backbone_sidechain_bonds,
            all_positions,
            atom_lookup,
        }
    }

    fn generate_sidechain_bonds<G>(
        coords: &Coords,
        atom_index_map: &HashMap<(u8, i32, String), u32>,
        get_bonds_fn: &G,
    ) -> Vec<(u32, u32)>
    where
        G: Fn(&str) -> Option<Vec<(&'static str, &'static str)>>,
    {
        let mut bonds: Vec<(u32, u32)> = Vec::new();
        let mut seen_residues: std::collections::HashSet<(u8, i32)> = std::collections::HashSet::new();

        for i in 0..coords.num_atoms {
            let atom_name = std::str::from_utf8(&coords.atom_names[i])
                .unwrap_or("")
                .trim();
            let chain_id = coords.chain_ids[i];
            let res_num = coords.res_nums[i];
            let res_name = std::str::from_utf8(&coords.res_names[i])
                .unwrap_or("UNK")
                .trim();

            if atom_name == "CA" && !seen_residues.contains(&(chain_id, res_num)) {
                seen_residues.insert((chain_id, res_num));

                if let Some(residue_bonds) = get_bonds_fn(res_name) {
                    for (a1, a2) in residue_bonds {
                        let key1 = (chain_id, res_num, a1.to_string());
                        let key2 = (chain_id, res_num, a2.to_string());

                        if let (Some(&idx1), Some(&idx2)) =
                            (atom_index_map.get(&key1), atom_index_map.get(&key2))
                        {
                            bonds.push((idx1, idx2));
                        }
                    }
                }
            }
        }

        bonds
    }

    pub fn sidechain_positions(&self) -> Vec<Vec3> {
        self.sidechain_atoms.iter().map(|a| a.position).collect()
    }

    pub fn sidechain_hydrophobicity(&self) -> Vec<bool> {
        self.sidechain_atoms.iter().map(|a| a.is_hydrophobic).collect()
    }

    pub fn sidechain_residue_indices(&self) -> Vec<u32> {
        self.sidechain_atoms.iter().map(|a| a.residue_idx).collect()
    }

    pub fn sidechain_atom_names(&self) -> Vec<String> {
        self.sidechain_atoms.iter().map(|a| a.atom_name.clone()).collect()
    }

    pub fn get_atom_position(&self, residue_idx: u32, atom_name: &str) -> Option<Vec3> {
        if atom_name == "N" || atom_name == "CA" || atom_name == "C" {
            return self.get_backbone_atom(residue_idx as usize, atom_name);
        }

        self.atom_lookup
            .get(&(residue_idx, atom_name.to_string()))
            .and_then(|&idx| self.sidechain_atoms.get(idx as usize))
            .map(|a| a.position)
    }

    fn get_backbone_atom(&self, residue_idx: usize, atom_name: &str) -> Option<Vec3> {
        let offset = match atom_name {
            "N" => 0,
            "CA" => 1,
            "C" => 2,
            _ => return None,
        };

        let mut current_idx = 0;
        for chain in &self.backbone_chains {
            let residues_in_chain = chain.len() / 3;
            if residue_idx < current_idx + residues_in_chain {
                let local_idx = residue_idx - current_idx;
                let atom_idx = local_idx * 3 + offset;
                return chain.get(atom_idx).copied();
            }
            current_idx += residues_in_chain;
        }
        None
    }

    pub fn ca_positions(&self) -> Vec<Vec3> {
        let mut cas = Vec::new();
        for chain in &self.backbone_chains {
            for (i, pos) in chain.iter().enumerate() {
                if i % 3 == 1 {
                    cas.push(*pos);
                }
            }
        }
        cas
    }

    pub fn find_closest_atom(
        &self,
        residue_idx: u32,
        reference_point: Vec3,
    ) -> Option<(Vec3, String)> {
        let mut closest: Option<(Vec3, String, f32)> = None;

        for name in ["N", "CA", "C"] {
            if let Some(pos) = self.get_backbone_atom(residue_idx as usize, name) {
                let dist = pos.distance_squared(reference_point);
                if closest.is_none() || dist < closest.as_ref().unwrap().2 {
                    closest = Some((pos, name.to_string(), dist));
                }
            }
        }

        for atom in &self.sidechain_atoms {
            if atom.residue_idx == residue_idx {
                let dist = atom.position.distance_squared(reference_point);
                if closest.is_none() || dist < closest.as_ref().unwrap().2 {
                    closest = Some((atom.position, atom.atom_name.clone(), dist));
                }
            }
        }

        closest.map(|(pos, name, _)| (pos, name))
    }

    pub fn residue_count(&self) -> usize {
        self.backbone_chains.iter().map(|c| c.len() / 3).sum()
    }
}

/// Extract amino acid sequences from Coords.
pub fn extract_sequences(coords: &Coords) -> (String, Vec<(u8, String)>) {
    let mut full_sequence = String::new();
    let mut chain_sequences: Vec<(u8, String)> = Vec::new();
    let mut current_chain_id: Option<u8> = None;
    let mut current_chain_seq = String::new();
    let mut last_res_key: Option<(u8, i32)> = None;

    for i in 0..coords.num_atoms {
        let atom_name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();
        let chain_id = coords.chain_ids[i];
        let res_num = coords.res_nums[i];
        let res_name = std::str::from_utf8(&coords.res_names[i])
            .unwrap_or("UNK")
            .trim();

        if atom_name == "CA" {
            let res_key = (chain_id, res_num);
            if last_res_key != Some(res_key) {
                if current_chain_id.is_some() && current_chain_id != Some(chain_id) {
                    if !current_chain_seq.is_empty() {
                        chain_sequences.push((current_chain_id.unwrap(), std::mem::take(&mut current_chain_seq)));
                    }
                }

                let aa = three_to_one(res_name);
                full_sequence.push(aa);
                current_chain_seq.push(aa);
                current_chain_id = Some(chain_id);
                last_res_key = Some(res_key);
            }
        }
    }

    if let Some(cid) = current_chain_id {
        if !current_chain_seq.is_empty() {
            chain_sequences.push((cid, current_chain_seq));
        }
    }

    (full_sequence, chain_sequences)
}

fn three_to_one(three: &str) -> char {
    match three {
        "ALA" => 'A', "CYS" => 'C', "ASP" => 'D', "GLU" => 'E',
        "PHE" => 'F', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
        "LYS" => 'K', "LEU" => 'L', "MET" => 'M', "ASN" => 'N',
        "PRO" => 'P', "GLN" => 'Q', "ARG" => 'R', "SER" => 'S',
        "THR" => 'T', "VAL" => 'V', "TRP" => 'W', "TYR" => 'Y',
        "MSE" => 'M',
        "HSD" | "HSE" | "HSP" => 'H',
        "CYX" => 'C',
        _ => 'X',
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::coords::CoordsAtom;

    fn make_test_coords() -> Coords {
        Coords {
            num_atoms: 5,
            atoms: vec![
                CoordsAtom { x: 0.0, y: 0.0, z: 0.0, occupancy: 1.0, b_factor: 0.0 },
                CoordsAtom { x: 1.5, y: 0.0, z: 0.0, occupancy: 1.0, b_factor: 0.0 },
                CoordsAtom { x: 2.5, y: 1.0, z: 0.0, occupancy: 1.0, b_factor: 0.0 },
                CoordsAtom { x: 2.5, y: 2.0, z: 0.0, occupancy: 1.0, b_factor: 0.0 },
                CoordsAtom { x: 1.5, y: -1.5, z: 0.0, occupancy: 1.0, b_factor: 0.0 },
            ],
            chain_ids: vec![b'A', b'A', b'A', b'A', b'A'],
            res_names: vec![*b"ALA", *b"ALA", *b"ALA", *b"ALA", *b"ALA"],
            res_nums: vec![1, 1, 1, 1, 1],
            atom_names: vec![*b"N   ", *b"CA  ", *b"C   ", *b"O   ", *b"CB  "],
            elements: vec![crate::types::coords::Element::Unknown; 5],
        }
    }

    #[test]
    fn test_extract_backbone() {
        let coords = make_test_coords();
        let render = RenderCoords::from_coords(&coords);
        assert_eq!(render.backbone_chains.len(), 1);
        assert_eq!(render.backbone_chains[0].len(), 3);
        assert_eq!(render.backbone_chains[0][1], Vec3::new(1.5, 0.0, 0.0));
    }

    #[test]
    fn test_extract_sidechains() {
        let coords = make_test_coords();
        let render = RenderCoords::from_coords(&coords);
        assert_eq!(render.sidechain_atoms.len(), 1);
        assert_eq!(render.sidechain_atoms[0].atom_name, "CB");
        assert_eq!(render.sidechain_atoms[0].residue_idx, 0);
    }

    #[test]
    fn test_atom_lookup() {
        let coords = make_test_coords();
        let render = RenderCoords::from_coords(&coords);
        assert_eq!(render.get_atom_position(0, "CA"), Some(Vec3::new(1.5, 0.0, 0.0)));
        assert_eq!(render.get_atom_position(0, "N"), Some(Vec3::new(0.0, 0.0, 0.0)));
        assert_eq!(render.get_atom_position(0, "CB"), Some(Vec3::new(1.5, -1.5, 0.0)));
        assert_eq!(render.get_atom_position(0, "CG"), None);
        assert_eq!(render.get_atom_position(1, "CA"), None);
    }

    #[test]
    fn test_find_closest_atom() {
        let coords = make_test_coords();
        let render = RenderCoords::from_coords(&coords);

        let result = render.find_closest_atom(0, Vec3::new(1.5, -1.0, 0.0));
        assert!(result.is_some());
        let (pos, name) = result.unwrap();
        assert_eq!(name, "CB");
        assert_eq!(pos, Vec3::new(1.5, -1.5, 0.0));

        let result = render.find_closest_atom(0, Vec3::new(0.1, 0.1, 0.0));
        assert!(result.is_some());
        let (_, name) = result.unwrap();
        assert_eq!(name, "N");
    }
}
