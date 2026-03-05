//! Molecule-type classification and per-entity coordinate splitting.
//!
//! Provides:
//! - `MoleculeType` — classification of residues into protein, DNA, RNA, ligand, ion, water
//! - `AtomSet` — core atom data (positions + chemistry, no PDB artifacts)
//! - `PolymerData` / `PolymerChain` / `Residue` — structured polymer hierarchy
//! - `EntityKind` — discriminated union of polymer, small molecule, and bulk entity data
//! - `MoleculeEntity` — a single entity with its own `EntityKind`
//! - `classify_residue()` — classify a residue name into a `MoleculeType`
//! - `split_into_entities()` — split flat `Coords` into per-entity groups
//! - `merge_entities()` — recombine entities back into flat `Coords`

use super::coords::{Coords, CoordsAtom, Element};
use crate::ops::transform::PROTEIN_RESIDUES;
use crate::render::backbone::{BackboneChain, ProteinBackbone};
use crate::render::sidechain::{SidechainAtomData, SidechainAtoms};
use glam::Vec3;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

/// Classification of molecule types found in structural biology files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoleculeType {
    Protein,
    DNA,
    RNA,
    Ligand,
    Ion,
    Water,
    Lipid,
    Cofactor,
    Solvent,
}

impl MoleculeType {
    /// Convert to a wire byte for ASSEM01 binary format.
    pub fn to_wire_byte(self) -> u8 {
        match self {
            MoleculeType::Protein => 0,
            MoleculeType::DNA => 1,
            MoleculeType::RNA => 2,
            MoleculeType::Ligand => 3,
            MoleculeType::Ion => 4,
            MoleculeType::Water => 5,
            MoleculeType::Lipid => 6,
            MoleculeType::Cofactor => 7,
            MoleculeType::Solvent => 8,
        }
    }

    /// Parse from a wire byte in ASSEM01 binary format.
    pub fn from_wire_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(MoleculeType::Protein),
            1 => Some(MoleculeType::DNA),
            2 => Some(MoleculeType::RNA),
            3 => Some(MoleculeType::Ligand),
            4 => Some(MoleculeType::Ion),
            5 => Some(MoleculeType::Water),
            6 => Some(MoleculeType::Lipid),
            7 => Some(MoleculeType::Cofactor),
            8 => Some(MoleculeType::Solvent),
            _ => None,
        }
    }
}

/// Axis-aligned bounding box (AABB).
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

impl Aabb {
    /// Geometric center of the box.
    pub fn center(&self) -> glam::Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Size along each axis (max - min).
    pub fn extents(&self) -> glam::Vec3 {
        self.max - self.min
    }

    /// Half-diagonal length (bounding sphere radius from center).
    pub fn radius(&self) -> f32 {
        self.extents().length() * 0.5
    }

    /// Merge two AABBs into one that contains both.
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Build AABB from positions. Returns `None` if the slice is empty.
    pub fn from_positions(positions: &[glam::Vec3]) -> Option<Aabb> {
        let first = *positions.first()?;
        let mut min = first;
        let mut max = first;
        for &p in &positions[1..] {
            min = min.min(p);
            max = max.max(p);
        }
        Some(Aabb { min, max })
    }

    /// Build unified AABB from multiple AABBs.
    pub fn from_aabbs(aabbs: &[Aabb]) -> Option<Aabb> {
        aabbs.iter().copied().reduce(|a, b| a.union(&b))
    }
}

// ---------------------------------------------------------------------------
// Core atom data
// ---------------------------------------------------------------------------

/// Core atom data — chemistry only, no format artifacts.
#[derive(Debug, Clone)]
pub struct AtomSet {
    pub atoms: Vec<CoordsAtom>,
    pub atom_names: Vec<[u8; 4]>,
    pub elements: Vec<Element>,
}

impl AtomSet {
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    pub fn positions(&self) -> Vec<Vec3> {
        self.atoms
            .iter()
            .map(|a| Vec3::new(a.x, a.y, a.z))
            .collect()
    }

    /// Convert to a minimal `Coords` (for bond inference compatibility).
    /// Chain IDs and residue metadata are set to dummy values.
    pub fn to_coords_minimal(&self) -> Coords {
        let n = self.atoms.len();
        Coords {
            num_atoms: n,
            atoms: self.atoms.clone(),
            chain_ids: vec![b' '; n],
            res_names: vec![[b' '; 3]; n],
            res_nums: vec![0; n],
            atom_names: self.atom_names.clone(),
            elements: self.elements.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Structured polymer data
// ---------------------------------------------------------------------------

/// A single residue within a polymer chain.
#[derive(Debug, Clone)]
pub struct Residue {
    /// 3-character residue name (e.g. b"ALA").
    pub name: [u8; 3],
    /// Residue sequence number.
    pub number: i32,
    /// Index range into the parent `PolymerData.atoms`.
    pub atom_range: Range<usize>,
}

/// A single polymer chain.
#[derive(Debug, Clone)]
pub struct PolymerChain {
    pub chain_id: u8,
    pub residues: Vec<Residue>,
}

/// Structured polymer data — chains containing residues containing atoms.
#[derive(Debug, Clone)]
pub struct PolymerData {
    pub atoms: AtomSet,
    pub chains: Vec<PolymerChain>,
}

// ---------------------------------------------------------------------------
// EntityKind
// ---------------------------------------------------------------------------

/// Discriminated union of entity data variants.
#[derive(Debug, Clone)]
pub enum EntityKind {
    /// Polymer chain (protein/DNA/RNA) — structured chain/residue hierarchy.
    Polymer(PolymerData),
    /// Single non-polymer molecule (ligand, cofactor, lipid, ion).
    SmallMolecule {
        atoms: AtomSet,
        residue_name: [u8; 3],
        display_name: String,
    },
    /// Bulk group (water, solvent) — many identical small molecules.
    Bulk {
        atoms: AtomSet,
        residue_name: [u8; 3],
        molecule_count: usize,
    },
}

// ---------------------------------------------------------------------------
// MoleculeEntity
// ---------------------------------------------------------------------------

/// A single entity: one logical molecule (a protein chain, a ligand, waters, etc.)
#[derive(Debug, Clone)]
pub struct MoleculeEntity {
    pub entity_id: u32,
    pub molecule_type: MoleculeType,
    pub kind: EntityKind,
}

/// Pre-extracted ring geometry for a single nucleotide base.
#[derive(Debug, Clone)]
pub struct NucleotideRing {
    /// Hexagonal ring atom positions in order: N1, C2, N3, C4, C5, C6
    pub hex_ring: Vec<glam::Vec3>,
    /// Pentagonal ring for purines: C4, C5, N7, C8, N9 (empty for pyrimidines)
    pub pent_ring: Vec<glam::Vec3>,
    /// NDB color for this base
    pub color: [f32; 3],
    /// C1' sugar carbon position (for anchoring stem to backbone spline).
    pub c1_prime: Option<glam::Vec3>,
}

const HEX_RING_ATOMS: &[&str] = &["N1", "C2", "N3", "C4", "C5", "C6"];
const PENT_RING_ATOMS: &[&str] = &["C4", "C5", "N7", "C8", "N9"];

fn ndb_base_color(res_name: &str) -> Option<[f32; 3]> {
    match res_name {
        "DA" | "A" | "ADE" | "RAD" => Some([0.85, 0.20, 0.20]),
        "DG" | "G" | "GUA" | "RGU" => Some([0.20, 0.80, 0.20]),
        "DC" | "C" | "CYT" | "RCY" => Some([0.90, 0.90, 0.20]),
        "DT" | "THY" => Some([0.20, 0.20, 0.85]),
        "DU" | "U" | "URA" => Some([0.20, 0.85, 0.85]),
        _ => None,
    }
}

fn is_purine(res_name: &str) -> bool {
    matches!(
        res_name,
        "DA" | "DG" | "DI" | "A" | "G" | "ADE" | "GUA" | "I" | "RAD" | "RGU"
    )
}

impl MoleculeEntity {
    // -- Convenience accessors --

    /// Reference to the underlying `AtomSet`.
    pub fn atom_set(&self) -> &AtomSet {
        match &self.kind {
            EntityKind::Polymer(data) => &data.atoms,
            EntityKind::SmallMolecule { atoms, .. } => atoms,
            EntityKind::Bulk { atoms, .. } => atoms,
        }
    }

    /// All atom positions as Vec3.
    pub fn positions(&self) -> Vec<glam::Vec3> {
        self.atom_set().positions()
    }

    /// Number of atoms in this entity.
    pub fn atom_count(&self) -> usize {
        self.atom_set().len()
    }

    /// Slice of all atoms.
    pub fn atoms(&self) -> &[CoordsAtom] {
        &self.atom_set().atoms
    }

    /// Slice of all elements.
    pub fn elements(&self) -> &[Element] {
        &self.atom_set().elements
    }

    /// Slice of all atom names.
    pub fn atom_names(&self) -> &[[u8; 4]] {
        &self.atom_set().atom_names
    }

    /// If this entity is a polymer, return the structured data.
    pub fn as_polymer(&self) -> Option<&PolymerData> {
        match &self.kind {
            EntityKind::Polymer(data) => Some(data),
            _ => None,
        }
    }

    /// Convert to a flat `Coords` for serialization or interop.
    pub fn to_coords(&self) -> Coords {
        match &self.kind {
            EntityKind::Polymer(data) => {
                let n = data.atoms.len();
                let mut chain_ids = Vec::with_capacity(n);
                let mut res_names = Vec::with_capacity(n);
                let mut res_nums = Vec::with_capacity(n);
                for chain in &data.chains {
                    for residue in &chain.residues {
                        for _ in residue.atom_range.clone() {
                            chain_ids.push(chain.chain_id);
                            res_names.push(residue.name);
                            res_nums.push(residue.number);
                        }
                    }
                }
                Coords {
                    num_atoms: n,
                    atoms: data.atoms.atoms.clone(),
                    chain_ids,
                    res_names,
                    res_nums,
                    atom_names: data.atoms.atom_names.clone(),
                    elements: data.atoms.elements.clone(),
                }
            }
            EntityKind::SmallMolecule {
                atoms,
                residue_name,
                ..
            } => {
                let n = atoms.len();
                Coords {
                    num_atoms: n,
                    atoms: atoms.atoms.clone(),
                    chain_ids: vec![b' '; n],
                    res_names: vec![*residue_name; n],
                    res_nums: vec![1; n],
                    atom_names: atoms.atom_names.clone(),
                    elements: atoms.elements.clone(),
                }
            }
            EntityKind::Bulk {
                atoms,
                residue_name,
                ..
            } => {
                let n = atoms.len();
                Coords {
                    num_atoms: n,
                    atoms: atoms.atoms.clone(),
                    chain_ids: vec![b' '; n],
                    res_names: vec![*residue_name; n],
                    res_nums: (1..=n as i32).collect(),
                    atom_names: atoms.atom_names.clone(),
                    elements: atoms.elements.clone(),
                }
            }
        }
    }

    /// Compute the axis-aligned bounding box for this entity's atoms.
    pub fn aabb(&self) -> Option<Aabb> {
        Aabb::from_positions(&self.positions())
    }

    /// Human-readable label (e.g. "Protein Chain A", "Ligand (ATP)", "Zn²⁺ Ion").
    pub fn label(&self) -> String {
        match self.molecule_type {
            MoleculeType::Protein => {
                if let EntityKind::Polymer(data) = &self.kind {
                    let chains: Vec<u8> = data.chains.iter().map(|c| c.chain_id).collect();
                    if chains.len() == 1 {
                        format!("Protein Chain {}", chains[0] as char)
                    } else {
                        let chain_str: String = chains.iter().map(|&c| c as char).collect();
                        format!("Protein Chains {}", chain_str)
                    }
                } else {
                    "Protein".to_string()
                }
            }
            MoleculeType::DNA => {
                if let EntityKind::Polymer(data) = &self.kind {
                    let chains: Vec<u8> = data.chains.iter().map(|c| c.chain_id).collect();
                    if chains.len() == 1 {
                        format!("DNA Chain {}", chains[0] as char)
                    } else {
                        "DNA".to_string()
                    }
                } else {
                    "DNA".to_string()
                }
            }
            MoleculeType::RNA => {
                if let EntityKind::Polymer(data) = &self.kind {
                    let chains: Vec<u8> = data.chains.iter().map(|c| c.chain_id).collect();
                    if chains.len() == 1 {
                        format!("RNA Chain {}", chains[0] as char)
                    } else {
                        "RNA".to_string()
                    }
                } else {
                    "RNA".to_string()
                }
            }
            MoleculeType::Ligand => {
                if let EntityKind::SmallMolecule { display_name, .. } = &self.kind {
                    format!("Ligand ({})", display_name)
                } else {
                    "Ligand".to_string()
                }
            }
            MoleculeType::Ion => {
                if let EntityKind::SmallMolecule { display_name, .. } = &self.kind {
                    format!("{} Ion", display_name)
                } else {
                    "Ion".to_string()
                }
            }
            MoleculeType::Water => {
                if let EntityKind::Bulk {
                    molecule_count, ..
                } = &self.kind
                {
                    format!("Water ({} molecules)", molecule_count)
                } else {
                    "Water".to_string()
                }
            }
            MoleculeType::Lipid => {
                if let EntityKind::SmallMolecule { display_name, .. } = &self.kind {
                    format!("Lipid ({})", display_name)
                } else {
                    format!("Lipid ({} molecules)", self.residue_count())
                }
            }
            MoleculeType::Cofactor => {
                if let EntityKind::SmallMolecule { display_name, .. } = &self.kind {
                    display_name.clone()
                } else {
                    "Cofactor".to_string()
                }
            }
            MoleculeType::Solvent => {
                if let EntityKind::Bulk {
                    molecule_count, ..
                } = &self.kind
                {
                    format!("Solvent ({} molecules)", molecule_count)
                } else {
                    "Solvent".to_string()
                }
            }
        }
    }

    /// Whether this entity type participates in tab-cycling focus.
    /// Protein: no (focused at group level). Water, Ion: no (ambient).
    /// Ligand, DNA, RNA: yes.
    pub fn is_focusable(&self) -> bool {
        !matches!(
            self.molecule_type,
            MoleculeType::Water | MoleculeType::Ion | MoleculeType::Solvent
        )
    }

    /// Extract phosphorus (P) atom positions grouped by chain ID.
    /// Chains are split at gaps where consecutive P-P distance exceeds ~8 Å.
    /// Only meaningful for DNA/RNA entities; returns empty for other molecule types.
    pub fn extract_p_atom_chains(&self) -> Vec<Vec<glam::Vec3>> {
        const MAX_PP_DIST_SQ: f32 = 8.0 * 8.0;

        if !matches!(self.molecule_type, MoleculeType::DNA | MoleculeType::RNA) {
            return Vec::new();
        }

        let data = match &self.kind {
            EntityKind::Polymer(data) => data,
            _ => return Vec::new(),
        };

        let mut raw_chains: BTreeMap<u8, Vec<glam::Vec3>> = BTreeMap::new();

        for chain in &data.chains {
            for residue in &chain.residues {
                for idx in residue.atom_range.clone() {
                    let name = std::str::from_utf8(&data.atoms.atom_names[idx])
                        .unwrap_or("")
                        .trim();
                    if name == "P" {
                        let a = &data.atoms.atoms[idx];
                        raw_chains
                            .entry(chain.chain_id)
                            .or_default()
                            .push(glam::Vec3::new(a.x, a.y, a.z));
                    }
                }
            }
        }

        // Split chains at large gaps (missing residues / chain breaks)
        let mut result = Vec::new();
        for chain in raw_chains.into_values() {
            let mut segment = Vec::new();
            for pos in chain {
                if let Some(&prev) = segment.last() {
                    if pos.distance_squared(prev) > MAX_PP_DIST_SQ {
                        if segment.len() >= 2 {
                            result.push(std::mem::take(&mut segment));
                        } else {
                            segment.clear();
                        }
                    }
                }
                segment.push(pos);
            }
            if segment.len() >= 2 {
                result.push(segment);
            }
        }

        result
    }

    /// Extract base ring geometry for each nucleotide residue.
    /// Only meaningful for DNA/RNA entities; returns empty for other molecule types.
    pub fn extract_base_rings(&self) -> Vec<NucleotideRing> {
        if !matches!(self.molecule_type, MoleculeType::DNA | MoleculeType::RNA) {
            return Vec::new();
        }

        let data = match &self.kind {
            EntityKind::Polymer(data) => data,
            _ => return Vec::new(),
        };

        let mut rings = Vec::new();
        let mut skipped_partial = 0u32;

        for chain in &data.chains {
            for residue in &chain.residues {
                let res_name = std::str::from_utf8(&residue.name)
                    .unwrap_or("")
                    .trim();

                let color = match ndb_base_color(res_name) {
                    Some(c) => c,
                    None => continue,
                };

                // Build atom_name -> position map for this residue
                let mut atom_map: HashMap<String, glam::Vec3> = HashMap::new();
                for idx in residue.atom_range.clone() {
                    let name = std::str::from_utf8(&data.atoms.atom_names[idx])
                        .unwrap_or("")
                        .trim()
                        .trim_matches('\0')
                        .to_owned();
                    let a = &data.atoms.atoms[idx];
                    atom_map.insert(name, glam::Vec3::new(a.x, a.y, a.z));
                }

                // Collect hex ring positions
                let hex_ring: Vec<glam::Vec3> = HEX_RING_ATOMS
                    .iter()
                    .filter_map(|name| atom_map.get(*name).copied())
                    .collect();
                if hex_ring.len() != 6 {
                    skipped_partial += 1;
                    continue;
                }

                // Collect pent ring for purines
                let pent_ring = if is_purine(res_name) {
                    let pent: Vec<glam::Vec3> = PENT_RING_ATOMS
                        .iter()
                        .filter_map(|name| atom_map.get(*name).copied())
                        .collect();
                    if pent.len() == 5 {
                        pent
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                let c1_prime = atom_map.get("C1'").or_else(|| atom_map.get("C1*")).copied();

                rings.push(NucleotideRing {
                    hex_ring,
                    pent_ring,
                    color,
                    c1_prime,
                });
            }
        }

        #[cfg(debug_assertions)]
        if skipped_partial > 0 {
            eprintln!(
                "[base_rings] {} rings extracted, {} residues skipped (missing ring atoms)",
                rings.len(),
                skipped_partial
            );
        }

        rings
    }

    /// Number of residues (for polymer/nucleic) or molecules (for small mol/ion/water).
    pub fn residue_count(&self) -> usize {
        match &self.kind {
            EntityKind::Polymer(data) => {
                data.chains.iter().map(|c| c.residues.len()).sum()
            }
            EntityKind::SmallMolecule { .. } => 1,
            EntityKind::Bulk {
                molecule_count, ..
            } => *molecule_count,
        }
    }

    /// Extract protein backbone chains (N-CA-C interleaved, split at chain breaks).
    ///
    /// Returns a [`ProteinBackbone`] containing one [`BackboneChain`] per
    /// contiguous polymer segment.
    pub fn extract_backbone(&self) -> ProteinBackbone {
        if self.molecule_type != MoleculeType::Protein {
            return ProteinBackbone {
                chains: Vec::new(),
                chain_ids: Vec::new(),
            };
        }

        let data = match &self.kind {
            EntityKind::Polymer(data) => data,
            _ => {
                return ProteinBackbone {
                    chains: Vec::new(),
                    chain_ids: Vec::new(),
                }
            }
        };

        let mut chains: Vec<Vec<Vec3>> = Vec::new();
        let mut chain_ids: Vec<u8> = Vec::new();

        for polymer_chain in &data.chains {
            let mut current_chain: Vec<Vec3> = Vec::new();
            let mut last_res_num: Option<i32> = None;

            for residue in &polymer_chain.residues {
                // Check for sequence gap → chain break
                let is_sequence_gap =
                    last_res_num.is_some_and(|r| (residue.number - r).abs() > 1);

                if is_sequence_gap && !current_chain.is_empty() {
                    chains.push(std::mem::take(&mut current_chain));
                    chain_ids.push(polymer_chain.chain_id);
                }

                // Collect backbone atoms (N, CA, C) for this residue
                for idx in residue.atom_range.clone() {
                    let atom_name = std::str::from_utf8(&data.atoms.atom_names[idx])
                        .unwrap_or("")
                        .trim();
                    if atom_name == "N" || atom_name == "CA" || atom_name == "C" {
                        let a = &data.atoms.atoms[idx];
                        current_chain.push(Vec3::new(a.x, a.y, a.z));
                    }
                }

                last_res_num = Some(residue.number);
            }

            if !current_chain.is_empty() {
                chains.push(current_chain);
                chain_ids.push(polymer_chain.chain_id);
            }
        }

        ProteinBackbone {
            chains: chains.into_iter().map(BackboneChain::new).collect(),
            chain_ids,
        }
    }

    /// Extract sidechain atom data with topology.
    pub fn extract_sidechains<F, G>(&self, is_hydrophobic: F, get_bonds: G) -> SidechainAtoms
    where
        F: Fn(&str) -> bool,
        G: Fn(&str) -> Option<Vec<(&'static str, &'static str)>>,
    {
        if self.molecule_type != MoleculeType::Protein {
            return SidechainAtoms::default();
        }

        let data = match &self.kind {
            EntityKind::Polymer(data) => data,
            _ => return SidechainAtoms::default(),
        };

        let mut atoms_out: Vec<SidechainAtomData> = Vec::new();
        let mut bonds_out: Vec<(u32, u32)> = Vec::new();
        let mut backbone_bonds: Vec<(Vec3, u32)> = Vec::new();

        // Map (chain_id, res_num, atom_name) -> sidechain index for bond generation
        let mut atom_index_map: HashMap<(u8, i32, String), u32> = HashMap::new();
        // Map (chain_id, res_num) -> sequential residue index
        let mut residue_idx_map: HashMap<(u8, i32), u32> = HashMap::new();
        let mut next_residue_idx: u32 = 0;

        // First pass: collect sidechain atoms and assign residue indices
        for chain in &data.chains {
            for residue in &chain.residues {
                let res_name = std::str::from_utf8(&residue.name)
                    .unwrap_or("UNK")
                    .trim();
                let res_key = (chain.chain_id, residue.number);

                for idx in residue.atom_range.clone() {
                    let atom_name = std::str::from_utf8(&data.atoms.atom_names[idx])
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let a = &data.atoms.atoms[idx];
                    let pos = Vec3::new(a.x, a.y, a.z);

                    match atom_name.as_str() {
                        "CA" => {
                            if let std::collections::hash_map::Entry::Vacant(e) =
                                residue_idx_map.entry(res_key)
                            {
                                e.insert(next_residue_idx);
                                next_residue_idx += 1;
                            }
                        }
                        "N" | "C" | "O" => {}
                        _ => {
                            // Skip hydrogens
                            let is_hydrogen = atom_name.starts_with('H')
                                || atom_name.starts_with("1H")
                                || atom_name.starts_with("2H")
                                || atom_name.starts_with("3H")
                                || (atom_name.len() >= 2
                                    && atom_name.chars().next().unwrap().is_ascii_digit()
                                    && atom_name.chars().nth(1) == Some('H'));

                            if !is_hydrogen {
                                let sidechain_idx = atoms_out.len() as u32;
                                atom_index_map.insert(
                                    (chain.chain_id, residue.number, atom_name.clone()),
                                    sidechain_idx,
                                );

                                let residue_idx =
                                    residue_idx_map.get(&res_key).copied().unwrap_or(0);
                                let hydrophobic = is_hydrophobic(res_name);

                                atoms_out.push(SidechainAtomData {
                                    position: pos,
                                    residue_idx,
                                    atom_name,
                                    is_hydrophobic: hydrophobic,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Second pass: generate CA→CB backbone-sidechain bonds
        for chain in &data.chains {
            for residue in &chain.residues {
                for idx in residue.atom_range.clone() {
                    let atom_name = std::str::from_utf8(&data.atoms.atom_names[idx])
                        .unwrap_or("")
                        .trim();
                    if atom_name == "CA" {
                        let a = &data.atoms.atoms[idx];
                        let ca_pos = Vec3::new(a.x, a.y, a.z);
                        let cb_key =
                            (chain.chain_id, residue.number, "CB".to_string());
                        if let Some(&cb_idx) = atom_index_map.get(&cb_key) {
                            backbone_bonds.push((ca_pos, cb_idx));
                        }
                    }
                }
            }
        }

        // Third pass: generate intra-residue sidechain bonds from topology
        for chain in &data.chains {
            for residue in &chain.residues {
                let res_name = std::str::from_utf8(&residue.name)
                    .unwrap_or("UNK")
                    .trim();
                if let Some(residue_bonds) = get_bonds(res_name) {
                    for (a1, a2) in residue_bonds {
                        let key1 =
                            (chain.chain_id, residue.number, a1.to_string());
                        let key2 =
                            (chain.chain_id, residue.number, a2.to_string());
                        if let (Some(&idx1), Some(&idx2)) =
                            (atom_index_map.get(&key1), atom_index_map.get(&key2))
                        {
                            bonds_out.push((idx1, idx2));
                        }
                    }
                }
            }
        }

        SidechainAtoms {
            atoms: atoms_out,
            bonds: bonds_out,
            backbone_bonds,
        }
    }
}

/// Standard DNA residue names (mmCIF convention + Rosetta internal name THY).
const DNA_RESIDUES: &[&str] = &["DA", "DC", "DG", "DT", "DU", "DI", "THY"];

/// Standard RNA residue names.
const RNA_RESIDUES: &[&str] = &[
    "A", "C", "G", "U", "ADE", "CYT", "GUA", "URA", "I", "RAD", "RCY", "RGU",
];

/// Water residue names.
const WATER_RESIDUES: &[&str] = &[
    "HOH", "WAT", "H2O", "DOD",
    // MD simulation water models (GROMACS, AMBER, CHARMM, etc.)
    "SOL", "TIP", "TP3", "TIP3", "T3P", "SPC", "TP4", "TIP4", "T4P", "TP5", "TIP5",
];

/// Known ion residue names. These are single-atom residues with well-known names.
const ION_RESIDUES: &[&str] = &[
    "ZN", "MG", "NA", "CL", "FE", "MN", "CO", "NI", "CU", "K", "CA", "BR", "I", "F", "LI", "CD",
    "SR", "BA", "CS", "RB", "PB", "HG", "PT", "AU", "AG",
];

/// Known lipid residue 3-char truncated names.
const LIPID_RESIDUES: &[&str] = &[
    // Phosphatidylcholines (DPPC, POPC, DOPC, DMPC, DSPC, DLPC)
    "DPP", "POP", "DOP", "DMP", "DSP", "DLP",
    // Phosphatidylethanolamines (DPPE, POPE, DOPE)
    "PPE", "DPE", // Phosphatidylglycerols (DPPG, POPG, DOPG)
    "PPG", "DPG", // Phosphatidylserines (DPPS, POPS, DOPS)
    "PPS", "DPS", // Cholesterol variants
    "CHO", "CHL", // Sphingomyelin, ceramide
    "SPH", "CER", // CHARMM-GUI lipid residue names (full 3-letter)
    "PAL", "OLE", "STE", "MYR", "LAU",
    // PDB crystallographic lipid codes (thylakoid/membrane lipids)
    "LHG", // dipalmitoyl phosphatidylglycerol
    "LMG", // monogalactosyl diglyceride (MGDG)
    "DGD", // digalactosyl diacyl glycerol (DGDG)
    "SQD", // sulfoquinovosyl diacylglycerol (SQDG)
    // PDB detergent codes (amphipathic, treated as lipid-like)
    "LMT", // dodecyl-beta-D-maltoside
    "HTG", // heptyl 1-thiohexopyranoside
];

/// Known cofactor residue names (exact match, checked before lipid truncation).
const COFACTOR_RESIDUES: &[&str] = &[
    // Porphyrins / chlorins
    "HEM", "HEC", "HEA", "HEB", "CLA", "CHL", "PHO", "BCR", "BCB", // Quinones
    "PL9", "PLQ", "UQ1", "UQ2", "MQ7", // Nucleotide cofactors
    "NAD", "NAP", "NAI", "NDP", "FAD", "FMN", "ATP", "ADP", "AMP", "ANP", "GTP", "GDP", "GMP",
    "GNP", // Other
    "SAM", "SAH", "COA", "ACO", "PLP", "PMP", "TPP", "TDP", "BTN", "BIO", "H4B", "BH4",
    // Fe-S clusters
    "SF4", "FES", "F3S",
];

/// Known solvent / crystallization artifact residue names (exact match).
const SOLVENT_RESIDUES: &[&str] = &[
    // Polyols / PEGs
    "GOL", "EDO", "PEG", "1PE", "P6G", "PG4", "PGE", // Salts / buffers
    "SO4", "SUL", "PO4", "ACT", "ACE", "CIT", "FMT", // Buffers
    "TRS", "MES", "EPE", "IMD", // Cryoprotectants
    "MPD", "DMS", "BME", "IPA", "EOH",
];

/// Human-readable display name for a cofactor residue code.
fn cofactor_display_name(res_name: &str) -> &str {
    match res_name {
        "CLA" => "Chlorophyll A",
        "CHL" => "Chlorophyll B",
        "BCR" => "Beta-Carotene",
        "BCB" => "Beta-Carotene B",
        "HEM" => "Heme",
        "HEC" => "Heme C",
        "HEA" => "Heme A",
        "HEB" => "Heme B",
        "PHO" => "Pheophytin",
        "PL9" => "Plastoquinone",
        "PLQ" => "Plastoquinone",
        "UQ1" | "UQ2" => "Ubiquinone",
        "MQ7" => "Menaquinone",
        "NAD" | "NAP" | "NAI" | "NDP" => "NAD",
        "FAD" => "FAD",
        "FMN" => "FMN",
        "ATP" | "ADP" | "AMP" | "ANP" => res_name,
        "GTP" | "GDP" | "GMP" | "GNP" => res_name,
        "SAM" | "SAH" => "SAM/SAH",
        "COA" | "ACO" => "Coenzyme A",
        "PLP" | "PMP" => "PLP",
        "TPP" | "TDP" => "Thiamine PP",
        "BTN" | "BIO" => "Biotin",
        "H4B" | "BH4" => "Tetrahydrobiopterin",
        "SF4" => "[4Fe-4S] Cluster",
        "FES" => "[2Fe-2S] Cluster",
        "F3S" => "[3Fe-4S] Cluster",
        _ => res_name,
    }
}

/// Display name for a small molecule, dispatching by molecule type.
fn small_molecule_display_name(mol_type: MoleculeType, res_name: &str) -> String {
    match mol_type {
        MoleculeType::Cofactor => cofactor_display_name(res_name).to_string(),
        _ => res_name.to_string(),
    }
}

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
    // Cofactor: exact match, checked before lipid truncation
    if COFACTOR_RESIDUES.contains(&name) {
        return MoleculeType::Cofactor;
    }
    // Solvent / crystallization artifacts: exact match
    if SOLVENT_RESIDUES.contains(&name) {
        return MoleculeType::Solvent;
    }
    // Check lipid residues: exact match on truncated 3-char names
    let truncated = if name.len() > 3 { &name[..3] } else { name };
    if LIPID_RESIDUES.contains(&truncated) {
        return MoleculeType::Lipid;
    }
    MoleculeType::Ligand
}

/// Check whether a set of atom indices (belonging to one residue) contains
/// protein backbone atoms N, CA, and C — the hallmark of an amino acid.
fn residue_has_backbone(indices: &[usize], coords: &Coords) -> bool {
    let mut has_n = false;
    let mut has_ca = false;
    let mut has_c = false;
    for &idx in indices {
        let name = &coords.atom_names[idx];
        match name {
            [b' ', b'N', b' ', b' '] | [b'N', b' ', b' ', b' '] => has_n = true,
            [b' ', b'C', b'A', b' '] | [b'C', b'A', b' ', b' '] => has_ca = true,
            [b' ', b'C', b' ', b' '] | [b'C', b' ', b' ', b' '] => has_c = true,
            _ => {}
        }
    }
    has_n && has_ca && has_c
}

// ---------------------------------------------------------------------------
// Entity splitting / merging
// ---------------------------------------------------------------------------

/// Key for grouping atoms into entities.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum EntityKey {
    /// A polymeric chain (protein, DNA, RNA) on a specific chain.
    Chain(u8, MoleculeTypeOrd),
    /// All water molecules consolidated into one entity.
    Water,
    /// All solvent molecules consolidated into one entity.
    Solvent,
    /// A single non-polymer molecule, keyed by (chain_id, res_num, type).
    SmallMolecule(u8, i32, MoleculeTypeOrd),
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
/// - Solvent: all solvent residues are consolidated into a single entity
/// - Ligand/Ion/Cofactor/Lipid: each unique (chain_id, res_num) is its own entity
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
            MoleculeType::Solvent => EntityKey::Solvent,
            MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA => {
                EntityKey::Chain(chain_id, MoleculeTypeOrd(mol_type))
            }
            MoleculeType::Ligand
            | MoleculeType::Ion
            | MoleculeType::Cofactor
            | MoleculeType::Lipid => {
                let res_num = coords.res_nums[i];
                EntityKey::SmallMolecule(chain_id, res_num, MoleculeTypeOrd(mol_type))
            }
        };

        groups.entry(key).or_default().push(i);
    }

    // Merge modified amino acids back into their protein chain.
    // A SmallMolecule group that has backbone atoms (N, CA, C) and whose chain
    // already has a protein entity is a modified residue, not a ligand.
    let protein_chains: Vec<u8> = groups
        .keys()
        .filter_map(|k| match k {
            EntityKey::Chain(cid, mt) if mt.0 == MoleculeType::Protein => Some(*cid),
            _ => None,
        })
        .collect();

    let merge_keys: Vec<EntityKey> = groups
        .iter()
        .filter_map(|(key, indices)| {
            if let EntityKey::SmallMolecule(chain_id, _, _) = key {
                if protein_chains.contains(chain_id)
                    && residue_has_backbone(indices, coords)
                {
                    return Some(key.clone());
                }
            }
            None
        })
        .collect();

    for key in merge_keys {
        if let EntityKey::SmallMolecule(chain_id, _, _) = &key {
            let chain_key =
                EntityKey::Chain(*chain_id, MoleculeTypeOrd(MoleculeType::Protein));
            if let Some(indices) = groups.remove(&key) {
                groups.entry(chain_key).or_default().extend(indices);
            }
        }
    }

    // Convert groups to entities
    groups
        .into_iter()
        .enumerate()
        .map(|(entity_id, (key, indices))| {
            let mol_type = match &key {
                EntityKey::Chain(_, mt) | EntityKey::SmallMolecule(_, _, mt) => mt.0,
                EntityKey::Water => MoleculeType::Water,
                EntityKey::Solvent => MoleculeType::Solvent,
            };

            let kind = match mol_type {
                MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA => {
                    build_polymer_kind(&indices, coords)
                }
                MoleculeType::Water | MoleculeType::Solvent => {
                    build_bulk_kind(&indices, coords)
                }
                _ => build_small_molecule_kind(mol_type, &indices, coords),
            };

            MoleculeEntity {
                entity_id: entity_id as u32,
                molecule_type: mol_type,
                kind,
            }
        })
        .collect()
}

/// Build `EntityKind::Polymer` from a set of atom indices belonging to one
/// polymer chain.
fn build_polymer_kind(indices: &[usize], coords: &Coords) -> EntityKind {
    // Group atoms by (chain_id, res_num) preserving insertion order within each
    let mut chain_residue_map: BTreeMap<u8, BTreeMap<i32, Vec<usize>>> = BTreeMap::new();
    for &idx in indices {
        chain_residue_map
            .entry(coords.chain_ids[idx])
            .or_default()
            .entry(coords.res_nums[idx])
            .or_default()
            .push(idx);
    }

    let mut atom_set_atoms = Vec::with_capacity(indices.len());
    let mut atom_set_names = Vec::with_capacity(indices.len());
    let mut atom_set_elements = Vec::with_capacity(indices.len());
    let mut chains = Vec::new();

    for (&chain_id, residues) in &chain_residue_map {
        let mut chain_residues = Vec::new();
        for (&res_num, atom_indices) in residues {
            let start = atom_set_atoms.len();
            let res_name = coords.res_names[atom_indices[0]];
            for &idx in atom_indices {
                atom_set_atoms.push(coords.atoms[idx].clone());
                atom_set_names.push(coords.atom_names[idx]);
                atom_set_elements.push(
                    coords
                        .elements
                        .get(idx)
                        .copied()
                        .unwrap_or(Element::Unknown),
                );
            }
            let end = atom_set_atoms.len();
            chain_residues.push(Residue {
                name: res_name,
                number: res_num,
                atom_range: start..end,
            });
        }
        chains.push(PolymerChain {
            chain_id,
            residues: chain_residues,
        });
    }

    EntityKind::Polymer(PolymerData {
        atoms: AtomSet {
            atoms: atom_set_atoms,
            atom_names: atom_set_names,
            elements: atom_set_elements,
        },
        chains,
    })
}

/// Build `EntityKind::SmallMolecule` from a set of atom indices.
fn build_small_molecule_kind(
    mol_type: MoleculeType,
    indices: &[usize],
    coords: &Coords,
) -> EntityKind {
    let mut atoms = Vec::with_capacity(indices.len());
    let mut atom_names = Vec::with_capacity(indices.len());
    let mut elements = Vec::with_capacity(indices.len());

    for &idx in indices {
        atoms.push(coords.atoms[idx].clone());
        atom_names.push(coords.atom_names[idx]);
        elements.push(
            coords
                .elements
                .get(idx)
                .copied()
                .unwrap_or(Element::Unknown),
        );
    }

    let residue_name = coords.res_names[indices[0]];
    let rn_str = std::str::from_utf8(&residue_name).unwrap_or("???").trim();
    let display_name = small_molecule_display_name(mol_type, rn_str);

    EntityKind::SmallMolecule {
        atoms: AtomSet {
            atoms,
            atom_names,
            elements,
        },
        residue_name,
        display_name,
    }
}

/// Build `EntityKind::Bulk` from a set of atom indices (water/solvent).
fn build_bulk_kind(indices: &[usize], coords: &Coords) -> EntityKind {
    let mut atoms = Vec::with_capacity(indices.len());
    let mut atom_names = Vec::with_capacity(indices.len());
    let mut elements = Vec::with_capacity(indices.len());

    // Count unique (chain_id, res_num) pairs for molecule_count
    let mut seen = std::collections::HashSet::new();
    for &idx in indices {
        atoms.push(coords.atoms[idx].clone());
        atom_names.push(coords.atom_names[idx]);
        elements.push(
            coords
                .elements
                .get(idx)
                .copied()
                .unwrap_or(Element::Unknown),
        );
        seen.insert((coords.chain_ids[idx], coords.res_nums[idx]));
    }

    let residue_name = coords.res_names[indices[0]];

    EntityKind::Bulk {
        atoms: AtomSet {
            atoms,
            atom_names,
            elements,
        },
        residue_name,
        molecule_count: seen.len(),
    }
}

/// Convert flat `Coords` + molecule type into an `EntityKind`.
///
/// Use this when you have raw Coords from deserialization and need to
/// construct the appropriate EntityKind based on molecule type.
pub fn coords_to_entity_kind(mol_type: MoleculeType, coords: Coords) -> EntityKind {
    match mol_type {
        MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA => {
            let indices: Vec<usize> = (0..coords.num_atoms).collect();
            build_polymer_kind(&indices, &coords)
        }
        MoleculeType::Water | MoleculeType::Solvent => {
            let indices: Vec<usize> = (0..coords.num_atoms).collect();
            build_bulk_kind(&indices, &coords)
        }
        _ => {
            let indices: Vec<usize> = (0..coords.num_atoms).collect();
            build_small_molecule_kind(mol_type, &indices, &coords)
        }
    }
}

/// Merge multiple entities back into a single flat `Coords`.
///
/// Entities are concatenated in order. Useful for recombining before
/// sending to backends that expect a single coordinate set (e.g., Rosetta).
pub fn merge_entities(entities: &[MoleculeEntity]) -> Coords {
    let total_atoms: usize = entities.iter().map(|e| e.atom_count()).sum();

    let mut atoms = Vec::with_capacity(total_atoms);
    let mut chain_ids = Vec::with_capacity(total_atoms);
    let mut res_names = Vec::with_capacity(total_atoms);
    let mut res_nums = Vec::with_capacity(total_atoms);
    let mut atom_names = Vec::with_capacity(total_atoms);
    let mut elements = Vec::with_capacity(total_atoms);

    for entity in entities {
        let c = entity.to_coords();
        atoms.extend(c.atoms);
        chain_ids.extend(c.chain_ids);
        res_names.extend(c.res_names);
        res_nums.extend(c.res_nums);
        atom_names.extend(c.atom_names);
        elements.extend(c.elements);
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

    let total_atoms: usize = matching.iter().map(|e| e.atom_count()).sum();
    let mut atoms = Vec::with_capacity(total_atoms);
    let mut chain_ids = Vec::with_capacity(total_atoms);
    let mut res_names = Vec::with_capacity(total_atoms);
    let mut res_nums = Vec::with_capacity(total_atoms);
    let mut atom_names = Vec::with_capacity(total_atoms);
    let mut elements = Vec::with_capacity(total_atoms);

    for entity in matching {
        let c = entity.to_coords();
        atoms.extend(c.atoms);
        chain_ids.extend(c.chain_ids);
        res_names.extend(c.res_names);
        res_nums.extend(c.res_nums);
        atom_names.extend(c.atom_names);
        elements.extend(c.elements);
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
        // ATP and HEM are now cofactors, not ligands
        assert_eq!(classify_residue("ATP"), MoleculeType::Cofactor);
        assert_eq!(classify_residue("HEM"), MoleculeType::Cofactor);
        // Unknown small molecules remain ligands
        assert_eq!(classify_residue("UNL"), MoleculeType::Ligand);
    }

    #[test]
    fn test_split_protein_only() {
        let coords = Coords {
            num_atoms: 6,
            atoms: (0..6).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A'; 6],
            res_names: vec![
                res_name("ALA"),
                res_name("ALA"),
                res_name("ALA"),
                res_name("GLY"),
                res_name("GLY"),
                res_name("GLY"),
            ],
            res_nums: vec![1, 1, 1, 2, 2, 2],
            atom_names: vec![
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
            ],
            elements: vec![Element::Unknown; 6],
        };

        let entities = split_into_entities(&coords);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].molecule_type, MoleculeType::Protein);
        assert_eq!(entities[0].atom_count(), 6);
        assert!(entities[0].as_polymer().is_some());
        let data = entities[0].as_polymer().unwrap();
        assert_eq!(data.chains.len(), 1);
        assert_eq!(data.chains[0].residues.len(), 2);
    }

    #[test]
    fn test_split_mixed() {
        let coords = Coords {
            num_atoms: 5,
            atoms: (0..5).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'B', b'C'],
            res_names: vec![
                res_name("ALA"),
                res_name("ALA"),
                res_name("HOH"),
                res_name("ATP"),
                res_name("HOH"),
            ],
            res_nums: vec![1, 1, 100, 1, 200],
            atom_names: vec![
                atom_name("N"),
                atom_name("CA"),
                atom_name("O"),
                atom_name("C1"),
                atom_name("O"),
            ],
            elements: vec![Element::Unknown; 5],
        };

        let entities = split_into_entities(&coords);
        // Protein(A), Water, Cofactor(ATP) = 3 entities
        assert_eq!(entities.len(), 3);

        let protein = entities
            .iter()
            .find(|e| e.molecule_type == MoleculeType::Protein)
            .unwrap();
        assert_eq!(protein.atom_count(), 2);

        let water = entities
            .iter()
            .find(|e| e.molecule_type == MoleculeType::Water)
            .unwrap();
        assert_eq!(water.atom_count(), 2);

        // ATP is now classified as Cofactor → SmallMolecule
        let cofactor = entities
            .iter()
            .find(|e| e.molecule_type == MoleculeType::Cofactor)
            .unwrap();
        assert_eq!(cofactor.atom_count(), 1);
        assert!(matches!(cofactor.kind, EntityKind::SmallMolecule { .. }));
    }

    #[test]
    fn test_merge_roundtrip() {
        let coords = Coords {
            num_atoms: 4,
            atoms: (0..4).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'B', b'B'],
            res_names: vec![
                res_name("ALA"),
                res_name("ALA"),
                res_name("HOH"),
                res_name("HOH"),
            ],
            res_nums: vec![1, 1, 100, 101],
            atom_names: vec![
                atom_name("N"),
                atom_name("CA"),
                atom_name("O"),
                atom_name("O"),
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

    #[test]
    fn test_classify_cofactor() {
        assert_eq!(classify_residue("CLA"), MoleculeType::Cofactor);
        assert_eq!(classify_residue("HEM"), MoleculeType::Cofactor);
        assert_eq!(classify_residue("FAD"), MoleculeType::Cofactor);
        assert_eq!(classify_residue("NAD"), MoleculeType::Cofactor);
        assert_eq!(classify_residue("SF4"), MoleculeType::Cofactor);
        assert_eq!(classify_residue("BCR"), MoleculeType::Cofactor);
        assert_eq!(classify_residue("PL9"), MoleculeType::Cofactor);
    }

    #[test]
    fn test_classify_solvent() {
        assert_eq!(classify_residue("GOL"), MoleculeType::Solvent);
        assert_eq!(classify_residue("EDO"), MoleculeType::Solvent);
        assert_eq!(classify_residue("SO4"), MoleculeType::Solvent);
        assert_eq!(classify_residue("PEG"), MoleculeType::Solvent);
        assert_eq!(classify_residue("MPD"), MoleculeType::Solvent);
        assert_eq!(classify_residue("DMS"), MoleculeType::Solvent);
    }

    #[test]
    fn test_split_cofactor_grouping() {
        // Two CLA residues on different chains should become 2 separate SmallMolecule entities
        let coords = Coords {
            num_atoms: 4,
            atoms: (0..4).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'D', b'D'],
            res_names: vec![
                res_name("CLA"),
                res_name("CLA"),
                res_name("CLA"),
                res_name("CLA"),
            ],
            res_nums: vec![1, 1, 2, 2],
            atom_names: vec![
                atom_name("MG"),
                atom_name("NA"),
                atom_name("MG"),
                atom_name("NA"),
            ],
            elements: vec![Element::Unknown; 4],
        };

        let entities = split_into_entities(&coords);
        // Each (chain_id, res_num) pair produces its own SmallMolecule entity
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].molecule_type, MoleculeType::Cofactor);
        assert_eq!(entities[1].molecule_type, MoleculeType::Cofactor);
        assert_eq!(entities[0].atom_count(), 2);
        assert_eq!(entities[1].atom_count(), 2);
        assert!(matches!(entities[0].kind, EntityKind::SmallMolecule { .. }));
    }

    #[test]
    fn test_split_solvent_consolidated() {
        // GOL and SO4 are both Solvent, should be consolidated into one entity
        let coords = Coords {
            num_atoms: 3,
            atoms: (0..3).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'B', b'C'],
            res_names: vec![res_name("GOL"), res_name("SO4"), res_name("GOL")],
            res_nums: vec![1, 2, 3],
            atom_names: vec![atom_name("O1"), atom_name("S"), atom_name("O1")],
            elements: vec![Element::Unknown; 3],
        };

        let entities = split_into_entities(&coords);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].molecule_type, MoleculeType::Solvent);
        assert_eq!(entities[0].atom_count(), 3);
        assert!(matches!(entities[0].kind, EntityKind::Bulk { .. }));
    }

    #[test]
    fn test_polymer_structure() {
        let coords = Coords {
            num_atoms: 6,
            atoms: (0..6).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'A', b'A', b'A'],
            res_names: vec![
                res_name("ALA"),
                res_name("ALA"),
                res_name("ALA"),
                res_name("GLY"),
                res_name("GLY"),
                res_name("GLY"),
            ],
            res_nums: vec![1, 1, 1, 2, 2, 2],
            atom_names: vec![
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
            ],
            elements: vec![Element::Unknown; 6],
        };

        let entities = split_into_entities(&coords);
        assert_eq!(entities.len(), 1);
        let data = entities[0].as_polymer().unwrap();
        assert_eq!(data.chains.len(), 1);
        assert_eq!(data.chains[0].chain_id, b'A');
        assert_eq!(data.chains[0].residues.len(), 2);
        assert_eq!(data.chains[0].residues[0].name, res_name("ALA"));
        assert_eq!(data.chains[0].residues[0].number, 1);
        assert_eq!(data.chains[0].residues[0].atom_range, 0..3);
        assert_eq!(data.chains[0].residues[1].name, res_name("GLY"));
        assert_eq!(data.chains[0].residues[1].number, 2);
        assert_eq!(data.chains[0].residues[1].atom_range, 3..6);
    }

    #[test]
    fn test_small_molecule_no_chain_residue() {
        let coords = Coords {
            num_atoms: 1,
            atoms: vec![make_atom(1.0)],
            chain_ids: vec![b'A'],
            res_names: vec![res_name("ZN")],
            res_nums: vec![300],
            atom_names: vec![atom_name("ZN")],
            elements: vec![Element::Zn],
        };

        let entities = split_into_entities(&coords);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].molecule_type, MoleculeType::Ion);
        match &entities[0].kind {
            EntityKind::SmallMolecule {
                atoms,
                residue_name,
                ..
            } => {
                assert_eq!(atoms.len(), 1);
                assert_eq!(*residue_name, res_name("ZN"));
            }
            _ => panic!("expected SmallMolecule"),
        }
    }

    #[test]
    fn test_to_coords_roundtrip() {
        let coords = Coords {
            num_atoms: 6,
            atoms: (0..6).map(|i| make_atom(i as f32)).collect(),
            chain_ids: vec![b'A', b'A', b'A', b'A', b'A', b'A'],
            res_names: vec![
                res_name("ALA"),
                res_name("ALA"),
                res_name("ALA"),
                res_name("GLY"),
                res_name("GLY"),
                res_name("GLY"),
            ],
            res_nums: vec![1, 1, 1, 2, 2, 2],
            atom_names: vec![
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
                atom_name("N"),
                atom_name("CA"),
                atom_name("C"),
            ],
            elements: vec![Element::Unknown; 6],
        };

        let entities = split_into_entities(&coords);
        let recovered = entities[0].to_coords();
        assert_eq!(recovered.num_atoms, 6);
        assert_eq!(recovered.chain_ids, vec![b'A'; 6]);
        assert_eq!(recovered.res_nums, vec![1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn test_aabb_from_positions() {
        let positions = vec![
            glam::Vec3::new(1.0, 2.0, 3.0),
            glam::Vec3::new(-1.0, 5.0, 0.0),
            glam::Vec3::new(3.0, -2.0, 7.0),
        ];
        let aabb = Aabb::from_positions(&positions).unwrap();
        assert_eq!(aabb.min, glam::Vec3::new(-1.0, -2.0, 0.0));
        assert_eq!(aabb.max, glam::Vec3::new(3.0, 5.0, 7.0));
    }

    #[test]
    fn test_aabb_empty() {
        assert!(Aabb::from_positions(&[]).is_none());
    }

    #[test]
    fn test_aabb_union() {
        let a = Aabb {
            min: glam::Vec3::new(0.0, 0.0, 0.0),
            max: glam::Vec3::new(1.0, 1.0, 1.0),
        };
        let b = Aabb {
            min: glam::Vec3::new(-1.0, 2.0, -3.0),
            max: glam::Vec3::new(0.5, 4.0, 0.5),
        };
        let merged = a.union(&b);
        assert_eq!(merged.min, glam::Vec3::new(-1.0, 0.0, -3.0));
        assert_eq!(merged.max, glam::Vec3::new(1.0, 4.0, 1.0));
    }

    #[test]
    fn test_aabb_from_aabbs() {
        let aabbs = vec![
            Aabb {
                min: glam::Vec3::ZERO,
                max: glam::Vec3::ONE,
            },
            Aabb {
                min: glam::Vec3::splat(2.0),
                max: glam::Vec3::splat(3.0),
            },
        ];
        let merged = Aabb::from_aabbs(&aabbs).unwrap();
        assert_eq!(merged.min, glam::Vec3::ZERO);
        assert_eq!(merged.max, glam::Vec3::splat(3.0));
        assert!(Aabb::from_aabbs(&[]).is_none());
    }

    #[test]
    fn test_aabb_center_extents_radius() {
        let aabb = Aabb {
            min: glam::Vec3::ZERO,
            max: glam::Vec3::new(4.0, 6.0, 8.0),
        };
        assert_eq!(aabb.center(), glam::Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(aabb.extents(), glam::Vec3::new(4.0, 6.0, 8.0));
        let expected_radius = glam::Vec3::new(4.0, 6.0, 8.0).length() * 0.5;
        assert!((aabb.radius() - expected_radius).abs() < 1e-6);
    }
}
