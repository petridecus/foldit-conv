//! Molecule-type classification and per-entity coordinate splitting.
//!
//! Provides:
//! - `MoleculeType` — classification of residues into protein, DNA, RNA, ligand, ion, water
//! - `MoleculeEntity` — a single entity (chain or group) with its own `Coords`
//! - `classify_residue()` — classify a residue name into a `MoleculeType`
//! - `split_into_entities()` — split flat `Coords` into per-entity groups
//! - `merge_entities()` — recombine entities back into flat `Coords`

use super::coords::{Coords, Element};
use crate::ops::transform::PROTEIN_RESIDUES;
use std::collections::{BTreeMap, HashMap};

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

/// A single entity: one logical molecule (a protein chain, a ligand, waters, etc.)
/// with its own coordinate set.
#[derive(Debug, Clone)]
pub struct MoleculeEntity {
    pub entity_id: u32,
    pub molecule_type: MoleculeType,
    pub coords: Coords,
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
    /// Compute the axis-aligned bounding box for this entity's atoms.
    pub fn aabb(&self) -> Option<Aabb> {
        Aabb::from_positions(&self.positions())
    }

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
                let mut chains: Vec<u8> = self.coords.chain_ids.to_vec();
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
            MoleculeType::Lipid => format!("Lipid ({} molecules)", self.residue_count()),
            MoleculeType::Cofactor => {
                if let Some(rn) = self.coords.res_names.first() {
                    let name = std::str::from_utf8(rn).unwrap_or("???").trim();
                    let display = cofactor_display_name(name);
                    let count = self.residue_count();
                    if count > 1 {
                        format!("{} ({} molecules)", display, count)
                    } else {
                        display.to_string()
                    }
                } else {
                    "Cofactor".to_string()
                }
            }
            MoleculeType::Solvent => format!("Solvent ({} molecules)", self.residue_count()),
        }
    }

    /// Whether this entity type participates in tab-cycling focus.
    /// Protein: no (focused at group level). Water, Ion: no (ambient).
    /// Ligand, DNA, RNA: yes.
    pub fn is_focusable(&self) -> bool {
        matches!(
            self.molecule_type,
            MoleculeType::Ligand | MoleculeType::DNA | MoleculeType::RNA | MoleculeType::Cofactor
        )
    }

    /// Extract phosphorus (P) atom positions grouped by chain ID.
    /// Returns one chain per polymer chain, each containing the P-atom positions in order.
    /// Chains are split at gaps where consecutive P-P distance exceeds ~8 Å
    /// (typical P-P distance is ~5.8–6.5 Å).
    /// Only meaningful for DNA/RNA entities; returns empty for other molecule types.
    pub fn extract_p_atom_chains(&self) -> Vec<Vec<glam::Vec3>> {
        const MAX_PP_DIST_SQ: f32 = 8.0 * 8.0;

        if !matches!(self.molecule_type, MoleculeType::DNA | MoleculeType::RNA) {
            return Vec::new();
        }

        let mut raw_chains: BTreeMap<u8, Vec<glam::Vec3>> = BTreeMap::new();

        for i in 0..self.coords.num_atoms {
            let name = std::str::from_utf8(&self.coords.atom_names[i])
                .unwrap_or("")
                .trim();
            if name == "P" {
                let a = &self.coords.atoms[i];
                raw_chains
                    .entry(self.coords.chain_ids[i])
                    .or_default()
                    .push(glam::Vec3::new(a.x, a.y, a.z));
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
    /// Returns one `NucleotideRing` per residue that has the required atoms.
    /// Only meaningful for DNA/RNA entities; returns empty for other molecule types.
    pub fn extract_base_rings(&self) -> Vec<NucleotideRing> {
        if !matches!(self.molecule_type, MoleculeType::DNA | MoleculeType::RNA) {
            return Vec::new();
        }

        // Group atom indices by (chain_id, res_num)
        let mut residue_atoms: BTreeMap<(u8, i32), Vec<usize>> = BTreeMap::new();
        for i in 0..self.coords.num_atoms {
            let key = (self.coords.chain_ids[i], self.coords.res_nums[i]);
            residue_atoms.entry(key).or_default().push(i);
        }

        let mut rings = Vec::new();
        let mut skipped_partial = 0u32;
        for indices in residue_atoms.values() {
            // Get residue name from first atom
            let res_name = std::str::from_utf8(&self.coords.res_names[indices[0]])
                .unwrap_or("")
                .trim();

            let color = match ndb_base_color(res_name) {
                Some(c) => c,
                None => continue,
            };

            // Build atom_name -> position map for this residue.
            // Use owned Strings with robust trimming (whitespace + null bytes)
            // to handle varying parser conventions.
            let mut atom_map: HashMap<String, glam::Vec3> = HashMap::new();
            for &idx in indices {
                let name = std::str::from_utf8(&self.coords.atom_names[idx])
                    .unwrap_or("")
                    .trim()
                    .trim_matches('\0')
                    .to_owned();
                let a = &self.coords.atoms[idx];
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

/// Standard DNA residue names (mmCIF convention + Rosetta internal name THY).
const DNA_RESIDUES: &[&str] = &["DA", "DC", "DG", "DT", "DU", "DI", "THY"];

/// Standard RNA residue names.
/// Single-letter names (A, C, G, U) are the mmCIF standard for RNA.
/// Three-letter variants (ADE, CYT, GUA, URA) are legacy PDB conventions
/// and also used by Rosetta for DNA exports (acceptable: both use NA renderer).
/// RAD/RCY/RGU are Rosetta internal names for RNA returned by the C++ backend.
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
/// Covers common lipids from CHARMM, AMBER, and GROMACS force fields.
/// Names are truncated to 3 characters to match PDB residue name conventions.
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
/// Covers porphyrins, quinones, nucleotide cofactors, and iron-sulfur clusters.
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

/// Key for grouping atoms into entities.
/// We use chain_id + molecule_type, except Water/Solvent which are consolidated.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum EntityKey {
    /// A polymeric chain (protein, DNA, RNA) on a specific chain.
    Chain(u8, MoleculeTypeOrd),
    /// All water molecules consolidated into one entity.
    Water,
    /// All lipid molecules consolidated into one entity.
    Lipid,
    /// All solvent molecules consolidated into one entity.
    Solvent,
    /// Cofactors grouped by residue name (e.g. all CLA → one entity).
    Cofactor([u8; 3]),
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
            MoleculeType::Lipid => EntityKey::Lipid,
            MoleculeType::Solvent => EntityKey::Solvent,
            MoleculeType::Cofactor => EntityKey::Cofactor(coords.res_names[i]),
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
                EntityKey::Lipid => MoleculeType::Lipid,
                EntityKey::Solvent => MoleculeType::Solvent,
                EntityKey::Cofactor(_) => MoleculeType::Cofactor,
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
                elements.push(
                    coords
                        .elements
                        .get(idx)
                        .copied()
                        .unwrap_or(Element::Unknown),
                );
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
        assert_eq!(entities[0].coords.num_atoms, 6);
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
        assert_eq!(protein.coords.num_atoms, 2);

        let water = entities
            .iter()
            .find(|e| e.molecule_type == MoleculeType::Water)
            .unwrap();
        assert_eq!(water.coords.num_atoms, 2);

        // ATP is now classified as Cofactor
        let cofactor = entities
            .iter()
            .find(|e| e.molecule_type == MoleculeType::Cofactor)
            .unwrap();
        assert_eq!(cofactor.coords.num_atoms, 1);
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
        // Two CLA residues on different chains should become one Cofactor entity
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
        // All CLA atoms go to one Cofactor entity (grouped by res_name)
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].molecule_type, MoleculeType::Cofactor);
        assert_eq!(entities[0].coords.num_atoms, 4);
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
        assert_eq!(entities[0].coords.num_atoms, 3);
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
