//! COORDS validation and introspection.
//!
//! Provides functions to check what atoms are present in COORDS data
//! and validate completeness against expected residue types.

use super::types::{Coords, ResidueAtoms, ValidationResult};
use std::collections::{HashMap, HashSet};

/// Expected heavy atoms for each standard amino acid residue type.
/// Returns the atom names that should be present (backbone + sidechain, no hydrogens).
pub fn expected_heavy_atoms(res_name: &str) -> &'static [&'static str] {
    match res_name.trim() {
        "ALA" => &["N", "CA", "C", "O", "CB"],
        "ARG" => &["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "ASN" => &["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
        "ASP" => &["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
        "CYS" => &["N", "CA", "C", "O", "CB", "SG"],
        "GLN" => &["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
        "GLU" => &["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
        "GLY" => &["N", "CA", "C", "O"],
        "HIS" => &["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "ILE" => &["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
        "LEU" => &["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
        "LYS" => &["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
        "MET" => &["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
        "PHE" => &["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "PRO" => &["N", "CA", "C", "O", "CB", "CG", "CD"],
        "SER" => &["N", "CA", "C", "O", "CB", "OG"],
        "THR" => &["N", "CA", "C", "O", "CB", "OG1", "CG2"],
        "TRP" => &["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "TYR" => &["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "VAL" => &["N", "CA", "C", "O", "CB", "CG1", "CG2"],
        // Unknown residues - just backbone
        _ => &["N", "CA", "C", "O"],
    }
}

/// Backbone atoms common to all residues (except GLY which has no CB).
pub fn backbone_atoms() -> &'static [&'static str] {
    &["N", "CA", "C", "O"]
}

/// Group atoms by residue from COORDS data.
pub fn atoms_by_residue(coords: &Coords) -> Vec<ResidueAtoms> {
    let mut residue_map: HashMap<(u8, i32), ResidueAtoms> = HashMap::new();

    for i in 0..coords.num_atoms {
        let chain_id = coords.chain_ids[i];
        let res_num = coords.res_nums[i];
        let res_name = coords.res_names[i];
        let atom_name = coords.atom_names[i];

        let key = (chain_id, res_num);
        residue_map
            .entry(key)
            .or_insert_with(|| ResidueAtoms {
                chain_id,
                res_num,
                res_name,
                atoms: Vec::new(),
            })
            .atoms
            .push(atom_name);
    }

    // Sort by chain and residue number
    let mut residues: Vec<ResidueAtoms> = residue_map.into_values().collect();
    residues.sort_by(|a, b| {
        a.chain_id.cmp(&b.chain_id).then(a.res_num.cmp(&b.res_num))
    });

    residues
}

/// Validate that COORDS has complete sidechains for all residues.
pub fn validate_completeness(coords: &Coords) -> ValidationResult {
    let residues = atoms_by_residue(coords);
    let mut missing_atoms = Vec::new();
    let mut extra_atoms = Vec::new();
    let mut incomplete_residues = 0;

    for residue in &residues {
        let res_name_str = std::str::from_utf8(&residue.res_name)
            .unwrap_or("UNK")
            .trim();

        let expected = expected_heavy_atoms(res_name_str);
        let expected_set: HashSet<&str> = expected.iter().copied().collect();

        // Convert present atoms to strings for comparison
        let present: HashSet<String> = residue
            .atoms
            .iter()
            .map(|a| {
                std::str::from_utf8(a)
                    .unwrap_or("")
                    .trim()
                    .to_string()
            })
            .collect();

        // Find missing atoms
        let missing: Vec<String> = expected_set
            .iter()
            .filter(|&a| !present.contains(*a))
            .map(|s| s.to_string())
            .collect();

        // Find extra atoms (not in expected list, excluding hydrogens)
        let extra: Vec<String> = present
            .iter()
            .filter(|a| {
                !expected_set.contains(a.as_str())
                    && !a.starts_with('H')
                    && !a.starts_with("1H")
                    && !a.starts_with("2H")
                    && !a.starts_with("3H")
            })
            .cloned()
            .collect();

        if !missing.is_empty() {
            missing_atoms.push((residue.res_num, res_name_str.to_string(), missing));
            incomplete_residues += 1;
        }

        if !extra.is_empty() {
            extra_atoms.push((residue.res_num, res_name_str.to_string(), extra));
        }
    }

    ValidationResult {
        is_complete: missing_atoms.is_empty(),
        missing_atoms,
        extra_atoms,
        total_residues: residues.len(),
        incomplete_residues,
    }
}

/// Generate a human-readable completeness report.
pub fn completeness_report(coords: &Coords) -> String {
    let validation = validate_completeness(coords);
    let mut report = String::new();

    report.push_str(&format!(
        "COORDS Completeness Report\n{}\n",
        "=".repeat(50)
    ));
    report.push_str(&format!(
        "Total residues: {}\n",
        validation.total_residues
    ));
    report.push_str(&format!(
        "Incomplete residues: {}\n",
        validation.incomplete_residues
    ));
    report.push_str(&format!(
        "Complete: {}\n\n",
        if validation.is_complete { "YES" } else { "NO" }
    ));

    if !validation.missing_atoms.is_empty() {
        report.push_str("Missing atoms:\n");
        for (res_num, res_name, atoms) in &validation.missing_atoms {
            report.push_str(&format!(
                "  Residue {} ({}): {}\n",
                res_num,
                res_name,
                atoms.join(", ")
            ));
        }
        report.push('\n');
    }

    if !validation.extra_atoms.is_empty() {
        report.push_str("Unexpected atoms:\n");
        for (res_num, res_name, atoms) in &validation.extra_atoms {
            report.push_str(&format!(
                "  Residue {} ({}): {}\n",
                res_num,
                res_name,
                atoms.join(", ")
            ));
        }
    }

    report
}

/// Quick check: does COORDS have at least backbone atoms for all residues?
pub fn has_complete_backbone(coords: &Coords) -> bool {
    let residues = atoms_by_residue(coords);
    let backbone = backbone_atoms();

    for residue in &residues {
        let present: HashSet<String> = residue
            .atoms
            .iter()
            .map(|a| std::str::from_utf8(a).unwrap_or("").trim().to_string())
            .collect();

        for &atom in backbone {
            if !present.contains(atom) {
                return false;
            }
        }
    }

    true
}

/// Count atoms by type in COORDS.
pub fn atom_counts(coords: &Coords) -> AtomCounts {
    let mut backbone = 0;
    let mut sidechain = 0;
    let mut hydrogen = 0;

    let backbone_set: HashSet<&str> = ["N", "CA", "C", "O"].iter().copied().collect();

    for i in 0..coords.num_atoms {
        let atom_name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();

        if atom_name.starts_with('H')
            || atom_name.starts_with("1H")
            || atom_name.starts_with("2H")
            || atom_name.starts_with("3H")
        {
            hydrogen += 1;
        } else if backbone_set.contains(atom_name) {
            backbone += 1;
        } else {
            sidechain += 1;
        }
    }

    AtomCounts {
        total: coords.num_atoms,
        backbone,
        sidechain,
        hydrogen,
    }
}

/// Summary of atom counts by type.
#[derive(Debug, Clone)]
pub struct AtomCounts {
    pub total: usize,
    pub backbone: usize,
    pub sidechain: usize,
    pub hydrogen: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_heavy_atoms() {
        assert_eq!(expected_heavy_atoms("GLY").len(), 4); // No CB
        assert_eq!(expected_heavy_atoms("ALA").len(), 5); // + CB
        assert_eq!(expected_heavy_atoms("ASP").len(), 8); // Full sidechain
        assert_eq!(expected_heavy_atoms("TRP").len(), 14); // Largest
    }

    #[test]
    fn test_backbone_atoms() {
        let bb = backbone_atoms();
        assert!(bb.contains(&"N"));
        assert!(bb.contains(&"CA"));
        assert!(bb.contains(&"C"));
        assert!(bb.contains(&"O"));
    }
}
