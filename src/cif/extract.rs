//! Typed extractors for CIF data blocks.
//!
//! Each extractor implements `TryFrom<&Block>`, pulling typed data out of the
//! untyped DOM. Extraction fails gracefully when required categories or tags
//! are missing.

use super::dom::{Block, Value};

/// Unit cell parameters.
#[derive(Debug, Clone)]
pub struct UnitCell {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

/// A single atom from the `_atom_site` category.
#[derive(Debug, Clone)]
pub struct AtomSite {
    /// `ATOM` or `HETATM`.
    pub group: String,
    /// Atom name (e.g. `CA`, `N`, `OG1`).
    pub label: String,
    /// Residue name (e.g. `ALA`, `GLY`).
    pub residue: String,
    /// Chain ID.
    pub chain: String,
    /// Residue sequence number.
    pub seq_id: Option<i32>,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Element symbol (e.g. `C`, `N`, `O`).
    pub element: String,
    pub occupancy: f64,
    pub b_factor: f64,
}

/// Coordinate data extracted from an mmCIF block.
#[derive(Debug, Clone)]
pub struct CoordinateData {
    pub atoms: Vec<AtomSite>,
    pub cell: Option<UnitCell>,
    pub spacegroup: Option<String>,
}

/// A single reflection from the `_refln` category.
#[derive(Debug, Clone)]
pub struct Reflection {
    pub h: i32,
    pub k: i32,
    pub l: i32,
    pub f_meas: Option<f64>,
    pub sigma_f_meas: Option<f64>,
    pub f_calc: Option<f64>,
    pub phase_calc: Option<f64>,
    /// R-free flag character (e.g. `o` = observed, `f` = free-set).
    pub status: Option<char>,
}

/// Structure factor / reflection data extracted from an SF-CIF block.
#[derive(Debug, Clone)]
pub struct ReflectionData {
    pub cell: UnitCell,
    pub spacegroup: Option<String>,
    pub reflections: Vec<Reflection>,
}

/// Auto-detected content of a CIF data block.
#[derive(Debug)]
pub enum CifContent {
    Coordinates(CoordinateData),
    Reflections(ReflectionData),
    Unknown(Block),
}

/// Errors from typed extraction.
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("missing category: {0}")]
    MissingCategory(String),
    #[error("missing required tag: {0}")]
    MissingTag(String),
    #[error("parse error in {tag} row {row}: {detail}")]
    ParseError {
        tag: String,
        row: usize,
        detail: String,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_cell(block: &Block) -> Option<UnitCell> {
    Some(UnitCell {
        a: block.get("_cell.length_a")?.as_f64()?,
        b: block.get("_cell.length_b")?.as_f64()?,
        c: block.get("_cell.length_c")?.as_f64()?,
        alpha: block.get("_cell.angle_alpha")?.as_f64()?,
        beta: block.get("_cell.angle_beta")?.as_f64()?,
        gamma: block.get("_cell.angle_gamma")?.as_f64()?,
    })
}

fn extract_spacegroup(block: &Block) -> Option<String> {
    block
        .get("_symmetry.space_group_name_H-M")
        .or_else(|| block.get("_space_group.name_H-M_alt"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn require_f64(v: &Value, tag: &str, row: usize) -> Result<f64, ExtractionError> {
    v.as_f64().ok_or_else(|| ExtractionError::ParseError {
        tag: tag.into(),
        row,
        detail: format!("expected float, got {:?}", v),
    })
}

fn require_i32(v: &Value, tag: &str, row: usize) -> Result<i32, ExtractionError> {
    v.as_i32().ok_or_else(|| ExtractionError::ParseError {
        tag: tag.into(),
        row,
        detail: format!("expected integer, got {:?}", v),
    })
}

// ---------------------------------------------------------------------------
// TryFrom impls
// ---------------------------------------------------------------------------

impl TryFrom<&Block> for UnitCell {
    type Error = ExtractionError;

    fn try_from(block: &Block) -> Result<Self, Self::Error> {
        extract_cell(block).ok_or_else(|| ExtractionError::MissingTag("_cell.length_*".into()))
    }
}

impl TryFrom<&Block> for CoordinateData {
    type Error = ExtractionError;

    fn try_from(block: &Block) -> Result<Self, Self::Error> {
        let cols = block
            .columns(&[
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
            ])
            .ok_or(ExtractionError::MissingCategory("_atom_site".into()))?;

        // Optional columns — collect into Vecs for indexed access
        let elements: Option<Vec<_>> = block.column("_atom_site.type_symbol").map(|c| c.collect());
        let bfactors: Option<Vec<_>> = block
            .column("_atom_site.B_iso_or_equiv")
            .map(|c| c.collect());
        let occupancies: Option<Vec<_>> = block.column("_atom_site.occupancy").map(|c| c.collect());
        let seq_ids: Option<Vec<_>> = block.column("_atom_site.label_seq_id").map(|c| c.collect());
        let groups: Option<Vec<_>> = block.column("_atom_site.group_PDB").map(|c| c.collect());

        let mut atoms = Vec::with_capacity(cols.nrows());
        for (i, row) in cols.iter().enumerate() {
            atoms.push(AtomSite {
                group: groups
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.as_str())
                    .unwrap_or("ATOM")
                    .to_string(),
                label: row[0].as_str().unwrap_or("").to_string(),
                residue: row[1].as_str().unwrap_or("").to_string(),
                chain: row[2].as_str().unwrap_or("").to_string(),
                seq_id: seq_ids
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.as_i32()),
                x: require_f64(row[3], "_atom_site.Cartn_x", i)?,
                y: require_f64(row[4], "_atom_site.Cartn_y", i)?,
                z: require_f64(row[5], "_atom_site.Cartn_z", i)?,
                element: elements
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                occupancy: occupancies
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0),
                b_factor: bfactors
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
            });
        }

        Ok(CoordinateData {
            atoms,
            cell: extract_cell(block),
            spacegroup: extract_spacegroup(block),
        })
    }
}

impl TryFrom<&Block> for ReflectionData {
    type Error = ExtractionError;

    fn try_from(block: &Block) -> Result<Self, Self::Error> {
        let cell = extract_cell(block)
            .ok_or_else(|| ExtractionError::MissingTag("_cell.length_*".into()))?;

        let cols = block
            .columns(&["_refln.index_h", "_refln.index_k", "_refln.index_l"])
            .ok_or(ExtractionError::MissingCategory("_refln".into()))?;

        // Optional columns
        let f_meas: Option<Vec<_>> = block.column("_refln.F_meas_au").map(|c| c.collect());
        let sigma: Option<Vec<_>> = block.column("_refln.F_meas_sigma_au").map(|c| c.collect());
        let f_calc: Option<Vec<_>> = block
            .column("_refln.F_calc_au")
            .or_else(|| block.column("_refln.F_calc"))
            .map(|c| c.collect());
        let phase: Option<Vec<_>> = block.column("_refln.phase_calc").map(|c| c.collect());
        let status: Option<Vec<_>> = block.column("_refln.status").map(|c| c.collect());

        let mut reflections = Vec::with_capacity(cols.nrows());
        for (i, row) in cols.iter().enumerate() {
            reflections.push(Reflection {
                h: require_i32(row[0], "_refln.index_h", i)?,
                k: require_i32(row[1], "_refln.index_k", i)?,
                l: require_i32(row[2], "_refln.index_l", i)?,
                f_meas: f_meas
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.as_f64()),
                sigma_f_meas: sigma
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.as_f64()),
                f_calc: f_calc
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.as_f64()),
                phase_calc: phase
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.as_f64()),
                status: status
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.chars().next()),
            });
        }

        Ok(ReflectionData {
            cell,
            spacegroup: extract_spacegroup(block),
            reflections,
        })
    }
}

impl From<Block> for CifContent {
    fn from(block: Block) -> Self {
        if let Ok(coords) = CoordinateData::try_from(&block) {
            return CifContent::Coordinates(coords);
        }
        if let Ok(reflns) = ReflectionData::try_from(&block) {
            return CifContent::Reflections(reflns);
        }
        CifContent::Unknown(block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cif::parse::parse;

    const MMCIF_SNIPPET: &str = r#"data_1ABC
_cell.length_a   50.000
_cell.length_b   60.000
_cell.length_c   70.000
_cell.angle_alpha 90.00
_cell.angle_beta  90.00
_cell.angle_gamma 90.00
_symmetry.space_group_name_H-M 'P 21 21 21'
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM N   ALA A 1 10.000 20.000 30.000 N 1.00 15.0
ATOM CA  ALA A 1 11.000 21.000 31.000 C 1.00 16.0
ATOM C   ALA A 1 12.000 22.000 32.000 C 1.00 17.0
HETATM O   HOH B 100 5.000 6.000 7.000 O 1.00 20.0
"#;

    const SF_SNIPPET: &str = r#"data_r1abcsf
_cell.length_a   50.000
_cell.length_b   60.000
_cell.length_c   70.000
_cell.angle_alpha 90.00
_cell.angle_beta  90.00
_cell.angle_gamma 90.00
_symmetry.space_group_name_H-M 'P 21 21 21'
loop_
_refln.index_h
_refln.index_k
_refln.index_l
_refln.F_meas_au
_refln.F_meas_sigma_au
_refln.status
1  0  0  100.5 2.3 o
0  1  0  200.1 3.4 o
0  0  1  150.7 2.8 f
-1 2  3  .     .   o
"#;

    #[test]
    fn extract_coordinates() {
        let doc = parse(MMCIF_SNIPPET).unwrap();
        let coords = CoordinateData::try_from(&doc.blocks[0]).unwrap();

        assert_eq!(coords.atoms.len(), 4);
        assert_eq!(coords.atoms[0].label, "N");
        assert_eq!(coords.atoms[0].residue, "ALA");
        assert_eq!(coords.atoms[0].chain, "A");
        assert_eq!(coords.atoms[0].seq_id, Some(1));
        assert!((coords.atoms[0].x - 10.0).abs() < 1e-6);
        assert!((coords.atoms[1].y - 21.0).abs() < 1e-6);
        assert_eq!(coords.atoms[1].element, "C");
        assert!((coords.atoms[2].b_factor - 17.0).abs() < 1e-6);

        // HETATM
        assert_eq!(coords.atoms[3].group, "HETATM");
        assert_eq!(coords.atoms[3].residue, "HOH");
        assert_eq!(coords.atoms[3].chain, "B");
    }

    #[test]
    fn extract_unit_cell() {
        let doc = parse(MMCIF_SNIPPET).unwrap();
        let cell = UnitCell::try_from(&doc.blocks[0]).unwrap();
        assert!((cell.a - 50.0).abs() < 1e-6);
        assert!((cell.b - 60.0).abs() < 1e-6);
        assert!((cell.c - 70.0).abs() < 1e-6);
        assert!((cell.alpha - 90.0).abs() < 1e-6);
    }

    #[test]
    fn extract_spacegroup_from_block() {
        let doc = parse(MMCIF_SNIPPET).unwrap();
        let coords = CoordinateData::try_from(&doc.blocks[0]).unwrap();
        assert_eq!(coords.spacegroup.as_deref(), Some("P 21 21 21"));
    }

    #[test]
    fn extract_reflections() {
        let doc = parse(SF_SNIPPET).unwrap();
        let sf = ReflectionData::try_from(&doc.blocks[0]).unwrap();

        assert_eq!(sf.reflections.len(), 4);
        assert_eq!(sf.reflections[0].h, 1);
        assert_eq!(sf.reflections[0].k, 0);
        assert_eq!(sf.reflections[0].l, 0);
        assert!((sf.reflections[0].f_meas.unwrap() - 100.5).abs() < 1e-6);
        assert!((sf.reflections[0].sigma_f_meas.unwrap() - 2.3).abs() < 1e-6);
        assert_eq!(sf.reflections[0].status, Some('o'));

        // Free-set flag
        assert_eq!(sf.reflections[2].status, Some('f'));

        // Negative index
        assert_eq!(sf.reflections[3].h, -1);

        // Inapplicable values → None
        assert!(sf.reflections[3].f_meas.is_none());
        assert!(sf.reflections[3].sigma_f_meas.is_none());
    }

    #[test]
    fn reflection_cell_required() {
        let input = "data_test\nloop_\n_refln.index_h\n_refln.index_k\n_refln.index_l\n1 0 0\n";
        let doc = parse(input).unwrap();
        let err = ReflectionData::try_from(&doc.blocks[0]).unwrap_err();
        assert!(matches!(err, ExtractionError::MissingTag(_)));
    }

    #[test]
    fn autodetect_coordinates() {
        let doc = parse(MMCIF_SNIPPET).unwrap();
        let content = CifContent::from(doc.blocks.into_iter().next().unwrap());
        assert!(matches!(content, CifContent::Coordinates(_)));
    }

    #[test]
    fn autodetect_reflections() {
        let doc = parse(SF_SNIPPET).unwrap();
        let content = CifContent::from(doc.blocks.into_iter().next().unwrap());
        assert!(matches!(content, CifContent::Reflections(_)));
    }

    #[test]
    fn autodetect_unknown() {
        let input = "data_mystery\n_some.tag value\n";
        let doc = parse(input).unwrap();
        let content = CifContent::from(doc.blocks.into_iter().next().unwrap());
        assert!(matches!(content, CifContent::Unknown(_)));
    }

    #[test]
    fn missing_atom_site_category() {
        let input = "data_empty\n_cell.length_a 50.0\n";
        let doc = parse(input).unwrap();
        let err = CoordinateData::try_from(&doc.blocks[0]).unwrap_err();
        assert!(matches!(err, ExtractionError::MissingCategory(_)));
    }

    #[test]
    fn optional_columns_absent() {
        // Minimal atom_site with only required columns
        let input = r#"data_min
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
CA ALA A 1.0 2.0 3.0
"#;
        let doc = parse(input).unwrap();
        let coords = CoordinateData::try_from(&doc.blocks[0]).unwrap();
        assert_eq!(coords.atoms.len(), 1);
        assert_eq!(coords.atoms[0].label, "CA");
        assert_eq!(coords.atoms[0].group, "ATOM"); // default
        assert_eq!(coords.atoms[0].element, ""); // absent
        assert!((coords.atoms[0].occupancy - 1.0).abs() < 1e-6); // default
        assert!((coords.atoms[0].b_factor - 0.0).abs() < 1e-6); // default
        assert!(coords.atoms[0].seq_id.is_none());
        assert!(coords.cell.is_none());
        assert!(coords.spacegroup.is_none());
    }

    #[test]
    fn cell_with_uncertainty() {
        let input = "data_test\n_cell.length_a 50.123(4)\n_cell.length_b 60.456(5)\n_cell.length_c 70.789(6)\n_cell.angle_alpha 90.00(1)\n_cell.angle_beta 90.00(1)\n_cell.angle_gamma 90.00(1)\n";
        let doc = parse(input).unwrap();
        let cell = UnitCell::try_from(&doc.blocks[0]).unwrap();
        assert!((cell.a - 50.123).abs() < 1e-6);
        assert!((cell.b - 60.456).abs() < 1e-6);
    }
}
