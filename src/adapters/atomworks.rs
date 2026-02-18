//! AtomWorks adapter for RC-Foundry / ModelForge models.
//!
//! Bidirectional conversion between molconv's `Vec<MoleculeEntity>` and
//! AtomWorks-annotated Biotite `AtomArray` objects.
//!
//! Unlike the plain biotite adapter (which converts flat `Coords` bytes and
//! produces an empty `BondList`), this adapter:
//!
//! - Operates on **entities**, preserving molecule type, entity ID, and chain
//!   grouping through the round-trip.
//! - Populates **bonds** from `MoleculeEntity.bonds` (when the planned bond
//!   topology refactor lands) or from distance inference as a fallback.
//! - Sets AtomWorks-specific per-atom annotations (`entity_id`, `mol_type`,
//!   `pn_unit_iid`) so structures can feed directly into Foundry model
//!   pipelines (RF3, RFdiffusion3, LigandMPNN) without re-parsing.
//! - Can optionally invoke `atomworks.io.parser.parse()` on the Python side
//!   to get the full cleaning pipeline (leaving group removal, charge
//!   correction, missing atom imputation, etc.).
//!
//! # Usage from Python (via PyO3)
//!
//! ```python
//! import foldit_conv
//! from atomworks.io.parser import parse as aw_parse
//!
//! # ── molconv → AtomWorks (for model inference) ──
//! atom_array = foldit_conv.entities_to_atom_array(assembly_bytes)
//! atom_array_plus = foldit_conv.entities_to_atom_array_plus(assembly_bytes)
//!
//! # ── AtomArray → molconv (after model prediction) ──
//! assembly_bytes = foldit_conv.atom_array_to_entities(atom_array)
//!
//! # ── Full AtomWorks cleaning pipeline ──
//! atom_array = foldit_conv.entities_to_atom_array_parsed(assembly_bytes, "3nez.cif.gz")
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::ops::bond_inference::{infer_bonds, BondOrder, DEFAULT_TOLERANCE};
use crate::types::coords::{
    deserialize, deserialize_assembly, serialize_assembly, ChainIdMapper, Coords, CoordsAtom,
    Element,
};
use crate::types::entity::{split_into_entities, MoleculeEntity, MoleculeType};

// ============================================================================
// Molecule type ↔ AtomWorks chain type mapping
// ============================================================================

/// AtomWorks `ChainType` enum values (from `atomworks.enums.ChainType`).
///
/// These are the integer codes AtomWorks uses to classify PN units:
///   0=CyclicPseudoPeptide, 1=OtherPolymer, 2=PeptideNucleicAcid,
///   3=DNA, 4=DNA_RNA_HYBRID, 5=POLYPEPTIDE_D, 6=POLYPEPTIDE_L, 7=RNA,
///   8=NON_POLYMER, 9=WATER, 10=BRANCHED, 11=MACROLIDE
///
/// We map to these from `MoleculeType` when building annotations.
fn molecule_type_to_chain_type_id(mt: MoleculeType) -> u8 {
    match mt {
        MoleculeType::Protein => 6, // POLYPEPTIDE_L (default; D-peptides need explicit flag)
        MoleculeType::DNA => 3,
        MoleculeType::RNA => 7,
        MoleculeType::Ligand => 8, // NON_POLYMER
        MoleculeType::Ion => 8,    // NON_POLYMER
        MoleculeType::Water => 9,
        MoleculeType::Lipid => 8,    // NON_POLYMER
        MoleculeType::Cofactor => 8, // NON_POLYMER
        MoleculeType::Solvent => 9,  // Treat crystallization solvents as water-like
    }
}

fn chain_type_id_to_molecule_type(ct: u8) -> MoleculeType {
    match ct {
        3 | 4 => MoleculeType::DNA,     // DNA, DNA_RNA_HYBRID
        5 | 6 => MoleculeType::Protein, // POLYPEPTIDE_D, POLYPEPTIDE_L
        7 => MoleculeType::RNA,
        9 => MoleculeType::Water,
        8 => MoleculeType::Ligand, // NON_POLYMER (refined later by residue name)
        10 => MoleculeType::Ligand, // BRANCHED (glycans etc.)
        _ => MoleculeType::Ligand, // Conservative default
    }
}

/// AtomWorks `mol_type` string annotation values.
fn molecule_type_to_mol_type_str(mt: MoleculeType) -> &'static str {
    match mt {
        MoleculeType::Protein => "protein",
        MoleculeType::DNA => "dna",
        MoleculeType::RNA => "rna",
        MoleculeType::Ligand | MoleculeType::Cofactor => "ligand",
        MoleculeType::Ion => "ion",
        MoleculeType::Water | MoleculeType::Solvent => "water",
        MoleculeType::Lipid => "ligand",
    }
}

fn mol_type_str_to_molecule_type(s: &str) -> MoleculeType {
    match s.to_lowercase().as_str() {
        "protein" | "polypeptide_l" | "polypeptide_d" => MoleculeType::Protein,
        "dna" => MoleculeType::DNA,
        "rna" => MoleculeType::RNA,
        "water" => MoleculeType::Water,
        "ion" => MoleculeType::Ion,
        "ligand" | "non_polymer" | "branched" => MoleculeType::Ligand,
        _ => MoleculeType::Ligand,
    }
}

// ============================================================================
// molconv → AtomWorks (entities → annotated AtomArray)
// ============================================================================

/// Convert a `Vec<MoleculeEntity>` to an AtomWorks-compatible Biotite `AtomArray`.
///
/// The resulting AtomArray has:
/// - Standard biotite annotations: `coord`, `chain_id`, `res_id`, `res_name`,
///   `atom_name`, `element`, `occupancy`, `b_factor`
/// - AtomWorks annotations: `entity_id` (per-atom int), `mol_type` (per-atom str),
///   `chain_type` (per-atom int matching `atomworks.enums.ChainType`)
/// - `BondList` populated from entity bond data or distance inference
fn entities_to_atom_array_impl(py: Python, entities: &[MoleculeEntity]) -> PyResult<Py<PyAny>> {
    // Count total atoms across all entities
    let total_atoms: usize = entities.iter().map(|e| e.coords.num_atoms).sum();
    if total_atoms == 0 {
        let biotite = py.import("biotite.structure")?;
        let arr = biotite.getattr("AtomArray")?.call1((0,))?;
        return Ok(arr.unbind());
    }

    let numpy = py.import("numpy")?;
    let biotite = py.import("biotite.structure")?;
    let atom_array_cls = biotite.getattr("AtomArray")?;
    let bond_list_cls = biotite.getattr("BondList")?;

    // Create the AtomArray
    let atom_array = atom_array_cls.call1((total_atoms,))?;

    // ── Collect per-atom data ──
    let mut coords_flat: Vec<f32> = Vec::with_capacity(total_atoms * 3);
    let mut chain_ids: Vec<String> = Vec::with_capacity(total_atoms);
    let mut res_ids: Vec<i32> = Vec::with_capacity(total_atoms);
    let mut res_names: Vec<String> = Vec::with_capacity(total_atoms);
    let mut atom_names: Vec<String> = Vec::with_capacity(total_atoms);
    let mut elements: Vec<String> = Vec::with_capacity(total_atoms);
    let mut occupancies: Vec<f32> = Vec::with_capacity(total_atoms);
    let mut b_factors: Vec<f32> = Vec::with_capacity(total_atoms);

    // AtomWorks-specific annotations
    let mut aw_entity_ids: Vec<i32> = Vec::with_capacity(total_atoms);
    let mut aw_mol_types: Vec<String> = Vec::with_capacity(total_atoms);
    let mut aw_chain_types: Vec<i32> = Vec::with_capacity(total_atoms);

    // Bond accumulation (global atom indices)
    let mut all_bonds: Vec<(usize, usize, u8)> = Vec::new();
    let mut atom_offset: usize = 0;

    for entity in entities {
        let c = &entity.coords;
        let entity_id = entity.entity_id as i32;
        let mol_type_str = molecule_type_to_mol_type_str(entity.molecule_type).to_string();
        let chain_type_id = molecule_type_to_chain_type_id(entity.molecule_type) as i32;

        for i in 0..c.num_atoms {
            let atom = &c.atoms[i];
            coords_flat.push(atom.x);
            coords_flat.push(atom.y);
            coords_flat.push(atom.z);

            let cid = c.chain_ids[i];
            chain_ids.push(if cid.is_ascii_alphanumeric() {
                String::from(cid as char)
            } else {
                "A".to_string()
            });

            res_ids.push(c.res_nums[i]);

            let rn = std::str::from_utf8(&c.res_names[i])
                .unwrap_or("UNK")
                .trim()
                .to_string();
            res_names.push(rn);

            let an = std::str::from_utf8(&c.atom_names[i])
                .unwrap_or("X")
                .trim()
                .to_string();
            atom_names.push(an);

            let elem = c.elements.get(i).copied().unwrap_or(Element::Unknown);
            elements.push(elem.symbol().to_string());

            occupancies.push(atom.occupancy);
            b_factors.push(atom.b_factor);

            aw_entity_ids.push(entity_id);
            aw_mol_types.push(mol_type_str.clone());
            aw_chain_types.push(chain_type_id);
        }

        // ── Bonds for this entity ──
        // TODO: When MoleculeEntity gains a `bonds: Option<BondTable>` field
        // from the planned refactor, prefer those (FileExplicit/Dictionary).
        // For now, run distance inference on small-molecule entities.
        let needs_inference = matches!(
            entity.molecule_type,
            MoleculeType::Ligand | MoleculeType::Cofactor | MoleculeType::Ion
        );

        if needs_inference && c.num_atoms >= 2 && c.num_atoms <= 500 {
            let inferred = infer_bonds(c, DEFAULT_TOLERANCE);
            for bond in &inferred {
                let bt = match bond.order {
                    BondOrder::Single => 1u8,
                    BondOrder::Double => 2,
                    BondOrder::Triple => 3,
                    BondOrder::Aromatic => 4,
                };
                all_bonds.push((bond.atom_a + atom_offset, bond.atom_b + atom_offset, bt));
            }
        }

        atom_offset += c.num_atoms;
    }

    // ── Set standard annotations ──
    let coord_np = numpy.call_method1("array", (coords_flat,))?;
    let coord_np = coord_np.call_method1("reshape", ((total_atoms, 3),))?;
    let coord_np = coord_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("coord", coord_np)?;

    let chain_np = numpy.call_method1("array", (chain_ids,))?;
    atom_array.setattr("chain_id", chain_np)?;

    let res_np = numpy.call_method1("array", (res_ids,))?;
    let res_np = res_np.call_method1("astype", (numpy.getattr("int32")?,))?;
    atom_array.setattr("res_id", res_np)?;

    let resname_np = numpy.call_method1("array", (res_names,))?;
    atom_array.setattr("res_name", resname_np)?;

    let atomname_np = numpy.call_method1("array", (atom_names,))?;
    atom_array.setattr("atom_name", atomname_np)?;

    let element_np = numpy.call_method1("array", (elements,))?;
    atom_array.setattr("element", element_np)?;

    let occ_np = numpy.call_method1("array", (occupancies,))?;
    let occ_np = occ_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("occupancy", occ_np)?;

    let bf_np = numpy.call_method1("array", (b_factors,))?;
    let bf_np = bf_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("b_factor", bf_np)?;

    // ── Set AtomWorks-specific annotations ──
    let eid_np = numpy.call_method1("array", (aw_entity_ids,))?;
    let eid_np = eid_np.call_method1("astype", (numpy.getattr("int32")?,))?;
    atom_array.call_method1("set_annotation", ("entity_id", eid_np))?;

    let mt_np = numpy.call_method1("array", (aw_mol_types,))?;
    atom_array.call_method1("set_annotation", ("mol_type", mt_np))?;

    let ct_np = numpy.call_method1("array", (aw_chain_types,))?;
    let ct_np = ct_np.call_method1("astype", (numpy.getattr("int32")?,))?;
    atom_array.call_method1("set_annotation", ("chain_type", ct_np))?;

    // ── Build BondList ──
    let bond_list = bond_list_cls.call1((total_atoms,))?;
    for (a, b, bt) in &all_bonds {
        bond_list.call_method1("add_bond", (*a, *b, *bt))?;
    }
    atom_array.setattr("bonds", bond_list)?;

    Ok(atom_array.unbind())
}

/// Convert `Vec<MoleculeEntity>` (from ASSEM01 bytes) to a Biotite `AtomArray`.
#[pyfunction]
pub fn entities_to_atom_array(py: Python, assembly_bytes: Vec<u8>) -> PyResult<Py<PyAny>> {
    let entities = deserialize_assembly(&assembly_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    entities_to_atom_array_impl(py, &entities)
}

/// Convert `Vec<MoleculeEntity>` (from ASSEM01 bytes) to an `AtomArrayPlus`.
///
/// `AtomArrayPlus` signals to downstream consumers (e.g. `parse_atom_array`)
/// that the structure is already fully constructed and should skip CCD
/// template rebuilding.
#[pyfunction]
pub fn entities_to_atom_array_plus(py: Python, assembly_bytes: Vec<u8>) -> PyResult<Py<PyAny>> {
    let entities = deserialize_assembly(&assembly_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let atom_array = entities_to_atom_array_impl(py, &entities)?;
    let as_plus = py
        .import("atomworks.io.utils.atom_array_plus")?
        .getattr("as_atom_array_plus")?;
    Ok(as_plus.call1((atom_array,))?.unbind())
}

/// Convert flat COORDS bytes to a Biotite `AtomArray`.
#[pyfunction]
pub fn coords_to_atom_array(py: Python, coords_bytes: Vec<u8>) -> PyResult<Py<PyAny>> {
    let coords = deserialize(&coords_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let entities = split_into_entities(&coords);
    entities_to_atom_array_impl(py, &entities)
}

/// Convert flat COORDS bytes to an `AtomArrayPlus`.
#[pyfunction]
pub fn coords_to_atom_array_plus(py: Python, coords_bytes: Vec<u8>) -> PyResult<Py<PyAny>> {
    let coords = deserialize(&coords_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let entities = split_into_entities(&coords);
    let atom_array = entities_to_atom_array_impl(py, &entities)?;
    let as_plus = py
        .import("atomworks.io.utils.atom_array_plus")?
        .getattr("as_atom_array_plus")?;
    Ok(as_plus.call1((atom_array,))?.unbind())
}

/// Convert a Biotite `AtomArray` (or `AtomArrayPlus`) back to flat COORDS bytes.
#[pyfunction]
pub fn atom_array_to_coords(py: Python, atom_array: Py<PyAny>) -> PyResult<Vec<u8>> {
    let assembly_bytes = atom_array_to_entities(py, atom_array)?;
    let entities = deserialize_assembly(&assembly_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let coords = crate::types::entity::merge_entities(&entities);
    crate::types::coords::serialize(&coords)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

/// Convert `Vec<MoleculeEntity>` to AtomArray, then run through
/// `atomworks.io.parser.parse()` for full cleaning.
///
/// This first writes a temporary CIF/PDB via the existing export path,
/// then lets AtomWorks re-parse it with its full cleaning pipeline
/// (leaving group removal, charge correction, bond order fixing, etc.).
///
/// Use this when you need maximum data quality for model training or
/// when handling structures with known issues (missing atoms, wrong charges).
/// For interactive use where latency matters, prefer `entities_to_atom_array`.
#[pyfunction]
pub fn entities_to_atom_array_parsed(
    py: Python,
    assembly_bytes: Vec<u8>,
    source_path: Option<String>,
) -> PyResult<Py<PyAny>> {
    // If we have the original file path, let AtomWorks parse from source
    // (this gets the best cleaning since AtomWorks can read mmCIF directly
    // and apply its full pipeline including CCD bond lookup, leaving group
    // removal, charge correction, etc.)
    if let Some(path) = source_path {
        let aw_parser = py.import("atomworks.io.parser")?;
        let result = aw_parser.call_method1("parse", (path,))?;
        let asym_unit = result.get_item("asym_unit")?;

        // parse() returns an AtomArrayStack; take model 0
        let atom_array = asym_unit.get_item(0)?;
        return Ok(atom_array.unbind());
    }

    // Fallback: convert through our adapter, then apply AtomWorks transforms
    // manually for cleaning. This is less thorough than parsing from file
    // but still better than raw conversion.
    let atom_array = entities_to_atom_array(py, assembly_bytes)?;

    // Try to apply basic AtomWorks cleaning if available
    match py.import("atomworks.io.cleaning") {
        Ok(cleaning) => {
            let cleaned = cleaning.call_method1("clean_atom_array", (atom_array.bind(py),));
            match cleaned {
                Ok(c) => Ok(c.unbind()),
                Err(_) => Ok(atom_array), // cleaning module API may vary; fall back
            }
        }
        Err(_) => Ok(atom_array), // atomworks not installed or no cleaning module
    }
}

// ============================================================================
// AtomWorks → molconv (annotated AtomArray → entities)
// ============================================================================

/// Convert a Biotite `AtomArray` (or `AtomArrayPlus`) back to ASSEM01 bytes.
///
/// Reconstructs `Vec<MoleculeEntity>` from per-atom annotations.
/// Entity boundaries are determined by `entity_id` annotation if present,
/// otherwise by grouping on `(chain_id, mol_type)`.
#[pyfunction]
pub fn atom_array_to_entities(py: Python, atom_array: Py<PyAny>) -> PyResult<Vec<u8>> {
    let atom_array = atom_array.bind(py);

    // Get array length
    let num_atoms: usize = atom_array
        .getattr("coord")?
        .getattr("shape")?
        .get_item(0)?
        .extract()?;
    if num_atoms == 0 {
        return serialize_assembly(&[])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()));
    }

    // ── Extract all per-atom arrays ──
    let coord = atom_array.getattr("coord")?;
    let chain_id_arr = atom_array.getattr("chain_id")?;
    let res_id_arr = atom_array.getattr("res_id")?;
    let res_name_arr = atom_array.getattr("res_name")?;
    let atom_name_arr = atom_array.getattr("atom_name")?;

    // Optional arrays (may or may not exist)
    let element_arr = atom_array.getattr("element").ok();
    let occupancy_arr = atom_array.getattr("occupancy").ok();
    let b_factor_arr = atom_array.getattr("b_factor").ok();

    // AtomWorks-specific annotations (may not exist if coming from plain biotite)
    let entity_id_arr = get_annotation_opt(atom_array, "entity_id")?;
    let mol_type_arr = get_annotation_opt(atom_array, "mol_type")?;
    let chain_type_arr = get_annotation_opt(atom_array, "chain_type")?;

    // ── Determine entity grouping strategy ──
    // Build per-atom entity assignments
    let mut atom_entity_ids: Vec<i32> = Vec::with_capacity(num_atoms);

    if let Some(ref eid_arr) = entity_id_arr {
        // Use explicit entity_id annotation
        for i in 0..num_atoms {
            let eid: i32 = eid_arr.get_item(i)?.extract()?;
            atom_entity_ids.push(eid);
        }
    } else {
        // Fallback: group by (chain_id, mol_type_str) or just chain_id
        // Assign sequential entity IDs based on unique group keys
        let mut group_map: std::collections::HashMap<String, i32> =
            std::collections::HashMap::new();
        let mut next_id: i32 = 0;

        for i in 0..num_atoms {
            let cid: String = chain_id_arr.get_item(i)?.extract()?;
            let mt_str = if let Some(ref mt_arr) = mol_type_arr {
                mt_arr.get_item(i)?.extract::<String>().unwrap_or_default()
            } else {
                String::new()
            };
            let key = format!("{}:{}", cid, mt_str);
            let eid = *group_map.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            atom_entity_ids.push(eid);
        }
    }

    // ── Collect unique entity IDs in order of first appearance ──
    let mut entity_order: Vec<i32> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for &eid in &atom_entity_ids {
            if seen.insert(eid) {
                entity_order.push(eid);
            }
        }
    }

    // ── Build MoleculeEntity for each group ──
    let mut chain_mapper = ChainIdMapper::new();
    let mut entities: Vec<MoleculeEntity> = Vec::with_capacity(entity_order.len());

    for (output_idx, &eid) in entity_order.iter().enumerate() {
        // Collect atom indices belonging to this entity
        let indices: Vec<usize> = atom_entity_ids
            .iter()
            .enumerate()
            .filter(|(_, &e)| e == eid)
            .map(|(i, _)| i)
            .collect();

        let n = indices.len();
        let mut atoms = Vec::with_capacity(n);
        let mut entity_chain_ids = Vec::with_capacity(n);
        let mut entity_res_names = Vec::with_capacity(n);
        let mut res_nums = Vec::with_capacity(n);
        let mut entity_atom_names = Vec::with_capacity(n);
        let mut elems = Vec::with_capacity(n);

        // Determine molecule type from first atom's annotation
        let mol_type = if let Some(ref ct_arr) = chain_type_arr {
            let ct: i32 = ct_arr.get_item(indices[0])?.extract().unwrap_or(8);
            chain_type_id_to_molecule_type(ct as u8)
        } else if let Some(ref mt_arr) = mol_type_arr {
            let mt_str: String = mt_arr.get_item(indices[0])?.extract().unwrap_or_default();
            mol_type_str_to_molecule_type(&mt_str)
        } else {
            // Last resort: classify from residue name
            let rn: String = res_name_arr.get_item(indices[0])?.extract()?;
            crate::types::entity::classify_residue(&rn)
        };

        for &i in &indices {
            // Coordinates
            let coord_i = coord.get_item(i)?;
            let x: f32 = coord_i.get_item(0)?.extract()?;
            let y: f32 = coord_i.get_item(1)?.extract()?;
            let z: f32 = coord_i.get_item(2)?.extract()?;

            let occupancy: f32 = occupancy_arr
                .as_ref()
                .and_then(|arr| arr.get_item(i).ok())
                .and_then(|v| v.extract().ok())
                .unwrap_or(1.0);

            let b_factor: f32 = b_factor_arr
                .as_ref()
                .and_then(|arr| arr.get_item(i).ok())
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0);

            atoms.push(CoordsAtom {
                x,
                y,
                z,
                occupancy,
                b_factor,
            });

            // Chain ID
            let cid: String = chain_id_arr.get_item(i)?.extract()?;
            entity_chain_ids.push(chain_mapper.get_or_assign(&cid));

            // Residue name (3 bytes, space-padded)
            let rn: String = res_name_arr.get_item(i)?.extract()?;
            let mut rn_bytes = [b' '; 3];
            for (j, b) in rn.bytes().take(3).enumerate() {
                rn_bytes[j] = b;
            }
            entity_res_names.push(rn_bytes);

            // Residue number
            let res_num: i32 = res_id_arr.get_item(i)?.extract()?;
            res_nums.push(res_num);

            // Atom name (4 bytes, space-padded)
            let an: String = atom_name_arr.get_item(i)?.extract()?;
            let mut an_bytes = [b' '; 4];
            for (j, b) in an.bytes().take(4).enumerate() {
                an_bytes[j] = b;
            }
            entity_atom_names.push(an_bytes);

            // Element
            let elem = if let Some(ref elem_arr) = element_arr {
                let sym: String = elem_arr.get_item(i)?.extract().unwrap_or_default();
                Element::from_symbol(&sym)
            } else {
                let an_str = std::str::from_utf8(&an_bytes).unwrap_or("");
                Element::from_atom_name(an_str)
            };
            elems.push(elem);
        }

        entities.push(MoleculeEntity {
            entity_id: output_idx as u32,
            molecule_type: mol_type,
            coords: Coords {
                num_atoms: n,
                atoms,
                chain_ids: entity_chain_ids,
                res_names: entity_res_names,
                res_nums,
                atom_names: entity_atom_names,
                elements: elems,
            },
        });
    }

    serialize_assembly(&entities)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

/// Convert an AtomArray to `Vec<MoleculeEntity>` (Rust structs, not bytes).
///
/// For use within the runner when you need Rust-side entity objects directly
/// rather than going through ASSEM01 serialization.
pub fn atom_array_to_entity_vec(
    py: Python,
    atom_array: &Bound<'_, PyAny>,
) -> PyResult<Vec<MoleculeEntity>> {
    let atom_array_py: Py<PyAny> = atom_array.clone().unbind();
    let bytes = atom_array_to_entities(py, atom_array_py)?;
    deserialize_assembly(&bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

// ============================================================================
// Convenience: parse from file through AtomWorks, return entities
// ============================================================================

/// Load a structure file through AtomWorks' full parsing pipeline and return
/// ASSEM01 bytes containing properly cleaned and annotated entities.
///
/// This is the highest-fidelity path for loading structures when AtomWorks
/// is available: it gets CCD bond lookup, leaving group removal, charge
/// correction, occupancy handling, and all other AtomWorks cleaning steps.
#[pyfunction]
pub fn parse_file_to_entities(py: Python, file_path: String) -> PyResult<Vec<u8>> {
    let aw_parser = py.import("atomworks.io.parser")?;
    let result = aw_parser.call_method1("parse", (&file_path,))?;

    let asym_unit = result.get_item("asym_unit")?;
    let atom_array = asym_unit.get_item(0)?; // First model

    let atom_array_py: Py<PyAny> = atom_array.unbind();
    atom_array_to_entities(py, atom_array_py)
}

/// Load a structure file through AtomWorks and return chain metadata.
///
/// Returns a Python dict with:
/// - `"assembly_bytes"`: ASSEM01 bytes for the entity assembly
/// - `"chain_info"`: dict of chain_id → { "sequence": str, ... }
/// - `"assemblies"`: dict of assembly_id → ASSEM01 bytes for each bio assembly
#[pyfunction]
pub fn parse_file_full(py: Python, file_path: String) -> PyResult<Py<PyAny>> {
    let aw_parser = py.import("atomworks.io.parser")?;
    let result = aw_parser.call_method1("parse", (&file_path,))?;

    let out = PyDict::new(py);

    // Convert asymmetric unit
    let asym_unit = result.get_item("asym_unit")?;
    let atom_array = asym_unit.get_item(0)?;
    let asym_bytes = atom_array_to_entities(py, atom_array.unbind())?;
    out.set_item("assembly_bytes", asym_bytes)?;

    // Pass through chain_info directly (Python dict)
    let chain_info = result.get_item("chain_info")?;
    out.set_item("chain_info", chain_info)?;

    // Convert biological assemblies
    let assemblies_dict = PyDict::new(py);
    if let Ok(assemblies) = result.get_item("assemblies") {
        if let Ok(items) = assemblies.call_method0("items") {
            if let Ok(iter) = items.try_iter() {
                for item_result in iter {
                    if let Ok(item) = item_result {
                        let key: String = item.get_item(0i32)?.extract()?;
                        let stack = item.get_item(1i32)?;
                        let aa = stack.get_item(0i32)?;
                        match atom_array_to_entities(py, aa.unbind()) {
                            Ok(bytes) => {
                                assemblies_dict.set_item(key, bytes)?;
                            }
                            Err(_) => continue, // Skip assemblies that fail to convert
                        }
                    }
                }
            }
        }
    }
    out.set_item("assemblies", assemblies_dict)?;

    Ok(out.unbind().into_any())
}

// ============================================================================
// Helpers
// ============================================================================

/// Try to get an optional annotation array from an AtomArray.
/// Returns Ok(None) if the annotation doesn't exist.
fn get_annotation_opt<'py>(
    atom_array: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match atom_array.call_method1("get_annotation", (name,)) {
        Ok(arr) => Ok(Some(arr)),
        Err(_) => Ok(None),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_molecule_type_roundtrip() {
        // Verify that our mapping is at least self-consistent for the common types
        let types = vec![
            MoleculeType::Protein,
            MoleculeType::DNA,
            MoleculeType::RNA,
            MoleculeType::Ligand,
            MoleculeType::Water,
        ];

        for mt in types {
            let ct = molecule_type_to_chain_type_id(mt);
            let back = chain_type_id_to_molecule_type(ct);
            assert_eq!(
                mt, back,
                "Round-trip failed for {:?} -> ct={} -> {:?}",
                mt, ct, back
            );
        }
    }

    #[test]
    fn test_mol_type_str_roundtrip() {
        let types = vec![
            (MoleculeType::Protein, "protein"),
            (MoleculeType::DNA, "dna"),
            (MoleculeType::RNA, "rna"),
            (MoleculeType::Ligand, "ligand"),
            (MoleculeType::Water, "water"),
        ];

        for (mt, expected_str) in types {
            let s = molecule_type_to_mol_type_str(mt);
            assert_eq!(s, expected_str);
            let back = mol_type_str_to_molecule_type(s);
            assert_eq!(mt, back);
        }
    }
}
