use thiserror::Error;

#[cfg(feature = "python")]
use crate::coords::{serialize_coords, Coords, CoordsAtom};

#[cfg(feature = "python")]
use numpy::{PyArrayMethods, PyUntypedArrayMethods};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Error, Debug)]
pub enum SimpleFoldError {
    #[error("Invalid array shape: {0}")]
    InvalidShape(String),
    #[error("Failed to convert: {0}")]
    ConversionError(String),
}

/// Decode atom name from Boltz int8[4] encoding.
/// Boltz encodes atom names as (ASCII - 32), so we add 32 to decode.
#[cfg(feature = "python")]
fn decode_boltz_atom_name(encoded: &[i8; 4]) -> [u8; 4] {
    let mut decoded = [b' '; 4]; // Default to spaces
    for (i, &b) in encoded.iter().enumerate() {
        if b != 0 {
            // Add 32 to decode from Boltz encoding to ASCII
            decoded[i] = ((b as i16 + 32) as u8).max(32).min(126);
        }
    }
    decoded
}

#[cfg(feature = "python")]
const ATOM37_NAMES: [&str; 37] = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2",
    "OG", "OG1", "OD1", "OD2", "CD", "CD1", "CD2",
    "ND1", "ND2", "OE1", "OE2", "NE", "NE1", "NE2",
    "CE", "CE1", "CE2", "CE3", "NH1", "NH2", "CZ",
    "CZ2", "CZ3", "CH2", "OH", "SG", "SD", "NZ", "X",
];

#[cfg(feature = "python")]
fn aatype_to_resname(aa_type: i32) -> &'static str {
    match aa_type {
        0 => "ALA",
        1 => "ARG",
        2 => "ASN",
        3 => "ASP",
        4 => "CYS",
        5 => "GLN",
        6 => "GLU",
        7 => "GLY",
        8 => "HIS",
        9 => "ILE",
        10 => "LEU",
        11 => "LYS",
        12 => "MET",
        13 => "PHE",
        14 => "PRO",
        15 => "SER",
        16 => "THR",
        17 => "TRP",
        18 => "TYR",
        19 => "VAL",
        _ => "UNK",
    }
}

/// Convert Boltz Structure object directly to COORDS format.
/// This handles the native Boltz Structure with its structured numpy arrays.
#[cfg(feature = "python")]
fn boltz_structure_to_coords(
    structure: &Bound<'_, PyAny>,
    plddt: Option<&Bound<'_, PyAny>>,
) -> Result<Vec<u8>, SimpleFoldError> {
    let py = structure.py();

    // Get atoms structured numpy array
    let atoms_array = structure
        .getattr("atoms")
        .map_err(|e| SimpleFoldError::ConversionError(format!("Failed to get 'atoms': {}", e)))?;

    let n_atoms: usize = atoms_array.len().map_err(|e| {
        SimpleFoldError::ConversionError(format!("Failed to get atoms length: {}", e))
    })?;

    if n_atoms == 0 {
        return Err(SimpleFoldError::ConversionError(
            "Empty Structure".to_string(),
        ));
    }

    // Import numpy for making arrays contiguous
    let numpy = py
        .import("numpy")
        .map_err(|e| SimpleFoldError::ConversionError(format!("Failed to import numpy: {}", e)))?;

    // Extract pLDDT array if provided
    let plddt_vec: Option<Vec<f32>> = if let Some(p) = plddt {
        if let Ok(arr) = p.cast::<numpy::PyArray1<f32>>() {
            let readonly = arr.readonly();
            readonly.as_slice().ok().map(|s| s.to_vec())
        } else {
            None
        }
    } else {
        None
    };

    // Get is_present field to filter atoms
    let is_present_field = atoms_array
        .call_method1("__getitem__", ("is_present",))
        .map_err(|e| {
            SimpleFoldError::ConversionError(format!("Failed to get is_present: {}", e))
        })?;
    let is_present_contiguous = numpy
        .call_method1("ascontiguousarray", (&is_present_field,))
        .map_err(|e| {
            SimpleFoldError::ConversionError(format!("Failed to make is_present contiguous: {}", e))
        })?;
    let is_present_arr = is_present_contiguous
        .cast::<numpy::PyArray1<bool>>()
        .map_err(|_| {
            SimpleFoldError::ConversionError("is_present field must be bool array".to_string())
        })?;
    let is_present_readonly = is_present_arr.readonly();
    let is_present_slice = is_present_readonly.as_slice().map_err(|e| {
        SimpleFoldError::ConversionError(format!("is_present not contiguous: {}", e))
    })?;

    // Count valid atoms
    let valid_count: usize = is_present_slice.iter().filter(|&&v| v).count();
    if valid_count == 0 {
        return Err(SimpleFoldError::ConversionError(
            "No valid atoms in Structure".to_string(),
        ));
    }

    // Get coords field [N, 3]
    let coords_field = atoms_array
        .call_method1("__getitem__", ("coords",))
        .map_err(|e| SimpleFoldError::ConversionError(format!("Failed to get coords: {}", e)))?;
    let coords_contiguous = numpy
        .call_method1("ascontiguousarray", (&coords_field,))
        .map_err(|e| {
            SimpleFoldError::ConversionError(format!("Failed to make coords contiguous: {}", e))
        })?;
    let coords_arr = coords_contiguous
        .cast::<numpy::PyArray2<f32>>()
        .map_err(|_| {
            SimpleFoldError::ConversionError("coords field must be float32 array".to_string())
        })?;
    let coords_readonly = coords_arr.readonly();
    // Verify shape is [N, 3]
    if coords_arr.ndim() != 2 || coords_arr.dims()[1] != 3 {
        return Err(SimpleFoldError::InvalidShape(
            "coords must have shape [N, 3]".to_string(),
        ));
    }
    let coords_slice = coords_readonly
        .as_slice()
        .map_err(|e| SimpleFoldError::ConversionError(format!("coords not contiguous: {}", e)))?;

    // Get atom name field [N, 4] int8 (Boltz encoding: ASCII - 32)
    let name_field = atoms_array
        .call_method1("__getitem__", ("name",))
        .map_err(|e| SimpleFoldError::ConversionError(format!("Failed to get name: {}", e)))?;
    let name_contiguous = numpy
        .call_method1("ascontiguousarray", (&name_field,))
        .map_err(|e| {
            SimpleFoldError::ConversionError(format!("Failed to make name contiguous: {}", e))
        })?;
    let name_arr = name_contiguous.cast::<numpy::PyArray2<i8>>().map_err(|_| {
        SimpleFoldError::ConversionError("name field must be int8 array".to_string())
    })?;
    let name_readonly = name_arr.readonly();
    let name_slice = name_readonly
        .as_slice()
        .map_err(|e| SimpleFoldError::ConversionError(format!("name not contiguous: {}", e)))?;

    // Build COORDS output
    let mut atoms = Vec::with_capacity(valid_count);
    let mut chain_ids = Vec::with_capacity(valid_count);
    let mut res_names = Vec::with_capacity(valid_count);
    let mut res_nums = Vec::with_capacity(valid_count);
    let mut atom_names = Vec::with_capacity(valid_count);

    let mut current_res_idx: i32 = 0;

    for i in 0..n_atoms {
        if !is_present_slice[i] {
            continue;
        }

        // Extract coordinates
        let x = coords_slice[i * 3];
        let y = coords_slice[i * 3 + 1];
        let z = coords_slice[i * 3 + 2];

        // Skip NaN coordinates
        if x.is_nan() || y.is_nan() || z.is_nan() {
            continue;
        }

        // Extract and decode atom name from Boltz encoding
        let encoded_name: [i8; 4] = [
            name_slice[i * 4],
            name_slice[i * 4 + 1],
            name_slice[i * 4 + 2],
            name_slice[i * 4 + 3],
        ];
        let decoded_name = decode_boltz_atom_name(&encoded_name);

        // Get B-factor from pLDDT if available (per-residue)
        let b_factor = plddt_vec
            .as_ref()
            .and_then(|p| p.get(current_res_idx as usize).copied())
            .map(|plddt_val| (1.0 - plddt_val) * 100.0)
            .unwrap_or(0.0);

        atoms.push(CoordsAtom {
            x,
            y,
            z,
            occupancy: 1.0,
            b_factor,
        });

        chain_ids.push(b'A'); // TODO: extract from chains if available

        // Use UNK for residue name (Boltz Structure doesn't provide this directly)
        res_names.push(*b"UNK");

        res_nums.push(current_res_idx + 1);

        atom_names.push(decoded_name);

        // Increment residue index when we see N atom (backbone nitrogen)
        if &decoded_name == b"N   " {
            current_res_idx += 1;
        }
    }

    if atoms.is_empty() {
        return Err(SimpleFoldError::ConversionError(
            "No valid atoms found".to_string(),
        ));
    }

    let coords = Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
    };

    serialize_coords(&coords).map_err(|e| SimpleFoldError::ConversionError(e.to_string()))
}

/// Convert atom37 dict format to COORDS.
/// This handles the dict format with positions, aatype, and atom37_atom_exists fields.
#[cfg(feature = "python")]
fn atom37_dict_to_coords(
    dict: &Bound<'_, pyo3::types::PyDict>,
    plddt: Option<&Bound<'_, PyAny>>,
) -> Result<Vec<u8>, SimpleFoldError> {
    let positions = dict
        .get_item("positions")
        .map_err(|_| SimpleFoldError::ConversionError("Failed to get 'positions'".to_string()))?
        .ok_or_else(|| SimpleFoldError::ConversionError("Missing 'positions' field".to_string()))?;

    let aatype = dict
        .get_item("aatype")
        .map_err(|_| SimpleFoldError::ConversionError("Failed to get 'aatype'".to_string()))?
        .ok_or_else(|| SimpleFoldError::ConversionError("Missing 'aatype' field".to_string()))?;

    let atom_exists = dict
        .get_item("atom37_atom_exists")
        .map_err(|_| {
            SimpleFoldError::ConversionError("Failed to get 'atom37_atom_exists'".to_string())
        })?
        .ok_or_else(|| {
            SimpleFoldError::ConversionError("Missing 'atom37_atom_exists' field".to_string())
        })?;

    let residue_index: Option<Bound<PyAny>> = dict.get_item("residue_index").map_err(|_| {
        SimpleFoldError::ConversionError("Failed to get 'residue_index'".to_string())
    })?;

    let positions_arr = positions.cast::<numpy::PyArray3<f32>>().map_err(|_| {
        SimpleFoldError::ConversionError("'positions' must be a 3D float array".to_string())
    })?;

    let aatype_arr = aatype.cast::<numpy::PyArray1<i32>>().map_err(|_| {
        SimpleFoldError::ConversionError("'aatype' must be a 1D int array".to_string())
    })?;

    let atom_exists_arr = atom_exists.cast::<numpy::PyArray2<bool>>().map_err(|_| {
        SimpleFoldError::ConversionError("'atom37_atom_exists' must be a 2D bool array".to_string())
    })?;

    let positions_view = unsafe { positions_arr.as_array() };
    let aatype_slice = unsafe { aatype_arr.as_slice() }
        .map_err(|e| SimpleFoldError::ConversionError(format!("aatype not contiguous: {}", e)))?;
    let atom_exists_view = unsafe { atom_exists_arr.as_array() };

    if positions_view.ndim() != 3 {
        return Err(SimpleFoldError::InvalidShape(
            "positions must be 3D [L, 37, 3]".to_string(),
        ));
    }

    let (seq_len, num_atoms, coords_dim) = positions_view.dim();
    if num_atoms != 37 || coords_dim != 3 {
        return Err(SimpleFoldError::InvalidShape(
            "positions must have shape [L, 37, 3]".to_string(),
        ));
    }

    let plddt_arr = plddt.and_then(|p| p.cast::<numpy::PyArray1<f32>>().ok().map(|b| b.clone()));
    let residue_index_arr =
        residue_index.and_then(|r| r.cast::<numpy::PyArray1<i32>>().ok().map(|b| b.clone()));

    let mut atoms = Vec::new();
    let mut chain_ids = Vec::new();
    let mut res_names = Vec::new();
    let mut res_nums = Vec::new();
    let mut atom_names = Vec::new();

    for res_idx in 0..seq_len {
        let aa_type = aatype_slice[res_idx];
        let res_num = residue_index_arr
            .as_ref()
            .and_then(|arr| unsafe { arr.get(res_idx).copied() })
            .unwrap_or((res_idx + 1) as i32);

        let res_name = aatype_to_resname(aa_type);

        for atom_idx in 0..37 {
            if !atom_exists_view[[res_idx, atom_idx]] {
                continue;
            }

            let x = positions_view[[res_idx, atom_idx, 0]];
            let y = positions_view[[res_idx, atom_idx, 1]];
            let z = positions_view[[res_idx, atom_idx, 2]];

            if x.is_nan() || y.is_nan() || z.is_nan() {
                continue;
            }

            let b_factor = plddt_arr
                .as_ref()
                .and_then(|arr| unsafe { arr.get(res_idx).copied() })
                .map(|p| (1.0 - p) * 100.0)
                .unwrap_or(0.0);

            atoms.push(CoordsAtom {
                x,
                y,
                z,
                occupancy: 1.0,
                b_factor,
            });

            chain_ids.push(b'A');

            let mut res_name_bytes = [0u8; 3];
            let res_name_str = res_name.as_bytes();
            for (i, b) in res_name_str.iter().take(3).enumerate() {
                res_name_bytes[i] = *b;
            }
            res_names.push(res_name_bytes);

            res_nums.push(res_num);

            let mut atom_name_bytes = [0u8; 4];
            let atom_name_str = ATOM37_NAMES[atom_idx].as_bytes();
            for (i, b) in atom_name_str.iter().take(4).enumerate() {
                atom_name_bytes[i] = *b;
            }
            atom_names.push(atom_name_bytes);
        }
    }

    if atoms.is_empty() {
        return Err(SimpleFoldError::ConversionError(
            "No valid atoms found".to_string(),
        ));
    }

    let coords = Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
    };

    serialize_coords(&coords).map_err(|e| SimpleFoldError::ConversionError(e.to_string()))
}

/// Convert SimpleFold structure to COORDS format.
/// Handles both:
/// 1. Boltz Structure objects (with atoms attribute containing structured numpy array)
/// 2. Dict format with positions, aatype, atom37_atom_exists fields
#[cfg(feature = "python")]
pub fn simplefold_to_coords_internal(
    structure: &Bound<'_, PyAny>,
    plddt: Option<&Bound<'_, PyAny>>,
) -> Result<Vec<u8>, SimpleFoldError> {
    // Check if this is a Boltz Structure object (has 'atoms' attribute)
    if structure.getattr("atoms").is_ok() {
        return boltz_structure_to_coords(structure, plddt);
    }

    // Otherwise, try to parse as dict format
    let dict = structure
        .cast::<pyo3::types::PyDict>()
        .map_err(|_| {
            SimpleFoldError::ConversionError(
                "Structure must be a Boltz Structure object or dict with positions/aatype/atom37_atom_exists".to_string()
            )
        })?;

    atom37_dict_to_coords(&dict, plddt)
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn simplefold_to_coords(
    py: Python,
    structure: Py<PyAny>,
    plddt: Option<Py<PyAny>>,
) -> PyResult<Vec<u8>> {
    simplefold_to_coords_internal(structure.bind(py), plddt.as_ref().map(|p| p.bind(py)))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}
