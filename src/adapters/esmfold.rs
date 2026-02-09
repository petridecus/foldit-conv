use thiserror::Error;

#[cfg(feature = "python")]
use crate::types::coords::{serialize as serialize_coords, Coords, CoordsAtom, Element};

#[cfg(feature = "python")]
use numpy::PyArrayMethods;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Error, Debug)]
pub enum ESMFoldError {
    #[error("Invalid array shape: {0}")]
    InvalidShape(String),
    #[error("Failed to convert: {0}")]
    ConversionError(String),
}

#[cfg(feature = "python")]
const ATOM37_NAMES: [&str; 37] = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OD1", "OD2", "CD", "CD1", "CD2", "ND1",
    "ND2", "OE1", "OE2", "NE", "NE1", "NE2", "CE", "CE1", "CE2", "CE3", "NH1", "NH2", "CZ", "CZ2",
    "CZ3", "CH2", "OH", "SG", "SD", "OD1", "OD2", "NE",
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

/// Convert ESMFold atom14 format directly to COORDS, handling atom14→atom37 conversion internally.
/// This eliminates the need for Python to do the scatter operation.
#[cfg(feature = "python")]
pub fn esmfold_atom14_to_coords_internal(
    positions_atom14: &Bound<'_, PyAny>,
    aatype: &Bound<'_, PyAny>,
    residx_atom14_to_atom37: &Bound<'_, PyAny>,
    atom14_atom_exists: &Bound<'_, PyAny>,
    plddt: Option<&Bound<'_, PyAny>>,
    residue_index: Option<&Bound<'_, PyAny>>,
    batch_idx: usize,
) -> Result<Vec<u8>, ESMFoldError> {
    // positions_atom14: [batch, L, 14, 3]
    // residx_atom14_to_atom37: [batch, L, 14] - maps atom14 index to atom37 index
    // atom14_atom_exists: [batch, L, 14] - mask for which atom14 positions exist

    let positions_arr = positions_atom14.cast::<numpy::PyArray4<f32>>().map_err(|_| {
        ESMFoldError::ConversionError("positions_atom14 must be a 4D float array".to_string())
    })?;

    let aatype_arr = aatype
        .cast::<numpy::PyArray1<i32>>()
        .map_err(|_| ESMFoldError::ConversionError("aatype must be a 1D int array".to_string()))?;

    let mapping_arr = residx_atom14_to_atom37.cast::<numpy::PyArray3<i64>>().map_err(|_| {
        ESMFoldError::ConversionError("residx_atom14_to_atom37 must be a 3D int array".to_string())
    })?;

    let exists_arr = atom14_atom_exists.cast::<numpy::PyArray3<bool>>().map_err(|_| {
        ESMFoldError::ConversionError("atom14_atom_exists must be a 3D bool array".to_string())
    })?;

    let positions_view = unsafe { positions_arr.as_array() };
    let aatype_slice = unsafe { aatype_arr.as_slice() }
        .map_err(|e| ESMFoldError::ConversionError(format!("aatype not contiguous: {}", e)))?;
    let mapping_view = unsafe { mapping_arr.as_array() };
    let exists_view = unsafe { exists_arr.as_array() };

    let (batch, seq_len, num_atom14, coords_dim) = positions_view.dim();
    if num_atom14 != 14 || coords_dim != 3 {
        return Err(ESMFoldError::InvalidShape(
            "positions_atom14 must have shape [batch, L, 14, 3]".to_string(),
        ));
    }

    if batch_idx >= batch {
        return Err(ESMFoldError::InvalidShape("batch_idx out of range".to_string()));
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

        // Iterate over atom14 positions and scatter to atom37
        for a14 in 0..14 {
            if !exists_view[[batch_idx, res_idx, a14]] {
                continue;
            }

            let a37 = mapping_view[[batch_idx, res_idx, a14]] as usize;
            if a37 >= 37 {
                continue;
            }

            let x = positions_view[[batch_idx, res_idx, a14, 0]];
            let y = positions_view[[batch_idx, res_idx, a14, 1]];
            let z = positions_view[[batch_idx, res_idx, a14, 2]];

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
            let atom_name_str = ATOM37_NAMES[a37].as_bytes();
            for (i, b) in atom_name_str.iter().take(4).enumerate() {
                atom_name_bytes[i] = *b;
            }
            atom_names.push(atom_name_bytes);
        }
    }

    if atoms.is_empty() {
        return Err(ESMFoldError::ConversionError("No valid atoms found".to_string()));
    }

    let elements = atom_names.iter().map(|n| {
        let s = std::str::from_utf8(n).unwrap_or("");
        Element::from_atom_name(s)
    }).collect();

    let coords = Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    };

    serialize_coords(&coords).map_err(|e| ESMFoldError::ConversionError(e.to_string()))
}

/// Python-callable function to convert ESMFold atom14 format directly to COORDS.
#[cfg(feature = "python")]
#[pyfunction]
pub fn esmfold_atom14_to_coords(
    py: Python,
    positions_atom14: Py<PyAny>,
    aatype: Py<PyAny>,
    residx_atom14_to_atom37: Py<PyAny>,
    atom14_atom_exists: Py<PyAny>,
    plddt: Option<Py<PyAny>>,
    residue_index: Option<Py<PyAny>>,
    batch_idx: usize,
) -> PyResult<Vec<u8>> {
    esmfold_atom14_to_coords_internal(
        positions_atom14.bind(py),
        aatype.bind(py),
        residx_atom14_to_atom37.bind(py),
        atom14_atom_exists.bind(py),
        plddt.as_ref().map(|p| p.bind(py)),
        residue_index.as_ref().map(|r| r.bind(py)),
        batch_idx,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[cfg(feature = "python")]
pub fn esmfold_to_coords_internal(
    positions: &Bound<'_, PyAny>,
    aatype: &Bound<'_, PyAny>,
    atom37_atom_exists: &Bound<'_, PyAny>,
    plddt: Option<&Bound<'_, PyAny>>,
    residue_index: Option<&Bound<'_, PyAny>>,
    batch_idx: usize,
) -> Result<Vec<u8>, ESMFoldError> {
    let positions_arr = positions.cast::<numpy::PyArray4<f32>>().map_err(|_| {
        ESMFoldError::ConversionError("positions must be a 4D float array".to_string())
    })?;

    let aatype_arr = aatype
        .cast::<numpy::PyArray1<i32>>()
        .map_err(|_| ESMFoldError::ConversionError("aatype must be a 1D int array".to_string()))?;

    let atom_exists_arr = atom37_atom_exists
        .cast::<numpy::PyArray4<bool>>()
        .map_err(|_| {
            ESMFoldError::ConversionError("atom37_atom_exists must be a 4D bool array".to_string())
        })?;

    let positions_view = unsafe { positions_arr.as_array() };
    let aatype_slice = unsafe { aatype_arr.as_slice() }
        .map_err(|e| ESMFoldError::ConversionError(format!("aatype not contiguous: {}", e)))?;
    let atom_exists_view = unsafe { atom_exists_arr.as_array() };

    let (batch, seq_len, num_atoms, coords_dim) = positions_view.dim();
    if num_atoms != 37 || coords_dim != 3 {
        return Err(ESMFoldError::InvalidShape(
            "positions must have shape [batch, L, 37, 3]".to_string(),
        ));
    }

    if batch_idx >= batch {
        return Err(ESMFoldError::InvalidShape(
            "batch_idx out of range".to_string(),
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
            if !atom_exists_view[[batch_idx, res_idx, atom_idx, 0]] {
                continue;
            }

            let x = positions_view[[batch_idx, res_idx, atom_idx, 0]];
            let y = positions_view[[batch_idx, res_idx, atom_idx, 1]];
            let z = positions_view[[batch_idx, res_idx, atom_idx, 2]];

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
        return Err(ESMFoldError::ConversionError(
            "No valid atoms found".to_string(),
        ));
    }

    let elements = atom_names.iter().map(|n| {
        let s = std::str::from_utf8(n).unwrap_or("");
        Element::from_atom_name(s)
    }).collect();

    let coords = Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    };

    serialize_coords(&coords).map_err(|e| ESMFoldError::ConversionError(e.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn esmfold_to_coords(
    py: Python,
    positions: Py<PyAny>,
    aatype: Py<PyAny>,
    atom37_atom_exists: Py<PyAny>,
    plddt: Option<Py<PyAny>>,
    residue_index: Option<Py<PyAny>>,
    batch_idx: usize,
) -> PyResult<Vec<u8>> {
    esmfold_to_coords_internal(
        positions.bind(py),
        aatype.bind(py),
        atom37_atom_exists.bind(py),
        plddt.as_ref().map(|p| p.bind(py)),
        residue_index.as_ref().map(|r| r.bind(py)),
        batch_idx,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}
