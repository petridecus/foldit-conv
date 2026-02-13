//! Biotite AtomArray adapter for RC-Foundry models.
//!
//! Used by: RFDiffusion3, RosettaFold3, LigandMPNN, and other models
//! that output Biotite AtomArray structures.

use pyo3::prelude::*;

use crate::types::coords::{deserialize, serialize, ChainIdMapper, Coords, CoordsAtom, Element};

/// Convert COORDS bytes directly to a Biotite AtomArray object.
#[pyfunction]
pub fn coords_to_biotite_atom_array(py: Python, coords_bytes: Vec<u8>) -> PyResult<Py<PyAny>> {
    let coords = deserialize(&coords_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let num_atoms = coords.num_atoms;

    // Import biotite.structure
    let biotite_structure = py.import("biotite.structure")?;
    let atom_array_class = biotite_structure.getattr("AtomArray")?;
    let bond_list_class = biotite_structure.getattr("BondList")?;

    // Import numpy for array creation
    let numpy = py.import("numpy")?;

    // Create AtomArray with correct length
    let atom_array = atom_array_class.call1((num_atoms,))?;

    // Coordinates
    let coords_data: Vec<f32> = coords
        .atoms
        .iter()
        .flat_map(|a| vec![a.x, a.y, a.z])
        .collect();
    let coord_np = numpy.call_method1("array", (coords_data,))?;
    let coord_np = coord_np.call_method1("reshape", ((num_atoms, 3),))?;
    let coord_np = coord_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("coord", coord_np)?;

    // Chain IDs — convert each byte to its ASCII character
    let chain_ids: Vec<String> = coords
        .chain_ids
        .iter()
        .map(|&c| {
            if c.is_ascii_alphanumeric() {
                String::from(c as char)
            } else {
                "A".to_string()
            }
        })
        .collect();
    let chain_np = numpy.call_method1("array", (chain_ids,))?;
    let chain_np = chain_np.call_method1("astype", ("U1",))?;
    atom_array.setattr("chain_id", chain_np)?;

    // Residue IDs
    let res_ids: Vec<i32> = coords.res_nums.clone();
    let res_np = numpy.call_method1("array", (res_ids,))?;
    let res_np = res_np.call_method1("astype", (numpy.getattr("int32")?,))?;
    atom_array.setattr("res_id", res_np)?;

    // Residue names
    let res_names: Vec<String> = coords
        .res_names
        .iter()
        .map(|b| std::str::from_utf8(b).unwrap_or("UNK").trim().to_string())
        .collect();
    let resname_np = numpy.call_method1("array", (res_names,))?;
    let resname_np = resname_np.call_method1("astype", ("U3",))?;
    atom_array.setattr("res_name", resname_np)?;

    // Atom names
    let atom_names: Vec<String> = coords
        .atom_names
        .iter()
        .map(|b| std::str::from_utf8(b).unwrap_or("X").trim().to_string())
        .collect();
    let atomname_np = numpy.call_method1("array", (atom_names.clone(),))?;
    let atomname_np = atomname_np.call_method1("astype", ("U4",))?;
    atom_array.setattr("atom_name", atomname_np)?;

    // Element (derive from atom name - first alphabetic character)
    let elements: Vec<String> = atom_names
        .iter()
        .map(|name| {
            name.chars()
                .find(|c| c.is_alphabetic())
                .map(|c| c.to_string())
                .unwrap_or_else(|| "X".to_string())
        })
        .collect();
    let element_np = numpy.call_method1("array", (elements,))?;
    let element_np = element_np.call_method1("astype", ("U2",))?;
    atom_array.setattr("element", element_np)?;

    // Occupancy — all atoms in COORDS are resolved, so set to 1.0.
    // Without this, biotite defaults to 0.0 and downstream transforms
    // (e.g. MaskResiduesWithSpecificUnresolvedAtoms) treat every atom as
    // unresolved, ultimately producing an empty AtomArray.
    let occupancy_data: Vec<f32> = coords.atoms.iter().map(|a| a.occupancy).collect();
    let occ_np = numpy.call_method1("array", (occupancy_data,))?;
    let occ_np = occ_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("occupancy", occ_np)?;

    // B-factor
    let bfactor_data: Vec<f32> = coords.atoms.iter().map(|a| a.b_factor).collect();
    let bf_np = numpy.call_method1("array", (bfactor_data,))?;
    let bf_np = bf_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("b_factor", bf_np)?;

    // Create empty BondList (required by some biotite functions)
    let bond_list = bond_list_class.call1((num_atoms,))?;
    atom_array.setattr("bonds", bond_list)?;

    Ok(atom_array.unbind())
}

/// Convert a Biotite AtomArray to COORDS bytes.
#[pyfunction]
pub fn atom_array_to_coords(py: Python, atom_array: Py<PyAny>) -> PyResult<Vec<u8>> {
    let atom_array = atom_array.bind(py);

    // Extract arrays from AtomArray
    let coord = atom_array.getattr("coord")?;
    let chain_id = atom_array.getattr("chain_id")?;
    let res_id = atom_array.getattr("res_id")?;
    let res_name = atom_array.getattr("res_name")?;
    let atom_name = atom_array.getattr("atom_name")?;

    // Get array length
    let num_atoms: usize = coord.getattr("shape")?.get_item(0)?.extract()?;

    let mut atoms = Vec::with_capacity(num_atoms);
    let mut chain_ids = Vec::with_capacity(num_atoms);
    let mut res_names = Vec::with_capacity(num_atoms);
    let mut res_nums = Vec::with_capacity(num_atoms);
    let mut atom_names_vec = Vec::with_capacity(num_atoms);
    let mut chain_mapper = ChainIdMapper::new();

    for i in 0..num_atoms {
        // Get coordinates
        let coord_i = coord.get_item(i)?;
        let x: f32 = coord_i.get_item(0)?.extract()?;
        let y: f32 = coord_i.get_item(1)?.extract()?;
        let z: f32 = coord_i.get_item(2)?.extract()?;

        // Get b_factor if available, otherwise default to 0
        let b_factor: f32 = atom_array
            .getattr("b_factor")
            .ok()
            .and_then(|bf| bf.get_item(i).ok())
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.0);

        atoms.push(CoordsAtom {
            x,
            y,
            z,
            occupancy: 1.0,
            b_factor,
        });

        // Chain ID
        let chain: String = chain_id.get_item(i)?.extract()?;
        chain_ids.push(chain_mapper.get_or_assign(&chain));

        // Residue name
        let resname: String = res_name.get_item(i)?.extract()?;
        let mut res_name_bytes = [b' '; 3];
        for (j, b) in resname.bytes().take(3).enumerate() {
            res_name_bytes[j] = b;
        }
        res_names.push(res_name_bytes);

        // Residue number
        let resnum: i32 = res_id.get_item(i)?.extract()?;
        res_nums.push(resnum);

        // Atom name
        let atomname: String = atom_name.get_item(i)?.extract()?;
        let mut atom_name_bytes = [b' '; 4];
        for (j, b) in atomname.bytes().take(4).enumerate() {
            atom_name_bytes[j] = b;
        }
        atom_names_vec.push(atom_name_bytes);
    }

    let elements = atom_names_vec.iter().map(|n| {
        let s = std::str::from_utf8(n).unwrap_or("");
        Element::from_atom_name(s)
    }).collect();

    let coords = Coords {
        num_atoms,
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names: atom_names_vec,
        elements,
    };

    serialize(&coords)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}
