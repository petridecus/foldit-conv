//! Coordinate transformation utilities.
//!
//! Provides functions for extracting, filtering, and aligning coordinates:
//! - Backbone/CA extraction
//! - Atom filtering
//! - Kabsch alignment

use crate::types::coords::{
    deserialize, deserialize_assembly, serialize, serialize_assembly, Coords, CoordsAtom,
    CoordsError, Element, ASSEMBLY_MAGIC,
};
use glam::{Mat3, Vec3};

/// Extract backbone chains from COORDS data.
/// Returns a vector of chains, where each chain is a sequence of N-CA-C positions.
/// Chain breaks are detected by chain ID change or residue number gap.
pub fn extract_backbone_chains(coords: &Coords) -> Vec<Vec<Vec3>> {
    let mut chains: Vec<Vec<Vec3>> = Vec::new();
    let mut current_chain: Vec<Vec3> = Vec::new();
    let mut last_chain_id: Option<u8> = None;
    let mut last_res_num: Option<i32> = None;

    for i in 0..coords.num_atoms {
        let atom_name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();

        // Only include N, CA, C for backbone spline (skip O and sidechains)
        if atom_name != "N" && atom_name != "CA" && atom_name != "C" {
            continue;
        }

        let chain_id = coords.chain_ids[i];
        let res_num = coords.res_nums[i];
        let pos = Vec3::new(coords.atoms[i].x, coords.atoms[i].y, coords.atoms[i].z);

        let is_chain_break = last_chain_id.map_or(false, |c| c != chain_id);
        let is_sequence_gap = last_res_num.map_or(false, |r| (res_num - r).abs() > 1);

        if (is_chain_break || is_sequence_gap) && !current_chain.is_empty() {
            chains.push(std::mem::take(&mut current_chain));
        }

        current_chain.push(pos);
        last_chain_id = Some(chain_id);

        if atom_name == "CA" {
            last_res_num = Some(res_num);
        }
    }

    if !current_chain.is_empty() {
        chains.push(current_chain);
    }

    chains
}

/// Extract CA positions from COORDS data.
pub fn extract_ca_positions(coords: &Coords) -> Vec<Vec3> {
    let mut ca_positions = Vec::new();
    for i in 0..coords.num_atoms {
        let atom_name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();
        if atom_name == "CA" {
            ca_positions.push(Vec3::new(
                coords.atoms[i].x,
                coords.atoms[i].y,
                coords.atoms[i].z,
            ));
        }
    }
    ca_positions
}

/// Extract CA positions from backbone chains (every 2nd element in N-CA-C pattern).
pub fn extract_ca_from_chains(chains: &[Vec<Vec3>]) -> Vec<Vec3> {
    let mut ca_positions = Vec::new();
    for chain in chains {
        // Backbone chains are N-CA-C pattern, so CA is every 3rd atom starting at index 1
        for (i, pos) in chain.iter().enumerate() {
            if i % 3 == 1 {
                ca_positions.push(*pos);
            }
        }
    }
    ca_positions
}

/// Get a single CA position by residue index from backbone chains.
/// Returns None if residue_idx is out of bounds.
pub fn get_ca_position_from_chains(chains: &[Vec<Vec3>], residue_idx: usize) -> Option<Vec3> {
    let mut current_idx = 0;
    for chain in chains {
        let residues_in_chain = chain.len() / 3;
        if residue_idx < current_idx + residues_in_chain {
            let local_idx = residue_idx - current_idx;
            let ca_idx = local_idx * 3 + 1; // CA is at index 1 in (N, CA, C)
            return chain.get(ca_idx).copied();
        }
        current_idx += residues_in_chain;
    }
    None
}

/// Get all backbone atom positions (N, CA, C) for a residue by index.
/// Returns None if residue_idx is out of bounds.
pub fn get_backbone_atoms_from_chains(
    chains: &[Vec<Vec3>],
    residue_idx: usize,
) -> Option<(Vec3, Vec3, Vec3)> {
    let mut current_idx = 0;
    for chain in chains {
        let residues_in_chain = chain.len() / 3;
        if residue_idx < current_idx + residues_in_chain {
            let local_idx = residue_idx - current_idx;
            let base_idx = local_idx * 3;
            let n = chain.get(base_idx).copied()?;
            let ca = chain.get(base_idx + 1).copied()?;
            let c = chain.get(base_idx + 2).copied()?;
            return Some((n, ca, c));
        }
        current_idx += residues_in_chain;
    }
    None
}

/// Get the closest backbone atom position to a reference point for a residue.
/// Returns the position of N, CA, or C - whichever is closest to `reference_point`.
/// Returns None if residue_idx is out of bounds.
pub fn get_closest_backbone_atom(
    chains: &[Vec<Vec3>],
    residue_idx: usize,
    reference_point: Vec3,
) -> Option<Vec3> {
    let (n, ca, c) = get_backbone_atoms_from_chains(chains, residue_idx)?;

    let dist_n = n.distance_squared(reference_point);
    let dist_ca = ca.distance_squared(reference_point);
    let dist_c = c.distance_squared(reference_point);

    if dist_n <= dist_ca && dist_n <= dist_c {
        Some(n)
    } else if dist_ca <= dist_c {
        Some(ca)
    } else {
        Some(c)
    }
}

/// Get the closest atom (backbone or sidechain) to a reference point for a residue.
pub fn get_closest_atom_for_residue(
    chains: &[Vec<Vec3>],
    sidechain_positions: &[Vec3],
    sidechain_residue_indices: &[u32],
    residue_idx: usize,
    reference_point: Vec3,
) -> Option<Vec3> {
    let mut closest: Option<(Vec3, f32)> = None;

    // Check backbone atoms (N, CA, C)
    if let Some((n, ca, c)) = get_backbone_atoms_from_chains(chains, residue_idx) {
        for pos in [n, ca, c] {
            let dist = pos.distance_squared(reference_point);
            if closest.is_none() || dist < closest.unwrap().1 {
                closest = Some((pos, dist));
            }
        }
    }

    // Check sidechain atoms for this residue
    for (i, &res_idx) in sidechain_residue_indices.iter().enumerate() {
        if res_idx as usize == residue_idx {
            if let Some(&pos) = sidechain_positions.get(i) {
                let dist = pos.distance_squared(reference_point);
                if closest.is_none() || dist < closest.unwrap().1 {
                    closest = Some((pos, dist));
                }
            }
        }
    }

    closest.map(|(pos, _)| pos)
}

/// Find the closest atom to a reference point within a residue, returning both position and atom name.
pub fn get_closest_atom_with_name(
    chains: &[Vec<Vec3>],
    sidechain_positions: &[Vec3],
    sidechain_residue_indices: &[u32],
    sidechain_atom_names: &[String],
    residue_idx: usize,
    reference_point: Vec3,
) -> Option<(Vec3, String)> {
    let mut closest: Option<(Vec3, String, f32)> = None;

    // Check backbone atoms (N, CA, C)
    if let Some((n, ca, c)) = get_backbone_atoms_from_chains(chains, residue_idx) {
        for (pos, name) in [(n, "N"), (ca, "CA"), (c, "C")] {
            let dist = pos.distance_squared(reference_point);
            if closest.is_none() || dist < closest.as_ref().unwrap().2 {
                closest = Some((pos, name.to_string(), dist));
            }
        }
    }

    // Check sidechain atoms for this residue
    for (i, &res_idx) in sidechain_residue_indices.iter().enumerate() {
        if res_idx as usize == residue_idx {
            if let (Some(&pos), Some(name)) =
                (sidechain_positions.get(i), sidechain_atom_names.get(i))
            {
                let dist = pos.distance_squared(reference_point);
                if closest.is_none() || dist < closest.as_ref().unwrap().2 {
                    closest = Some((pos, name.clone(), dist));
                }
            }
        }
    }

    closest.map(|(pos, name, _)| (pos, name))
}

/// Filter COORDS to only heavy atoms (exclude hydrogens).
pub fn heavy_atoms_only(coords: &Coords) -> Coords {
    filter_atoms(coords, |name| {
        let name_str = std::str::from_utf8(name).unwrap_or("").trim();
        !name_str.starts_with('H')
            && !name_str.starts_with("1H")
            && !name_str.starts_with("2H")
            && !name_str.starts_with("3H")
    })
}

/// Filter COORDS to only backbone atoms (N, CA, C, O).
pub fn backbone_only(coords: &Coords) -> Coords {
    filter_atoms(coords, |name| {
        let name_str = std::str::from_utf8(name).unwrap_or("").trim();
        matches!(name_str, "N" | "CA" | "C" | "O")
    })
}

/// Filter atoms by predicate on atom name.
pub fn filter_atoms(coords: &Coords, predicate: impl Fn(&[u8; 4]) -> bool) -> Coords {
    let mut atoms = Vec::new();
    let mut chain_ids = Vec::new();
    let mut res_names = Vec::new();
    let mut res_nums = Vec::new();
    let mut atom_names = Vec::new();
    let mut elements = Vec::new();

    for i in 0..coords.num_atoms {
        if predicate(&coords.atom_names[i]) {
            atoms.push(coords.atoms[i].clone());
            chain_ids.push(coords.chain_ids[i]);
            res_names.push(coords.res_names[i]);
            res_nums.push(coords.res_nums[i]);
            atom_names.push(coords.atom_names[i]);
            elements.push(coords.elements.get(i).copied().unwrap_or(Element::Unknown));
        }
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

/// Filter atoms by predicate on residue name.
pub fn filter_residues(coords: &Coords, predicate: impl Fn(&[u8; 3]) -> bool) -> Coords {
    let mut atoms = Vec::new();
    let mut chain_ids = Vec::new();
    let mut res_names = Vec::new();
    let mut res_nums = Vec::new();
    let mut atom_names = Vec::new();
    let mut elements = Vec::new();

    for i in 0..coords.num_atoms {
        if predicate(&coords.res_names[i]) {
            atoms.push(coords.atoms[i].clone());
            chain_ids.push(coords.chain_ids[i]);
            res_names.push(coords.res_names[i]);
            res_nums.push(coords.res_nums[i]);
            atom_names.push(coords.atom_names[i]);
            elements.push(coords.elements.get(i).copied().unwrap_or(Element::Unknown));
        }
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

/// Standard amino acid residue names
pub const PROTEIN_RESIDUES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", // Non-standard but protein-like
    "MSE", "SEC", "PYL",
];

/// Filter COORDS to only protein residues (remove water, ligands, etc.).
pub fn protein_only(coords: &Coords) -> Coords {
    filter_residues(coords, |res_name| {
        let name_str = std::str::from_utf8(res_name).unwrap_or("").trim();
        PROTEIN_RESIDUES.contains(&name_str)
    })
}

/// Compute centroid of a point set.
pub fn centroid(points: &[Vec3]) -> Vec3 {
    if points.is_empty() {
        return Vec3::ZERO;
    }
    let sum: Vec3 = points.iter().copied().sum();
    sum / points.len() as f32
}

/// Kabsch algorithm: find optimal rotation and translation to align target to reference.
/// Returns (rotation_matrix, translation) such that: aligned = rotation * target + translation
pub fn kabsch_alignment(reference: &[Vec3], target: &[Vec3]) -> Option<(Mat3, Vec3)> {
    if reference.len() != target.len() || reference.len() < 3 {
        return None;
    }

    let ref_centroid = centroid(reference);
    let tgt_centroid = centroid(target);

    let ref_centered: Vec<Vec3> = reference.iter().map(|p| *p - ref_centroid).collect();
    let tgt_centered: Vec<Vec3> = target.iter().map(|p| *p - tgt_centroid).collect();

    let mut h = [[0.0f32; 3]; 3];
    for k in 0..reference.len() {
        let t = tgt_centered[k];
        let r = ref_centered[k];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += t[i] * r[j];
            }
        }
    }

    let (u, _s, v) = svd_3x3(h);

    let u_mat = Mat3::from_cols(
        Vec3::new(u[0][0], u[1][0], u[2][0]),
        Vec3::new(u[0][1], u[1][1], u[2][1]),
        Vec3::new(u[0][2], u[1][2], u[2][2]),
    );
    let v_mat = Mat3::from_cols(
        Vec3::new(v[0][0], v[1][0], v[2][0]),
        Vec3::new(v[0][1], v[1][1], v[2][1]),
        Vec3::new(v[0][2], v[1][2], v[2][2]),
    );

    let mut rotation = v_mat * u_mat.transpose();

    if rotation.determinant() < 0.0 {
        let v_flipped = Mat3::from_cols(v_mat.col(0), v_mat.col(1), -v_mat.col(2));
        rotation = v_flipped * u_mat.transpose();
    }

    let translation = ref_centroid - rotation * tgt_centroid;

    Some((rotation, translation))
}

/// Kabsch-Umeyama algorithm: find optimal rotation, translation, AND scale.
pub fn kabsch_alignment_with_scale(
    reference: &[Vec3],
    target: &[Vec3],
) -> Option<(Mat3, Vec3, f32)> {
    if reference.len() != target.len() || reference.len() < 3 {
        return None;
    }

    let ref_centroid = centroid(reference);
    let tgt_centroid = centroid(target);

    let ref_centered: Vec<Vec3> = reference.iter().map(|p| *p - ref_centroid).collect();
    let tgt_centered: Vec<Vec3> = target.iter().map(|p| *p - tgt_centroid).collect();

    let _ref_var: f32 =
        ref_centered.iter().map(|p| p.length_squared()).sum::<f32>() / reference.len() as f32;
    let tgt_var: f32 =
        tgt_centered.iter().map(|p| p.length_squared()).sum::<f32>() / target.len() as f32;

    if tgt_var < 1e-10 {
        return None;
    }

    let mut h = [[0.0f32; 3]; 3];
    for k in 0..reference.len() {
        let t = tgt_centered[k];
        let r = ref_centered[k];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += t[i] * r[j];
            }
        }
    }

    let (u, s, v) = svd_3x3(h);

    let u_mat = Mat3::from_cols(
        Vec3::new(u[0][0], u[1][0], u[2][0]),
        Vec3::new(u[0][1], u[1][1], u[2][1]),
        Vec3::new(u[0][2], u[1][2], u[2][2]),
    );
    let v_mat = Mat3::from_cols(
        Vec3::new(v[0][0], v[1][0], v[2][0]),
        Vec3::new(v[0][1], v[1][1], v[2][1]),
        Vec3::new(v[0][2], v[1][2], v[2][2]),
    );

    let mut rotation = v_mat * u_mat.transpose();
    let mut sign = 1.0f32;

    if rotation.determinant() < 0.0 {
        let v_flipped = Mat3::from_cols(v_mat.col(0), v_mat.col(1), -v_mat.col(2));
        rotation = v_flipped * u_mat.transpose();
        sign = -1.0;
    }

    let trace_sd = s[0] + s[1] + sign * s[2];
    let scale = trace_sd / (tgt_var * reference.len() as f32);
    let scale = scale.clamp(0.1, 10.0);

    let translation = ref_centroid - scale * (rotation * tgt_centroid);

    Some((rotation, translation, scale))
}

/// Apply transformation to all atoms in COORDS.
pub fn transform_coords(coords: &mut Coords, rotation: Mat3, translation: Vec3) {
    for atom in &mut coords.atoms {
        let pos = Vec3::new(atom.x, atom.y, atom.z);
        let transformed = rotation * pos + translation;
        atom.x = transformed.x;
        atom.y = transformed.y;
        atom.z = transformed.z;
    }
}

/// Apply transformation with scale to all atoms in COORDS.
pub fn transform_coords_with_scale(
    coords: &mut Coords,
    rotation: Mat3,
    translation: Vec3,
    scale: f32,
) {
    for atom in &mut coords.atoms {
        let pos = Vec3::new(atom.x, atom.y, atom.z);
        let transformed = rotation * (pos * scale) + translation;
        atom.x = transformed.x;
        atom.y = transformed.y;
        atom.z = transformed.z;
    }
}

/// Align COORDS to match reference CA positions using Kabsch algorithm.
pub fn align_to_reference(coords: &mut Coords, reference_ca: &[Vec3]) -> Result<(), CoordsError> {
    let predicted_ca = extract_ca_positions(coords);

    if predicted_ca.len() != reference_ca.len() {
        return Err(CoordsError::InvalidFormat(format!(
            "CA count mismatch: reference={}, coords={}",
            reference_ca.len(),
            predicted_ca.len()
        )));
    }

    let (rotation, translation) = kabsch_alignment(reference_ca, &predicted_ca)
        .ok_or_else(|| CoordsError::InvalidFormat("Kabsch alignment failed".to_string()))?;

    transform_coords(coords, rotation, translation);
    Ok(())
}

/// Align coordinate bytes to match reference CA positions.
/// Supports both COORDS and ASSEM01 formats — detects automatically.
/// Returns new aligned bytes in the same format as the input.
pub fn align_coords_bytes(
    coords_bytes: &[u8],
    reference_ca: &[Vec3],
) -> Result<Vec<u8>, CoordsError> {
    if coords_bytes.len() >= 8 && &coords_bytes[0..8] == ASSEMBLY_MAGIC {
        align_assembly_bytes(coords_bytes, reference_ca)
    } else {
        let mut coords = deserialize(coords_bytes)?;
        align_to_reference(&mut coords, reference_ca)?;
        serialize(&coords)
    }
}

/// Align ASSEM01 bytes to match reference CA positions.
/// Merges entities to extract CA, computes Kabsch, applies to each entity,
/// then re-serializes as ASSEM01.
fn align_assembly_bytes(
    coords_bytes: &[u8],
    reference_ca: &[Vec3],
) -> Result<Vec<u8>, CoordsError> {
    let mut entities = deserialize_assembly(coords_bytes)?;

    // Merge all entities into flat coords to extract CA positions
    let merged = crate::types::entity::merge_entities(&entities);
    let predicted_ca = extract_ca_positions(&merged);

    if predicted_ca.len() != reference_ca.len() {
        return Err(CoordsError::InvalidFormat(format!(
            "CA count mismatch: reference={}, assembly={}",
            reference_ca.len(),
            predicted_ca.len()
        )));
    }

    let (rotation, translation) = kabsch_alignment(reference_ca, &predicted_ca)
        .ok_or_else(|| CoordsError::InvalidFormat("Kabsch alignment failed".to_string()))?;

    // Apply same transform to every entity
    for entity in &mut entities {
        transform_coords(&mut entity.coords, rotation, translation);
    }

    serialize_assembly(&entities)
}

// ============================================================================
// SVD Implementation (Jacobi iteration for 3x3 matrices)
// ============================================================================

fn svd_3x3(a: [[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3], [[f32; 3]; 3]) {
    let mut ata = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    let (eigenvalues, v) = jacobi_eigendecomposition(ata);

    let s = [
        eigenvalues[0].max(0.0).sqrt(),
        eigenvalues[1].max(0.0).sqrt(),
        eigenvalues[2].max(0.0).sqrt(),
    ];

    let mut u = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            if s[j] > 1e-10 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += a[i][k] * v[k][j];
                }
                u[i][j] = sum / s[j];
            }
        }
    }

    orthonormalize(&mut u);

    (u, s, v)
}

fn jacobi_eigendecomposition(mut a: [[f32; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut v = [[0.0f32; 3]; 3];
    for i in 0..3 {
        v[i][i] = 1.0;
    }

    const MAX_ITER: usize = 50;
    for _ in 0..MAX_ITER {
        let mut max_val = 0.0f32;
        let mut p = 0;
        let mut q = 1;
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        let diff = a[q][q] - a[p][p];
        let theta = if diff.abs() < 1e-10 {
            std::f32::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / diff).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        let mut new_a = a;
        new_a[p][p] = c * c * a[p][p] - 2.0 * s * c * a[p][q] + s * s * a[q][q];
        new_a[q][q] = s * s * a[p][p] + 2.0 * s * c * a[p][q] + c * c * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;

        for i in 0..3 {
            if i != p && i != q {
                new_a[i][p] = c * a[i][p] - s * a[i][q];
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = s * a[i][p] + c * a[i][q];
                new_a[q][i] = new_a[i][q];
            }
        }
        a = new_a;

        for i in 0..3 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }

    let eigenvalues = [a[0][0], a[1][1], a[2][2]];

    let mut indices = [0usize, 1, 2];
    indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

    let sorted_eigenvalues = [
        eigenvalues[indices[0]],
        eigenvalues[indices[1]],
        eigenvalues[indices[2]],
    ];

    let mut sorted_v = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            sorted_v[i][j] = v[i][indices[j]];
        }
    }

    (sorted_eigenvalues, sorted_v)
}

fn orthonormalize(m: &mut [[f32; 3]; 3]) {
    let mut norm = 0.0f32;
    for i in 0..3 {
        norm += m[i][0] * m[i][0];
    }
    norm = norm.sqrt();
    if norm > 1e-10 {
        for i in 0..3 {
            m[i][0] /= norm;
        }
    }

    let mut dot = 0.0f32;
    for i in 0..3 {
        dot += m[i][1] * m[i][0];
    }
    for i in 0..3 {
        m[i][1] -= dot * m[i][0];
    }
    norm = 0.0;
    for i in 0..3 {
        norm += m[i][1] * m[i][1];
    }
    norm = norm.sqrt();
    if norm > 1e-10 {
        for i in 0..3 {
            m[i][1] /= norm;
        }
    }

    dot = 0.0;
    for i in 0..3 {
        dot += m[i][2] * m[i][0];
    }
    for i in 0..3 {
        m[i][2] -= dot * m[i][0];
    }
    dot = 0.0;
    for i in 0..3 {
        dot += m[i][2] * m[i][1];
    }
    for i in 0..3 {
        m[i][2] -= dot * m[i][1];
    }
    norm = 0.0;
    for i in 0..3 {
        norm += m[i][2] * m[i][2];
    }
    norm = norm.sqrt();
    if norm > 1e-10 {
        for i in 0..3 {
            m[i][2] /= norm;
        }
    }
}

/// Linear interpolation between two Coords instances.
pub fn interpolate_coords(start: &Coords, end: &Coords, t: f32) -> Option<Coords> {
    if start.num_atoms != end.num_atoms {
        return None;
    }

    let t = t.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;

    let atoms = start
        .atoms
        .iter()
        .zip(end.atoms.iter())
        .map(|(s, e)| CoordsAtom {
            x: s.x * one_minus_t + e.x * t,
            y: s.y * one_minus_t + e.y * t,
            z: s.z * one_minus_t + e.z * t,
            occupancy: s.occupancy * one_minus_t + e.occupancy * t,
            b_factor: s.b_factor * one_minus_t + e.b_factor * t,
        })
        .collect();

    Some(Coords {
        num_atoms: start.num_atoms,
        atoms,
        chain_ids: start.chain_ids.clone(),
        res_names: start.res_names.clone(),
        res_nums: start.res_nums.clone(),
        atom_names: start.atom_names.clone(),
        elements: start.elements.clone(),
    })
}

/// Interpolate Coords with a collapse/expand effect through a collapse point.
pub fn interpolate_coords_collapse<F>(
    start: &Coords,
    end: &Coords,
    t: f32,
    collapse_fn: F,
) -> Option<Coords>
where
    F: Fn(i32, u8) -> Vec3,
{
    if start.num_atoms != end.num_atoms {
        return None;
    }

    let t = t.clamp(0.0, 1.0);

    let atoms = start
        .atoms
        .iter()
        .zip(end.atoms.iter())
        .enumerate()
        .map(|(i, (s, e))| {
            let start_pos = Vec3::new(s.x, s.y, s.z);
            let end_pos = Vec3::new(e.x, e.y, e.z);
            let collapse_point = collapse_fn(start.res_nums[i], start.chain_ids[i]);

            let interpolated = if t < 0.5 {
                let phase_t = t * 2.0;
                start_pos.lerp(collapse_point, phase_t)
            } else {
                let phase_t = (t - 0.5) * 2.0;
                collapse_point.lerp(end_pos, phase_t)
            };

            CoordsAtom {
                x: interpolated.x,
                y: interpolated.y,
                z: interpolated.z,
                occupancy: s.occupancy * (1.0 - t) + e.occupancy * t,
                b_factor: s.b_factor * (1.0 - t) + e.b_factor * t,
            }
        })
        .collect();

    Some(Coords {
        num_atoms: start.num_atoms,
        atoms,
        chain_ids: start.chain_ids.clone(),
        res_names: start.res_names.clone(),
        res_nums: start.res_nums.clone(),
        atom_names: start.atom_names.clone(),
        elements: start.elements.clone(),
    })
}

/// Get atom position by index.
pub fn get_atom_position(coords: &Coords, index: usize) -> Option<Vec3> {
    coords.atoms.get(index).map(|a| Vec3::new(a.x, a.y, a.z))
}

/// Set atom position by index.
pub fn set_atom_position(coords: &mut Coords, index: usize, pos: Vec3) {
    if let Some(atom) = coords.atoms.get_mut(index) {
        atom.x = pos.x;
        atom.y = pos.y;
        atom.z = pos.z;
    }
}

/// Get position of a specific atom by residue number, chain ID, and atom name.
pub fn get_atom_by_name(
    coords: &Coords,
    res_num: i32,
    chain_id: u8,
    atom_name: &str,
) -> Option<Vec3> {
    for i in 0..coords.num_atoms {
        if coords.res_nums[i] == res_num && coords.chain_ids[i] == chain_id {
            let name = std::str::from_utf8(&coords.atom_names[i])
                .unwrap_or("")
                .trim();
            if name == atom_name {
                return Some(Vec3::new(
                    coords.atoms[i].x,
                    coords.atoms[i].y,
                    coords.atoms[i].z,
                ));
            }
        }
    }
    None
}

/// Get CA position for a specific residue by residue number and chain ID.
pub fn get_ca_for_residue(coords: &Coords, res_num: i32, chain_id: u8) -> Option<Vec3> {
    get_atom_by_name(coords, res_num, chain_id, "CA")
}

/// Build a map of (chain_id, res_num) -> CA position for efficient lookup.
pub fn build_ca_position_map(coords: &Coords) -> std::collections::HashMap<(u8, i32), Vec3> {
    let mut map = std::collections::HashMap::new();
    for i in 0..coords.num_atoms {
        let name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();
        if name == "CA" {
            let key = (coords.chain_ids[i], coords.res_nums[i]);
            map.insert(
                key,
                Vec3::new(coords.atoms[i].x, coords.atoms[i].y, coords.atoms[i].z),
            );
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_coords() {
        let start = Coords {
            num_atoms: 2,
            atoms: vec![
                CoordsAtom {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    occupancy: 1.0,
                    b_factor: 0.0,
                },
                CoordsAtom {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    occupancy: 1.0,
                    b_factor: 0.0,
                },
            ],
            chain_ids: vec![b'A', b'A'],
            res_names: vec![*b"ALA", *b"ALA"],
            res_nums: vec![1, 1],
            atom_names: vec![*b"N   ", *b"CA  "],
            elements: vec![Element::Unknown; 2],
        };

        let end = Coords {
            num_atoms: 2,
            atoms: vec![
                CoordsAtom {
                    x: 0.0,
                    y: 10.0,
                    z: 0.0,
                    occupancy: 1.0,
                    b_factor: 0.0,
                },
                CoordsAtom {
                    x: 1.0,
                    y: 10.0,
                    z: 0.0,
                    occupancy: 1.0,
                    b_factor: 0.0,
                },
            ],
            chain_ids: vec![b'A', b'A'],
            res_names: vec![*b"ALA", *b"ALA"],
            res_nums: vec![1, 1],
            atom_names: vec![*b"N   ", *b"CA  "],
            elements: vec![Element::Unknown; 2],
        };

        let mid = interpolate_coords(&start, &end, 0.5).unwrap();
        assert!((mid.atoms[0].y - 5.0).abs() < 0.001);
        assert!((mid.atoms[1].y - 5.0).abs() < 0.001);

        let at_start = interpolate_coords(&start, &end, 0.0).unwrap();
        assert!((at_start.atoms[0].y - 0.0).abs() < 0.001);

        let at_end = interpolate_coords(&start, &end, 1.0).unwrap();
        assert!((at_end.atoms[0].y - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_centroid() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let c = centroid(&points);
        assert!((c.x - 0.667).abs() < 0.01);
        assert!((c.y - 0.667).abs() < 0.01);
        assert!(c.z.abs() < 0.01);
    }

    #[test]
    fn test_kabsch_identity() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let (rotation, translation) = kabsch_alignment(&points, &points).unwrap();
        assert!((rotation.determinant() - 1.0).abs() < 0.01);
        assert!(translation.length() < 0.01);
    }
}
