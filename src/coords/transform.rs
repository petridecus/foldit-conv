//! Coordinate transformation utilities.
//!
//! Provides functions for extracting, filtering, and aligning coordinates:
//! - Backbone/CA extraction
//! - Atom filtering
//! - Kabsch alignment

use super::binary::{deserialize, serialize};
use super::types::{Coords, CoordsError};
use glam::{Mat3, Vec3};

/// Extract backbone chains from COORDS data.
/// Returns a vector of chains, where each chain is a sequence of N-CA-C positions.
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

        // Check for chain break
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

    for i in 0..coords.num_atoms {
        if predicate(&coords.atom_names[i]) {
            atoms.push(coords.atoms[i].clone());
            chain_ids.push(coords.chain_ids[i]);
            res_names.push(coords.res_names[i]);
            res_nums.push(coords.res_nums[i]);
            atom_names.push(coords.atom_names[i]);
        }
    }

    Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
    }
}

/// Filter atoms by predicate on residue name.
pub fn filter_residues(coords: &Coords, predicate: impl Fn(&[u8; 3]) -> bool) -> Coords {
    let mut atoms = Vec::new();
    let mut chain_ids = Vec::new();
    let mut res_names = Vec::new();
    let mut res_nums = Vec::new();
    let mut atom_names = Vec::new();

    for i in 0..coords.num_atoms {
        if predicate(&coords.res_names[i]) {
            atoms.push(coords.atoms[i].clone());
            chain_ids.push(coords.chain_ids[i]);
            res_names.push(coords.res_names[i]);
            res_nums.push(coords.res_nums[i]);
            atom_names.push(coords.atom_names[i]);
        }
    }

    Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
    }
}

/// Standard amino acid residue names
const PROTEIN_RESIDUES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    // Non-standard but protein-like
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

    // 1. Center both point sets
    let ref_centroid = centroid(reference);
    let tgt_centroid = centroid(target);

    let ref_centered: Vec<Vec3> = reference.iter().map(|p| *p - ref_centroid).collect();
    let tgt_centered: Vec<Vec3> = target.iter().map(|p| *p - tgt_centroid).collect();

    // 2. Compute covariance matrix H = sum(tgt * ref^T)
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

    // 3. SVD: H = U * S * V^T
    let (u, _s, v) = svd_3x3(h);

    // 4. Rotation R = V * U^T
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

    // Handle reflection (if det(R) < 0)
    if rotation.determinant() < 0.0 {
        let v_flipped = Mat3::from_cols(v_mat.col(0), v_mat.col(1), -v_mat.col(2));
        rotation = v_flipped * u_mat.transpose();
    }

    // 5. Translation t = ref_centroid - R * tgt_centroid
    let translation = ref_centroid - rotation * tgt_centroid;

    Some((rotation, translation))
}

/// Kabsch-Umeyama algorithm: find optimal rotation, translation, AND scale.
/// Returns (rotation_matrix, translation, scale) such that: aligned = rotation * (target * scale) + translation
pub fn kabsch_alignment_with_scale(
    reference: &[Vec3],
    target: &[Vec3],
) -> Option<(Mat3, Vec3, f32)> {
    if reference.len() != target.len() || reference.len() < 3 {
        return None;
    }

    // 1. Center both point sets
    let ref_centroid = centroid(reference);
    let tgt_centroid = centroid(target);

    let ref_centered: Vec<Vec3> = reference.iter().map(|p| *p - ref_centroid).collect();
    let tgt_centered: Vec<Vec3> = target.iter().map(|p| *p - tgt_centroid).collect();

    // 2. Compute variances for scale estimation
    let _ref_var: f32 =
        ref_centered.iter().map(|p| p.length_squared()).sum::<f32>() / reference.len() as f32;
    let tgt_var: f32 =
        tgt_centered.iter().map(|p| p.length_squared()).sum::<f32>() / target.len() as f32;

    // Avoid division by zero
    if tgt_var < 1e-10 {
        return None;
    }

    // 3. Compute covariance matrix H = sum(tgt * ref^T)
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

    // 4. SVD: H = U * S * V^T
    let (u, s, v) = svd_3x3(h);

    // 5. Rotation R = V * U^T
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

    // Handle reflection (if det(R) < 0)
    if rotation.determinant() < 0.0 {
        let v_flipped = Mat3::from_cols(v_mat.col(0), v_mat.col(1), -v_mat.col(2));
        rotation = v_flipped * u_mat.transpose();
        sign = -1.0;
    }

    // 6. Compute optimal scale: c = trace(S * D) / var(target)
    let trace_sd = s[0] + s[1] + sign * s[2];
    let scale = trace_sd / (tgt_var * reference.len() as f32);

    // Clamp scale to reasonable range (0.1x to 10x)
    let scale = scale.clamp(0.1, 10.0);

    // 7. Translation t = ref_centroid - scale * R * tgt_centroid
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
/// This transforms ALL atoms in the COORDS data.
pub fn align_to_reference(
    coords: &mut Coords,
    reference_ca: &[Vec3],
) -> Result<(), CoordsError> {
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

/// Align COORDS bytes to match reference CA positions.
/// Returns new aligned COORDS bytes.
pub fn align_coords_bytes(
    coords_bytes: &[u8],
    reference_ca: &[Vec3],
) -> Result<Vec<u8>, CoordsError> {
    let mut coords = deserialize(coords_bytes)?;
    align_to_reference(&mut coords, reference_ca)?;
    serialize(&coords)
}

// ============================================================================
// SVD Implementation (Jacobi iteration for 3x3 matrices)
// ============================================================================

/// Compute SVD of a 3x3 matrix: A = U * diag(S) * V^T
fn svd_3x3(a: [[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3], [[f32; 3]; 3]) {
    // Compute A^T * A
    let mut ata = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    // Find eigenvalues and eigenvectors of A^T * A using Jacobi iteration
    let (eigenvalues, v) = jacobi_eigendecomposition(ata);

    // Singular values are sqrt of eigenvalues
    let s = [
        eigenvalues[0].max(0.0).sqrt(),
        eigenvalues[1].max(0.0).sqrt(),
        eigenvalues[2].max(0.0).sqrt(),
    ];

    // Compute U = A * V * S^-1
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

    // Orthonormalize U using Gram-Schmidt
    orthonormalize(&mut u);

    (u, s, v)
}

/// Jacobi eigendecomposition for symmetric 3x3 matrix
fn jacobi_eigendecomposition(mut a: [[f32; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut v = [[0.0f32; 3]; 3];
    for i in 0..3 {
        v[i][i] = 1.0;
    }

    const MAX_ITER: usize = 50;
    for _ in 0..MAX_ITER {
        // Find largest off-diagonal element
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

        // Compute rotation angle
        let diff = a[q][q] - a[p][p];
        let theta = if diff.abs() < 1e-10 {
            std::f32::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / diff).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to A
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

        // Apply rotation to V
        for i in 0..3 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }

    // Extract eigenvalues (diagonal of A)
    let eigenvalues = [a[0][0], a[1][1], a[2][2]];

    // Sort by decreasing eigenvalue
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

/// Gram-Schmidt orthonormalization for 3x3 matrix (columns as vectors)
fn orthonormalize(m: &mut [[f32; 3]; 3]) {
    // Normalize first column
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

    // Second column: subtract projection onto first, normalize
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

    // Third column: subtract projections onto first two, normalize
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

#[cfg(test)]
mod tests {
    use super::*;

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
        // Should be identity rotation and zero translation
        assert!((rotation.determinant() - 1.0).abs() < 0.01);
        assert!(translation.length() < 0.01);
    }
}
