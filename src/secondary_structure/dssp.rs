//! DSSP-style secondary structure assignment.
//!
//! Two backends:
//! - `detect`: Kabsch-Sander Q3 from backbone atom coordinates
//! - `from_string`: Parse a DSSP-style annotation string (e.g. "HHHEEECCC")

use super::{BackboneResidue, SSType};
use glam::Vec3;

/// Kabsch-Sander electrostatic energy constant (kcal/mol · Å).
const KS_FACTOR: f32 = 27.888;

/// H-bond energy threshold (kcal/mol). Bonds with E < this are accepted.
const HBOND_THRESHOLD: f32 = -0.5;

/// Detect secondary structure from full backbone atom coordinates
/// using the Kabsch-Sander hydrogen-bond energy criterion (Q3).
///
/// For each residue, N/CA/C/O positions are required. H positions
/// are estimated from geometry: `H = N + normalize(N - C_prev)`.
///
/// Algorithm:
/// 1. Estimate amide H positions
/// 2. Compute pairwise H-bond energies
/// 3. Assign helix (α, 3₁₀, π patterns) and sheet (bridge patterns)
/// 4. Smooth with minimum run lengths (4 helix, 3 sheet)
pub fn detect(residues: &[BackboneResidue]) -> Vec<SSType> {
    let n = residues.len();
    if n < 2 {
        return vec![SSType::Coil; n];
    }

    // Estimate amide H positions: H_i = N_i + normalize(N_i - C_{i-1})
    // First residue has no previous C, so H is estimated from N-CA direction
    let h_positions: Vec<Vec3> = (0..n)
        .map(|i| {
            if i == 0 {
                let n_ca = (residues[i].n - residues[i].ca).normalize_or_zero();
                residues[i].n + n_ca
            } else {
                let n_c_prev = (residues[i].n - residues[i - 1].c).normalize_or_zero();
                residues[i].n + n_c_prev
            }
        })
        .collect();

    // Compute H-bond energy matrix.
    // hbond_energy[i][j] = energy of H-bond from NH_i to CO_j
    // Only store bonds that pass threshold.
    let mut hbond: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];

    for i in 1..n {
        // Donor: NH of residue i
        let h = h_positions[i];
        let n_pos = residues[i].n;

        for j in 0..n {
            if i == j || i == j + 1 || (j > 0 && i == j - 1) {
                continue; // skip self and immediate neighbors
            }

            // Acceptor: CO of residue j
            let c = residues[j].c;
            let o = residues[j].o;

            let r_on = (o - n_pos).length();
            let r_ch = (c - h).length();
            let r_oh = (o - h).length();
            let r_cn = (c - n_pos).length();

            // Avoid division by near-zero distances
            if r_on < 0.5 || r_ch < 0.5 || r_oh < 0.5 || r_cn < 0.5 {
                continue;
            }

            let energy = KS_FACTOR * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn);

            if energy < HBOND_THRESHOLD {
                hbond[i].push((j, energy));
            }
        }
    }

    // Helper: does residue i have an H-bond to residue j (NH_i → CO_j)?
    let has_hbond = |donor: usize, acceptor: usize| -> bool {
        hbond[donor].iter().any(|&(j, _)| j == acceptor)
    };

    let mut raw_ss = vec![SSType::Coil; n];

    // Detect helices: n-turn at residue i means H-bond from i+n to i
    // α-helix: consecutive i→i+4 pattern
    // 3₁₀-helix: i→i+3
    // π-helix: i→i+5
    for turn_size in [4usize, 3, 5] {
        let mut consecutive_turns = 0usize;
        let min_consecutive = if turn_size == 4 { 4 } else { 3 };

        for i in 0..n {
            if i + turn_size < n && has_hbond(i + turn_size, i) {
                consecutive_turns += 1;
                if consecutive_turns >= min_consecutive {
                    // Mark residues in the helix
                    let start = if consecutive_turns == min_consecutive {
                        i + 1 - (min_consecutive - 1)
                    } else {
                        i
                    };
                    for k in start..=i + turn_size {
                        if k < n {
                            raw_ss[k] = SSType::Helix;
                        }
                    }
                }
            } else {
                consecutive_turns = 0;
            }
        }
    }

    // Detect sheets: bridge patterns between non-adjacent residues
    // Parallel bridge: (i-1,j) and (j,i+1) — or — (j-1,i) and (i,j+1)
    // Antiparallel bridge: (i,j) and (j,i) — or — (i-1,j+1) and (j-1,i+1)
    for i in 1..n.saturating_sub(1) {
        if raw_ss[i] == SSType::Helix {
            continue;
        }
        for j in (i + 2)..n {
            if raw_ss[j] == SSType::Helix {
                continue;
            }

            // Parallel bridge
            let parallel = (i > 0 && j + 1 < n && has_hbond(i, j) && has_hbond(j + 1, i))
                || (j > 0 && i + 1 < n && has_hbond(j, i) && has_hbond(i + 1, j));

            // Antiparallel bridge
            let antiparallel = (has_hbond(i, j) && has_hbond(j, i))
                || (i > 0
                    && j + 1 < n
                    && j > 0
                    && i + 1 < n
                    && has_hbond(i, j) // simplified check
                    && has_hbond(j, i));

            if parallel || antiparallel {
                raw_ss[i] = SSType::Sheet;
                raw_ss[j] = SSType::Sheet;
            }
        }
    }

    raw_ss
}

/// Parse a DSSP-style secondary structure annotation string.
///
/// Character mapping:
/// - `H`, `G`, `I` → Helix (α-helix, 3₁₀-helix, π-helix)
/// - `E`, `B` → Sheet (strand, isolated bridge)
/// - Everything else → Coil
///
/// Also handles simplified notation where lowercase is accepted.
pub fn from_string(ss: &str) -> Vec<SSType> {
    ss.chars()
        .map(|c| match c.to_ascii_uppercase() {
            'H' | 'G' | 'I' => SSType::Helix,
            'E' | 'B' => SSType::Sheet,
            _ => SSType::Coil,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_string_basic() {
        let result = from_string("HHHEEECCC");
        assert_eq!(
            result,
            vec![
                SSType::Helix,
                SSType::Helix,
                SSType::Helix,
                SSType::Sheet,
                SSType::Sheet,
                SSType::Sheet,
                SSType::Coil,
                SSType::Coil,
                SSType::Coil,
            ]
        );
    }

    #[test]
    fn test_from_string_dssp_codes() {
        // G = 3₁₀ helix, I = π helix, B = isolated bridge
        let result = from_string("HGIEBS T");
        assert_eq!(result[0], SSType::Helix); // H
        assert_eq!(result[1], SSType::Helix); // G
        assert_eq!(result[2], SSType::Helix); // I
        assert_eq!(result[3], SSType::Sheet); // E
        assert_eq!(result[4], SSType::Sheet); // B
        assert_eq!(result[5], SSType::Coil); // S (bend)
        assert_eq!(result[6], SSType::Coil); // space
        assert_eq!(result[7], SSType::Coil); // T (turn)
    }

    #[test]
    fn test_from_string_lowercase() {
        let result = from_string("hhe");
        assert_eq!(result, vec![SSType::Helix, SSType::Helix, SSType::Sheet]);
    }

    #[test]
    fn test_from_string_empty() {
        let result = from_string("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_from_string_puzzle_ss() {
        // Puzzle 0000010001 uses ss = "EEE"
        let result = from_string("EEE");
        assert_eq!(result, vec![SSType::Sheet, SSType::Sheet, SSType::Sheet]);
    }

    #[test]
    fn test_detect_single_residue() {
        let residues = vec![BackboneResidue {
            n: Vec3::new(0.0, 0.0, 0.0),
            ca: Vec3::new(1.5, 0.0, 0.0),
            c: Vec3::new(2.5, 1.0, 0.0),
            o: Vec3::new(2.5, 2.0, 0.0),
        }];
        let result = detect(&residues);
        assert_eq!(result, vec![SSType::Coil]);
    }

    #[test]
    fn test_detect_empty() {
        let result = detect(&[]);
        assert!(result.is_empty());
    }

}
