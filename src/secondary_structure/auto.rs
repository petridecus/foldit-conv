//! CA-distance heuristic for secondary structure detection.
//!
//! Detects alpha helices and beta sheets based on Cα-Cα distances
//! and assigns per-residue Q3 classifications.

use super::SSType;
use glam::Vec3;

/// Detect secondary structure from Cα positions.
///
/// Uses distance-based heuristics:
/// - Helix: Cα(i)-Cα(i+3) ≈ 5.0-5.5Å, Cα(i)-Cα(i+4) ≈ 5.5-6.5Å
/// - Sheet: Extended conformation with Cα(i)-Cα(i+2) ≈ 6.0-7.5Å
///
/// Returns a Vec of SSType, one per residue (same length as `ca_positions`).
pub fn detect(ca_positions: &[Vec3]) -> Vec<SSType> {
    let n = ca_positions.len();
    let mut ss: Vec<SSType> = vec![SSType::Coil; n];

    for i in 0..n {
        // Check for helix pattern (need i+3 and i+4)
        if i + 4 < n {
            let d_i3 = (ca_positions[i] - ca_positions[i + 3]).length();
            let d_i4 = (ca_positions[i] - ca_positions[i + 4]).length();

            // Helix: Cα(i)-Cα(i+3) ~ 5.0-5.5Å, Cα(i)-Cα(i+4) ~ 5.5-6.5Å
            let is_helix = (4.5..=6.0).contains(&d_i3) && (5.0..=7.0).contains(&d_i4);

            if is_helix {
                ss[i] = SSType::Helix;
            }
        }

        // Check for sheet pattern (extended conformation)
        if i + 2 < n && ss[i] != SSType::Helix {
            let d_i1 = (ca_positions[i] - ca_positions[i + 1]).length();
            let d_i2 = (ca_positions[i] - ca_positions[i + 2]).length();

            // Sheet: Extended, so Cα(i)-Cα(i+1) ~ 3.8Å and Cα(i)-Cα(i+2) ~ 6.5-7.5Å
            let is_extended = (3.5..=4.1).contains(&d_i1) && (6.0..=8.0).contains(&d_i2);

            if is_extended {
                ss[i] = SSType::Sheet;
            }
        }
    }

    ss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_chain() {
        let result = detect(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_short_chain() {
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)];
        let result = detect(&positions);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&s| s == SSType::Coil));
    }

    #[test]
    fn test_extended_sheet() {
        // Create an extended chain with ~3.8Å spacing (sheet-like)
        let positions: Vec<Vec3> = (0..8)
            .map(|i| Vec3::new(i as f32 * 3.8, 0.0, 0.0))
            .collect();
        let result = detect(&positions);
        assert_eq!(result.len(), 8);
        // Extended chain should have some sheet residues
        assert!(result.iter().any(|&s| s == SSType::Sheet));
    }
}
