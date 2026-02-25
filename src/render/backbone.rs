//! Protein backbone domain types.
//!
//! [`BackboneChain`] wraps a single interleaved N-CA-C backbone chain.
//! [`ProteinBackbone`] collects multiple chains extracted from a protein entity.

use glam::Vec3;

/// A single protein backbone chain stored as interleaved N, CA, C positions.
///
/// Length is always `3 * residue_count`. Index layout:
/// ```text
/// [N0, CA0, C0, N1, CA1, C1, ...]
/// ```
#[derive(Debug, Clone)]
pub struct BackboneChain {
    atoms: Vec<Vec3>,
}

impl BackboneChain {
    /// Create from an interleaved N-CA-C position vector.
    ///
    /// The length should be a multiple of 3; a trailing incomplete residue
    /// is silently truncated.
    pub fn new(mut atoms: Vec<Vec3>) -> Self {
        let remainder = atoms.len() % 3;
        if remainder != 0 {
            atoms.truncate(atoms.len() - remainder);
        }
        Self { atoms }
    }

    /// Number of residues in this chain.
    pub fn residue_count(&self) -> usize {
        self.atoms.len() / 3
    }

    /// Iterator over CA positions.
    pub fn ca_positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.atoms.chunks_exact(3).map(|chunk| chunk[1])
    }

    /// Iterator over N positions.
    pub fn n_positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.atoms.chunks_exact(3).map(|chunk| chunk[0])
    }

    /// Iterator over C positions.
    pub fn c_positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.atoms.chunks_exact(3).map(|chunk| chunk[2])
    }

    /// Get the backbone triple (N, CA, C) for the residue at `idx`.
    pub fn backbone_triple(&self, idx: usize) -> Option<(Vec3, Vec3, Vec3)> {
        let base = idx * 3;
        let n = *self.atoms.get(base)?;
        let ca = *self.atoms.get(base + 1)?;
        let c = *self.atoms.get(base + 2)?;
        Some((n, ca, c))
    }

    /// Escape hatch: borrow the raw interleaved slice for GPU code.
    pub fn as_slice(&self) -> &[Vec3] {
        &self.atoms
    }

    /// Consume and return the inner Vec.
    pub fn into_inner(self) -> Vec<Vec3> {
        self.atoms
    }
}

/// Multi-chain protein backbone extracted from one or more protein entities.
#[derive(Debug, Clone)]
pub struct ProteinBackbone {
    pub chains: Vec<BackboneChain>,
    pub chain_ids: Vec<u8>,
}

impl ProteinBackbone {
    /// Flattened CA positions across all chains.
    pub fn ca_positions(&self) -> Vec<Vec3> {
        self.chains
            .iter()
            .flat_map(|c| c.ca_positions())
            .collect()
    }

    /// Total residue count across all chains.
    pub fn residue_count(&self) -> usize {
        self.chains.iter().map(|c| c.residue_count()).sum()
    }

    /// Get the CA position for a global residue index.
    pub fn ca_position(&self, global_idx: usize) -> Option<Vec3> {
        let mut remaining = global_idx;
        for chain in &self.chains {
            let n = chain.residue_count();
            if remaining < n {
                return chain.backbone_triple(remaining).map(|(_, ca, _)| ca);
            }
            remaining -= n;
        }
        None
    }

    /// Get the (N, CA, C) triple for a global residue index.
    pub fn backbone_triple(&self, global_idx: usize) -> Option<(Vec3, Vec3, Vec3)> {
        let mut remaining = global_idx;
        for chain in &self.chains {
            let n = chain.residue_count();
            if remaining < n {
                return chain.backbone_triple(remaining);
            }
            remaining -= n;
        }
        None
    }

    /// Get each chain as a raw `&[Vec3]` slice (for renderer compatibility).
    pub fn as_chain_slices(&self) -> Vec<&[Vec3]> {
        self.chains.iter().map(|c| c.as_slice()).collect()
    }

    /// Convert into owned `Vec<Vec<Vec3>>` (for APIs that need owned data).
    pub fn into_chain_vecs(self) -> Vec<Vec<Vec3>> {
        self.chains.into_iter().map(|c| c.into_inner()).collect()
    }

    /// Borrow chains as `Vec<Vec<Vec3>>` references (cloning the outer structure).
    pub fn to_chain_vecs(&self) -> Vec<Vec<Vec3>> {
        self.chains.iter().map(|c| c.as_slice().to_vec()).collect()
    }
}

/// Extract CA positions from interleaved N-CA-C backbone chains.
///
/// Each chain stores atoms as `[N0, CA0, C0, N1, CA1, C1, ...]`. This
/// function flattens all chains and returns only the CA positions.
pub fn ca_positions_from_chains(chains: &[Vec<Vec3>]) -> Vec<Vec3> {
    chains
        .iter()
        .flat_map(|chain| chain.chunks(3).filter_map(|chunk| chunk.get(1).copied()))
        .collect()
}

/// Construct a [`ProteinBackbone`] from the legacy `Vec<Vec<Vec3>>` + chain IDs
/// representation used throughout the codebase.
impl From<(Vec<Vec<Vec3>>, Vec<u8>)> for ProteinBackbone {
    fn from((chains, chain_ids): (Vec<Vec<Vec3>>, Vec<u8>)) -> Self {
        Self {
            chains: chains.into_iter().map(BackboneChain::new).collect(),
            chain_ids,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    #[test]
    fn test_backbone_chain_basic() {
        let chain = BackboneChain::new(vec![
            v(0.0, 0.0, 0.0), // N0
            v(1.0, 0.0, 0.0), // CA0
            v(2.0, 0.0, 0.0), // C0
            v(3.0, 0.0, 0.0), // N1
            v(4.0, 0.0, 0.0), // CA1
            v(5.0, 0.0, 0.0), // C1
        ]);
        assert_eq!(chain.residue_count(), 2);

        let cas: Vec<_> = chain.ca_positions().collect();
        assert_eq!(cas, vec![v(1.0, 0.0, 0.0), v(4.0, 0.0, 0.0)]);

        let ns: Vec<_> = chain.n_positions().collect();
        assert_eq!(ns, vec![v(0.0, 0.0, 0.0), v(3.0, 0.0, 0.0)]);

        assert_eq!(
            chain.backbone_triple(0),
            Some((v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(2.0, 0.0, 0.0)))
        );
        assert_eq!(chain.backbone_triple(2), None);
    }

    #[test]
    fn test_backbone_chain_truncates_incomplete() {
        let chain = BackboneChain::new(vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(2.0, 0.0, 0.0),
            v(3.0, 0.0, 0.0), // incomplete residue — truncated
        ]);
        assert_eq!(chain.residue_count(), 1);
        assert_eq!(chain.as_slice().len(), 3);
    }

    #[test]
    fn test_protein_backbone() {
        let bb = ProteinBackbone {
            chains: vec![
                BackboneChain::new(vec![
                    v(0.0, 0.0, 0.0),
                    v(1.0, 0.0, 0.0),
                    v(2.0, 0.0, 0.0),
                ]),
                BackboneChain::new(vec![
                    v(10.0, 0.0, 0.0),
                    v(11.0, 0.0, 0.0),
                    v(12.0, 0.0, 0.0),
                    v(13.0, 0.0, 0.0),
                    v(14.0, 0.0, 0.0),
                    v(15.0, 0.0, 0.0),
                ]),
            ],
            chain_ids: vec![b'A', b'B'],
        };

        assert_eq!(bb.residue_count(), 3);

        let cas = bb.ca_positions();
        assert_eq!(cas.len(), 3);
        assert_eq!(cas[0], v(1.0, 0.0, 0.0));
        assert_eq!(cas[1], v(11.0, 0.0, 0.0));
        assert_eq!(cas[2], v(14.0, 0.0, 0.0));

        assert_eq!(bb.ca_position(0), Some(v(1.0, 0.0, 0.0)));
        assert_eq!(bb.ca_position(1), Some(v(11.0, 0.0, 0.0)));
        assert_eq!(bb.ca_position(2), Some(v(14.0, 0.0, 0.0)));
        assert_eq!(bb.ca_position(3), None);
    }
}
