//! Distance-based bond inference for small molecules.
//!
//! Infers covalent bonds from atom positions and element covalent radii.
//! Used for ligands, waters, and other non-protein entities where
//! bond topology is not provided by a dictionary.

use crate::types::coords::{Coords, Element};
use glam::Vec3;

/// Bond order classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
    Aromatic,
}

/// An inferred bond between two atoms.
#[derive(Debug, Clone)]
pub struct InferredBond {
    /// Index of the first atom in the Coords arrays
    pub atom_a: usize,
    /// Index of the second atom in the Coords arrays
    pub atom_b: usize,
    /// Inferred bond order
    pub order: BondOrder,
}

/// Infer covalent bonds from atom positions and element types.
///
/// O(n^2) is fine for small molecules (ligands typically <100 atoms).
pub fn infer_bonds(coords: &Coords, tolerance: f32) -> Vec<InferredBond> {
    let n = coords.num_atoms;
    if n < 2 {
        return Vec::new();
    }

    let positions: Vec<Vec3> = coords
        .atoms
        .iter()
        .map(|a| Vec3::new(a.x, a.y, a.z))
        .collect();

    let mut bonds = Vec::new();

    for i in 0..n {
        let elem_i = coords.elements.get(i).copied().unwrap_or(Element::Unknown);
        if elem_i == Element::H {
            continue;
        }
        let cov_i = elem_i.covalent_radius();

        for j in (i + 1)..n {
            let elem_j = coords.elements.get(j).copied().unwrap_or(Element::Unknown);
            if elem_i == Element::H && elem_j == Element::H {
                continue;
            }
            let cov_j = elem_j.covalent_radius();

            let dist = positions[i].distance(positions[j]);
            let sum_cov = cov_i + cov_j;
            let single_threshold = sum_cov + tolerance;

            if dist <= single_threshold && dist > 0.4 {
                let order = if dist < sum_cov * 0.9 {
                    BondOrder::Double
                } else {
                    BondOrder::Single
                };

                bonds.push(InferredBond {
                    atom_a: i,
                    atom_b: j,
                    order,
                });
            }
        }
    }

    bonds
}

/// Default tolerance for bond inference (0.4 angstroms).
pub const DEFAULT_TOLERANCE: f32 = 0.4;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::coords::CoordsAtom;

    fn make_atom(x: f32, y: f32, z: f32) -> CoordsAtom {
        CoordsAtom {
            x,
            y,
            z,
            occupancy: 1.0,
            b_factor: 0.0,
        }
    }

    #[test]
    fn test_simple_bond() {
        let coords = Coords {
            num_atoms: 2,
            atoms: vec![make_atom(0.0, 0.0, 0.0), make_atom(1.5, 0.0, 0.0)],
            chain_ids: vec![b'A', b'A'],
            res_names: vec![*b"LIG", *b"LIG"],
            res_nums: vec![1, 1],
            atom_names: vec![*b"C1  ", *b"C2  "],
            elements: vec![Element::C, Element::C],
        };

        let bonds = infer_bonds(&coords, DEFAULT_TOLERANCE);
        assert_eq!(bonds.len(), 1);
        assert_eq!(bonds[0].atom_a, 0);
        assert_eq!(bonds[0].atom_b, 1);
        assert_eq!(bonds[0].order, BondOrder::Single);
    }

    #[test]
    fn test_double_bond() {
        let coords = Coords {
            num_atoms: 2,
            atoms: vec![make_atom(0.0, 0.0, 0.0), make_atom(1.23, 0.0, 0.0)],
            chain_ids: vec![b'A', b'A'],
            res_names: vec![*b"LIG", *b"LIG"],
            res_nums: vec![1, 1],
            atom_names: vec![*b"C1  ", *b"O1  "],
            elements: vec![Element::C, Element::O],
        };

        let bonds = infer_bonds(&coords, DEFAULT_TOLERANCE);
        assert_eq!(bonds.len(), 1);
        assert_eq!(bonds[0].order, BondOrder::Double);
    }

    #[test]
    fn test_no_bond_far_apart() {
        let coords = Coords {
            num_atoms: 2,
            atoms: vec![make_atom(0.0, 0.0, 0.0), make_atom(5.0, 0.0, 0.0)],
            chain_ids: vec![b'A', b'A'],
            res_names: vec![*b"LIG", *b"LIG"],
            res_nums: vec![1, 1],
            atom_names: vec![*b"C1  ", *b"C2  "],
            elements: vec![Element::C, Element::C],
        };

        let bonds = infer_bonds(&coords, DEFAULT_TOLERANCE);
        assert!(bonds.is_empty());
    }

    #[test]
    fn test_water_bonds() {
        let coords = Coords {
            num_atoms: 3,
            atoms: vec![
                make_atom(0.0, 0.0, 0.0),
                make_atom(0.757, 0.586, 0.0),
                make_atom(-0.757, 0.586, 0.0),
            ],
            chain_ids: vec![b'A', b'A', b'A'],
            res_names: vec![*b"HOH", *b"HOH", *b"HOH"],
            res_nums: vec![1, 1, 1],
            atom_names: vec![*b"O   ", *b"H1  ", *b"H2  "],
            elements: vec![Element::O, Element::H, Element::H],
        };

        let bonds = infer_bonds(&coords, DEFAULT_TOLERANCE);
        assert_eq!(bonds.len(), 2);
    }
}
