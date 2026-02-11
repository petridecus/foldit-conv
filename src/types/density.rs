//! Data types for volumetric density maps (MRC/CCP4 format).

use ndarray::Array3;

/// Parsed volumetric density map.
///
/// Grid data is stored in spatial XYZ order after axis reordering from the
/// file's column/row/section layout. The `data` array is indexed as
/// `data[[x, y, z]]`.
#[derive(Debug, Clone)]
pub struct DensityMap {
    /// Spatial grid dimensions (X, Y, Z) after axis reordering.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    /// Grid start indices along X, Y, Z.
    pub nxstart: i32,
    pub nystart: i32,
    pub nzstart: i32,

    /// Unit cell grid sampling (intervals along each cell edge).
    pub mx: usize,
    pub my: usize,
    pub mz: usize,

    /// Unit cell dimensions a, b, c in Angstroms.
    pub cell_dims: [f32; 3],
    /// Unit cell angles alpha, beta, gamma in degrees.
    pub cell_angles: [f32; 3],

    /// Minimum density value.
    pub dmin: f32,
    /// Maximum density value.
    pub dmax: f32,
    /// Mean density value.
    pub dmean: f32,
    /// RMS deviation from mean density.
    pub rms: f32,

    /// Origin in Angstroms (MRC2014 words 50-52).
    pub origin: [f32; 3],

    /// Space group number.
    pub space_group: u32,

    /// 3D grid of density values, indexed as `data[[x, y, z]]`.
    pub data: Array3<f32>,
}

impl DensityMap {
    /// Angstroms per voxel along each axis: `[cell_a/mx, cell_b/my, cell_c/mz]`.
    pub fn voxel_size(&self) -> [f32; 3] {
        [
            self.cell_dims[0] / self.mx as f32,
            self.cell_dims[1] / self.my as f32,
            self.cell_dims[2] / self.mz as f32,
        ]
    }

    /// Convert a grid index to Cartesian coordinates in Angstroms.
    ///
    /// Accounts for both `origin` (MRC2014 words 50-52) and `nxstart/nystart/nzstart`.
    pub fn grid_to_cartesian(&self, ix: usize, iy: usize, iz: usize) -> [f32; 3] {
        let vs = self.voxel_size();
        [
            self.origin[0] + (self.nxstart as f32 + ix as f32) * vs[0],
            self.origin[1] + (self.nystart as f32 + iy as f32) * vs[1],
            self.origin[2] + (self.nzstart as f32 + iz as f32) * vs[2],
        ]
    }

    /// Density threshold at a given sigma level: `dmean + sigma * rms`.
    pub fn sigma_level(&self, sigma: f32) -> f32 {
        self.dmean + sigma * self.rms
    }
}

/// Errors that can occur when parsing a density map.
#[derive(Debug, thiserror::Error)]
pub enum DensityError {
    #[error("invalid density map format: {0}")]
    InvalidFormat(String),

    #[error("unsupported MRC data mode: {0}")]
    UnsupportedMode(i32),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
