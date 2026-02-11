//! MRC/CCP4 density map parser.
//!
//! Parses the MRC2014 / CCP4 binary format: a 1024-byte header followed by a
//! 3D grid of density values. Supports modes 0 (i8), 1 (i16), 2 (f32), and
//! 6 (u16), with automatic endianness detection via MACHST.

use std::fs;
use std::io::{Cursor, Read};
use std::path::Path;

use ndarray::Array3;

use crate::types::density::{DensityError, DensityMap};

const HEADER_SIZE: usize = 1024;
const MAP_MAGIC: &[u8; 4] = b"MAP ";

/// Parse an MRC/CCP4 density map from a file path.
pub fn mrc_file_to_density(path: &Path) -> Result<DensityMap, DensityError> {
    let bytes = fs::read(path)?;
    mrc_to_density(&bytes)
}

/// Parse an MRC/CCP4 density map from raw bytes.
pub fn mrc_to_density(bytes: &[u8]) -> Result<DensityMap, DensityError> {
    if bytes.len() < HEADER_SIZE {
        return Err(DensityError::InvalidFormat(format!(
            "file too small for MRC header: {} bytes (need at least {HEADER_SIZE})",
            bytes.len()
        )));
    }

    let header = &bytes[..HEADER_SIZE];

    // Validate MAP_ magic at words 53 (byte offset 208)
    let magic = &header[208..212];
    if magic != MAP_MAGIC {
        return Err(DensityError::InvalidFormat(format!(
            "missing MAP magic at offset 208: got {:?}",
            magic
        )));
    }

    // Detect endianness from MACHST (word 54, byte offset 212)
    let little_endian = detect_endianness(header)?;

    let i32_at = |offset: usize| -> i32 {
        let b = &header[offset..offset + 4];
        if little_endian {
            i32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            i32::from_be_bytes([b[0], b[1], b[2], b[3]])
        }
    };
    let f32_at = |offset: usize| -> f32 {
        let b = &header[offset..offset + 4];
        if little_endian {
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            f32::from_be_bytes([b[0], b[1], b[2], b[3]])
        }
    };

    // Words 1-3: NC, NR, NS (grid cols/rows/sections as stored in file)
    let nc = i32_at(0);
    let nr = i32_at(4);
    let ns = i32_at(8);

    if nc <= 0 || nr <= 0 || ns <= 0 {
        return Err(DensityError::InvalidFormat(format!(
            "non-positive grid dimensions: NC={nc}, NR={nr}, NS={ns}"
        )));
    }

    // Word 4: MODE
    let mode = i32_at(12);
    if !matches!(mode, 0 | 1 | 2 | 6) {
        return Err(DensityError::UnsupportedMode(mode));
    }

    // Words 5-7: NXSTART, NYSTART, NZSTART (in file axis order)
    let ncstart = i32_at(16);
    let nrstart = i32_at(20);
    let nsstart = i32_at(24);

    // Words 8-10: MX, MY, MZ
    let mx = i32_at(28);
    let my = i32_at(32);
    let mz = i32_at(36);

    if mx <= 0 || my <= 0 || mz <= 0 {
        return Err(DensityError::InvalidFormat(format!(
            "non-positive unit cell sampling: MX={mx}, MY={my}, MZ={mz}"
        )));
    }

    // Words 11-16: cell dimensions and angles
    let cell_a = f32_at(40);
    let cell_b = f32_at(44);
    let cell_c = f32_at(48);
    let cell_alpha = f32_at(52);
    let cell_beta = f32_at(56);
    let cell_gamma = f32_at(60);

    // Words 17-19: MAPC, MAPR, MAPS (axis correspondence: 1=X, 2=Y, 3=Z)
    let mapc = i32_at(64);
    let mapr = i32_at(68);
    let maps = i32_at(72);

    if !((1..=3).contains(&mapc) && (1..=3).contains(&mapr) && (1..=3).contains(&maps))
        || mapc == mapr
        || mapc == maps
        || mapr == maps
    {
        return Err(DensityError::InvalidFormat(format!(
            "invalid axis mapping: MAPC={mapc}, MAPR={mapr}, MAPS={maps}"
        )));
    }

    // Words 20-22: DMIN, DMAX, DMEAN
    let dmin = f32_at(76);
    let dmax = f32_at(80);
    let dmean = f32_at(84);

    // Word 23: ISPG (space group)
    let space_group = i32_at(88) as u32;

    // Word 24: NSYMBT (extended header size in bytes)
    let nsymbt = i32_at(92);
    if nsymbt < 0 {
        return Err(DensityError::InvalidFormat(format!(
            "negative extended header size: {nsymbt}"
        )));
    }

    // Words 50-52: origin X, Y, Z (byte offsets 196-208)
    let origin_x = f32_at(196);
    let origin_y = f32_at(200);
    let origin_z = f32_at(204);

    // Word 55: RMS (byte offset 216)
    let rms = f32_at(216);

    // --- Read density data ---
    let data_offset = HEADER_SIZE + nsymbt as usize;
    let nc = nc as usize;
    let nr = nr as usize;
    let ns = ns as usize;
    let total_voxels = nc * nr * ns;

    let data_bytes = &bytes[data_offset..];
    let flat = read_density_values(data_bytes, mode, total_voxels, little_endian)?;

    // --- Axis reordering ---
    // File stores data as sections(slow) -> rows(medium) -> columns(fast).
    // MAPC tells which spatial axis columns map to, MAPR -> rows, MAPS -> sections.
    // We need to build an Array3 in [X, Y, Z] order.

    // Determine spatial dimensions: spatial_dim[axis-1] = file_dim for that axis
    let mapc = mapc as usize;
    let mapr = mapr as usize;
    let maps = maps as usize;

    let mut spatial_size = [0usize; 3];
    spatial_size[mapc - 1] = nc;
    spatial_size[mapr - 1] = nr;
    spatial_size[maps - 1] = ns;

    let nx = spatial_size[0];
    let ny = spatial_size[1];
    let nz = spatial_size[2];

    // Also reorder start indices
    let file_starts = [ncstart, nrstart, nsstart];
    let nxstart = file_starts[axis_file_index(mapc, mapr, maps, 1)];
    let nystart = file_starts[axis_file_index(mapc, mapr, maps, 2)];
    let nzstart = file_starts[axis_file_index(mapc, mapr, maps, 3)];

    let data = if mapc == 1 && mapr == 2 && maps == 3 {
        // Fast path: file order matches XYZ — just reshape
        // File order: section(Z) -> row(Y) -> col(X), which is Z-major = [nz, ny, nx]
        // We want [nx, ny, nz], so we need to permute
        let arr = Array3::from_shape_vec((ns, nr, nc), flat).map_err(|e| {
            DensityError::InvalidFormat(format!("array reshape failed: {e}"))
        })?;
        arr.permuted_axes([2, 1, 0])
            .as_standard_layout()
            .to_owned()
    } else {
        // General case: scatter file voxels into spatial positions
        let mut arr = Array3::<f32>::zeros((nx, ny, nz));
        for s in 0..ns {
            for r in 0..nr {
                let row_start = (s * nr + r) * nc;
                for c in 0..nc {
                    let mut xyz = [0usize; 3];
                    xyz[mapc - 1] = c;
                    xyz[mapr - 1] = r;
                    xyz[maps - 1] = s;
                    arr[[xyz[0], xyz[1], xyz[2]]] = flat[row_start + c];
                }
            }
        }
        arr
    };

    Ok(DensityMap {
        nx,
        ny,
        nz,
        nxstart,
        nystart,
        nzstart,
        mx: mx as usize,
        my: my as usize,
        mz: mz as usize,
        cell_dims: [cell_a, cell_b, cell_c],
        cell_angles: [cell_alpha, cell_beta, cell_gamma],
        dmin,
        dmax,
        dmean,
        rms,
        origin: [origin_x, origin_y, origin_z],
        space_group,
        data,
    })
}

/// Given axis mapping values, return the file dimension index (0=col, 1=row, 2=section)
/// for a given spatial axis (1=X, 2=Y, 3=Z).
fn axis_file_index(mapc: usize, mapr: usize, maps: usize, spatial: usize) -> usize {
    if mapc == spatial {
        0
    } else if mapr == spatial {
        1
    } else {
        debug_assert_eq!(maps, spatial);
        2
    }
}

/// Detect endianness from MACHST (byte offset 212-213).
fn detect_endianness(header: &[u8]) -> Result<bool, DensityError> {
    let machst = header[212];
    match machst {
        0x44 => Ok(true),  // little-endian
        0x11 => Ok(false), // big-endian
        _ => {
            // Fallback: try LE, check if MODE is valid
            let mode_le = i32::from_le_bytes([
                header[12],
                header[13],
                header[14],
                header[15],
            ]);
            if matches!(mode_le, 0 | 1 | 2 | 6) {
                Ok(true)
            } else {
                let mode_be = i32::from_be_bytes([
                    header[12],
                    header[13],
                    header[14],
                    header[15],
                ]);
                if matches!(mode_be, 0 | 1 | 2 | 6) {
                    Ok(false)
                } else {
                    Err(DensityError::InvalidFormat(format!(
                        "cannot determine endianness: MACHST={machst:#x}, MODE(LE)={mode_le}, MODE(BE)={mode_be}"
                    )))
                }
            }
        }
    }
}

/// Read and convert density data to f32 based on MODE.
fn read_density_values(
    data: &[u8],
    mode: i32,
    count: usize,
    little_endian: bool,
) -> Result<Vec<f32>, DensityError> {
    let bytes_per_voxel = match mode {
        0 => 1,
        1 | 6 => 2,
        2 => 4,
        _ => return Err(DensityError::UnsupportedMode(mode)),
    };

    let needed = count * bytes_per_voxel;
    if data.len() < needed {
        return Err(DensityError::InvalidFormat(format!(
            "not enough data: need {needed} bytes for {count} voxels (mode {mode}), got {}",
            data.len()
        )));
    }

    let mut cursor = Cursor::new(&data[..needed]);
    let mut values = Vec::with_capacity(count);

    match mode {
        0 => {
            // Mode 0: signed 8-bit integers
            let mut buf = [0u8; 1];
            for _ in 0..count {
                cursor.read_exact(&mut buf)?;
                values.push(buf[0] as i8 as f32);
            }
        }
        1 => {
            // Mode 1: signed 16-bit integers
            let mut buf = [0u8; 2];
            for _ in 0..count {
                cursor.read_exact(&mut buf)?;
                let v = if little_endian {
                    i16::from_le_bytes(buf)
                } else {
                    i16::from_be_bytes(buf)
                };
                values.push(v as f32);
            }
        }
        2 => {
            // Mode 2: 32-bit floats
            let mut buf = [0u8; 4];
            for _ in 0..count {
                cursor.read_exact(&mut buf)?;
                let v = if little_endian {
                    f32::from_le_bytes(buf)
                } else {
                    f32::from_be_bytes(buf)
                };
                values.push(v);
            }
        }
        6 => {
            // Mode 6: unsigned 16-bit integers
            let mut buf = [0u8; 2];
            for _ in 0..count {
                cursor.read_exact(&mut buf)?;
                let v = if little_endian {
                    u16::from_le_bytes(buf)
                } else {
                    u16::from_be_bytes(buf)
                };
                values.push(v as f32);
            }
        }
        _ => unreachable!(),
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid MRC header + data for testing.
    fn make_test_mrc(
        nc: i32,
        nr: i32,
        ns: i32,
        mode: i32,
        mapc: i32,
        mapr: i32,
        maps: i32,
        data_values: &[f32],
    ) -> Vec<u8> {
        let mut header = vec![0u8; HEADER_SIZE];

        let put_i32 = |buf: &mut [u8], offset: usize, val: i32| {
            buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        };
        let put_f32 = |buf: &mut [u8], offset: usize, val: f32| {
            buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        };

        // Grid dimensions
        put_i32(&mut header, 0, nc);
        put_i32(&mut header, 4, nr);
        put_i32(&mut header, 8, ns);

        // Mode
        put_i32(&mut header, 12, mode);

        // Start indices (all 0)
        put_i32(&mut header, 16, 0);
        put_i32(&mut header, 20, 0);
        put_i32(&mut header, 24, 0);

        // MX, MY, MZ = grid dims
        put_i32(&mut header, 28, nc);
        put_i32(&mut header, 32, nr);
        put_i32(&mut header, 36, ns);

        // Cell dimensions (10 Å each)
        put_f32(&mut header, 40, 10.0);
        put_f32(&mut header, 44, 10.0);
        put_f32(&mut header, 48, 10.0);

        // Cell angles (90°)
        put_f32(&mut header, 52, 90.0);
        put_f32(&mut header, 56, 90.0);
        put_f32(&mut header, 60, 90.0);

        // Axis mapping
        put_i32(&mut header, 64, mapc);
        put_i32(&mut header, 68, mapr);
        put_i32(&mut header, 72, maps);

        // DMIN, DMAX, DMEAN
        put_f32(&mut header, 76, 0.0);
        put_f32(&mut header, 80, 1.0);
        put_f32(&mut header, 84, 0.5);

        // Space group
        put_i32(&mut header, 88, 1);

        // NSYMBT = 0
        put_i32(&mut header, 92, 0);

        // Origin (0, 0, 0)
        put_f32(&mut header, 196, 0.0);
        put_f32(&mut header, 200, 0.0);
        put_f32(&mut header, 204, 0.0);

        // MAP_ magic
        header[208..212].copy_from_slice(b"MAP ");

        // MACHST: little-endian
        header[212] = 0x44;

        // RMS
        put_f32(&mut header, 216, 0.25);

        // Append data as mode-2 f32
        let mut bytes = header;
        for &v in data_values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        bytes
    }

    #[test]
    fn parse_basic_xyz() {
        // 2x3x4 grid, XYZ axis order (MAPC=1, MAPR=2, MAPS=3)
        let nc = 2;
        let nr = 3;
        let ns = 4;
        let total = (nc * nr * ns) as usize;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();

        let mrc = make_test_mrc(nc, nr, ns, 2, 1, 2, 3, &data);
        let map = mrc_to_density(&mrc).unwrap();

        assert_eq!(map.nx, 2);
        assert_eq!(map.ny, 3);
        assert_eq!(map.nz, 4);
        assert_eq!(map.dmin, 0.0);
        assert_eq!(map.dmax, 1.0);
        assert_eq!(map.dmean, 0.5);
        assert_eq!(map.rms, 0.25);

        // File stores section(Z)->row(Y)->col(X):
        // index in flat = z * nr * nc + y * nc + x
        // So flat[0] = data[z=0,y=0,x=0], flat[1] = data[z=0,y=0,x=1], etc.
        assert_eq!(map.data[[0, 0, 0]], 0.0);
        assert_eq!(map.data[[1, 0, 0]], 1.0); // x=1, col index 1
        assert_eq!(map.data[[0, 1, 0]], 2.0); // y=1, row index 1 -> offset nc
        assert_eq!(map.data[[0, 0, 1]], 6.0); // z=1, section index 1 -> offset nr*nc
    }

    #[test]
    fn parse_zxy_axis_reorder() {
        // File stores cols=Z, rows=X, sections=Y (MAPC=3, MAPR=1, MAPS=2)
        // NC=4(Z), NR=2(X), NS=3(Y)
        let nc = 4; // maps to Z
        let nr = 2; // maps to X
        let ns = 3; // maps to Y

        let total = (nc * nr * ns) as usize;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();

        let mrc = make_test_mrc(nc, nr, ns, 2, 3, 1, 2, &data);
        let map = mrc_to_density(&mrc).unwrap();

        // Spatial: nx=NR=2, ny=NS=3, nz=NC=4
        assert_eq!(map.nx, 2);
        assert_eq!(map.ny, 3);
        assert_eq!(map.nz, 4);

        // Flat index for file (s, r, c) = s * nr * nc + r * nc + c
        // Spatial: xyz[MAPC-1]=xyz[2]=c, xyz[MAPR-1]=xyz[0]=r, xyz[MAPS-1]=xyz[1]=s
        // So map[x,y,z] should equal flat[y * nr * nc + x * nc + z]

        // (x=0,y=0,z=0) -> (s=0,r=0,c=0) -> flat[0] = 0
        assert_eq!(map.data[[0, 0, 0]], 0.0);
        // (x=1,y=0,z=0) -> (s=0,r=1,c=0) -> flat[0*2*4 + 1*4 + 0] = 4
        assert_eq!(map.data[[1, 0, 0]], 4.0);
        // (x=0,y=1,z=0) -> (s=1,r=0,c=0) -> flat[1*2*4 + 0] = 8
        assert_eq!(map.data[[0, 1, 0]], 8.0);
        // (x=0,y=0,z=1) -> (s=0,r=0,c=1) -> flat[1] = 1
        assert_eq!(map.data[[0, 0, 1]], 1.0);
    }

    #[test]
    fn voxel_size_and_grid_to_cartesian() {
        let nc = 2;
        let nr = 3;
        let ns = 4;
        let total = (nc * nr * ns) as usize;
        let data: Vec<f32> = vec![0.0; total];

        let mut mrc = make_test_mrc(nc, nr, ns, 2, 1, 2, 3, &data);

        // Set cell dims to 20, 30, 40 Å with MX=2, MY=3, MZ=4
        mrc[40..44].copy_from_slice(&20.0f32.to_le_bytes());
        mrc[44..48].copy_from_slice(&30.0f32.to_le_bytes());
        mrc[48..52].copy_from_slice(&40.0f32.to_le_bytes());

        let map = mrc_to_density(&mrc).unwrap();

        let vs = map.voxel_size();
        assert!((vs[0] - 10.0).abs() < 1e-6); // 20/2
        assert!((vs[1] - 10.0).abs() < 1e-6); // 30/3
        assert!((vs[2] - 10.0).abs() < 1e-6); // 40/4

        let pos = map.grid_to_cartesian(1, 2, 3);
        assert!((pos[0] - 10.0).abs() < 1e-6);
        assert!((pos[1] - 20.0).abs() < 1e-6);
        assert!((pos[2] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn sigma_level() {
        let nc = 1;
        let nr = 1;
        let ns = 1;
        let data = make_test_mrc(nc, nr, ns, 2, 1, 2, 3, &[0.5f32]);

        // dmean=0.5, rms=0.25
        let map = mrc_to_density(&data).unwrap();
        let level = map.sigma_level(2.0);
        assert!((level - 1.0).abs() < 1e-6); // 0.5 + 2.0 * 0.25 = 1.0
    }

    #[test]
    fn reject_invalid_magic() {
        let mut mrc = make_test_mrc(1, 1, 1, 2, 1, 2, 3, &[0.0]);
        mrc[208..212].copy_from_slice(b"NOPE");

        let err = mrc_to_density(&mrc).unwrap_err();
        assert!(matches!(err, DensityError::InvalidFormat(_)));
    }

    #[test]
    fn reject_unsupported_mode() {
        let mut mrc = make_test_mrc(1, 1, 1, 2, 1, 2, 3, &[0.0]);
        // Set mode to 3 (complex i16, unsupported)
        mrc[12..16].copy_from_slice(&3i32.to_le_bytes());

        let err = mrc_to_density(&mrc).unwrap_err();
        assert!(matches!(err, DensityError::UnsupportedMode(3)));
    }

    #[test]
    fn reject_too_small() {
        let err = mrc_to_density(&[0u8; 100]).unwrap_err();
        assert!(matches!(err, DensityError::InvalidFormat(_)));
    }

    #[test]
    fn mode_1_i16() {
        // Test mode 1 (signed 16-bit)
        let mut header = make_test_mrc(2, 1, 1, 1, 1, 2, 3, &[]);
        // Remove the empty f32 data appended by make_test_mrc (none since &[] was passed)
        // Now manually add i16 data
        let values: [i16; 2] = [100, -200];
        for v in &values {
            header.extend_from_slice(&v.to_le_bytes());
        }

        let map = mrc_to_density(&header).unwrap();
        assert_eq!(map.data[[0, 0, 0]], 100.0);
        assert_eq!(map.data[[1, 0, 0]], -200.0);
    }

    #[test]
    fn mode_0_i8() {
        let mut header = make_test_mrc(2, 1, 1, 0, 1, 2, 3, &[]);
        header.push(50u8);           // 50 as i8
        header.push((-30i8) as u8);  // -30 as i8

        let map = mrc_to_density(&header).unwrap();
        assert_eq!(map.data[[0, 0, 0]], 50.0);
        assert_eq!(map.data[[1, 0, 0]], -30.0);
    }

    #[test]
    fn mode_6_u16() {
        let mut header = make_test_mrc(2, 1, 1, 6, 1, 2, 3, &[]);
        let values: [u16; 2] = [1000, 65000];
        for v in &values {
            header.extend_from_slice(&v.to_le_bytes());
        }

        let map = mrc_to_density(&header).unwrap();
        assert_eq!(map.data[[0, 0, 0]], 1000.0);
        assert_eq!(map.data[[1, 0, 0]], 65000.0);
    }

    #[test]
    fn endianness_fallback() {
        // Set MACHST to unknown value, but leave mode as valid LE
        let mut mrc = make_test_mrc(1, 1, 1, 2, 1, 2, 3, &[1.0]);
        mrc[212] = 0x00; // unknown MACHST

        let map = mrc_to_density(&mrc).unwrap();
        assert_eq!(map.data[[0, 0, 0]], 1.0);
    }

    #[test]
    fn extended_header_skip() {
        let nc = 1;
        let nr = 1;
        let ns = 1;

        let mut mrc = make_test_mrc(nc, nr, ns, 2, 1, 2, 3, &[]);

        // Set NSYMBT = 64 (skip 64 bytes of extended header)
        mrc[92..96].copy_from_slice(&64i32.to_le_bytes());

        // Add 64 bytes of garbage extended header
        mrc.extend_from_slice(&[0xAB; 64]);

        // Then the actual data
        mrc.extend_from_slice(&42.0f32.to_le_bytes());

        let map = mrc_to_density(&mrc).unwrap();
        assert_eq!(map.data[[0, 0, 0]], 42.0);
    }

    #[test]
    fn origin_and_start_indices() {
        let nc = 2;
        let nr = 2;
        let ns = 2;
        let total = (nc * nr * ns) as usize;
        let data: Vec<f32> = vec![0.0; total];

        let mut mrc = make_test_mrc(nc, nr, ns, 2, 1, 2, 3, &data);

        // Set origin to (5.0, 10.0, 15.0)
        mrc[196..200].copy_from_slice(&5.0f32.to_le_bytes());
        mrc[200..204].copy_from_slice(&10.0f32.to_le_bytes());
        mrc[204..208].copy_from_slice(&15.0f32.to_le_bytes());

        // Set start indices to (1, 2, 3) -- these are in file axis order
        mrc[16..20].copy_from_slice(&1i32.to_le_bytes());
        mrc[20..24].copy_from_slice(&2i32.to_le_bytes());
        mrc[24..28].copy_from_slice(&3i32.to_le_bytes());

        let map = mrc_to_density(&mrc).unwrap();

        assert_eq!(map.nxstart, 1);
        assert_eq!(map.nystart, 2);
        assert_eq!(map.nzstart, 3);

        // voxel_size = 10/2 = 5 Å per axis
        // grid_to_cartesian(0,0,0) = origin + start * voxel_size
        let pos = map.grid_to_cartesian(0, 0, 0);
        assert!((pos[0] - 10.0).abs() < 1e-6); // 5.0 + 1*5.0
        assert!((pos[1] - 20.0).abs() < 1e-6); // 10.0 + 2*5.0
        assert!((pos[2] - 30.0).abs() < 1e-6); // 15.0 + 3*5.0
    }
}
