//! DCD trajectory file parser (CHARMM/NAMD binary format).
//!
//! DCD files store molecular dynamics trajectories as Fortran-style binary records:
//! - Header: magic `CORD`, frame/atom counts, timestep, title
//! - Per-frame: optional unit cell, then X[N], Y[N], Z[N] as f32 arrays
//!
//! No atom metadata is stored — topology must come from the loaded structure.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Parsed DCD file header.
#[derive(Debug, Clone)]
pub struct DcdHeader {
    pub num_frames: u32,
    pub num_atoms: u32,
    pub start_step: u32,
    pub step_interval: u32,
    pub timestep: f32,
    pub has_extra_block: bool,
    pub has_four_dims: bool,
    pub title: String,
}

/// A single trajectory frame: flat arrays of x, y, z positions.
#[derive(Debug, Clone)]
pub struct DcdFrame {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
}

/// Streaming DCD reader over any `Read + Seek` source.
pub struct DcdReader<R: Read + Seek> {
    reader: R,
    pub header: DcdHeader,
    frames_read: u32,
}

impl DcdReader<BufReader<File>> {
    /// Open a DCD file and parse its header.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::new(reader)
    }
}

impl<R: Read + Seek> DcdReader<R> {
    /// Parse header from any seekable reader.
    pub fn new(mut reader: R) -> io::Result<Self> {
        let header = parse_header(&mut reader)?;
        Ok(Self {
            reader,
            header,
            frames_read: 0,
        })
    }

    /// Read the next frame. Returns `None` at EOF.
    pub fn read_frame(&mut self) -> io::Result<Option<DcdFrame>> {
        if self.frames_read >= self.header.num_frames {
            return Ok(None);
        }

        let n = self.header.num_atoms as usize;

        // Skip unit cell record if present
        if self.header.has_extra_block {
            skip_fortran_record(&mut self.reader)?;
        }

        // Read X, Y, Z coordinate arrays
        let x = read_f32_fortran_record(&mut self.reader, n)?;
        let y = read_f32_fortran_record(&mut self.reader, n)?;
        let z = read_f32_fortran_record(&mut self.reader, n)?;

        // Skip 4th dimension if present
        if self.header.has_four_dims {
            skip_fortran_record(&mut self.reader)?;
        }

        self.frames_read += 1;
        Ok(Some(DcdFrame { x, y, z }))
    }

    /// Read all remaining frames into memory.
    pub fn read_all_frames(&mut self) -> io::Result<Vec<DcdFrame>> {
        let remaining = (self.header.num_frames - self.frames_read) as usize;
        let mut frames = Vec::with_capacity(remaining);
        while let Some(frame) = self.read_frame()? {
            frames.push(frame);
        }
        Ok(frames)
    }
}

/// Convenience: open a DCD file, parse header and all frames.
pub fn dcd_file_to_frames(path: &Path) -> io::Result<(DcdHeader, Vec<DcdFrame>)> {
    let mut reader = DcdReader::open(path)?;
    let header = reader.header.clone();
    let frames = reader.read_all_frames()?;
    Ok((header, frames))
}

// ---------------------------------------------------------------------------
// Internal parsing helpers
// ---------------------------------------------------------------------------

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

/// Read a Fortran record: i32 size, N bytes payload, i32 size (must match).
fn read_fortran_record(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let size = read_i32(r)?;
    if size < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("negative Fortran record size: {size}"),
        ));
    }
    let size = size as usize;
    let mut buf = vec![0u8; size];
    r.read_exact(&mut buf)?;
    let end_size = read_i32(r)? as usize;
    if size != end_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Fortran record size mismatch: start={size}, end={end_size}"),
        ));
    }
    Ok(buf)
}

/// Skip over a Fortran record without allocating.
fn skip_fortran_record(r: &mut (impl Read + Seek)) -> io::Result<()> {
    let size = read_i32(r)?;
    if size < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("negative Fortran record size: {size}"),
        ));
    }
    r.seek(SeekFrom::Current(size as i64))?;
    let end_size = read_i32(r)?;
    if size != end_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Fortran record size mismatch: start={size}, end={end_size}"),
        ));
    }
    Ok(())
}

/// Read a Fortran record containing exactly `n` f32 values.
fn read_f32_fortran_record(r: &mut impl Read, n: usize) -> io::Result<Vec<f32>> {
    let data = read_fortran_record(r)?;
    let expected = n * 4;
    if data.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "f32 record size mismatch: expected {expected} bytes ({n} floats), got {}",
                data.len()
            ),
        ));
    }
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(floats)
}

/// Parse the DCD header (first three Fortran records).
fn parse_header(r: &mut impl Read) -> io::Result<DcdHeader> {
    // --- Record 1: 84-byte control block ---
    let rec1 = read_fortran_record(r)?;
    if rec1.len() != 84 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("DCD header record 1: expected 84 bytes, got {}", rec1.len()),
        ));
    }

    // Magic: first 4 bytes must be "CORD"
    let magic = &rec1[0..4];
    if magic != b"CORD" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("DCD magic: expected CORD, got {:?}", magic),
        ));
    }

    // 20 i32 control values (indices 0..19, each 4 bytes, starting at offset 4)
    let icntrl: Vec<i32> = rec1[4..84]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let num_frames = icntrl[0] as u32;
    let start_step = icntrl[1] as u32;
    let step_interval = icntrl[2] as u32;
    let delta_bits = icntrl[9]; // icntrl[9] holds timestep as raw f32 bits
    let has_extra_block = icntrl[10] != 0;
    let has_four_dims = icntrl[11] != 0;

    let timestep = f32::from_bits(delta_bits as u32);

    // --- Record 2: title block ---
    let rec2 = read_fortran_record(r)?;
    let title = if rec2.len() >= 4 {
        let ntitle = i32::from_le_bytes([rec2[0], rec2[1], rec2[2], rec2[3]]) as usize;
        let mut lines = Vec::with_capacity(ntitle);
        for i in 0..ntitle {
            let start = 4 + i * 80;
            let end = start + 80;
            if end <= rec2.len() {
                let line = String::from_utf8_lossy(&rec2[start..end])
                    .trim()
                    .to_string();
                if !line.is_empty() {
                    lines.push(line);
                }
            }
        }
        lines.join("\n")
    } else {
        String::new()
    };

    // --- Record 3: atom count ---
    let rec3 = read_fortran_record(r)?;
    if rec3.len() != 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("DCD natom record: expected 4 bytes, got {}", rec3.len()),
        ));
    }
    let num_atoms = i32::from_le_bytes([rec3[0], rec3[1], rec3[2], rec3[3]]) as u32;

    Ok(DcdHeader {
        num_frames,
        num_atoms,
        start_step,
        step_interval,
        timestep,
        has_extra_block,
        has_four_dims,
        title,
    })
}
