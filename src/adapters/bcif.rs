//! BinaryCIF (.bcif) format decoder.
//!
//! BinaryCIF is a column-oriented binary encoding of mmCIF, used by RCSB PDB
//! as the standard binary format (replacing MMTF). Files are MessagePack-encoded
//! with optional gzip compression.
//!
//! Reference: https://github.com/molstar/BinaryCIF/blob/master/encoding.md

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::types::coords::{Coords, CoordsAtom, CoordsError, Element};

/// Load a BinaryCIF file and convert to Coords.
pub fn bcif_file_to_coords(path: &Path) -> Result<Coords, CoordsError> {
    let bytes = std::fs::read(path)
        .map_err(|e| CoordsError::InvalidFormat(format!("Failed to read file: {}", e)))?;
    bcif_to_coords(&bytes)
}

/// Decode BinaryCIF bytes (possibly gzipped) into a Coords struct.
pub fn bcif_to_coords(bytes: &[u8]) -> Result<Coords, CoordsError> {
    let data = decompress_if_gzip(bytes)?;
    let root = decode_msgpack(&data)?;
    parse_bcif_to_coords(&root)
}

// ---------------------------------------------------------------------------
// Gzip detection & decompression
// ---------------------------------------------------------------------------

fn decompress_if_gzip(bytes: &[u8]) -> Result<Vec<u8>, CoordsError> {
    if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        let mut decoder = flate2::read::GzDecoder::new(bytes);
        let mut out = Vec::new();
        decoder
            .read_to_end(&mut out)
            .map_err(|e| CoordsError::InvalidFormat(format!("Gzip decompression failed: {}", e)))?;
        Ok(out)
    } else {
        Ok(bytes.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Lightweight MessagePack value tree
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum MsgVal {
    Nil,
    Bool(bool),
    Int(i64),
    Uint(u64),
    F32(f32),
    F64(f64),
    Str(String),
    Bin(Vec<u8>),
    Array(Vec<MsgVal>),
    Map(Vec<(MsgVal, MsgVal)>),
}

impl MsgVal {
    fn as_str(&self) -> Option<&str> {
        match self {
            MsgVal::Str(s) => Some(s),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            MsgVal::Int(v) => Some(*v),
            MsgVal::Uint(v) => Some(*v as i64),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            MsgVal::F64(v) => Some(*v),
            MsgVal::F32(v) => Some(*v as f64),
            MsgVal::Int(v) => Some(*v as f64),
            MsgVal::Uint(v) => Some(*v as f64),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            MsgVal::Bool(v) => Some(*v),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[MsgVal]> {
        match self {
            MsgVal::Array(a) => Some(a),
            _ => None,
        }
    }

    fn as_bin(&self) -> Option<&[u8]> {
        match self {
            MsgVal::Bin(b) => Some(b),
            _ => None,
        }
    }

    fn get(&self, key: &str) -> Option<&MsgVal> {
        match self {
            MsgVal::Map(pairs) => {
                for (k, v) in pairs {
                    if let MsgVal::Str(s) = k {
                        if s == key {
                            return Some(v);
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// MessagePack decoder
// ---------------------------------------------------------------------------

fn decode_msgpack(data: &[u8]) -> Result<MsgVal, CoordsError> {
    let mut cursor = std::io::Cursor::new(data);
    read_value(&mut cursor)
}

fn read_bytes<const N: usize>(rd: &mut std::io::Cursor<&[u8]>) -> Result<[u8; N], CoordsError> {
    let mut buf = [0u8; N];
    rd.read_exact(&mut buf)
        .map_err(|e| CoordsError::InvalidFormat(format!("msgpack read {N} bytes: {e}")))?;
    Ok(buf)
}

fn read_value(rd: &mut std::io::Cursor<&[u8]>) -> Result<MsgVal, CoordsError> {
    use rmp::Marker;

    let marker = rmp::decode::read_marker(rd)
        .map_err(|e| CoordsError::InvalidFormat(format!("msgpack marker: {:?}", e)))?;

    match marker {
        Marker::Null => Ok(MsgVal::Nil),
        Marker::True => Ok(MsgVal::Bool(true)),
        Marker::False => Ok(MsgVal::Bool(false)),

        Marker::FixPos(v) => Ok(MsgVal::Uint(v as u64)),
        Marker::FixNeg(v) => Ok(MsgVal::Int(v as i64)),

        Marker::U8 => Ok(MsgVal::Uint(read_bytes::<1>(rd)?[0] as u64)),
        Marker::U16 => Ok(MsgVal::Uint(u16::from_be_bytes(read_bytes(rd)?) as u64)),
        Marker::U32 => Ok(MsgVal::Uint(u32::from_be_bytes(read_bytes(rd)?) as u64)),
        Marker::U64 => Ok(MsgVal::Uint(u64::from_be_bytes(read_bytes(rd)?))),
        Marker::I8 => Ok(MsgVal::Int(read_bytes::<1>(rd)?[0] as i8 as i64)),
        Marker::I16 => Ok(MsgVal::Int(i16::from_be_bytes(read_bytes(rd)?) as i64)),
        Marker::I32 => Ok(MsgVal::Int(i32::from_be_bytes(read_bytes(rd)?) as i64)),
        Marker::I64 => Ok(MsgVal::Int(i64::from_be_bytes(read_bytes(rd)?))),
        Marker::F32 => Ok(MsgVal::F32(f32::from_be_bytes(read_bytes(rd)?))),
        Marker::F64 => Ok(MsgVal::F64(f64::from_be_bytes(read_bytes(rd)?))),

        Marker::FixStr(len) => read_string(rd, len as usize),
        Marker::Str8 => {
            let len = read_bytes::<1>(rd)?[0] as usize;
            read_string(rd, len)
        }
        Marker::Str16 => {
            let len = u16::from_be_bytes(read_bytes(rd)?) as usize;
            read_string(rd, len)
        }
        Marker::Str32 => {
            let len = u32::from_be_bytes(read_bytes(rd)?) as usize;
            read_string(rd, len)
        }

        Marker::Bin8 => {
            let len = read_bytes::<1>(rd)?[0] as usize;
            read_bin(rd, len)
        }
        Marker::Bin16 => {
            let len = u16::from_be_bytes(read_bytes(rd)?) as usize;
            read_bin(rd, len)
        }
        Marker::Bin32 => {
            let len = u32::from_be_bytes(read_bytes(rd)?) as usize;
            read_bin(rd, len)
        }

        Marker::FixArray(len) => read_array(rd, len as usize),
        Marker::Array16 => {
            let len = u16::from_be_bytes(read_bytes(rd)?) as usize;
            read_array(rd, len)
        }
        Marker::Array32 => {
            let len = u32::from_be_bytes(read_bytes(rd)?) as usize;
            read_array(rd, len)
        }

        Marker::FixMap(len) => read_map(rd, len as usize),
        Marker::Map16 => {
            let len = u16::from_be_bytes(read_bytes(rd)?) as usize;
            read_map(rd, len)
        }
        Marker::Map32 => {
            let len = u32::from_be_bytes(read_bytes(rd)?) as usize;
            read_map(rd, len)
        }

        other => Err(CoordsError::InvalidFormat(format!(
            "Unsupported msgpack marker: {:?}",
            other
        ))),
    }
}

fn read_string(rd: &mut std::io::Cursor<&[u8]>, len: usize) -> Result<MsgVal, CoordsError> {
    let mut buf = vec![0u8; len];
    rd.read_exact(&mut buf)
        .map_err(|e| CoordsError::InvalidFormat(format!("msgpack string read: {}", e)))?;
    let s = String::from_utf8(buf)
        .map_err(|e| CoordsError::InvalidFormat(format!("msgpack string utf8: {}", e)))?;
    Ok(MsgVal::Str(s))
}

fn read_bin(rd: &mut std::io::Cursor<&[u8]>, len: usize) -> Result<MsgVal, CoordsError> {
    let mut buf = vec![0u8; len];
    rd.read_exact(&mut buf)
        .map_err(|e| CoordsError::InvalidFormat(format!("msgpack bin read: {}", e)))?;
    Ok(MsgVal::Bin(buf))
}

fn read_array(rd: &mut std::io::Cursor<&[u8]>, len: usize) -> Result<MsgVal, CoordsError> {
    let mut arr = Vec::with_capacity(len);
    for _ in 0..len {
        arr.push(read_value(rd)?);
    }
    Ok(MsgVal::Array(arr))
}

fn read_map(rd: &mut std::io::Cursor<&[u8]>, len: usize) -> Result<MsgVal, CoordsError> {
    let mut pairs = Vec::with_capacity(len);
    for _ in 0..len {
        let k = read_value(rd)?;
        let v = read_value(rd)?;
        pairs.push((k, v));
    }
    Ok(MsgVal::Map(pairs))
}

// ---------------------------------------------------------------------------
// BinaryCIF encoding chain decoder
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum ColData {
    IntArray(Vec<i32>),
    FloatArray(Vec<f64>),
    StringArray(Vec<String>),
    Bytes(Vec<u8>),
}

fn decode_column(data_node: &MsgVal) -> Result<ColData, CoordsError> {
    let raw_bytes = data_node
        .get("data")
        .and_then(|v| v.as_bin())
        .ok_or_else(|| CoordsError::InvalidFormat("Column missing 'data' bytes".into()))?;

    let encodings = data_node
        .get("encoding")
        .and_then(|v| v.as_array())
        .ok_or_else(|| CoordsError::InvalidFormat("Column missing 'encoding' array".into()))?;

    if encodings.is_empty() {
        return Ok(ColData::Bytes(raw_bytes.to_vec()));
    }

    let first_kind = encodings[0]
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if first_kind == "StringArray" {
        return decode_string_array_column(raw_bytes, &encodings[0], &encodings[1..]);
    }

    let mut current = ColData::Bytes(raw_bytes.to_vec());
    for enc in encodings.iter().rev() {
        let kind = enc
            .get("kind")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CoordsError::InvalidFormat("Encoding missing 'kind'".into()))?;

        current = match kind {
            "ByteArray" => decode_byte_array(current, enc)?,
            "FixedPoint" => decode_fixed_point(current, enc)?,
            "IntervalQuantization" => decode_interval_quantization(current, enc)?,
            "RunLength" => decode_run_length(current, enc)?,
            "Delta" => decode_delta(current, enc)?,
            "IntegerPacking" => decode_integer_packing(current, enc)?,
            other => {
                return Err(CoordsError::InvalidFormat(format!(
                    "Unknown encoding kind: {}", other
                )))
            }
        };
    }

    Ok(current)
}

fn decode_byte_array(input: ColData, enc: &MsgVal) -> Result<ColData, CoordsError> {
    let bytes = match input {
        ColData::Bytes(b) => b,
        _ => return Err(CoordsError::InvalidFormat("ByteArray expects bytes input".into())),
    };

    let type_id = enc
        .get("type")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| CoordsError::InvalidFormat("ByteArray missing 'type'".into()))?
        as u8;

    match type_id {
        1 => Ok(ColData::IntArray(bytes.iter().map(|&b| b as i8 as i32).collect())),
        2 => Ok(ColData::IntArray(bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i32).collect())),
        3 => Ok(ColData::IntArray(bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())),
        4 => Ok(ColData::IntArray(bytes.iter().map(|&b| b as i32).collect())),
        5 => Ok(ColData::IntArray(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as i32).collect())),
        6 => Ok(ColData::IntArray(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i32).collect())),
        32 => Ok(ColData::FloatArray(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect())),
        33 => Ok(ColData::FloatArray(bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())),
        _ => Err(CoordsError::InvalidFormat(format!("Unknown ByteArray type: {}", type_id))),
    }
}

fn decode_fixed_point(input: ColData, enc: &MsgVal) -> Result<ColData, CoordsError> {
    let ints = match input {
        ColData::IntArray(v) => v,
        _ => return Err(CoordsError::InvalidFormat("FixedPoint expects int array".into())),
    };

    let factor = enc.get("factor").and_then(|v| v.as_f64())
        .ok_or_else(|| CoordsError::InvalidFormat("FixedPoint missing 'factor'".into()))?;

    let inv = 1.0 / factor;
    Ok(ColData::FloatArray(ints.iter().map(|&v| v as f64 * inv).collect()))
}

fn decode_interval_quantization(input: ColData, enc: &MsgVal) -> Result<ColData, CoordsError> {
    let ints = match input {
        ColData::IntArray(v) => v,
        _ => return Err(CoordsError::InvalidFormat("IntervalQuantization expects int array".into())),
    };

    let min = enc.get("min").and_then(|v| v.as_f64())
        .ok_or_else(|| CoordsError::InvalidFormat("IntervalQuantization missing 'min'".into()))?;
    let max = enc.get("max").and_then(|v| v.as_f64())
        .ok_or_else(|| CoordsError::InvalidFormat("IntervalQuantization missing 'max'".into()))?;
    let num_steps = enc.get("numSteps").and_then(|v| v.as_i64())
        .ok_or_else(|| CoordsError::InvalidFormat("IntervalQuantization missing 'numSteps'".into()))? as f64;

    let delta = (max - min) / (num_steps - 1.0);
    Ok(ColData::FloatArray(ints.iter().map(|&v| min + v as f64 * delta).collect()))
}

fn decode_run_length(input: ColData, _enc: &MsgVal) -> Result<ColData, CoordsError> {
    let ints = match input {
        ColData::IntArray(v) => v,
        _ => return Err(CoordsError::InvalidFormat("RunLength expects int array".into())),
    };

    if ints.len() % 2 != 0 {
        return Err(CoordsError::InvalidFormat("RunLength array length must be even".into()));
    }

    let mut result = Vec::new();
    for pair in ints.chunks_exact(2) {
        let value = pair[0];
        let count = pair[1] as usize;
        result.extend(std::iter::repeat(value).take(count));
    }
    Ok(ColData::IntArray(result))
}

fn decode_delta(input: ColData, enc: &MsgVal) -> Result<ColData, CoordsError> {
    let mut ints = match input {
        ColData::IntArray(v) => v,
        _ => return Err(CoordsError::InvalidFormat("Delta expects int array".into())),
    };

    let origin = enc.get("origin").and_then(|v| v.as_i64()).unwrap_or(0) as i32;

    if !ints.is_empty() {
        ints[0] += origin;
        for i in 1..ints.len() {
            ints[i] += ints[i - 1];
        }
    }
    Ok(ColData::IntArray(ints))
}

fn decode_integer_packing(input: ColData, enc: &MsgVal) -> Result<ColData, CoordsError> {
    let packed = match input {
        ColData::IntArray(v) => v,
        _ => return Err(CoordsError::InvalidFormat("IntegerPacking expects int array".into())),
    };

    let byte_count = enc.get("byteCount").and_then(|v| v.as_i64())
        .ok_or_else(|| CoordsError::InvalidFormat("IntegerPacking missing 'byteCount'".into()))? as i32;
    let src_size = enc.get("srcSize").and_then(|v| v.as_i64())
        .ok_or_else(|| CoordsError::InvalidFormat("IntegerPacking missing 'srcSize'".into()))? as usize;
    let is_unsigned = enc.get("isUnsigned").and_then(|v| v.as_bool()).unwrap_or(false);

    let (upper_limit, lower_limit) = if is_unsigned {
        if byte_count == 1 { (0xFF_i32, 0_i32) } else { (0xFFFF_i32, 0_i32) }
    } else if byte_count == 1 {
        (0x7F_i32, -0x7F_i32)
    } else {
        (0x7FFF_i32, -0x7FFF_i32)
    };

    let mut result = Vec::with_capacity(src_size);
    let mut i = 0;

    while i < packed.len() && result.len() < src_size {
        let mut value: i32 = 0;
        let mut t = packed[i];
        while t == upper_limit || t == lower_limit {
            value += t;
            i += 1;
            if i >= packed.len() { break; }
            t = packed[i];
        }
        value += t;
        i += 1;
        result.push(value);
    }

    Ok(ColData::IntArray(result))
}

fn decode_string_array_column(
    raw_bytes: &[u8],
    sa_enc: &MsgVal,
    remaining_encodings: &[MsgVal],
) -> Result<ColData, CoordsError> {
    let string_data = sa_enc.get("stringData").and_then(|v| v.as_str())
        .ok_or_else(|| CoordsError::InvalidFormat("StringArray missing 'stringData'".into()))?;

    let offset_bytes = sa_enc.get("offsets").and_then(|v| v.as_bin())
        .ok_or_else(|| CoordsError::InvalidFormat("StringArray missing 'offsets'".into()))?;
    let offset_encoding = sa_enc.get("offsetEncoding").and_then(|v| v.as_array())
        .ok_or_else(|| CoordsError::InvalidFormat("StringArray missing 'offsetEncoding'".into()))?;

    let offset_data_node = build_encoded_data(offset_bytes, offset_encoding);
    let offsets = match decode_column(&offset_data_node)? {
        ColData::IntArray(v) => v,
        _ => return Err(CoordsError::InvalidFormat("StringArray offsets must decode to int array".into())),
    };

    let mut strings: Vec<&str> = Vec::with_capacity(if offsets.is_empty() { 0 } else { offsets.len() - 1 });
    for w in offsets.windows(2) {
        let start = w[0] as usize;
        let end = w[1] as usize;
        if end > string_data.len() || start > end {
            return Err(CoordsError::InvalidFormat("StringArray offset out of bounds".into()));
        }
        strings.push(&string_data[start..end]);
    }

    let data_encoding = sa_enc.get("dataEncoding").and_then(|v| v.as_array())
        .ok_or_else(|| CoordsError::InvalidFormat("StringArray missing 'dataEncoding'".into()))?;

    let mut index_encodings = Vec::new();
    index_encodings.extend_from_slice(data_encoding);
    index_encodings.extend_from_slice(remaining_encodings);

    let index_data_node = build_encoded_data(raw_bytes, &index_encodings);
    let indices = match decode_column(&index_data_node)? {
        ColData::IntArray(v) => v,
        _ => return Err(CoordsError::InvalidFormat("StringArray indices must decode to int array".into())),
    };

    let result: Vec<String> = indices
        .iter()
        .map(|&idx| {
            let idx = idx as usize;
            if idx < strings.len() { strings[idx].to_string() } else { String::new() }
        })
        .collect();

    Ok(ColData::StringArray(result))
}

fn build_encoded_data(bytes: &[u8], encodings: &[MsgVal]) -> MsgVal {
    MsgVal::Map(vec![
        (MsgVal::Str("data".into()), MsgVal::Bin(bytes.to_vec())),
        (MsgVal::Str("encoding".into()), MsgVal::Array(encodings.to_vec())),
    ])
}

// ---------------------------------------------------------------------------
// BinaryCIF → Coords conversion
// ---------------------------------------------------------------------------

fn parse_bcif_to_coords(root: &MsgVal) -> Result<Coords, CoordsError> {
    let data_blocks = root.get("dataBlocks").and_then(|v| v.as_array())
        .ok_or_else(|| CoordsError::InvalidFormat("Missing 'dataBlocks'".into()))?;

    if data_blocks.is_empty() {
        return Err(CoordsError::InvalidFormat("No data blocks found".into()));
    }

    let block = &data_blocks[0];
    let categories = block.get("categories").and_then(|v| v.as_array())
        .ok_or_else(|| CoordsError::InvalidFormat("Missing 'categories' in data block".into()))?;

    let atom_site = categories
        .iter()
        .find(|cat| cat.get("name").and_then(|v| v.as_str()).map(|s| s == "_atom_site").unwrap_or(false))
        .ok_or_else(|| CoordsError::InvalidFormat("No '_atom_site' category found".into()))?;

    let row_count = atom_site.get("rowCount").and_then(|v| v.as_i64())
        .ok_or_else(|| CoordsError::InvalidFormat("Missing 'rowCount'".into()))? as usize;

    let columns = atom_site.get("columns").and_then(|v| v.as_array())
        .ok_or_else(|| CoordsError::InvalidFormat("Missing 'columns'".into()))?;

    let col_map: HashMap<&str, &MsgVal> = columns
        .iter()
        .filter_map(|col| {
            let name = col.get("name")?.as_str()?;
            Some((name, col))
        })
        .collect();

    let cartn_x = decode_float_col(&col_map, "Cartn_x", row_count)?;
    let cartn_y = decode_float_col(&col_map, "Cartn_y", row_count)?;
    let cartn_z = decode_float_col(&col_map, "Cartn_z", row_count)?;
    let label_atom_id = decode_string_col(&col_map, "label_atom_id", row_count)?;
    let label_comp_id = decode_string_col(&col_map, "label_comp_id", row_count)?;
    let label_asym_id = decode_string_col(&col_map, "label_asym_id", row_count)?;
    let label_seq_id = decode_int_col(&col_map, "label_seq_id", row_count)?;

    let occupancy = decode_float_col_opt(&col_map, "occupancy", row_count, 1.0);
    let b_factor = decode_float_col_opt(&col_map, "B_iso_or_equiv", row_count, 0.0);

    let type_symbol = decode_string_col(&col_map, "type_symbol", row_count).ok();

    let mut atoms = Vec::with_capacity(row_count);
    let mut chain_ids = Vec::with_capacity(row_count);
    let mut res_names = Vec::with_capacity(row_count);
    let mut res_nums = Vec::with_capacity(row_count);
    let mut atom_names = Vec::with_capacity(row_count);
    let mut elements = Vec::with_capacity(row_count);

    for i in 0..row_count {
        atoms.push(CoordsAtom {
            x: cartn_x[i] as f32,
            y: cartn_y[i] as f32,
            z: cartn_z[i] as f32,
            occupancy: occupancy[i] as f32,
            b_factor: b_factor[i] as f32,
        });

        chain_ids.push(label_asym_id[i].bytes().next().unwrap_or(b'A'));

        let mut rn = [b' '; 3];
        for (j, b) in label_comp_id[i].bytes().take(3).enumerate() {
            rn[j] = b;
        }
        res_names.push(rn);

        res_nums.push(label_seq_id[i]);

        let mut an = [b' '; 4];
        for (j, b) in label_atom_id[i].bytes().take(4).enumerate() {
            an[j] = b;
        }
        atom_names.push(an);

        let elem = if let Some(ref ts) = type_symbol {
            Element::from_symbol(&ts[i])
        } else {
            Element::from_atom_name(&label_atom_id[i])
        };
        elements.push(elem);
    }

    if atoms.is_empty() {
        return Err(CoordsError::InvalidFormat("No ATOM records found in BinaryCIF".into()));
    }

    Ok(Coords {
        num_atoms: atoms.len(),
        atoms,
        chain_ids,
        res_names,
        res_nums,
        atom_names,
        elements,
    })
}

// ---------------------------------------------------------------------------
// Column extraction helpers
// ---------------------------------------------------------------------------

fn get_col_data<'a>(col_map: &HashMap<&str, &'a MsgVal>, name: &str) -> Result<&'a MsgVal, CoordsError> {
    let col = col_map.get(name)
        .ok_or_else(|| CoordsError::InvalidFormat(format!("Missing column '{}'", name)))?;
    col.get("data")
        .ok_or_else(|| CoordsError::InvalidFormat(format!("Column '{}' has no data", name)))
}

fn decode_float_col(col_map: &HashMap<&str, &MsgVal>, name: &str, expected: usize) -> Result<Vec<f64>, CoordsError> {
    let data = get_col_data(col_map, name)?;
    match decode_column(data)? {
        ColData::FloatArray(v) if v.len() == expected => Ok(v),
        ColData::FloatArray(v) => Err(CoordsError::InvalidFormat(format!("Column '{}': expected {} rows, got {}", name, expected, v.len()))),
        ColData::IntArray(v) if v.len() == expected => Ok(v.iter().map(|&x| x as f64).collect()),
        _ => Err(CoordsError::InvalidFormat(format!("Column '{}': expected float array", name))),
    }
}

fn decode_float_col_opt(col_map: &HashMap<&str, &MsgVal>, name: &str, count: usize, default: f64) -> Vec<f64> {
    decode_float_col(col_map, name, count).unwrap_or_else(|_| vec![default; count])
}

fn decode_int_col(col_map: &HashMap<&str, &MsgVal>, name: &str, expected: usize) -> Result<Vec<i32>, CoordsError> {
    let data = get_col_data(col_map, name)?;
    match decode_column(data)? {
        ColData::IntArray(v) if v.len() == expected => Ok(v),
        ColData::IntArray(v) => Err(CoordsError::InvalidFormat(format!("Column '{}': expected {} rows, got {}", name, expected, v.len()))),
        _ => Err(CoordsError::InvalidFormat(format!("Column '{}': expected int array", name))),
    }
}

fn decode_string_col(col_map: &HashMap<&str, &MsgVal>, name: &str, expected: usize) -> Result<Vec<String>, CoordsError> {
    let data = get_col_data(col_map, name)?;
    match decode_column(data)? {
        ColData::StringArray(v) if v.len() == expected => Ok(v),
        ColData::StringArray(v) => Err(CoordsError::InvalidFormat(format!("Column '{}': expected {} rows, got {}", name, expected, v.len()))),
        _ => Err(CoordsError::InvalidFormat(format!("Column '{}': expected string array", name))),
    }
}
