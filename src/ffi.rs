use std::ffi::{c_char, CString};
use std::ptr;

#[repr(C)]
pub struct CoordsResult {
    pub data: *const u8,
    pub len: usize,
    pub data_len: usize,
    pub error: *const c_char,
}

impl CoordsResult {
    pub fn success(data: Vec<u8>) -> Self {
        let len = data.len();
        let data_ptr = if data.is_empty() {
            ptr::null()
        } else {
            let boxed = data.into_boxed_slice();
            let ptr = boxed.as_ptr();
            std::mem::forget(boxed);
            ptr
        };
        Self {
            data: data_ptr,
            len,
            data_len: len,
            error: ptr::null(),
        }
    }

    pub fn error(msg: &str) -> Self {
        Self {
            data: ptr::null(),
            len: 0,
            data_len: 0,
            error: CString::new(msg).unwrap().into_raw(),
        }
    }
}

#[no_mangle]
pub extern "C" fn coords_free_result(result: *const CoordsResult) {
    if result.is_null() {
        return;
    }
    unsafe {
        let r = &*result;
        if !r.data.is_null() && r.len > 0 {
            let _ = Vec::from_raw_parts(r.data as *mut u8, r.len, r.len);
        }
        if !r.error.is_null() {
            let _ = CString::from_raw(r.error as *mut i8);
        }
    }
}

#[no_mangle]
pub extern "C" fn coords_free_string(s: *const c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s as *mut i8);
        }
    }
}

#[no_mangle]
pub extern "C" fn pdb_to_coords_bytes(pdb_ptr: *const c_char, pdb_len: usize) -> CoordsResult {
    if pdb_ptr.is_null() {
        return CoordsResult::error("PDB string is null");
    }
    unsafe {
        let pdb_slice = std::slice::from_raw_parts(pdb_ptr as *const u8, pdb_len);
        let pdb_str = match std::str::from_utf8(pdb_slice) {
            Ok(s) => s,
            Err(_) => return CoordsResult::error("Invalid UTF-8 in PDB string"),
        };
        match crate::adapters::pdb::pdb_to_coords(pdb_str) {
            Ok(coords_bytes) => CoordsResult::success(coords_bytes),
            Err(e) => CoordsResult::error(&e.to_string()),
        }
    }
}

#[no_mangle]
pub extern "C" fn coords_to_pdb(
    coords_ptr: *const u8,
    coords_len: usize,
    out_len: *mut usize,
) -> *const c_char {
    if coords_ptr.is_null() {
        return std::ffi::CString::new("COORDS data is null")
            .unwrap()
            .into_raw();
    }
    if out_len.is_null() {
        return std::ffi::CString::new("out_len is null")
            .unwrap()
            .into_raw();
    }
    unsafe {
        let coords_slice = std::slice::from_raw_parts(coords_ptr, coords_len);
        match crate::adapters::pdb::coords_to_pdb(coords_slice) {
            Ok(pdb_string) => {
                *out_len = pdb_string.len();
                std::ffi::CString::new(pdb_string).unwrap().into_raw()
            }
            Err(e) => std::ffi::CString::new(e.to_string()).unwrap().into_raw(),
        }
    }
}

#[no_mangle]
pub extern "C" fn coords_from_coords(coords_ptr: *const u8, coords_len: usize) -> CoordsResult {
    if coords_ptr.is_null() {
        return CoordsResult::error("Coords data is null");
    }
    unsafe {
        let coords_slice = std::slice::from_raw_parts(coords_ptr, coords_len);
        match crate::types::coords::deserialize(coords_slice)
            .and_then(|coords| crate::types::coords::serialize(&coords))
        {
            Ok(coords_bytes) => CoordsResult::success(coords_bytes),
            Err(e) => CoordsResult::error(&e.to_string()),
        }
    }
}

#[no_mangle]
pub extern "C" fn coords_from_backbone(
    positions: *const f32,
    _num_res: usize,
    _sequence: *const c_char,
    _chain_breaks: *const i32,
    _chain_break_count: usize,
) -> CoordsResult {
    if positions.is_null() {
        return CoordsResult::error("Null pointer provided");
    }
    CoordsResult::error("Not implemented: use model-specific adapters")
}
