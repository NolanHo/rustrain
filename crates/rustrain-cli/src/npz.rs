//! A minimal `.npz` writer — numpy's container format — with no dependency beyond the standard
//! library.
//!
//! The `.npy` payload is a header plus raw little-endian data, and the container is a plain zip
//! with STORED (uncompressed) entries. Both are written here by hand so the comparison script can
//! read the candidate dump with `numpy.load` on the verification host, and `compare` stays one
//! command.
//!
//! The writer is deterministic: fixed zip version/date fields and entry order, so the unit test
//! pins the exact bytes of a small dump.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

/// One named array of a dump. `descr` is numpy's dtype spelling (`<f4` for f32, `<i8` for i64).
pub(crate) struct Npy<'a> {
    pub name: &'a str,
    pub descr: &'static str,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

/// The `.npy` payload for one array: magic, version, padded header, then the raw bytes.
pub(crate) fn npy_payload(descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_text = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let dict = match shape.len() {
        // Canonical numpy spellings: scalars `()`, one-element tuples `(n,)`,
        // longer tuples without a trailing comma.
        0 => format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': (), }}"),
        1 => format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_text},)}}"),
        _ => format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_text})}}"),
    };
    // Header length must satisfy 10 + len ≡ 0 (mod 64); the last byte is a newline.
    let mut header_len = dict.len() + 1;
    while (10 + header_len) % 64 != 0 {
        header_len += 1;
    }
    let mut out = Vec::with_capacity(10 + header_len + data.len());
    out.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0]);
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat_n(b' ', header_len - dict.len() - 1));
    out.push(b'\n');
    out.extend_from_slice(data);
    out
}

/// Writes `entries` as a `.npz` — a zip of STORED `.npy` files.
pub(crate) fn write_npz(path: &Path, entries: &[Npy<'_>]) -> Result<()> {
    let payloads: Vec<(String, Vec<u8>)> = entries
        .iter()
        .map(|e| {
            (
                format!("{}.npy", e.name),
                npy_payload(e.descr, e.shape, e.data),
            )
        })
        .collect();

    let mut out = Vec::new();
    let mut central = Vec::new();
    let mut offsets: Vec<u32> = Vec::new();
    let mut crcs: Vec<u32> = Vec::new();
    let mut sizes: Vec<u32> = Vec::new();

    for (name, payload) in &payloads {
        let offset = out.len() as u32;
        let size = payload.len() as u32;
        let crc = crc32(payload);
        offsets.push(offset);
        crcs.push(crc);
        sizes.push(size);

        // Local file header (fixed fields for deterministic output).
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0x21u16.to_le_bytes()); // mod date: 1980-01-01
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(payload);
    }

    for (index, (name, payload)) in payloads.iter().enumerate() {
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes()); // version made by
        central.extend_from_slice(&20u16.to_le_bytes()); // version needed
        central.extend_from_slice(&0u16.to_le_bytes()); // flags
        central.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        central.extend_from_slice(&0u16.to_le_bytes()); // mod time
        central.extend_from_slice(&0x21u16.to_le_bytes()); // mod date
        central.extend_from_slice(&crcs[index].to_le_bytes());
        central.extend_from_slice(&sizes[index].to_le_bytes());
        central.extend_from_slice(&sizes[index].to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes()); // extra len
        central.extend_from_slice(&0u16.to_le_bytes()); // comment len
        central.extend_from_slice(&0u16.to_le_bytes()); // disk number
        central.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        central.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        central.extend_from_slice(&offsets[index].to_le_bytes());
        central.extend_from_slice(name.as_bytes());
        let _ = payload;
    }

    let central_offset = out.len() as u32;
    let central_size = central.len() as u32;
    out.extend_from_slice(&central);

    // End of central directory.
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // disk number
    out.extend_from_slice(&0u16.to_le_bytes()); // central dir disk
    out.extend_from_slice(&(payloads.len() as u16).to_le_bytes());
    out.extend_from_slice(&(payloads.len() as u16).to_le_bytes());
    out.extend_from_slice(&central_size.to_le_bytes());
    out.extend_from_slice(&central_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment len

    let mut file = std::fs::File::create(path)
        .with_context(|| format!("creating the dump {}", path.display()))?;
    file.write_all(&out)
        .with_context(|| format!("writing the dump {}", path.display()))?;
    Ok(())
}

/// CRC-32 (the zip polynomial), table-based.
fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *entry = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc = table[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// The f32 dump contract (numpy `<f4`).
pub(crate) const F4: &str = "<f4";
/// The i64 dump contract (numpy `<i8`).
pub(crate) const I8: &str = "<i8";

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard CRC-32 check value.
    #[test]
    fn crc32_matches_the_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    /// The `.npy` header is byte-exact: magic, version, length, padded dict.
    #[test]
    fn npy_header_bytes_are_exact() {
        let payload = npy_payload("<f4", &[2, 3], &[]);
        assert_eq!(&payload[..8], &[0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0]);
        let header_len = u16::from_le_bytes([payload[8], payload[9]]) as usize;
        assert_eq!(
            (10 + header_len) % 64,
            0,
            "the header must be 64-byte aligned"
        );
        let header = std::str::from_utf8(&payload[10..10 + header_len]).unwrap();
        assert!(header.starts_with("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3)"));
        assert!(header.trim_end().ends_with('}'));
        assert!(header.ends_with('\n'));
        // One-element shapes have no trailing comma; scalar shapes are `()`.
        let payload = npy_payload("<i8", &[8], &[]);
        let header = std::str::from_utf8(&payload[10..]).unwrap();
        assert!(header.contains("'shape': (8,)"));
        let scalar = npy_payload("<f4", &[], &[]);
        let header = std::str::from_utf8(&scalar[10..]).unwrap();
        assert!(header.contains("'shape': ()"));
    }

    /// The whole dump is byte-deterministic — two writes of the same arrays are identical, and
    /// the zip landmarks sit where they must.
    #[test]
    fn a_dump_is_byte_deterministic_and_zip_well_formed() {
        let a_bytes = 1.0f32.to_le_bytes();
        let b_bytes = 7i64.to_le_bytes();
        let entries = vec![
            Npy {
                name: "a",
                descr: "<f4",
                shape: &[2],
                data: &a_bytes,
            },
            Npy {
                name: "b",
                descr: "<i8",
                shape: &[1],
                data: &b_bytes,
            },
        ];
        let dir = std::env::temp_dir();
        let p1 = dir.join("rustrain-npz-test-a.npz");
        let p2 = dir.join("rustrain-npz-test-b.npz");
        write_npz(&p1, &entries).unwrap();
        write_npz(&p2, &entries).unwrap();
        let a = std::fs::read(&p1).unwrap();
        let b = std::fs::read(&p2).unwrap();
        assert_eq!(a, b, "the writer must be deterministic");
        // Local header signature at 0, central directory at the recorded offset,
        // EOCD at the very end.
        assert_eq!(&a[0..4], &0x0403_4b50u32.to_le_bytes());
        let eocd = a.len() - 22;
        assert_eq!(&a[eocd..eocd + 4], &0x0605_4b50u32.to_le_bytes());
        let cd_offset = u32::from_le_bytes(a[eocd + 16..eocd + 20].try_into().unwrap());
        assert_eq!(
            &a[cd_offset as usize..cd_offset as usize + 4],
            &0x0201_4b50u32.to_le_bytes()
        );
        std::fs::remove_file(&p1).ok();
        std::fs::remove_file(&p2).ok();
    }
}
