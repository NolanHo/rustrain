//! Quantization operators: `quantize`, `dequantize`, `amax_update`, plus the
//! faithful f32 emulation of the `f8e4m3` / `f8e5m2` formats.
//!
//! Data-driven by contract (R-3): the scheme (`per_tensor` | `per_token` |
//! `per_block`) and the format (`f8e4m3` | `f8e5m2`) are attributes on the
//! call, never inferred from tensor shapes. The scale/amax tensors flow as
//! explicit input/output tensors — the `rs_tensor.scale`/`amax` sidecar
//! pointer fields are not used by `reference.f32`.
//!
//! The fp8 conversion below is an **emulation**, not the hardware
//! instruction: values are rounded to nearest, ties to even, onto the
//! target's exponent/mantissa grid (subnormals included), computed in f64 so
//! the rounding itself is exact. Overflow follows the format: `f8e4m3`
//! (OCP E4M3, no infinities) saturates past its grid to NaN; `f8e5m2` goes
//! to ±inf. `quantize` never overflows because the scale is
//! `max|x| / max_finite(format)` by construction.
//!
//! NaN inputs are rejected by `quantize` and `amax_update`: a NaN would
//! poison the running max/scale silently, and a reference backend must not
//! do that quietly.

use rustrain_abi::ffi::{RsAttrs, RsDtype, RsTensor, MAX_RANK};

use crate::attrs::{attr_i64s, require_str_of};
use crate::dispatch::{Call, run};
use crate::error::{OpResult, err};
use crate::tensor::{SmallShape, expect_out, set_output_desc};

macro_rules! infer_entry {
    ($name:ident, $op:literal, $body:path) => {
        pub(crate) unsafe extern "C" fn $name(
            in_: *const *const RsTensor,
            n_in: u32,
            out: *const *mut RsTensor,
            n_out: u32,
            attrs: *const RsAttrs,
        ) -> i32 {
            // SAFETY: ABI contract; pointers are the framework's.
            unsafe { run($op, in_, n_in, out, n_out, attrs, $body) }
        }
    };
}
macro_rules! exec_entry {
    ($name:ident, $op:literal, $body:path) => {
        pub(crate) unsafe extern "C" fn $name(
            ctx: *mut rustrain_abi::ffi::RsCtx,
            in_: *const *const RsTensor,
            n_in: u32,
            out: *const *mut RsTensor,
            n_out: u32,
            attrs: *const RsAttrs,
        ) -> i32 {
            let _ = ctx;
            // SAFETY: ABI contract; pointers are the framework's.
            unsafe { run($op, in_, n_in, out, n_out, attrs, $body) }
        }
    };
}

pub(crate) const SCHEMES: &[&str] = &["per_tensor", "per_token", "per_block"];
pub(crate) const FORMATS: &[&str] = &["f8e4m3", "f8e5m2"];

// ── fp8 emulation ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fp8Format {
    E4M3,
    E5M2,
}

impl Fp8Format {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f8e4m3" => Some(Self::E4M3),
            "f8e5m2" => Some(Self::E5M2),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::E4M3 => "f8e4m3",
            Self::E5M2 => "f8e5m2",
        }
    }

    pub fn dtype(self) -> RsDtype {
        match self {
            Self::E4M3 => RsDtype::F8E4M3,
            Self::E5M2 => RsDtype::F8E5M2,
        }
    }

    /// Explicit mantissa bits (the stored fraction width).
    pub const fn mant_bits(self) -> i32 {
        match self {
            Self::E4M3 => 3,
            Self::E5M2 => 2,
        }
    }

    /// Smallest normal exponent (subnormals live below it).
    pub const fn emin(self) -> i32 {
        match self {
            Self::E4M3 => -6,
            Self::E5M2 => -14,
        }
    }

    /// Largest finite exponent.
    pub const fn emax(self) -> i32 {
        match self {
            // OCP E4M3 keeps exponent field 15 (unbiased 8) for normals with
            // mantissas 0..6 and reserves (15, 111) for NaN — no infinities.
            Self::E4M3 => 8,
            Self::E5M2 => 15,
        }
    }

    /// Largest finite magnitude: (2 - 2^-p) * 2^emax, with E4M3's top binade
    /// capped at mantissa 110b (448).
    pub const fn max_finite(self) -> f32 {
        match self {
            Self::E4M3 => 448.0,
            Self::E5M2 => 57344.0,
        }
    }

    pub const fn bias(self) -> i32 {
        match self {
            Self::E4M3 => 7,
            Self::E5M2 => 15,
        }
    }
}

/// Rounds an f32 onto the target fp8 grid (RNE) and returns the 8-bit
/// payload. NaN in → NaN pattern; ±inf in → format's infinity (E5M2) or NaN
/// (E4M3, which has no infinity); overflow past the finite grid behaves the
/// same way.
pub fn f32_to_fp8(x: f32, fmt: Fp8Format) -> u8 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u8;
    let p = fmt.mant_bits();
    if x.is_nan() {
        // Canonical NaN payloads: E4M3 (15,111)=0x7F; E5M2 (31,01)=0x7D.
        return match fmt {
            Fp8Format::E4M3 => 0x7F,
            Fp8Format::E5M2 => 0x7D,
        };
    }
    let ax = x.abs();
    if ax == f32::INFINITY {
        return match fmt {
            Fp8Format::E4M3 => 0x7F,
            Fp8Format::E5M2 => 0x7C | (sign << 7),
        };
    }
    if ax == 0.0 {
        return sign << 7;
    }

    // Work in f64 so the significand rounding is exact. The floor(log2) of a
    // value one f32-ulp below a power of two may round up to the power, but
    // the resulting significand is then < 1 and rounds back up through the
    // carry below — the same final bits either way.
    let m = ax as f64;
    let g = m.log2().floor() as i32;
    let scaled = m / 2f64.powi(g); // in [1, 2)
    let mut k = f64::round_ties_even(scaled * (1i64 << p) as f64) as i64; // [2^p, 2^(p+1)]
    let mut e = g;
    if k == (1i64 << (p + 1)) {
        // Rounded up out of the binade: value becomes 2^(g+1).
        k = 1i64 << p;
        e += 1;
    }
    let mant = (k - (1i64 << p)) as u8;

    if e > fmt.emax() {
        // Overflow.
        return match fmt {
            Fp8Format::E4M3 => 0x7F,
            Fp8Format::E5M2 => 0x7C | (sign << 7),
        };
    }
    if fmt == Fp8Format::E4M3 && e == fmt.emax() && mant == 0b111 {
        // E4M3's top binade stops at mantissa 110b; 111b is the NaN pattern.
        return 0x7F;
    }
    if e >= fmt.emin() {
        // Normal.
        let exp_field = (e + fmt.bias()) as u8;
        return (sign << 7) | (exp_field << p) | mant;
    }
    // Subnormal: grid unit 2^(emin - p). Count units from zero.
    let units = f64::round_ties_even(m * 2f64.powi(p - fmt.emin())) as i64;
    if units >= (1i64 << p) {
        // Rounds up to the smallest normal.
        let exp_field = (fmt.emin() + fmt.bias()) as u8;
        return (sign << 7) | (exp_field << p);
    }
    (sign << 7) | units as u8
}

/// Decodes an fp8 payload to the exact f32 value it denotes. All fp8 values
/// have <= 4 significant bits, so the f64 intermediate casts exactly.
pub fn fp8_to_f32(b: u8, fmt: Fp8Format) -> f32 {
    let sign = if b >> 7 != 0 { -1.0f64 } else { 1.0f64 };
    let p = fmt.mant_bits();
    let (e, mant) = (b >> p, b & ((1u8 << p) - 1));
    let m = mant as f64;
    let v = match fmt {
        Fp8Format::E4M3 => {
            if e == 0b1111 {
                if mant == 0b111 {
                    return f32::NAN;
                }
                // Top binade: 1.m * 2^8, mant in 0..=6.
                (1.0 + m / 8.0) * 256.0
            } else if e == 0 {
                m / 512.0
            } else {
                (1.0 + m / 8.0) * 2f64.powi(e as i32 - 7)
            }
        }
        Fp8Format::E5M2 => {
            if e == 0b11111 {
                return if mant == 0 {
                    (sign as f32) * f32::INFINITY
                } else {
                    f32::NAN
                };
            } else if e == 0 {
                m / 65536.0
            } else {
                (1.0 + m / 4.0) * 2f64.powi(e as i32 - 15)
            }
        }
    };
    (sign * v) as f32
}

// ── scheme plumbing shared by quantize / dequantize / amax_update ───────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scheme {
    PerTensor,
    PerToken,
    PerBlock,
}

fn parse_scheme(a: &RsAttrs, op: &'static str) -> OpResult<Scheme> {
    let s = require_str_of(a, "scheme", SCHEMES, op)?;
    Ok(match s {
        "per_tensor" => Scheme::PerTensor,
        "per_token" => Scheme::PerToken,
        _ => Scheme::PerBlock,
    })
}

fn parse_format(a: &RsAttrs, op: &'static str) -> OpResult<Fp8Format> {
    let f = require_str_of(a, "format", FORMATS, op)?;
    Fp8Format::parse(f).ok_or_else(|| err(op, format!("unknown format '{f}'")))
}

/// The block attribute `[m, n]`, required for `per_block`.
fn block_attr(a: &RsAttrs, op: &'static str) -> OpResult<(u32, u32)> {
    let b = attr_i64s(a, "block").ok_or_else(|| {
        err(
            op,
            "attribute 'block' (list of two i64, e.g. [128, 128]) is required \
             for scheme 'per_block'",
        )
    })?;
    if b.len() != 2 || b[0] <= 0 || b[1] <= 0 {
        return Err(err(
            op,
            format!("attribute 'block' must be [m, n] with m > 0 and n > 0, got {b:?}"),
        ));
    }
    Ok((b[0] as u32, b[1] as u32))
}

/// Scale-tensor shape implied by `scheme` for a data tensor of `shape`.
/// Pure (no allocation); also validates per-block divisibility.
fn scale_shape(
    scheme: Scheme,
    shape: &[i64],
    block: Option<(u32, u32)>,
    op: &'static str,
) -> OpResult<SmallShape> {
    let rank = shape.len();
    if rank == 0 {
        return Err(err(op, "quantized data must have rank >= 1"));
    }
    match scheme {
        Scheme::PerTensor => Ok(SmallShape {
            len: 0,
            dims: [0; MAX_RANK],
        }),
        Scheme::PerToken => {
            // One scale per element of the data shape with the last dim
            // removed: per-token quantizes along the last axis.
            let mut s = SmallShape {
                len: rank - 1,
                dims: [0; MAX_RANK],
            };
            s.dims[..rank - 1].copy_from_slice(&shape[..rank - 1]);
            Ok(s)
        }
        Scheme::PerBlock => {
            if rank < 2 {
                return Err(err(op, "per_block requires data rank >= 2"));
            }
            let (m, n) = block.expect("validated by caller");
            let (rows, cols) = (shape[rank - 2], shape[rank - 1]);
            if rows % m as i64 != 0 || cols % n as i64 != 0 {
                return Err(err(
                    op,
                    format!(
                        "per_block block [{m}, {n}] must divide the last two dims \
                         [{rows}, {cols}] exactly (no partial edge blocks)"
                    ),
                ));
            }
            // One block grid per outer element: shape[..-2] + [rows/m, cols/n].
            let mut s = SmallShape {
                len: rank,
                dims: [0; MAX_RANK],
            };
            s.dims[..rank - 2].copy_from_slice(&shape[..rank - 2]);
            s.dims[rank - 2] = rows / m as i64;
            s.dims[rank - 1] = cols / n as i64;
            Ok(s)
        }
    }
}

/// Validates a scale/amax tensor against the *declared* scheme: the scheme
/// dictates the scale shape, never the other way round (contract R-3).
fn check_scale_shape(
    scale: &RsTensor,
    scheme: Scheme,
    data_dims: &[i64],
    block: Option<(u32, u32)>,
    op: &'static str,
) -> OpResult<SmallShape> {
    let want = scale_shape(scheme, data_dims, block, op)?;
    let ok = match scheme {
        Scheme::PerTensor => scale.numel() == 1,
        Scheme::PerToken => scale.dims() == want.as_slice(),
        Scheme::PerBlock => scale.dims() == want.as_slice(),
    };
    if !ok {
        return Err(err(
            op,
            format!(
                "scale tensor shape {:?} is inconsistent with declared scheme '{:?}' \
                 (expected {:?}) — the scheme is declared, never inferred from shapes",
                scale.dims(),
                scheme,
                want.as_slice()
            ),
        ));
    }
    Ok(want)
}

// ── quantize ────────────────────────────────────────────────────────────────

fn quantize_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(2)?;
    let x = c.in_t(0);
    if x.dtype != RsDtype::F32 {
        return Err(err(c.op, format!("input 'x' has dtype {}, expected f32", x.dtype)));
    }
    let scheme = parse_scheme(a, c.op)?;
    let fmt = parse_format(a, c.op)?;
    let block = if scheme == Scheme::PerBlock {
        Some(block_attr(a, c.op)?)
    } else {
        None
    };
    let sshape = scale_shape(scheme, x.dims(), block, c.op)?;
    // Out 0: the quantized payload, one byte per element in the target grid.
    set_output_desc(c.out_t(0), fmt.dtype(), x.dims());
    // Out 1: the f32 scale tensor whose shape the declared scheme dictates.
    set_output_desc(c.out_t(1), RsDtype::F32, sshape.as_slice());
    Ok(())
}

fn quantize_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(2)?;
    let x = c.in_t(0);
    let scheme = parse_scheme(a, c.op)?;
    let fmt = parse_format(a, c.op)?;
    let block = if scheme == Scheme::PerBlock {
        Some(block_attr(a, c.op)?)
    } else {
        None
    };
    let sshape = scale_shape(scheme, x.dims(), block, c.op)?;
    expect_out(c.out_t(0), c.op, fmt.dtype(), x.dims())?;
    expect_out(c.out_t(1), c.op, RsDtype::F32, sshape.as_slice())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let mut qv = unsafe { crate::tensor::u8_out(c.op, c.out_t(0)) }?;
    let mut sv = unsafe { crate::tensor::f32_out(c.op, c.out_t(1)) }?;
    let maxf = fmt.max_finite();

    // scale = amax / max_finite, so every x/scale lands inside the finite
    // grid and the emulation never overflows. A zero amax gets scale 1.0 to
    // avoid 0/0; zero quantizes to zero under any scale.
    // Uniform element order everywhere (ascending), per-scheme aggregation.
    match scheme {
        Scheme::PerTensor => {
            let mut amax = 0.0f32;
            for &v in xv.iter() {
                if v.is_nan() {
                    return Err(err(c.op, "input 'x' contains NaN; quantize refuses NaN"));
                }
                amax = amax.max(v.abs());
            }
            let scale = if amax == 0.0 { 1.0 } else { amax / maxf };
            sv.iter_mut().next().map(|s| *s = scale);
            for (q, &v) in qv.iter_mut().zip(xv.iter()) {
                *q = f32_to_fp8(v / scale, fmt);
            }
        }
        Scheme::PerToken => {
            let last = xv.shape()[xv.ndim() - 1];
            let outer = xv.len() / last;
            for oi in 0..outer {
                let mut amax = 0.0f32;
                for j in 0..last {
                    let v = xv[oi * last + j];
                    if v.is_nan() {
                        return Err(err(c.op, "input 'x' contains NaN; quantize refuses NaN"));
                    }
                    amax = amax.max(v.abs());
                }
                let scale = if amax == 0.0 { 1.0 } else { amax / maxf };
                sv[oi] = scale;
                for j in 0..last {
                    qv[oi * last + j] = f32_to_fp8(xv[oi * last + j] / scale, fmt);
                }
            }
        }
        Scheme::PerBlock => {
            let (bm, bn) = block.expect("validated");
            let (bm, bn) = (bm as usize, bn as usize);
            let last_two = [xv.shape()[xv.ndim() - 2], xv.shape()[xv.ndim() - 1]];
            let (dr, dc) = (last_two[0], last_two[1]);
            let outer = xv.len() / (dr * dc);
            let (gr, gc) = (dr / bm, dc / bn);
            for oi in 0..outer {
                for br in 0..gr {
                    for bc in 0..gc {
                        let mut amax = 0.0f32;
                        for i in 0..bm {
                            for j in 0..bn {
                                let v = xv[oi * dr * dc + (br * bm + i) * dc + (bc * bn + j)];
                                if v.is_nan() {
                                    return Err(
                                        err(c.op, "input 'x' contains NaN; quantize refuses NaN"),
                                    );
                                }
                                amax = amax.max(v.abs());
                            }
                        }
                        let scale = if amax == 0.0 { 1.0 } else { amax / maxf };
                        sv[oi * gr * gc + br * gc + bc] = scale;
                        for i in 0..bm {
                            for j in 0..bn {
                                let idx = oi * dr * dc + (br * bm + i) * dc + (bc * bn + j);
                                qv[idx] = f32_to_fp8(xv[idx] / scale, fmt);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

// ── dequantize ──────────────────────────────────────────────────────────────

fn dequantize_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let q = c.in_t(0);
    let fmt = parse_format(a, c.op)?;
    if q.dtype != fmt.dtype() {
        return Err(err(
            c.op,
            format!(
                "input 'q' has dtype {}, expected {} for format '{}'",
                q.dtype,
                fmt.dtype(),
                fmt.name()
            ),
        ));
    }
    let scale = c.in_t(1);
    if scale.dtype != RsDtype::F32 {
        return Err(err(
            c.op,
            format!("input 'scale' has dtype {}, expected f32", scale.dtype),
        ));
    }
    let scheme = parse_scheme(a, c.op)?;
    let block = if scheme == Scheme::PerBlock {
        Some(block_attr(a, c.op)?)
    } else {
        None
    };
    check_scale_shape(scale, scheme, q.dims(), block, c.op)?;
    let o = c.out_t(0);
    set_output_desc(o, RsDtype::F32, q.dims());
    Ok(())
}

fn dequantize_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let q = c.in_t(0);
    let fmt = parse_format(a, c.op)?;
    let scheme = parse_scheme(a, c.op)?;
    let block = if scheme == Scheme::PerBlock {
        Some(block_attr(a, c.op)?)
    } else {
        None
    };
    check_scale_shape(c.in_t(1), scheme, q.dims(), block, c.op)?;
    expect_out(c.out_t(0), c.op, RsDtype::F32, q.dims())?;
    if q.rank == 0 {
        return Err(err(c.op, "dequantize expects q with rank >= 1"));
    }
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let qb = unsafe { crate::tensor::u8_in(c.op, "q", q) }?;
    let sv = unsafe { crate::tensor::f32_in(c.op, "scale", c.in_t(1)) }?;
    let mut xv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;

    match scheme {
        Scheme::PerTensor => {
            let s = sv.iter().next().copied().unwrap_or(1.0);
            for (o, &b) in xv.iter_mut().zip(qb.iter()) {
                *o = fp8_to_f32(b, fmt) * s;
            }
        }
        Scheme::PerToken => {
            let last = q.dims()[q.rank as usize - 1] as usize;
            let outer = qb.len() / last;
            for oi in 0..outer {
                let s = sv[oi];
                for j in 0..last {
                    xv[oi * last + j] = fp8_to_f32(qb[oi * last + j], fmt) * s;
                }
            }
        }
        Scheme::PerBlock => {
            let (bm, bn) = block.expect("validated");
            let (bm, bn) = (bm as usize, bn as usize);
            let rank = q.rank as usize;
            let dr = q.dims()[rank - 2] as usize;
            let dc = q.dims()[rank - 1] as usize;
            let outer = qb.len() / (dr * dc);
            let (gr, gc) = (dr / bm, dc / bn);
            for oi in 0..outer {
                for br in 0..gr {
                    for bc in 0..gc {
                        let s = sv[oi * gr * gc + br * gc + bc];
                        for i in 0..bm {
                            for j in 0..bn {
                                let idx = oi * dr * dc + (br * bm + i) * dc + (bc * bn + j);
                                xv[idx] = fp8_to_f32(qb[idx], fmt) * s;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

// ── amax_update ─────────────────────────────────────────────────────────────

fn amax_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let amax = c.in_t(1);
    if x.dtype != RsDtype::F32 || amax.dtype != RsDtype::F32 {
        return Err(err(c.op, "amax_update expects f32 inputs"));
    }
    let scheme = parse_scheme(a, c.op)?;
    let block = if scheme == Scheme::PerBlock {
        Some(block_attr(a, c.op)?)
    } else {
        None
    };
    check_scale_shape(amax, scheme, x.dims(), block, c.op)?;
    // The running max keeps its shape.
    set_output_desc(c.out_t(0), RsDtype::F32, amax.dims());
    Ok(())
}

fn amax_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let amax_in = c.in_t(1);
    let scheme = parse_scheme(a, c.op)?;
    let block = if scheme == Scheme::PerBlock {
        Some(block_attr(a, c.op)?)
    } else {
        None
    };
    check_scale_shape(amax_in, scheme, x.dims(), block, c.op)?;
    expect_out(c.out_t(0), c.op, RsDtype::F32, amax_in.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let av = unsafe { crate::tensor::f32_in(c.op, "amax", amax_in) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;

    // amax' = max(amax, |x|) aggregated under the declared scheme. NaN is
    // rejected: it would silently poison the running max for all future steps.
    match scheme {
        Scheme::PerTensor => {
            let mut m = av.iter().next().copied().unwrap_or(0.0);
            for &v in xv.iter() {
                if v.is_nan() {
                    return Err(err(c.op, "input 'x' contains NaN; amax_update refuses NaN"));
                }
                m = m.max(v.abs());
            }
            if let Some(o) = yv.iter_mut().next() {
                *o = m;
            }
        }
        Scheme::PerToken => {
            let last = xv.shape()[xv.ndim() - 1];
            let outer = xv.len() / last;
            for oi in 0..outer {
                let mut m = av[oi];
                for j in 0..last {
                    let v = xv[oi * last + j];
                    if v.is_nan() {
                        return Err(err(c.op, "input 'x' contains NaN; amax_update refuses NaN"));
                    }
                    m = m.max(v.abs());
                }
                yv[oi] = m;
            }
        }
        Scheme::PerBlock => {
            let (bm, bn) = block.expect("validated");
            let (bm, bn) = (bm as usize, bn as usize);
            let rank = x.rank as usize;
            let dr = x.dims()[rank - 2] as usize;
            let dc = x.dims()[rank - 1] as usize;
            let outer = xv.len() / (dr * dc);
            let (gr, gc) = (dr / bm, dc / bn);
            for oi in 0..outer {
                for br in 0..gr {
                    for bc in 0..gc {
                        let s_idx = oi * gr * gc + br * gc + bc;
                        let mut m = av[s_idx];
                        for i in 0..bm {
                            for j in 0..bn {
                                let v = xv[oi * dr * dc + (br * bm + i) * dc + (bc * bn + j)];
                                if v.is_nan() {
                                    return Err(
                                        err(c.op, "input 'x' contains NaN; amax_update refuses NaN"),
                                    );
                                }
                                m = m.max(v.abs());
                            }
                        }
                        yv[s_idx] = m;
                    }
                }
            }
        }
    }
    Ok(())
}

infer_entry!(quantize_infer, "quantize", quantize_infer_body);
exec_entry!(quantize_exec, "quantize", quantize_exec_body);
infer_entry!(dequantize_infer, "dequantize", dequantize_infer_body);
exec_entry!(dequantize_exec, "dequantize", dequantize_exec_body);
infer_entry!(amax_infer, "amax_update", amax_infer_body);
exec_entry!(amax_exec, "amax_update", amax_exec_body);
