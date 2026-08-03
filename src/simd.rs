//! Runtime-dispatched numerical kernels.
//!
//! Profiling showed that the dominant cost in this crate is not the algorithm but the
//! instruction set the hot loops get compiled for: a plain `-C target-cpu=native` build was
//! ~1.45x faster than the default baseline-x86-64 build. A published library cannot require
//! its users to set `RUSTFLAGS`, so the four hot kernels are compiled once per supported
//! instruction set and selected at first use with runtime feature detection.
//!
//! The bodies are written once, as `#[inline(always)]` portable functions, and instantiated
//! into `#[target_feature]` wrappers. Inlining into a `#[target_feature]` function is what
//! makes the body get compiled with those features enabled.
//!
//! Building with the `reference` feature pins everything to the scalar path, which produces
//! identical results on every machine.

use crate::{BandTables, Complex, NB_BANDS};

/// The instruction set the hot kernels are using.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// Portable code, with whatever the compiler's baseline target offers.
    Scalar,
    /// x86-64 AVX2 with fused multiply-add.
    Avx2Fma,
    /// AArch64 Advanced SIMD.
    Neon,
    /// WebAssembly fixed-width SIMD.
    ///
    /// Unlike the others this is a compile-time decision: WebAssembly has no runtime feature
    /// detection, so `simd128` has to be enabled when the module is built.
    Simd128,
}

impl std::fmt::Display for Isa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Isa::Scalar => "scalar",
            Isa::Avx2Fma => "avx2+fma",
            Isa::Neon => "neon",
            Isa::Simd128 => "simd128",
        };
        f.write_str(s)
    }
}

/// Function pointers for the kernels, resolved once from CPU feature detection.
pub(crate) struct Kernels {
    pub(crate) isa: Isa,
    /// Inner product of two equal-length slices.
    pub(crate) dot: fn(&[f32], &[f32]) -> f32,
    /// `out[i] = sum_j xs[j] * ys[i + j]`, for every `i` in `0..out.len()`.
    pub(crate) xcorr: fn(&[f32], &[f32], &mut [f32]),
    /// `out[j] += sum_i w[i * stride + offset + j] * input[i]`. Unused under `low-memory`,
    /// which keeps the weights quantized and goes through `matvec_i8` instead.
    #[cfg_attr(feature = "low-memory", allow(dead_code))]
    pub(crate) matvec: fn(&[f32], usize, usize, &mut [f32], &[f32]),
    /// As `matvec`, but widening `i8` weights on the fly. Only read under `low-memory`.
    #[cfg_attr(not(feature = "low-memory"), allow(dead_code))]
    pub(crate) matvec_i8: fn(&[i8], usize, usize, &mut [f32], &[f32]),
    /// Band-aggregated correlation between two spectra.
    pub(crate) band_corr: fn(&mut [f32], &[Complex], &[Complex], &BandTables),
}

impl Kernels {
    pub(crate) fn detect() -> Kernels {
        #[cfg(not(feature = "reference"))]
        {
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return Kernels {
                    isa: Isa::Avx2Fma,
                    dot: avx2::dot,
                    xcorr: avx2::xcorr,
                    matvec: avx2::matvec,
                    matvec_i8: avx2::matvec_i8,
                    band_corr: avx2::band_corr,
                };
            }

            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Kernels {
                    isa: Isa::Neon,
                    dot: neon::dot,
                    xcorr: neon::xcorr,
                    matvec: neon::matvec,
                    matvec_i8: neon::matvec_i8,
                    band_corr: neon::band_corr,
                };
            }
        }

        // WebAssembly cannot probe for features at runtime, so the best we can do is report
        // what the module was compiled with. The portable bodies vectorize fine under
        // `-C target-feature=+simd128`.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let isa = Isa::Simd128;
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        let isa = Isa::Scalar;

        Kernels {
            isa,
            dot: dot_body,
            xcorr: xcorr_body,
            matvec: matvec_body,
            matvec_i8: matvec_i8_body,
            band_corr: band_corr_body,
        }
    }
}

// ---------------------------------------------------------------------------------------
// Kernel bodies. Written once, portable, and instantiated per instruction set below.
// ---------------------------------------------------------------------------------------

/// Number of independent accumulators. Rust will not reassociate floating-point additions,
/// so a naive `sum()` compiles to a serial dependency chain that no vector unit can help
/// with. Eight lanes fills one AVX register and splits cleanly on narrower units.
const LANES: usize = 8;

#[inline(always)]
fn dot_body(xs: &[f32], ys: &[f32]) -> f32 {
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    let n_lanes = n - n % LANES;

    let mut acc = [0.0f32; LANES];
    for (x, y) in xs[..n_lanes]
        .chunks_exact(LANES)
        .zip(ys[..n_lanes].chunks_exact(LANES))
    {
        for k in 0..LANES {
            acc[k] += x[k] * y[k];
        }
    }

    let mut sum = 0.0;
    for &a in &acc {
        sum += a;
    }
    for (&x, &y) in xs[n_lanes..].iter().zip(&ys[n_lanes..]) {
        sum += x * y;
    }
    sum
}

#[inline(always)]
fn xcorr_body(xs: &[f32], ys: &[f32], xcorr: &mut [f32]) {
    // Computing four lags at a time lets each loaded `ys` value be reused four times, which
    // is what makes this memory-bound loop fast.
    let xcorr_len_4 = xcorr.len() - xcorr.len() % 4;
    let xs_len_4 = xs.len() - xs.len() % 4;

    for i in (0..xcorr_len_4).step_by(4) {
        let mut c0 = 0.0;
        let mut c1 = 0.0;
        let mut c2 = 0.0;
        let mut c3 = 0.0;

        let mut y0 = ys[i];
        let mut y1 = ys[i + 1];
        let mut y2 = ys[i + 2];
        let mut y3 = ys[i + 3];

        for (x, y) in xs.chunks_exact(4).zip(ys[(i + 4)..].chunks_exact(4)) {
            c0 += x[0] * y0;
            c1 += x[0] * y1;
            c2 += x[0] * y2;
            c3 += x[0] * y3;

            y0 = y[0];
            c0 += x[1] * y1;
            c1 += x[1] * y2;
            c2 += x[1] * y3;
            c3 += x[1] * y0;

            y1 = y[1];
            c0 += x[2] * y2;
            c1 += x[2] * y3;
            c2 += x[2] * y0;
            c3 += x[2] * y1;

            y2 = y[2];
            c0 += x[3] * y3;
            c1 += x[3] * y0;
            c2 += x[3] * y1;
            c3 += x[3] * y2;

            y3 = y[3];
        }

        for j in xs_len_4..xs.len() {
            c0 += xs[j] * ys[i + j];
            c1 += xs[j] * ys[i + 1 + j];
            c2 += xs[j] * ys[i + 2 + j];
            c3 += xs[j] * ys[i + 3 + j];
        }
        xcorr[i] = c0;
        xcorr[i + 1] = c1;
        xcorr[i + 2] = c2;
        xcorr[i + 3] = c3;
    }

    for i in xcorr_len_4..xcorr.len() {
        xcorr[i] = dot_body(xs, &ys[i..(i + xs.len())]);
    }
}

#[inline(always)]
fn matvec_body(w: &[f32], stride: usize, offset: usize, out: &mut [f32], input: &[f32]) {
    let n = out.len();
    debug_assert!(offset + n <= stride);
    for (col, &inp) in w.chunks_exact(stride).zip(input) {
        let col = &col[offset..(offset + n)];
        for (&x, o) in col.iter().zip(out.iter_mut()) {
            *o += x * inp;
        }
    }
}

#[inline(always)]
fn matvec_i8_body(w: &[i8], stride: usize, offset: usize, out: &mut [f32], input: &[f32]) {
    let n = out.len();
    debug_assert!(offset + n <= stride);
    for (col, &inp) in w.chunks_exact(stride).zip(input) {
        let col = &col[offset..(offset + n)];
        for (&x, o) in col.iter().zip(out.iter_mut()) {
            *o += (x as f32) * inp;
        }
    }
}

#[inline(always)]
fn band_corr_body(out: &mut [f32], x: &[Complex], p: &[Complex], bands: &BandTables) {
    for y in out.iter_mut() {
        *y = 0.0;
    }

    let frac = bands.frac();
    for i in 0..(NB_BANDS - 1) {
        let r = BandTables::band_range(i);
        let mut lo = 0.0f32;
        let mut hi = 0.0f32;
        for ((xj, pj), &f) in x[r.clone()].iter().zip(&p[r.clone()]).zip(&frac[r]) {
            let corr = xj.re * pj.re + xj.im * pj.im;
            hi += f * corr;
            lo += corr - f * corr;
        }
        out[i] += lo;
        out[i + 1] += hi;
    }
    out[0] *= 2.0;
    out[NB_BANDS - 1] *= 2.0;
}

// ---------------------------------------------------------------------------------------
// Instruction-set instantiations.
// ---------------------------------------------------------------------------------------

/// Defines safe wrappers around `#[target_feature]` instantiations of the kernel bodies.
///
/// The wrappers are only ever installed into `Kernels` after the corresponding runtime
/// feature check has passed, which is what makes the `unsafe` calls sound.
// Unused when the `reference` feature pins everything to the scalar path.
#[allow(unused_macros)]
macro_rules! instantiate {
    ($module:ident, $($feature:literal),+) => {
        mod $module {
            use super::*;

            $(#[target_feature(enable = $feature)])+
            unsafe fn dot_impl(xs: &[f32], ys: &[f32]) -> f32 { dot_body(xs, ys) }
            $(#[target_feature(enable = $feature)])+
            unsafe fn xcorr_impl(xs: &[f32], ys: &[f32], out: &mut [f32]) { xcorr_body(xs, ys, out) }
            $(#[target_feature(enable = $feature)])+
            unsafe fn matvec_impl(w: &[f32], s: usize, o: usize, out: &mut [f32], i: &[f32]) {
                matvec_body(w, s, o, out, i)
            }
            $(#[target_feature(enable = $feature)])+
            unsafe fn matvec_i8_impl(w: &[i8], s: usize, o: usize, out: &mut [f32], i: &[f32]) {
                matvec_i8_body(w, s, o, out, i)
            }
            $(#[target_feature(enable = $feature)])+
            unsafe fn band_corr_impl(
                out: &mut [f32], x: &[Complex], p: &[Complex], b: &BandTables,
            ) {
                band_corr_body(out, x, p, b)
            }

            pub(super) fn dot(xs: &[f32], ys: &[f32]) -> f32 {
                unsafe { dot_impl(xs, ys) }
            }
            pub(super) fn xcorr(xs: &[f32], ys: &[f32], out: &mut [f32]) {
                unsafe { xcorr_impl(xs, ys, out) }
            }
            pub(super) fn matvec(w: &[f32], s: usize, o: usize, out: &mut [f32], i: &[f32]) {
                unsafe { matvec_impl(w, s, o, out, i) }
            }
            pub(super) fn matvec_i8(w: &[i8], s: usize, o: usize, out: &mut [f32], i: &[f32]) {
                unsafe { matvec_i8_impl(w, s, o, out, i) }
            }
            pub(super) fn band_corr(
                out: &mut [f32], x: &[Complex], p: &[Complex], b: &BandTables,
            ) {
                unsafe { band_corr_impl(out, x, p, b) }
            }
        }
    };
}

#[cfg(all(target_arch = "x86_64", not(feature = "reference")))]
instantiate!(avx2, "avx2", "fma");

#[cfg(all(target_arch = "aarch64", not(feature = "reference")))]
instantiate!(neon, "neon");

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 16) as i32 - 32768) as f32 / 32768.0
            })
            .collect()
    }

    /// The dispatched kernels must agree with the portable bodies. If the CPU has no SIMD
    /// support this compares the scalar path against itself, which is still worth running.
    #[test]
    fn dispatched_kernels_agree_with_scalar() {
        let k = Kernels::detect();
        let bands = BandTables::new();

        for n in [1usize, 3, 8, 17, 64, 480] {
            let a = pseudo_random(n, 1);
            let b = pseudo_random(n, 2);
            let got = (k.dot)(&a, &b);
            let want = dot_body(&a, &b);
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "dot n={n}: {got} vs {want}"
            );
        }

        // cross-correlation
        let xs = pseudo_random(64, 3);
        let ys = pseudo_random(128, 4);
        let mut got = vec![0.0; 32];
        let mut want = vec![0.0; 32];
        (k.xcorr)(&xs, &ys, &mut got);
        xcorr_body(&xs, &ys, &mut want);
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "xcorr: {g} vs {w}"
            );
        }

        // matvec
        let (nin, n, stride, offset) = (42usize, 24usize, 72usize, 24usize);
        let w = pseudo_random(nin * stride, 5);
        let input = pseudo_random(nin, 6);
        let mut got = vec![0.0; n];
        let mut want = vec![0.0; n];
        (k.matvec)(&w, stride, offset, &mut got, &input);
        matvec_body(&w, stride, offset, &mut want, &input);
        for (g, wv) in got.iter().zip(&want) {
            assert!(
                (g - wv).abs() <= 1e-3 * wv.abs().max(1.0),
                "matvec: {g} vs {wv}"
            );
        }

        // band correlation
        let spec: Vec<Complex> = pseudo_random(crate::FREQ_SIZE * 2, 7)
            .chunks_exact(2)
            .map(|c| Complex::new(c[0], c[1]))
            .collect();
        let mut got = [0.0; NB_BANDS];
        let mut want = [0.0; NB_BANDS];
        (k.band_corr)(&mut got, &spec, &spec, &bands);
        band_corr_body(&mut want, &spec, &spec, &bands);
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "band_corr: {g} vs {w}"
            );
        }
    }

    /// `band_corr` of a spectrum against itself is the band energy, which cannot be negative.
    #[test]
    fn band_energies_are_non_negative() {
        let bands = BandTables::new();
        let spec: Vec<Complex> = pseudo_random(crate::FREQ_SIZE * 2, 11)
            .chunks_exact(2)
            .map(|c| Complex::new(c[0], c[1]))
            .collect();
        let mut e = [0.0; NB_BANDS];
        band_corr_body(&mut e, &spec, &spec, &bands);
        assert!(e.iter().all(|&x| x >= 0.0), "{e:?}");
    }
}
