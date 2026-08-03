//! Shared signal generation and metrics for the integration tests.
//!
//! Everything here is deterministic, so a failing test can be reproduced exactly.

#![allow(dead_code)]

pub const SAMPLE_RATE: f32 = 48_000.0;

/// A small linear-congruential generator, so the tests do not need a dependency and produce
/// the same audio on every platform.
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Rng {
        Rng(seed)
    }

    /// Uniform in `-1.0..1.0`.
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        ((self.0 >> 16) as i32 - 32768) as f32 / 32768.0
    }
}

/// Synthetic voiced speech: a harmonic stack with a moving pitch, a syllable-rate envelope
/// and pauses between "words".
///
/// This is not a substitute for real recordings, but it does have the properties the
/// algorithm keys off: harmonic structure, a detectable pitch, onsets, and silences.
pub fn speech(n: usize, seed: u32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let jitter = rng.next_f32();
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE;
            // Syllables at ~3.5Hz, with real gaps between them.
            let syl = (2.0 * std::f32::consts::PI * 3.5 * t).sin();
            let env = if syl > 0.0 { syl.powf(0.6) } else { 0.0 };
            // A pitch that drifts, like an intonation contour.
            let f0 = 145.0 + 25.0 * (2.0 * std::f32::consts::PI * 0.7 * t + jitter).sin();
            let mut s = 0.0;
            for h in 1..=14 {
                let a = 1.0 / (h as f32).powf(1.15);
                s += a * (2.0 * std::f32::consts::PI * f0 * h as f32 * t).sin();
            }
            s * env * 6000.0
        })
        .collect()
}

/// Stationary coloured noise, roughly like a ventilation or hiss floor.
pub fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut lp = 0.0f32;
    (0..n)
        .map(|_| {
            let w = rng.next_f32();
            lp = 0.85 * lp + 0.15 * w;
            w * 0.5 + lp * 2.0
        })
        .collect()
}

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64).sqrt() as f32
}

pub fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

/// Mixes `clean` and `noise` at the requested signal-to-noise ratio, scaling the noise.
pub fn mix_at_snr(clean: &[f32], noise: &[f32], snr_db: f32) -> Vec<f32> {
    let target_noise = rms(clean) / 10f32.powf(snr_db / 20.0);
    let scale = target_noise / rms(noise).max(1e-12);
    clean
        .iter()
        .zip(noise)
        .map(|(&c, &n)| c + n * scale)
        .collect()
}

/// Scale-invariant signal-to-distortion ratio, in dB.
///
/// This is the standard separation metric: it projects the estimate onto the reference, so it
/// is not fooled by an overall level change, only by actual distortion.
pub fn si_sdr(estimate: &[f32], reference: &[f32]) -> f32 {
    let n = estimate.len().min(reference.len());
    let (e, r) = (&estimate[..n], &reference[..n]);

    let dot: f64 = e.iter().zip(r).map(|(&a, &b)| a as f64 * b as f64).sum();
    let energy: f64 = r.iter().map(|&b| (b as f64) * (b as f64)).sum();
    if energy <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let alpha = dot / energy;

    let mut target = 0.0f64;
    let mut noise = 0.0f64;
    for (&a, &b) in e.iter().zip(r) {
        let t = alpha * b as f64;
        target += t * t;
        let d = a as f64 - t;
        noise += d * d;
    }
    if noise <= 0.0 {
        return f32::INFINITY;
    }
    (10.0 * (target / noise).log10()) as f32
}

/// Segmental SNR: the SNR computed per short segment and then averaged in dB.
///
/// Averaging in the log domain stops loud passages from dominating, which makes this track
/// perceived quality better than a single global SNR does. Segments where the reference is
/// essentially silent are skipped, and each segment is clamped to a sane range, both of which
/// are standard for this measure.
pub fn segmental_snr(estimate: &[f32], reference: &[f32], seg: usize) -> f32 {
    let n = estimate.len().min(reference.len());
    let floor = rms(&reference[..n]) * 0.01;

    let mut total = 0.0f64;
    let mut count = 0usize;
    for start in (0..n).step_by(seg) {
        let end = (start + seg).min(n);
        if end - start < seg / 2 {
            break;
        }
        let r = &reference[start..end];
        if rms(r) < floor {
            continue;
        }
        let mut sig = 0.0f64;
        let mut err = 0.0f64;
        for (&a, &b) in estimate[start..end].iter().zip(r) {
            sig += (b as f64) * (b as f64);
            let d = a as f64 - b as f64;
            err += d * d;
        }
        // An exact match has zero error; count it as the clamp ceiling rather than
        // skipping it, or a perfect estimate would score `-inf` for want of any segments.
        total += if err <= 0.0 {
            35.0
        } else {
            (10.0 * (sig / err).log10()).clamp(-10.0, 35.0)
        };
        count += 1;
    }
    if count == 0 {
        return f32::NEG_INFINITY;
    }
    (total / count as f64) as f32
}

/// Denoises a whole signal, aligned with the input. Thin wrapper over the library helper so
/// the tests exercise the same code path users would.
pub fn denoise_aligned(params: nnnoiseless::DenoiseParams, input: &[f32]) -> Vec<f32> {
    nnnoiseless::denoise_offline(params, input)
}
