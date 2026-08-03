//! Guards the signal path against unintended change.
//!
//! `tests/fixtures/rnnoise_v1_reference.txt` holds per-frame statistics captured from
//! nnnoiseless 0.1.0, which was a faithful transcription of the original RNNoise. Optimization
//! work has to leave the output alone, and this is what proves it.
//!
//! The comparison is by tolerance rather than bit-for-bit. Reassociating a sum, contracting a
//! multiply-add into an FMA, or planning an FFT differently all perturb the last few bits, and
//! because the network is recurrent those perturbations accumulate. What must not change is
//! the audible result, so the fixture stores frame energy, frame peak and the voice-activity
//! probability, and the test bounds how far each may drift.
//!
//! If you change the algorithm on purpose, regenerate the fixture and say so in the commit.

use nnnoiseless::{DenoiseState, FRAME_SIZE};

/// The exact signal the fixture was captured from: harmonics with a slow envelope, coloured
/// noise, and a short broadband burst to exercise transient handling.
fn fixture_input() -> Vec<f32> {
    let n = 48_000 * 3;
    let mut seed = 0x2545f491u32;
    let mut lp = 0.0f32;
    (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let w = ((seed >> 16) as i32 - 32768) as f32 / 32768.0;
            lp = 0.85 * lp + 0.15 * w;
            let t = i as f32 / 48_000.0;
            let env = (0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin()).powf(1.5);
            let mut s = 0.0;
            for h in 1..=12 {
                s += (2.0 * std::f32::consts::PI * 150.0 * h as f32 * t).sin()
                    / (h as f32).powf(1.2);
            }
            let burst = if (24_000..24_400).contains(&i) {
                8000.0
            } else {
                0.0
            };
            s * env * 6000.0 + (w * 0.5 + lp * 2.0) * 1200.0 + burst * w
        })
        .collect()
}

struct FrameStats {
    rms: f32,
    peak: f32,
    vad: f32,
}

fn reference() -> Vec<FrameStats> {
    let text = include_str!("fixtures/rnnoise_v1_reference.txt");
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            let mut next = || it.next().unwrap().parse::<f32>().unwrap();
            FrameStats {
                rms: next(),
                peak: next(),
                vad: next(),
            }
        })
        .collect()
}

fn measure() -> Vec<FrameStats> {
    let input = fixture_input();
    let mut state = DenoiseState::new();
    let mut out = vec![0.0; FRAME_SIZE];
    input
        .chunks_exact(FRAME_SIZE)
        .map(|frame| {
            let vad = state.process_frame(&mut out, frame);
            FrameStats {
                rms: (out.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
                    / FRAME_SIZE as f64)
                    .sqrt() as f32,
                peak: out.iter().fold(0.0f32, |a, b| a.max(b.abs())),
                vad,
            }
        })
        .collect()
}

/// Frame-by-frame agreement with the original implementation.
#[test]
fn output_still_matches_the_original_rnnoise_path() {
    let want = reference();
    let got = measure();
    assert_eq!(got.len(), want.len(), "frame count changed");

    // The signal is around 2700 RMS, so this floor keeps near-silent frames (where a relative
    // comparison is meaningless) from producing spurious failures.
    let floor = 50.0f32;

    let mut worst_rms = 0.0f32;
    let mut worst_vad = 0.0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let denom = w.rms.max(floor);
        let rel = (g.rms - w.rms).abs() / denom;
        worst_rms = worst_rms.max(rel);
        assert!(
            rel < 0.05,
            "frame {i}: RMS drifted {:.1}% ({} vs {})",
            rel * 100.0,
            g.rms,
            w.rms
        );

        let peak_rel = (g.peak - w.peak).abs() / w.peak.max(floor);
        assert!(
            peak_rel < 0.10,
            "frame {i}: peak drifted {:.1}% ({} vs {})",
            peak_rel * 100.0,
            g.peak,
            w.peak
        );

        let vad_delta = (g.vad - w.vad).abs();
        worst_vad = worst_vad.max(vad_delta);
        assert!(
            vad_delta < 0.05,
            "frame {i}: VAD drifted by {vad_delta} ({} vs {})",
            g.vad,
            w.vad
        );
    }

    println!("worst frame RMS drift: {:.3}%", worst_rms * 100.0);
    println!("worst VAD drift:       {worst_vad:.6}");
}

/// Aggregate energy must be preserved much more tightly than any single frame.
#[test]
fn total_energy_matches_the_original() {
    let want = reference();
    let got = measure();

    let total = |s: &[FrameStats]| -> f64 {
        (s.iter()
            .map(|f| (f.rms as f64) * (f.rms as f64))
            .sum::<f64>()
            / s.len() as f64)
            .sqrt()
    };
    let (g, w) = (total(&got), total(&want));
    let rel = ((g - w) / w).abs();
    println!("overall RMS: {g:.3} vs {w:.3} ({:.4}% drift)", rel * 100.0);
    assert!(rel < 0.005, "overall energy drifted by {:.3}%", rel * 100.0);
}

/// Processing is deterministic: the same input must always give the same output.
#[test]
fn processing_is_deterministic() {
    let a = measure();
    let b = measure();
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.rms, y.rms, "frame {i} differed between runs");
        assert_eq!(x.vad, y.vad, "frame {i} VAD differed between runs");
    }
}

/// Feeding audio in one frame at a time and in bigger batches must give the same answer, since
/// the state is supposed to carry everything across calls.
#[test]
fn streaming_is_independent_of_call_pattern() {
    let input = fixture_input();
    let mut a = DenoiseState::new();
    let mut out = vec![0.0; FRAME_SIZE];
    let mut first = Vec::new();
    for frame in input.chunks_exact(FRAME_SIZE) {
        a.process_frame(&mut out, frame);
        first.extend_from_slice(&out);
    }

    // Same data, but the caller happens to hold it in one big buffer.
    let mut b = DenoiseState::new();
    let mut second = Vec::new();
    let mut buf = vec![0.0; FRAME_SIZE];
    for chunk in input.chunks_exact(FRAME_SIZE * 7) {
        for frame in chunk.chunks_exact(FRAME_SIZE) {
            b.process_frame(&mut buf, frame);
            second.extend_from_slice(&buf);
        }
    }
    let n = second.len();
    assert_eq!(&first[..n], &second[..], "call pattern changed the output");
}
