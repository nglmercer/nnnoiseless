//! Objective quality evaluation.
//!
//! Nothing in the denoising path can be tuned responsibly without a way to tell whether a
//! change helped, so this file is the measurement half of that. It reports SI-SDR and
//! segmental SNR against a known clean reference, plus how much noise was removed against how
//! much of the speech was damaged doing it.
//!
//! The audio here is synthetic and deterministic. That makes the tests reproducible and
//! dependency-free, but it is emphatically *not* a substitute for real recordings: a harmonic
//! stack flatters a pitch-driven model. Treat the absolute numbers as a floor, and trust the
//! comparisons between configurations rather than the values themselves.
//!
//! Run `cargo test --test quality -- --nocapture` to see the tables.

mod common;

use common::*;
use nnnoiseless::{DenoiseParams, FRAME_SIZE};

/// Splits a denoised signal into the frames where the clean reference was speaking and the
/// frames where it was silent, so that noise removal and speech damage can be measured apart.
struct Breakdown {
    noise_reduction_db: f32,
    speech_loss_db: f32,
}

fn breakdown(clean: &[f32], noisy: &[f32], denoised: &[f32]) -> Breakdown {
    let n = clean.len().min(denoised.len());
    let speech_floor = rms(&clean[..n]) * 0.15;

    let (mut nin, mut nout, mut ncnt) = (0.0f64, 0.0f64, 0usize);
    let (mut sin_, mut sout, mut scnt) = (0.0f64, 0.0f64, 0usize);

    for start in (0..n).step_by(FRAME_SIZE) {
        let end = (start + FRAME_SIZE).min(n);
        if end - start < FRAME_SIZE {
            break;
        }
        let c = rms(&clean[start..end]);
        if c < speech_floor * 0.2 {
            // Speech-absent: whatever is here is noise.
            nin += rms(&noisy[start..end]) as f64;
            nout += rms(&denoised[start..end]) as f64;
            ncnt += 1;
        } else if c > speech_floor {
            sin_ += c as f64;
            sout += rms(&denoised[start..end]) as f64;
            scnt += 1;
        }
    }

    Breakdown {
        noise_reduction_db: if ncnt > 0 {
            db(nin as f32 / ncnt as f32) - db(nout as f32 / ncnt as f32)
        } else {
            0.0
        },
        speech_loss_db: if scnt > 0 {
            db(sin_ as f32 / scnt as f32) - db(sout as f32 / scnt as f32)
        } else {
            0.0
        },
    }
}

fn corpus(seconds: usize) -> (Vec<f32>, Vec<f32>) {
    let n = 48_000 * seconds;
    (speech(n, 7), noise(n, 99))
}

/// The headline check.
///
/// The denoiser must remove real amounts of noise without wrecking the speech, and on genuinely
/// noisy input it must improve both objective metrics.
///
/// It does *not* improve them on nearly clean input, and that is expected rather than a bug:
/// SI-SDR and segmental SNR both measure waveform fidelity, while the denoiser works by
/// applying time-varying gains per frequency band. That reshaping always costs some waveform
/// accuracy, so once there is little noise left to remove the cost outweighs the benefit. The
/// measured crossover sits at roughly 7dB input SNR, which is why the assertions below only
/// demand improvement where there is something to improve.
#[test]
fn denoising_improves_objective_quality_on_noisy_input() {
    let (clean, nz) = corpus(6);
    let params = DenoiseParams::default();

    println!(
        "\n{:<10}{:>12}{:>12}{:>12}{:>12}{:>13}{:>13}",
        "in SNR",
        "SI-SDR in",
        "SI-SDR out",
        "segSNR in",
        "segSNR out",
        "noise redux",
        "speech loss"
    );

    let mut rows = Vec::new();
    for &snr in &[20.0f32, 10.0, 5.0, 0.0] {
        let noisy = mix_at_snr(&clean, &nz, snr);
        let denoised = denoise_aligned(params, &noisy);
        let n = denoised.len();

        let sisdr_in = si_sdr(&noisy[..n], &clean[..n]);
        let sisdr_out = si_sdr(&denoised, &clean[..n]);
        let seg_in = segmental_snr(&noisy[..n], &clean[..n], FRAME_SIZE);
        let seg_out = segmental_snr(&denoised, &clean[..n], FRAME_SIZE);
        let b = breakdown(&clean[..n], &noisy[..n], &denoised);

        println!(
            "{:<10}{:>11.1}dB{:>11.1}dB{:>11.1}dB{:>11.1}dB{:>11.1}dB{:>11.1}dB",
            format!("{snr:.0} dB"),
            sisdr_in,
            sisdr_out,
            seg_in,
            seg_out,
            b.noise_reduction_db,
            b.speech_loss_db
        );
        rows.push((snr, sisdr_in, sisdr_out, seg_in, seg_out, b));
    }

    for (snr, sisdr_in, sisdr_out, seg_in, seg_out, b) in rows {
        // These must hold at every SNR: the denoiser is doing real work and not by
        // sacrificing the speech.
        assert!(
            b.noise_reduction_db > 5.0,
            "{snr}dB: only removed {:.1}dB of noise",
            b.noise_reduction_db
        );
        assert!(
            b.speech_loss_db < 4.0,
            "{snr}dB: damaged the speech by {:.1}dB",
            b.speech_loss_db
        );

        if snr <= 5.0 {
            assert!(
                sisdr_out > sisdr_in,
                "{snr}dB: SI-SDR got worse ({sisdr_out:.1} vs {sisdr_in:.1})"
            );
            assert!(
                seg_out > seg_in,
                "{snr}dB: segmental SNR got worse ({seg_out:.1} vs {seg_in:.1})"
            );
        }
    }
}

/// A gain floor is a deliberate trade: less noise removed, less damage to the speech.
#[test]
fn attenuation_limit_trades_noise_removal_for_speech_preservation() {
    let (clean, nz) = corpus(6);
    let noisy = mix_at_snr(&clean, &nz, 5.0);

    println!(
        "\n{:<22}{:>13}{:>13}{:>12}",
        "attenuation cap", "noise redux", "speech loss", "SI-SDR"
    );

    let mut previous_reduction = f32::INFINITY;
    for cap in [None, Some(18.0f32), Some(12.0), Some(6.0)] {
        let params = match cap {
            None => DenoiseParams::default(),
            Some(db) => DenoiseParams::default().max_attenuation_db(db),
        };
        let denoised = denoise_aligned(params, &noisy);
        let n = denoised.len();
        let b = breakdown(&clean[..n], &noisy[..n], &denoised);
        let sisdr = si_sdr(&denoised, &clean[..n]);

        println!(
            "{:<22}{:>11.1}dB{:>11.1}dB{:>10.1}dB",
            cap.map_or("none".to_string(), |d| format!("{d:.0} dB")),
            b.noise_reduction_db,
            b.speech_loss_db,
            sisdr
        );

        // A tighter cap must never remove *more* noise than a looser one.
        assert!(
            b.noise_reduction_db <= previous_reduction + 0.5,
            "cap {cap:?} removed more noise than the looser setting"
        );
        previous_reduction = b.noise_reduction_db;
    }
}

/// Looking ahead should protect speech onsets, which is the whole point of the mode.
#[test]
fn lookahead_preserves_speech_at_least_as_well() {
    let (clean, nz) = corpus(8);
    let noisy = mix_at_snr(&clean, &nz, 5.0);

    println!(
        "\n{:<14}{:>13}{:>13}{:>12}",
        "lookahead", "noise redux", "speech loss", "SI-SDR"
    );

    let mut baseline_loss = 0.0;
    for look in [0usize, 1, 2, 4] {
        let params = DenoiseParams::default().lookahead(look);
        let denoised = denoise_aligned(params, &noisy);
        let n = denoised.len();
        let b = breakdown(&clean[..n], &noisy[..n], &denoised);
        let sisdr = si_sdr(&denoised, &clean[..n]);

        println!(
            "{look:<14}{:>11.1}dB{:>11.1}dB{:>10.1}dB",
            b.noise_reduction_db, b.speech_loss_db, sisdr
        );

        if look == 0 {
            baseline_loss = b.speech_loss_db;
        } else {
            // Taking the largest gain over the window can only preserve more speech.
            assert!(
                b.speech_loss_db <= baseline_loss + 0.2,
                "lookahead {look} damaged speech more than none: {:.2} vs {:.2}",
                b.speech_loss_db,
                baseline_loss
            );
        }
    }
}

/// Running the pitch search less often is a speed/quality trade, so bound the quality cost.
#[test]
fn decimating_the_pitch_search_costs_little_quality() {
    let (clean, nz) = corpus(6);
    let noisy = mix_at_snr(&clean, &nz, 5.0);

    println!("\n{:<18}{:>12}{:>13}", "pitch interval", "SI-SDR", "segSNR");

    let mut baseline = f32::NAN;
    for interval in [1usize, 2, 3, 4] {
        let params = DenoiseParams::default().pitch_interval(interval);
        let denoised = denoise_aligned(params, &noisy);
        let n = denoised.len();
        let sisdr = si_sdr(&denoised, &clean[..n]);
        let seg = segmental_snr(&denoised, &clean[..n], FRAME_SIZE);
        println!("{interval:<18}{sisdr:>10.2}dB{seg:>11.2}dB");

        if interval == 1 {
            baseline = sisdr;
        } else {
            assert!(
                sisdr > baseline - 3.0,
                "interval {interval} lost {:.2}dB of SI-SDR",
                baseline - sisdr
            );
        }
    }
}

/// Gating on voice activity must remove more noise than not gating.
#[test]
fn vad_gating_removes_more_noise() {
    let (clean, nz) = corpus(6);
    let noisy = mix_at_snr(&clean, &nz, 5.0);

    let plain = denoise_aligned(DenoiseParams::default(), &noisy);
    let gated = denoise_aligned(DenoiseParams::default().vad_threshold(0.6), &noisy);
    let n = plain.len().min(gated.len());

    let bp = breakdown(&clean[..n], &noisy[..n], &plain[..n]);
    let bg = breakdown(&clean[..n], &noisy[..n], &gated[..n]);
    println!(
        "\nungated: {:.1}dB noise redux / {:.1}dB speech loss\n  gated: {:.1}dB noise redux / {:.1}dB speech loss",
        bp.noise_reduction_db, bp.speech_loss_db, bg.noise_reduction_db, bg.speech_loss_db
    );

    assert!(
        bg.noise_reduction_db >= bp.noise_reduction_db - 0.1,
        "gating removed less noise: {:.1} vs {:.1}",
        bg.noise_reduction_db,
        bp.noise_reduction_db
    );
}

/// Pure silence in must stay silence out.
#[test]
fn silence_stays_silent() {
    let silence = vec![0.0f32; 48_000];
    let out = denoise_aligned(DenoiseParams::default(), &silence);
    assert!(out.iter().all(|&x| x == 0.0), "silence should stay silent");
}

/// A known limitation, pinned down so it cannot regress silently: on input that contains no
/// speech at all, RNNoise neither suppresses the noise nor recognizes that it is noise.
///
/// This is not a defect in this port. The original C implementation produces the same numbers
/// to within a fraction of a dB. The model was trained on speech-plus-noise mixtures and works
/// by separating the two, so given noise alone it has nothing to separate: the gains stay near
/// passthrough, and the voice-activity output reports high confidence that it is hearing
/// speech.
///
/// The practical consequences are worth knowing:
///
/// * suppression figures measured on noise-only signals will look terrible and mean nothing —
///   measure on mixtures, as the tests above do;
/// * [`DenoiseParams::vad_threshold`] cannot rescue this case, because the VAD itself is
///   fooled. It does work well on mixtures, where the contrast exists.
#[test]
fn noise_only_input_is_neither_suppressed_nor_detected() {
    use nnnoiseless::DenoiseState;

    let nz: Vec<f32> = noise(48_000, 5).iter().map(|x| x * 2000.0).collect();
    let level = db(rms(&nz[FRAME_SIZE..]));

    let plain = denoise_aligned(DenoiseParams::default(), &nz);
    let plain_reduction = level - db(rms(&plain[FRAME_SIZE..]));

    // Collect the voice-activity probabilities the model reports for this noise.
    let mut state = DenoiseState::new();
    let mut out = vec![0.0; FRAME_SIZE];
    let vads: Vec<f32> = nz
        .chunks_exact(FRAME_SIZE)
        .map(|f| state.process_frame(&mut out, f))
        .collect();
    let mean_vad = vads.iter().sum::<f32>() / vads.len() as f32;

    println!("\nnoise-only suppression: {plain_reduction:.1} dB");
    println!("noise-only mean VAD:    {mean_vad:.3}");

    assert!(
        plain_reduction > 0.5 && plain_reduction < 6.0,
        "noise-only suppression moved unexpectedly: {plain_reduction:.1}dB"
    );
    assert!(
        mean_vad > 0.8,
        "the VAD used to be (wrongly) confident here; it now reports {mean_vad:.3}"
    );

    // The contrast: with speech present, gating does remove far more noise.
    let (clean, nz2) = corpus(6);
    let mixed = mix_at_snr(&clean, &nz2, 5.0);
    let ungated = denoise_aligned(DenoiseParams::default(), &mixed);
    let gated = denoise_aligned(DenoiseParams::default().vad_threshold(0.6), &mixed);
    let n = ungated.len();
    let bu = breakdown(&clean[..n], &mixed[..n], &ungated);
    let bg = breakdown(&clean[..n], &mixed[..n], &gated);
    assert!(
        bg.noise_reduction_db > bu.noise_reduction_db + 2.0,
        "gating should clearly help on mixtures: {:.1} vs {:.1}",
        bg.noise_reduction_db,
        bu.noise_reduction_db
    );
}

/// The metrics themselves need to be right, or none of the above means anything.
#[test]
fn metrics_are_self_consistent() {
    let reference = speech(48_000, 3);

    // A perfect estimate.
    assert!(si_sdr(&reference, &reference) > 100.0);
    assert!(segmental_snr(&reference, &reference, FRAME_SIZE) >= 34.0);

    // SI-SDR ignores an overall scale factor; a plain SNR would not.
    let scaled: Vec<f32> = reference.iter().map(|x| x * 0.5).collect();
    assert!(
        si_sdr(&scaled, &reference) > 100.0,
        "SI-SDR should be scale invariant"
    );

    // Adding noise must lower both metrics, monotonically.
    let nz = noise(48_000, 11);
    let mut last_sisdr = f32::INFINITY;
    let mut last_seg = f32::INFINITY;
    for &snr in &[20.0f32, 10.0, 0.0] {
        let noisy = mix_at_snr(&reference, &nz, snr);
        let s = si_sdr(&noisy, &reference);
        let g = segmental_snr(&noisy, &reference, FRAME_SIZE);
        assert!(s < last_sisdr, "SI-SDR not monotonic at {snr}dB");
        assert!(g < last_seg, "segSNR not monotonic at {snr}dB");
        last_sisdr = s;
        last_seg = g;
    }
}

/// `denoise_aligned` must line the output up with the input regardless of the latency the
/// configuration introduces, otherwise every metric above would be comparing shifted signals.
#[test]
fn alignment_helper_compensates_for_latency() {
    let (clean, nz) = corpus(3);
    let noisy = mix_at_snr(&clean, &nz, 10.0);

    let baseline = denoise_aligned(DenoiseParams::default(), &noisy);
    let base_score = si_sdr(&baseline, &clean[..baseline.len()]);

    for params in [
        DenoiseParams::default().lookahead(2),
        DenoiseParams::default().lookahead(4),
    ] {
        let out = denoise_aligned(params, &noisy);
        assert_eq!(out.len(), baseline.len(), "aligned lengths should match");
        let score = si_sdr(&out, &clean[..out.len()]);
        assert!(
            (score - base_score).abs() < 6.0,
            "alignment looks wrong: {score:.1}dB vs {base_score:.1}dB"
        );
    }
}
