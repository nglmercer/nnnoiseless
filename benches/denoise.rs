//! Per-stage benchmark for the denoising pipeline.
//!
//! Run with `cargo bench`. This is deliberately dependency-free rather than built on
//! `criterion`: the measurements here are tens of microseconds over thousands of frames, so
//! the run-to-run noise is already small, and the numbers are more useful reported per frame
//! and as a realtime factor than as a distribution.
//!
//! ```text
//! cargo bench
//! RUSTFLAGS="-C target-cpu=native" cargo bench   # compare against a tuned build
//! ```

use std::time::{Duration, Instant};

use nnnoiseless::{ChannelLink, DenoiseParams, DenoiseState, MultiDenoiser, Resampler, FRAME_SIZE};

const SECONDS: usize = 20;
const SAMPLE_RATE: usize = 48_000;

/// Voiced-sounding input: a harmonic stack with a slow envelope, plus coloured noise.
fn make_input(n: usize) -> Vec<f32> {
    let mut seed = 0x12345678u32;
    let mut lp = 0.0f32;
    (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let w = ((seed >> 16) as i32 - 32768) as f32 / 32768.0;
            lp = 0.85 * lp + 0.15 * w;
            let t = i as f32 / SAMPLE_RATE as f32;
            let env = (0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin()).powf(1.5);
            let mut s = 0.0;
            for h in 1..=12 {
                s += (2.0 * std::f32::consts::PI * 150.0 * h as f32 * t).sin()
                    / (h as f32).powf(1.2);
            }
            s * env * 6000.0 + (w * 0.5 + lp * 2.0) * 1500.0
        })
        .collect()
}

/// Runs `f` a few times and keeps the fastest, which is the most stable estimator here.
fn best_of<F: FnMut()>(rounds: usize, mut f: F) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..rounds {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
    }
    best
}

fn report(name: &str, elapsed: Duration, frames: usize) {
    let per_frame_us = elapsed.as_secs_f64() * 1e6 / frames as f64;
    let audio_secs = frames as f64 * FRAME_SIZE as f64 / SAMPLE_RATE as f64;
    let realtime = audio_secs / elapsed.as_secs_f64();
    println!(
        "{name:<34}{:>9.1} ms{:>11.2} us/frame{:>10.0}x realtime",
        elapsed.as_secs_f64() * 1000.0,
        per_frame_us,
        realtime
    );
}

fn bench_params(name: &str, params: DenoiseParams, input: &[f32], frames: usize) {
    let mut out = vec![0.0; FRAME_SIZE];
    let mut state = DenoiseState::with_params(params);
    // Warm up, so that lazily built tables and FFT plans are not counted.
    for f in input.chunks_exact(FRAME_SIZE).take(50) {
        state.process_frame(&mut out, f);
    }
    let elapsed = best_of(3, || {
        let mut state = DenoiseState::with_params(params);
        for f in input.chunks_exact(FRAME_SIZE) {
            state.process_frame(&mut out, f);
        }
        std::hint::black_box(&out);
    });
    report(name, elapsed, frames);
}

fn main() {
    let n = SAMPLE_RATE * SECONDS;
    let input = make_input(n);
    let frames = n / FRAME_SIZE;

    println!(
        "nnnoiseless bench - {SECONDS}s of audio, {frames} frames, kernels: {}\n",
        nnnoiseless::active_isa()
    );

    bench_params("default", DenoiseParams::default(), &input, frames);
    bench_params(
        "pitch_interval(2)",
        DenoiseParams::default().pitch_interval(2),
        &input,
        frames,
    );
    bench_params(
        "pitch_interval(3)",
        DenoiseParams::default().pitch_interval(3),
        &input,
        frames,
    );
    bench_params(
        "no pitch filter",
        DenoiseParams::default().pitch_filter(false),
        &input,
        frames,
    );
    bench_params(
        "lookahead(2)",
        DenoiseParams::default().lookahead(2),
        &input,
        frames,
    );
    bench_params(
        "max_attenuation_db(12)",
        DenoiseParams::default().max_attenuation_db(12.0),
        &input,
        frames,
    );

    // Stereo, with and without gain linking.
    for (name, link) in [
        ("stereo, independent", ChannelLink::Independent),
        ("stereo, max-linked", ChannelLink::Max),
    ] {
        let mut bufs = vec![vec![0.0f32; FRAME_SIZE]; 2];
        let elapsed = best_of(3, || {
            let mut d = MultiDenoiser::new(2, link);
            for f in input.chunks_exact(FRAME_SIZE) {
                let ins: Vec<&[f32]> = vec![f, f];
                let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|b| &mut b[..]).collect();
                d.process_frame(&mut outs, &ins);
            }
            std::hint::black_box(&bufs);
        });
        report(name, elapsed, frames);
    }

    // Resampling, reported against the same frame count for comparability.
    for &rate in &[16_000.0f64, 44_100.0] {
        let src: Vec<f32> = make_input((rate as usize) * SECONDS);
        let elapsed = best_of(3, || {
            let mut r = Resampler::to_denoiser_rate(rate, 1);
            let mut out = Vec::with_capacity(n);
            r.process(&src, &mut out);
            r.flush(&mut out);
            std::hint::black_box(&out);
        });
        report(&format!("resample {rate:.0} -> 48000"), elapsed, frames);
    }
}
