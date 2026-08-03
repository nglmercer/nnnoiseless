//! Backend benchmark for the released Hush ONNX bundle.
//!
//! ```text
//! HUSH_MODEL=/path/to/advanced_dfnet16k_model_best_onnx.tar.gz \
//!   cargo bench --features hush --bench hush
//! ```

use std::env;
use std::time::{Duration, Instant};

use nnnoiseless::{HushModel, HUSH_FRAME_SIZE, HUSH_SAMPLE_RATE};

const SECONDS: usize = 20;

fn make_input() -> Vec<f32> {
    let n = SECONDS * HUSH_SAMPLE_RATE;
    let mut seed = 0x12345678u32;
    let mut lp = 0.0f32;
    (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let white = ((seed >> 16) as i32 - 32768) as f32 / 32768.0;
            lp = 0.85 * lp + 0.15 * white;
            let t = i as f32 / HUSH_SAMPLE_RATE as f32;
            let env = (0.5
                + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin())
                .max(0.0)
                .powf(1.5);
            let mut speech = 0.0;
            for harmonic in 1..=12 {
                speech += (2.0
                    * std::f32::consts::PI
                    * 150.0
                    * harmonic as f32
                    * t)
                    .sin()
                    / (harmonic as f32).powf(1.2);
            }
            speech * env * 0.08 + (white * 0.5 + lp * 2.0) * 0.03
        })
        .collect()
}

fn best_of<F: FnMut()>(rounds: usize, mut f: F) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..rounds {
        let started = Instant::now();
        f();
        best = best.min(started.elapsed());
    }
    best
}

fn main() {
    let Some(path) = env::var_os("HUSH_MODEL") else {
        eprintln!("HUSH_MODEL is not set; Hush benchmark skipped");
        return;
    };
    let input = make_input();
    let frames = input.len() / HUSH_FRAME_SIZE;

    let started = Instant::now();
    let model = HushModel::from_path(path).expect("Hush model should load");
    let load_time = started.elapsed();

    let mut warmup = model.denoiser().expect("Hush runtime should initialize");
    let mut output = vec![0.0f32; HUSH_FRAME_SIZE];
    for frame in input.chunks_exact(HUSH_FRAME_SIZE).take(20) {
        warmup
            .process_frame(&mut output, frame)
            .expect("Hush warmup frame should process");
    }

    let elapsed = best_of(3, || {
        let mut denoiser = model.denoiser().expect("Hush runtime should initialize");
        for frame in input.chunks_exact(HUSH_FRAME_SIZE) {
            denoiser
                .process_frame(&mut output, frame)
                .expect("Hush frame should process");
        }
        std::hint::black_box(&output);
    });
    let per_frame_us = elapsed.as_secs_f64() * 1e6 / frames as f64;
    let audio_seconds = frames as f64 * HUSH_FRAME_SIZE as f64 / HUSH_SAMPLE_RATE as f64;
    let realtime = audio_seconds / elapsed.as_secs_f64();
    println!(
        "Hush backend — model load: {:.1} ms, {:.2} us/frame, {:.0}x realtime",
        load_time.as_secs_f64() * 1000.0,
        per_frame_us,
        realtime
    );
}
