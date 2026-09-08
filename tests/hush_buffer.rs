#![cfg(feature = "hush")]

use std::env;
use std::fs;

use nnnoiseless::{denoise_hush_buffer, HUSH_SAMPLE_RATE};

fn model_bytes() -> Option<Vec<u8>> {
    let path = env::var_os("HUSH_MODEL").filter(|path| !path.is_empty())?;
    Some(fs::read(path).expect("HUSH_MODEL should point to a readable Hush bundle"))
}

fn speech_like_signal(seconds: usize) -> Vec<f32> {
    let samples = seconds * HUSH_SAMPLE_RATE;
    let rate = HUSH_SAMPLE_RATE as f32;
    (0..samples)
        .map(|index| {
            let t = index as f32 / rate;
            let f0 = 145.0 + 24.0 * (2.0 * std::f32::consts::PI * 0.7 * t).sin();
            let syllable = (2.0 * std::f32::consts::PI * 3.2 * t).sin();
            let envelope = 0.35 + 0.65 * syllable.max(0.0).powf(0.6);
            let harmonics = (1..=12).fold(0.0, |sum, harmonic| {
                sum + (2.0 * std::f32::consts::PI * f0 * harmonic as f32 * t).sin()
                    / (harmonic as f32).powf(1.15)
            });
            harmonics * envelope * 0.04
        })
        .collect()
}

#[test]
fn denoise_hush_buffer_preserves_and_flushes_the_tail() {
    let Some(model) = model_bytes() else {
        eprintln!("skipping Hush buffer integration test: set HUSH_MODEL to the ONNX bundle");
        return;
    };

    let input = speech_like_signal(2);
    let output = denoise_hush_buffer(&input, HUSH_SAMPLE_RATE as f32, 0.0, &model)
        .expect("Hush buffer processing should succeed");

    assert_eq!(output.len(), input.len());
    assert!(
        output.iter().any(|sample| sample.abs() > 0.0),
        "Hush output was completely silent"
    );

    let tail = &output[output.len() - 320..];
    assert!(
        tail.iter().any(|sample| sample.abs() > 1e-6),
        "Hush output lost tail audio"
    );
}
