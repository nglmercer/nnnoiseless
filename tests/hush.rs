#![cfg(feature = "hush")]

use std::env;

use nnnoiseless::{HushModel, HUSH_FRAME_SIZE, HUSH_SAMPLE_RATE};

fn model_path() -> Option<String> {
    env::var("HUSH_MODEL").ok().filter(|path| !path.is_empty())
}

#[test]
fn released_hush_bundle_processes_streaming_frames() {
    let Some(path) = model_path() else {
        eprintln!("skipping Hush integration test: set HUSH_MODEL to the ONNX bundle");
        return;
    };

    let model = HushModel::from_path(path).expect("Hush model should load");
    let mut denoiser = model.denoiser().expect("Hush runtime should initialize");
    assert_eq!(denoiser.sample_rate(), HUSH_SAMPLE_RATE);
    assert_eq!(denoiser.frame_size(), HUSH_FRAME_SIZE);

    let mut phase = 0.0f32;
    let mut output = [0.0f32; HUSH_FRAME_SIZE];
    let mut finite_lsnr = false;
    let mut input_rms = 0.0f32;
    let mut output_rms = 0.0f32;
    for frame in 0..120 {
        let mut input = [0.0f32; HUSH_FRAME_SIZE];
        for sample in &mut input {
            phase += 2.0 * std::f32::consts::PI * 180.0 / HUSH_SAMPLE_RATE as f32;
            if phase > 2.0 * std::f32::consts::PI {
                phase -= 2.0 * std::f32::consts::PI;
            }
            *sample = phase.sin() * 0.12;
        }
        let lsnr = denoiser
            .process_frame(&mut output, &input)
            .expect("Hush frame should process");
        finite_lsnr |= lsnr.is_finite();
        if frame > 10 {
            input_rms += input.iter().map(|x| x * x).sum::<f32>();
            output_rms += output.iter().map(|x| x * x).sum::<f32>();
        }
        assert!(output.iter().all(|x| x.is_finite()));
    }

    assert!(finite_lsnr);
    assert!(input_rms > 0.0);
    assert!(output_rms > 0.0, "Hush output was silent after warmup");
    denoiser.reset().expect("Hush reset should work");
}
