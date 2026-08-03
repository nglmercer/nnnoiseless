//! Record the default microphone and denoise the recorded WAV.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example mic_denoise --features mic-example -- 5 recorded.wav denoised.wav
//! ```

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample, SampleFormat, SizedSample, I24, U24,
};
use hound::{SampleFormat as WavSampleFormat, WavReader, WavSpec, WavWriter};
use nnnoiseless::{DenoiseState, FRAME_SIZE};
use std::{
    env,
    error::Error,
    io::{Error as IoError, ErrorKind},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

const TARGET_SAMPLE_RATE: u32 = 48_000;
const DEFAULT_DURATION_SECONDS: f64 = 5.0;
type RecordedAudio = (Vec<f32>, u16, u32, u64);

fn main() -> Result<(), Box<dyn Error>> {
    let (duration, recorded_path, denoised_path) = parse_args()?;

    println!("Recording for {duration:.1} seconds...");
    let (samples, channels, sample_rate, dropped_callbacks) = record_microphone(duration)?;
    if samples.is_empty() {
        return Err(invalid_input("the microphone produced no samples"));
    }

    write_recorded_wav(&recorded_path, &samples, channels, sample_rate)?;
    println!(
        "Recorded {} samples to {}",
        samples.len() / channels as usize,
        recorded_path.display()
    );
    if dropped_callbacks > 0 {
        eprintln!(
            "Warning: dropped {dropped_callbacks} audio callback buffer(s) because the recorder was busy"
        );
    }

    let output_frames = denoise_wav(&recorded_path, &denoised_path)?;
    println!(
        "Denoised {output_frames} samples to {}",
        denoised_path.display()
    );
    Ok(())
}

fn parse_args() -> Result<(f64, PathBuf, PathBuf), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    if matches!(args.next().as_deref(), Some("--help" | "-h")) {
        println!(
            "Usage: cargo run --release --example mic_denoise --features mic-example -- [SECONDS] [RECORDED.wav] [DENOISED.wav]\n\nDefaults: 5 recorded.wav denoised.wav"
        );
        std::process::exit(0);
    }

    let duration = match args.next() {
        Some(value) => value
            .parse::<f64>()
            .map_err(|_| invalid_input("duration must be a positive number"))?,
        None => DEFAULT_DURATION_SECONDS,
    };
    if !duration.is_finite() || duration <= 0.0 {
        return Err(invalid_input("duration must be a positive number"));
    }

    let recorded_path = PathBuf::from(args.next().unwrap_or_else(|| "recorded.wav".into()));
    let denoised_path = PathBuf::from(args.next().unwrap_or_else(|| "denoised.wav".into()));
    if args.next().is_some() {
        return Err(invalid_input("too many arguments; use --help for usage"));
    }

    Ok((duration, recorded_path, denoised_path))
}

fn record_microphone(duration: f64) -> Result<RecordedAudio, Box<dyn Error>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| invalid_input("no default input device was found"))?;
    let supported_config = device.default_input_config()?;
    let channels = supported_config.channels();
    let sample_rate = supported_config.sample_rate();
    if channels == 0 || sample_rate == 0 {
        return Err(invalid_input(
            "the input device returned an invalid configuration",
        ));
    }

    println!("Input device: {device}");
    println!("Input config: {supported_config:?}");

    let expected_samples = (duration * sample_rate as f64 * channels as f64) as usize;
    let samples = Arc::new(Mutex::new(Vec::with_capacity(expected_samples)));
    let dropped_callbacks = Arc::new(AtomicU64::new(0));
    let config: cpal::StreamConfig = supported_config.into();

    let stream = match supported_config.sample_format() {
        SampleFormat::I8 => build_stream::<i8, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::I16 => build_stream::<i16, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::I24 => build_stream::<I24, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::I32 => build_stream::<i32, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::I64 => build_stream::<i64, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::U8 => build_stream::<u8, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::U16 => build_stream::<u16, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::U24 => build_stream::<U24, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::U32 => build_stream::<u32, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::U64 => build_stream::<u64, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::F32 => build_stream::<f32, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        SampleFormat::F64 => build_stream::<f64, _>(
            &device,
            &config,
            Arc::clone(&samples),
            Arc::clone(&dropped_callbacks),
        )?,
        format => {
            return Err(invalid_input(format!(
                "unsupported microphone format: {format}"
            )))
        }
    };

    stream.play()?;
    thread::sleep(Duration::from_secs_f64(duration));
    drop(stream);

    let samples = samples
        .lock()
        .map_err(|_| invalid_input("microphone sample buffer was poisoned"))?
        .clone();
    Ok((
        samples,
        channels,
        sample_rate,
        dropped_callbacks.load(Ordering::Relaxed),
    ))
}

fn build_stream<T, D>(
    device: &D,
    config: &cpal::StreamConfig,
    samples: Arc<Mutex<Vec<f32>>>,
    dropped_callbacks: Arc<AtomicU64>,
) -> Result<D::Stream, cpal::Error>
where
    T: SizedSample,
    f32: FromSample<T>,
    D: DeviceTrait,
{
    device.build_input_stream(
        *config,
        move |data: &[T], _| capture_samples(data, &samples, &dropped_callbacks),
        move |error| eprintln!("input stream error: {error}"),
        None,
    )
}

fn capture_samples<T>(input: &[T], samples: &Arc<Mutex<Vec<f32>>>, dropped_callbacks: &AtomicU64)
where
    T: Sample,
    f32: FromSample<T>,
{
    let Ok(mut samples) = samples.try_lock() else {
        dropped_callbacks.fetch_add(1, Ordering::Relaxed);
        return;
    };
    samples.extend(input.iter().copied().map(f32::from_sample));
}

fn write_recorded_wav(
    path: &Path,
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
) -> Result<(), Box<dyn Error>> {
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: WavSampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn denoise_wav(input_path: &Path, output_path: &Path) -> Result<usize, Box<dyn Error>> {
    let mut reader = WavReader::open(input_path)?;
    let spec = reader.spec();
    if spec.sample_format != WavSampleFormat::Float || spec.bits_per_sample != 32 {
        return Err(invalid_input(
            "the recorded WAV must contain 32-bit float samples",
        ));
    }
    if spec.channels == 0 || spec.sample_rate == 0 {
        return Err(invalid_input("the recorded WAV has an invalid format"));
    }

    let samples = reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?;
    let mono = downmix_to_mono(&samples, spec.channels as usize);
    let resampled = resample_linear(&mono, spec.sample_rate, TARGET_SAMPLE_RATE);
    let frame_count = resampled.len().div_ceil(FRAME_SIZE);

    let output_spec = WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: WavSampleFormat::Int,
    };
    let mut writer = WavWriter::create(output_path, output_spec)?;
    let mut state = DenoiseState::new();
    let mut input_frame = [0.0; FRAME_SIZE];
    let mut output_frame = [0.0; FRAME_SIZE];
    let mut written_samples = 0;

    for frame_index in 0..frame_count {
        input_frame.fill(0.0);
        let start = frame_index * FRAME_SIZE;
        let end = (start + FRAME_SIZE).min(resampled.len());
        let frame_len = end.saturating_sub(start);
        input_frame[..frame_len].copy_from_slice(&resampled[start..end]);
        input_frame
            .iter_mut()
            .for_each(|sample| *sample *= 32_768.0);

        state.process_frame(&mut output_frame, &input_frame);
        if frame_index == 0 {
            continue;
        }
        for &sample in &output_frame {
            let sample = sample.clamp(i16::MIN as f32, i16::MAX as f32).round() as i16;
            writer.write_sample(sample)?;
            written_samples += 1;
        }
    }
    writer.finalize()?;
    Ok(written_samples)
}

fn downmix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == target_rate {
        return samples.to_vec();
    }

    let output_len =
        ((samples.len() as f64 * target_rate as f64 / source_rate as f64).round() as usize).max(1);
    let scale = source_rate as f64 / target_rate as f64;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let position = index as f64 * scale;
        let left = position.floor() as usize;
        let left = left.min(samples.len() - 1);
        let right = (left + 1).min(samples.len() - 1);
        let fraction = (position - left as f64) as f32;
        output.push(samples[left] * (1.0 - fraction) + samples[right] * fraction);
    }
    output
}

fn invalid_input(message: impl Into<String>) -> Box<IoError> {
    Box::new(IoError::new(ErrorKind::InvalidInput, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn downmixes_interleaved_channels() {
        assert_eq!(downmix_to_mono(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
    }

    #[test]
    fn reads_recorded_wav_and_writes_denoised_wav() {
        let stem = format!("nnnoiseless-mic-example-{}", std::process::id());
        let input_path = env::temp_dir().join(format!("{stem}-input.wav"));
        let output_path = env::temp_dir().join(format!("{stem}-output.wav"));
        let input_samples: Vec<f32> = (0..1_920)
            .map(|index| (index as f32 * 0.03).sin() * 0.1)
            .collect();

        write_recorded_wav(&input_path, &input_samples, 2, 24_000).unwrap();
        let written = denoise_wav(&input_path, &output_path).unwrap();
        assert_eq!(written, FRAME_SIZE * 3);

        let reader = WavReader::open(&output_path).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, TARGET_SAMPLE_RATE);
        assert_eq!(reader.spec().bits_per_sample, 16);

        fs::remove_file(input_path).unwrap();
        fs::remove_file(output_path).unwrap();
    }
}
