use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use anyhow::{anyhow, Context, Error};
use clap::{arg, crate_version, Command};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use nnnoiseless::{ChannelLink, DenoiseParams, DenoiseState, MultiDenoiser, Resampler, RnnModel};

const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;
const DENOISER_RATE: f64 = 48_000.0;

/// Reads interleaved samples from a source, a whole frame at a time.
///
/// The previous version of this handed out one sample at a time through a `Box<dyn Trait>`,
/// which meant a virtual call per sample. Batching at frame granularity removes ~480 indirect
/// calls per frame.
trait FrameSource {
    /// Fills `out` with up to `out.len()` interleaved samples, returning how many were written.
    fn read_frame(&mut self, out: &mut [f32]) -> Result<usize, Error>;
}

/// Pulls samples from an iterator and resamples them to 48kHz if necessary.
struct SampleReader<I> {
    samples: I,
    channels: usize,
    resampler: Option<Resampler>,
    /// Resampled output waiting to be handed out.
    ready: Vec<f32>,
    ready_pos: usize,
    /// Staging area for input handed to the resampler.
    staging: Vec<f32>,
    exhausted: bool,
}

/// How many input frames to hand the resampler at a time.
const RESAMPLE_CHUNK_FRAMES: usize = 2048;

impl<I: Iterator<Item = Result<f32, Error>>> SampleReader<I> {
    fn new(samples: I, channels: usize, sample_rate: f64) -> SampleReader<I> {
        let resampler = if (sample_rate - DENOISER_RATE).abs() > f64::EPSILON {
            Some(Resampler::to_denoiser_rate(sample_rate, channels))
        } else {
            None
        };
        SampleReader {
            samples,
            channels,
            resampler,
            ready: Vec::new(),
            ready_pos: 0,
            staging: Vec::with_capacity(RESAMPLE_CHUNK_FRAMES * channels),
            exhausted: false,
        }
    }

    /// Tops up `ready`, returning false once the source is finished and drained.
    fn refill(&mut self) -> Result<bool, Error> {
        if self.ready_pos < self.ready.len() {
            return Ok(true);
        }
        self.ready.clear();
        self.ready_pos = 0;

        while self.ready.is_empty() {
            if self.exhausted {
                return Ok(false);
            }

            self.staging.clear();
            for _ in 0..(RESAMPLE_CHUNK_FRAMES * self.channels) {
                match self.samples.next() {
                    Some(Ok(s)) => self.staging.push(s),
                    Some(Err(e)) => return Err(e),
                    None => {
                        self.exhausted = true;
                        break;
                    }
                }
            }
            if !self.staging.len().is_multiple_of(self.channels) {
                return Err(anyhow!(
                    "Unexpected end of input (expected a multiple of {} samples)",
                    self.channels
                ));
            }

            match self.resampler.as_mut() {
                None => std::mem::swap(&mut self.ready, &mut self.staging),
                Some(r) => {
                    r.process(&self.staging, &mut self.ready);
                    if self.exhausted {
                        r.flush(&mut self.ready);
                    }
                }
            }
        }
        Ok(true)
    }
}

impl<I: Iterator<Item = Result<f32, Error>>> FrameSource for SampleReader<I> {
    fn read_frame(&mut self, out: &mut [f32]) -> Result<usize, Error> {
        let mut written = 0;
        while written < out.len() {
            if !self.refill()? {
                break;
            }
            let available = self.ready.len() - self.ready_pos;
            let take = available.min(out.len() - written);
            out[written..(written + take)]
                .copy_from_slice(&self.ready[self.ready_pos..(self.ready_pos + take)]);
            self.ready_pos += take;
            written += take;
        }
        Ok(written)
    }
}

// TODO: support either endianness
struct RawSampleIter<R: Read> {
    reader: R,
}

impl<R: Read> Iterator for RawSampleIter<R> {
    type Item = Result<f32, Error>;

    fn next(&mut self) -> Option<Result<f32, Error>> {
        let mut first = [0u8; 1];
        match self.reader.read(&mut first) {
            Ok(0) => return None,
            Ok(1) => {}
            Ok(_) => unreachable!("a one-byte read cannot return more than one byte"),
            Err(e) => return Some(Err(e.into())),
        }

        let mut second = [0u8; 1];
        if let Err(e) = self.reader.read_exact(&mut second) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                return Some(Err(anyhow!(
                    "Unexpected end of input (expected an even number of bytes)"
                )));
            }
            return Some(Err(e.into()));
        }

        Some(Ok(i16::from_le_bytes([first[0], second[0]]) as f32))
    }
}

trait FrameWriter {
    fn write_frame(&mut self, buf: &[f32]) -> Result<(), Error>;
    fn finalize(&mut self) -> Result<(), Error>;
}

struct RawFrameWriter<W: Write> {
    writer: W,
    buf: Vec<u8>,
}

struct WavFrameWriter<W: Write + Seek> {
    writer: WavWriter<W>,
}

impl<W: Write> FrameWriter for RawFrameWriter<W> {
    fn write_frame(&mut self, buf: &[f32]) -> Result<(), Error> {
        self.buf.resize(buf.len() * 2, 0);
        for (dst, src) in self.buf.chunks_mut(2).zip(buf) {
            let bytes =
                (src.max(i16::MIN as f32).min(i16::MAX as f32).round() as i16).to_le_bytes();
            dst[0] = bytes[0];
            dst[1] = bytes[1];
        }
        self.writer.write_all(&self.buf[..]).map_err(|e| e.into())
    }

    fn finalize(&mut self) -> Result<(), Error> {
        self.writer.flush()?;
        Ok(())
    }
}

impl<W: Write + Seek> FrameWriter for WavFrameWriter<W> {
    fn write_frame(&mut self, buf: &[f32]) -> Result<(), Error> {
        let mut w = self.writer.get_i16_writer(buf.len() as u32);
        for &x in buf {
            w.write_sample(x.max(i16::MIN as f32).min(i16::MAX as f32).round() as i16);
        }
        w.flush().map_err(|e| e.into())
    }

    fn finalize(&mut self) -> Result<(), Error> {
        self.writer.flush().map_err(|e| e.into())
    }
}

fn raw_samples<R: Read + 'static>(r: R, channels: usize, sample_rate: f64) -> Box<dyn FrameSource> {
    Box::new(SampleReader::new(
        RawSampleIter { reader: r },
        channels,
        sample_rate,
    ))
}

fn wav_samples<R: Read + 'static>(wav: WavReader<R>) -> Box<dyn FrameSource> {
    let sample_rate = wav.spec().sample_rate as f64;
    let channels = wav.spec().channels as usize;
    match wav.spec().sample_format {
        SampleFormat::Int => {
            let bits_per_sample = wav.spec().bits_per_sample;
            assert!(bits_per_sample <= 32);

            let iter = wav.into_samples::<i32>().map(move |s| {
                s.map(|s| {
                    if bits_per_sample < 16 {
                        (s << (16 - bits_per_sample)) as f32
                    } else {
                        (s >> (bits_per_sample - 16)) as f32
                    }
                })
                .map_err(|e| e.into())
            });
            Box::new(SampleReader::new(iter, channels, sample_rate))
        }
        SampleFormat::Float => {
            let iter = wav
                .into_samples::<f32>()
                .map(|s| s.map(|s| s * 32767.0).map_err(|e| e.into()));
            Box::new(SampleReader::new(iter, channels, sample_rate))
        }
    }
}

fn parse_positive<T: std::str::FromStr + PartialOrd + Default>(s: &str) -> Result<(), String> {
    match s.parse::<T>() {
        Ok(v) if v > T::default() => Ok(()),
        Ok(_) => Err("must be greater than zero".to_string()),
        Err(_) => Err("not a number".to_string()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("nnnoiseless")
        .version(crate_version!())
        .about("Remove noise from audio files")
        .arg(arg!(<INPUT> "input audio file"))
        .arg(arg!(<OUTPUT> "output audio file"))
        .arg(arg!(--"wav-in" "the input is a wav file (default is to detect wav files by their filename"))
        .arg(arg!(--"wav-out" "the output is a wav file (default is to detect wav files by their filename)"))
        .arg(arg!(--"sample-rate" <RATE> "for raw input, the sample rate of the input (defaults to 48kHz)").required(false)
                .validator(parse_positive::<f64>))
        .arg(arg!(--channels <CHANNELS> "for raw input, the number of channels (defaults to 1)")
                .required(false)
                .validator(parse_positive::<u16>))
        .arg(arg!(--model <PATH> "path to a custom model file").required(false))
        .arg(arg!(--"max-attenuation" <DB> "limit suppression to this many dB, leaving a noise floor (default: unlimited)")
                .required(false)
                .validator(|s| s.parse::<f32>().map(|_| ())))
        .arg(arg!(--"vad-threshold" <PROB> "attenuate frames whose speech probability is below this (0..1, default 0)")
                .required(false)
                .validator(|s| s.parse::<f32>().map(|_| ())))
        .arg(arg!(--lookahead <FRAMES> "look this many 10ms frames ahead to protect speech onsets (default 0)")
                .required(false)
                .validator(|s| s.parse::<usize>().map(|_| ())))
        .arg(arg!(--"pitch-interval" <N> "run the pitch search every N frames; faster, slightly lower quality (default 1)")
                .required(false)
                .validator(parse_positive::<usize>))
        .arg(arg!(--"link-channels" <MODE> "how to combine gains across channels: independent, max, mean (default: max)")
                .required(false)
                .possible_values(["independent", "max", "mean"]))
        .get_matches();

    let in_name = matches.value_of("INPUT").unwrap();
    let out_name = matches.value_of("OUTPUT").unwrap();
    let in_file = BufReader::new(
        File::open(in_name)
            .with_context(|| format!("Failed to open input file \"{}\"", in_name))?,
    );
    let out_file = BufWriter::new(
        File::create(out_name)
            .with_context(|| format!("Failed to open output file \"{}\"", out_name))?,
    );
    let in_wav =
        matches.is_present("wav-in") || Path::new(in_name).extension() == Some("wav".as_ref());
    let out_wav =
        matches.is_present("wav-out") || Path::new(out_name).extension() == Some("wav".as_ref());

    let (mut samples, channels) = if in_wav {
        let wav_reader = WavReader::new(in_file)?;
        if wav_reader.spec().channels == 0 || wav_reader.spec().sample_rate == 0 {
            return Err(anyhow!(
                "input WAV must have at least one channel and a non-zero sample rate"
            )
            .into());
        }
        let channels = wav_reader.spec().channels;
        (wav_samples(wav_reader), channels)
    } else {
        let sample_rate: f64 = matches.value_of_t("sample-rate").unwrap_or(48_000.0);
        let channels = matches.value_of_t("channels").unwrap_or(1);
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(anyhow!("sample rate must be a finite positive number").into());
        }
        if channels == 0 {
            return Err(anyhow!("channels must be greater than zero").into());
        }
        (
            raw_samples(in_file, channels as usize, sample_rate),
            channels,
        )
    };

    let mut frame_writer: Box<dyn FrameWriter> = if out_wav {
        let spec = WavSpec {
            channels,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let writer = WavWriter::new(out_file, spec)?;
        Box::new(WavFrameWriter { writer })
    } else {
        Box::new(RawFrameWriter {
            writer: out_file,
            buf: Vec::new(),
        })
    };

    let model = if let Some(model_path) = matches.value_of("model") {
        let data = std::fs::read(model_path).context("Failed to open model file")?;
        RnnModel::from_bytes(&data).context("Failed to parse model file")?
    } else {
        RnnModel::default()
    };

    let mut params = DenoiseParams::default();
    if let Some(db) = matches.value_of("max-attenuation") {
        params = params.max_attenuation_db(db.parse()?);
    }
    if let Some(p) = matches.value_of("vad-threshold") {
        params = params.vad_threshold(p.parse()?);
    }
    if let Some(f) = matches.value_of("lookahead") {
        params = params.lookahead(f.parse()?);
    }
    if let Some(n) = matches.value_of("pitch-interval") {
        params = params.pitch_interval(n.parse()?);
    }
    let link = match matches.value_of("link-channels").unwrap_or("max") {
        "independent" => ChannelLink::Independent,
        "mean" => ChannelLink::Mean,
        _ => ChannelLink::Max,
    };

    let channels = channels as usize;
    let mut denoiser = MultiDenoiser::with_model(channels, link, &model, params);
    let latency = denoiser.latency_frames();

    let mut interleaved = vec![0.0; FRAME_SIZE * channels];
    let mut in_bufs = vec![vec![0.0; FRAME_SIZE]; channels];
    let mut out_bufs = vec![vec![0.0; FRAME_SIZE]; channels];
    let mut out_buf = vec![0.0; FRAME_SIZE * channels];

    // Frames still inside the denoiser's delay line when the input runs out.
    let mut pending = latency;
    let mut emitted = 0usize;

    loop {
        let read = samples.read_frame(&mut interleaved)?;
        if read == 0 {
            if pending == 0 {
                break;
            }
            // Push silence through so the tail of the signal comes back out.
            pending -= 1;
            interleaved.fill(0.0);
        } else if read < interleaved.len() {
            interleaved[read..].fill(0.0);
        }

        for (ch, buf) in in_bufs.iter_mut().enumerate() {
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = interleaved[i * channels + ch];
            }
        }

        {
            let ins: Vec<&[f32]> = in_bufs.iter().map(|b| &b[..]).collect();
            let mut outs: Vec<&mut [f32]> = out_bufs.iter_mut().map(|b| &mut b[..]).collect();
            denoiser.process_frame(&mut outs, &ins);
        }

        // Drop the leading frames that are still the denoiser warming up, so the output lines
        // up with the input.
        emitted += 1;
        if emitted > latency {
            for i in 0..FRAME_SIZE {
                for j in 0..channels {
                    out_buf[i * channels + j] = out_bufs[j][i];
                }
            }
            frame_writer.write_frame(&out_buf[..])?;
        }
    }
    frame_writer.finalize()?;

    Ok(())
}
