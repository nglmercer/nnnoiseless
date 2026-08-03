# nnnoiseless

`nnnoiseless` is a Rust implementation of the RNNoise signal path for
real-time speech denoising. The denoising library itself has no dependency on
the original C RNNoise project, C headers, or an FFI bridge. It includes the
DSP pipeline, pitch analysis, feature extraction, GRU inference, and overlap-
add synthesis in Rust.

The crate also provides:

- tunable denoising parameters, including an attenuation limit and lookahead;
- multi-channel denoising with linked gains;
- a windowed-sinc resampler, so input at other sample rates can be converted;
- a WebAssembly build with JavaScript bindings, and a browser demo;
- a WAV/RAW command-line program;
- an optional DASP `Signal` adapter;
- an optional CPAL microphone example that records a WAV and writes a
  denoised WAV.

## Audio contract

The denoiser operates on mono, 48 kHz PCM in frames of 480 samples (10 ms).
The public API uses `f32` values with the scale of signed 16-bit PCM:
`-32768.0..=32767.0`. It does not expect normalized audio in
`-1.0..=1.0`.

Output lags input by at least one frame. This is intrinsic: reconstructing
input frame *k* needs the analysis window covering frames *k* and *k+1*, so no
causal implementation can emit frame *k* before it has been given frame *k+1*.
`DenoiseState::latency_frames` reports the total delay, including any lookahead
you requested. If you have the whole signal in memory, `denoise_offline`
handles the bookkeeping and returns a buffer aligned with the input.

## Quick start

From this directory:

```bash
cargo build --release
cargo test --all-targets
cargo run --release -- input.wav output.wav
```

The command-line program detects WAV files by their `.wav` extension. For RAW
PCM, specify the input format explicitly:

```bash
cargo run --release -- \
  --sample-rate 48000 \
  --channels 1 \
  --wav-out \
  input.raw output.wav
```

RAW input is signed, 16-bit, little-endian, interleaved PCM. WAV input may be
multi-channel and may use another sample rate; the CLI resamples it to 48 kHz
before processing. Output is 16-bit, 48 kHz WAV or RAW PCM, the same length as
the input.

### Tuning the CLI

| Flag | Effect |
| --- | --- |
| `--max-attenuation <DB>` | Leave a noise floor instead of suppressing to silence. |
| `--vad-threshold <PROB>` | Attenuate frames whose speech probability is below this. |
| `--lookahead <FRAMES>` | Protect speech onsets, at the cost of latency. |
| `--pitch-interval <N>` | Run the pitch search every `N` frames. Faster, slightly worse. |
| `--link-channels <MODE>` | `independent`, `max` (default) or `mean`. |
| `--model <PATH>` | Use a custom model file. |

```bash
# A gentler setting that usually sounds more natural than full suppression.
cargo run --release -- --max-attenuation 12 --lookahead 2 noisy.wav clean.wav
```

## Library usage

Use `DenoiseState` when you already have 48 kHz audio frames:

```rust
use nnnoiseless::{DenoiseState, FRAME_SIZE};

let mut denoise = DenoiseState::new();
let mut output = [0.0f32; FRAME_SIZE];
let input = [0.0f32; FRAME_SIZE];

let vad_probability = denoise.process_frame(&mut output, &input);
assert!((0.0..=1.0).contains(&vad_probability));
```

For a whole signal that is already in memory, `denoise_offline` deals with the
latency for you and returns output the same length as the input:

```rust
use nnnoiseless::{denoise_offline, DenoiseParams};

let noisy: Vec<f32> = vec![0.0; 48_000];
let clean = denoise_offline(DenoiseParams::default().lookahead(2), &noisy);
assert_eq!(clean.len(), noisy.len());
```

For streaming, keep one state per channel and feed it complete 480-sample
frames in order, discarding `latency_frames()` frames of output at the start.

### Tuning

`DenoiseParams` exposes the constants that used to be baked into the signal
path. Every default reproduces the original RNNoise behaviour exactly, so
`DenoiseParams::default()` changes nothing.

```rust
use nnnoiseless::{DenoiseParams, DenoiseState};

let params = DenoiseParams::default()
    // Leave a 12 dB noise floor rather than suppressing all the way down. This
    // is the usual fix for "musical noise" and pumping.
    .max_attenuation_db(12.0)
    // Gate frames the model is confident contain no speech.
    .vad_threshold(0.5)
    // Look two frames ahead so speech onsets are not clipped (offline only).
    .lookahead(2);

let mut state = DenoiseState::with_params(params);
# let _ = &mut state;
```

| Parameter | Default | Purpose |
| --- | --- | --- |
| `max_attenuation_db` | unlimited | Cap suppression, leaving a noise floor. |
| `gain_decay` | `0.6` | How fast suppression may engage. |
| `gain_rise` | `1.0` (no limit) | How fast suppression may let go. |
| `vad_threshold` | `0.0` (off) | Gate on speech probability. |
| `silence_gain_decay` | `1.0` (off) | Fade remembered gains during silence. |
| `pitch_filter` | `true` | Comb filter reinforcing the detected pitch. |
| `pitch_interval` | `1` | Run the pitch search every `N` frames. |
| `lookahead` | `0` | Frames of lookahead for gain decisions. |

### Multiple channels

Running one `DenoiseState` per channel lets the channels make different
decisions, so the residual noise floor wanders between them. `MultiDenoiser`
links their gains:

```rust
use nnnoiseless::{ChannelLink, MultiDenoiser, DenoiseState};

let mut denoiser = MultiDenoiser::new(2, ChannelLink::Max);
let input = vec![vec![0.0f32; DenoiseState::FRAME_SIZE]; 2];
let mut output = vec![vec![0.0f32; DenoiseState::FRAME_SIZE]; 2];

let inputs: Vec<&[f32]> = input.iter().map(|c| &c[..]).collect();
let mut outputs: Vec<&mut [f32]> = output.iter_mut().map(|c| &mut c[..]).collect();
denoiser.process_frame(&mut outputs, &inputs);
```

### Other sample rates

`Resampler` converts to and from the 48 kHz the denoiser requires. It is a
Kaiser-windowed sinc interpolator whose cutoff tracks the conversion ratio, so
downsampling does not fold high-frequency content back into the audible band.

```rust
use nnnoiseless::Resampler;

let mut r = Resampler::to_denoiser_rate(16_000.0, 1);
let mut at_48k = Vec::new();
r.process(&vec![0.0f32; 1600], &mut at_48k);
r.flush(&mut at_48k);
```

### Custom models

The built-in model is embedded in the crate. A custom model can be loaded from
disk and owned by a state:

```rust
use nnnoiseless::{DenoiseState, RnnModel};

let model_bytes = std::fs::read("weights.rnn")?;
let model = RnnModel::from_bytes(&model_bytes)
    .ok_or("invalid nnnoiseless model")?;
let mut state = DenoiseState::from_model(model);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Two on-disk formats are understood, and `RnnModel::from_bytes` detects which
one it was given:

- **v1**, the original RNNoise layout. Each layer dimension is a single signed
  byte, so it cannot describe a layer wider than 127 neurons.
- **v2**, the same weight data behind a short header with 32-bit dimensions,
  which removes that limit. `RnnModel::to_bytes` writes this format, and a
  v1 → v2 → v1 round trip is lossless.

Layer widths are otherwise unconstrained, so larger models than the built-in
one can be loaded. The structural requirements that remain are fixed by the DSP
around the network: 42 input features, 22 output band gains, one VAD output,
and consistent wiring between layers.

For an embedded model, use `RnnModel::from_static_bytes`. A parsed model can
also be shared by multiple `DenoiseState::with_model` instances; each state
keeps its own recurrent and DSP history.

### DASP integration

Enable the `dasp` feature to use `DenoiseSignal` with a DASP `Signal`:

```bash
cargo test --no-default-features --features dasp
```

## Performance

The hot kernels — the GRU matrix-vector products, the pitch cross-correlations
and the band aggregation — are compiled once per supported instruction set and
selected at first use from runtime CPU feature detection. Model weights are
widened from `i8` to `f32` once at load time so the inner loops do no
conversion. `nnnoiseless::active_isa()` reports what was selected.

Measured on an AMD BC-250 (AVX2), 20 s of synthetic voiced audio:

| build | µs per 10 ms frame | realtime factor |
| --- | ---: | ---: |
| 0.1.0, default target | 73.3 | 136x |
| 0.1.0, `-C target-cpu=native` | 50.2 | 199x |
| this version, default target | 45.9 | 218x |

Runtime dispatch means the default build now beats what the previous version
achieved only with a machine-specific `RUSTFLAGS`. Setting
`RUSTFLAGS="-C target-cpu=native"` on top of this version is still worth a few
percent, mostly in code the kernels do not cover.

`cargo bench` reports the per-configuration table.

### Numerical reproducibility

This version is not bit-for-bit identical to 0.1.0: the FFT plan, the order of
accumulation in the dot products, and FMA contraction all perturb the low bits,
and the recurrent network accumulates those perturbations. What is preserved is
the audible result — `tests/regression.rs` holds per-frame statistics captured
from 0.1.0 and checks against them. Measured drift is 0.0001% on overall energy,
0.36% worst case on any single frame, and 1e-6 on the voice-activity output.

Build with the `reference` feature to pin the kernels to the portable scalar
path, which produces identical results on every machine.

## Feature flags

| Feature | Purpose |
| --- | --- |
| `bin` | Builds the WAV/RAW `nnnoiseless` command-line program. |
| `dasp` | Enables the DASP streaming adapter. |
| `mic-example` | Builds the CPAL microphone-recording example. |
| `reference` | Forces the scalar kernels, for cross-machine reproducibility. |
| `low-memory` | Keeps weights quantized as `i8`; ~4x less model memory, slower. |

The default feature set is `bin,dasp`. The microphone example is deliberately
opt-in because it adds a platform audio backend.

## Known limitations

**Noise-only input is barely suppressed.** The model separates speech from
noise, so given a signal with no speech anywhere it has nothing to separate: it
leaves the level roughly alone and its voice-activity output reports high
confidence that it is hearing speech. The original C implementation behaves the
same way. Suppression figures should be measured on speech-plus-noise mixtures,
where 15–20 dB is typical. `vad_threshold` helps on mixtures but cannot rescue
noise-only material, because the VAD itself is fooled.

**Objective metrics get worse on nearly clean input.** SI-SDR and segmental SNR
measure waveform fidelity, while the denoiser applies time-varying per-band
gains. Above roughly 7 dB input SNR the reshaping costs more waveform accuracy
than the removed noise is worth. This is expected, and is why the quality tests
only require improvement below that point.

## In the browser

The crate compiles to WebAssembly, and `web/` holds a Vite demo that denoises
both a decoded clip and live microphone input:

```bash
cd web
npm install
npm run wasm     # wasm-pack build, both targets
npm run dev      # http://localhost:5173
```

The bindings live behind the `wasm` feature: a streaming `Denoiser` class for
live audio, which buffers internally so it can be fed the 128-sample blocks an
`AudioWorklet` delivers, and a `denoiseBuffer` function that handles a whole
clip and resamples to 48 kHz and back. `web/smoke-test.mjs` verifies the build
headlessly with `npm test`. See [web/README.md](web/README.md).

The module is about 441 kB (208 kB gzipped) and runs at roughly 100x realtime
in Node with `simd128` enabled.

## Record a microphone and denoise it

The `mic_denoise` example uses [CPAL](https://docs.rs/cpal/latest/cpal/) to
capture the default input device. It records the device's native sample rate
and channel count to a float WAV, then reads that WAV, downmixes to mono,
resamples to 48 kHz, runs the `nnnoiseless` library, and writes a 16-bit
denoised WAV.

Run it for five seconds with the default filenames:

```bash
cargo run --release --example mic_denoise --features mic-example
```

Or choose the duration and output paths:

```bash
cargo run --release \
  --example mic_denoise \
  --features mic-example \
  -- 10 microphone.wav microphone-denoised.wav
```

The example prints the selected device and configuration. It may require an
audio-backend development package on Linux, such as `libasound2-dev` on
Debian/Ubuntu or the equivalent ALSA package on another distribution. You
also need a working microphone and permission for the process to access it.

## Verify the implementation

Run the complete local checks:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo test --no-default-features
cargo test --no-default-features --features dasp
cargo test --no-default-features --features reference
cargo test --no-default-features --features low-memory
cargo check --example mic_denoise --features mic-example
cargo doc --no-deps --all-features
cargo build --release
cargo bench

# WebAssembly
cargo check --no-default-features --features wasm --target wasm32-unknown-unknown
(cd web && npm install && npm run build)
```

The quality tests print their measurements:

```bash
cargo test --test quality --release -- --nocapture --test-threads=1
```

To verify the CLI without a microphone, create a synthetic WAV with SoX:

```bash
sox -n -r 48000 -c 1 -b 16 /tmp/nnnoiseless-input.wav synth 1 sine 440
cargo run --release -- \
  /tmp/nnnoiseless-input.wav \
  /tmp/nnnoiseless-output.wav
sox --i /tmp/nnnoiseless-output.wav
```

The output should report a 48 kHz, 16-bit WAV with the same duration as the
input.

## Source layout

- `src/lib.rs` — shared constants, FFT windowing, Bark-band aggregation, and
  public exports;
- `src/simd.rs` — runtime-dispatched numerical kernels;
- `src/util.rs` — high-pass filter and activation approximations;
- `src/params.rs` — tunable denoising parameters;
- `src/features.rs` — spectral, cepstral, pitch-filter, and synthesis state;
- `src/pitch.rs` — multi-resolution pitch search;
- `src/rnn.rs` — dense/GRU layers, model parser, and recurrent inference;
- `src/denoise.rs` — frame-level orchestration and the offline helper;
- `src/multi.rs` — multi-channel denoising with linked gains;
- `src/resample.rs` — Kaiser-windowed sinc resampler;
- `src/nnnoiseless.rs` — WAV/RAW command-line interface;
- `benches/denoise.rs` — per-configuration benchmark;
- `tests/quality.rs` — objective quality evaluation;
- `tests/regression.rs` — agreement with the original signal path;
- `examples/mic_denoise.rs` — microphone recording and WAV denoising workflow.

## License

BSD-3-Clause. See [COPYING](COPYING).
