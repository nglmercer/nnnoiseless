# nnnoiseless

`nnnoiseless` is a Rust implementation of the RNNoise signal path for
real-time speech denoising. The denoising library itself has no dependency on
the original C RNNoise project, C headers, or an FFI bridge. It includes the
DSP pipeline, pitch analysis, feature extraction, GRU inference, and overlap-
add synthesis in Rust.

The crate also provides:

- a small WAV/RAW command-line program;
- an optional DASP `Signal` adapter;
- an optional CPAL microphone example that records a WAV and writes a
  denoised WAV.

## Audio contract

The denoiser operates on mono, 48 kHz PCM in frames of 480 samples (10 ms).
The public API uses `f32` values with the scale of signed 16-bit PCM:
`-32768.0..=32767.0`. It does not expect normalized audio in
`-1.0..=1.0`.

The first processed frame is a warm-up frame because the algorithm uses a
window and history from previous frames. Streaming applications should
discard that first output frame. The CLI and microphone example do this for
you.

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
before processing. Output is 16-bit, 48 kHz WAV or RAW PCM.

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

For a stream of samples, keep one state for each channel and process complete
480-sample frames in order:

```rust
use nnnoiseless::{DenoiseState, FRAME_SIZE};

fn denoise_mono(input: &[f32]) -> Vec<f32> {
    let mut state = DenoiseState::new();
    let mut output = Vec::with_capacity(input.len());
    let mut frame_output = [0.0; FRAME_SIZE];

    for frame in input.chunks_exact(FRAME_SIZE) {
        state.process_frame(&mut frame_output, frame);
        output.extend_from_slice(&frame_output);
    }

    output
}
```

In production, discard the first `FRAME_SIZE` values from `output` or use an
equivalent stream delay policy. Inputs whose length is not a multiple of 480
samples should be padded or buffered until a complete frame is available.

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

For an embedded model, use `RnnModel::from_static_bytes`. A parsed model can
also be shared by multiple `DenoiseState::with_model` instances; each state
keeps its own recurrent and DSP history.

### DASP integration

Enable the `dasp` feature to use `DenoiseSignal` with a DASP `Signal`:

```bash
cargo test --no-default-features --features dasp
```

The adapter converts the signal to the denoiser's 16-bit PCM scale and returns
floating-point output samples.

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
Use `--help` to see the positional arguments:

```bash
cargo run --example mic_denoise --features mic-example -- --help
```

The `mic-example` feature is optional because microphone backends are
platform-specific. The core library and the normal CLI do not need it.

## Verify the implementation

Run the complete local checks:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo test --no-default-features
cargo test --no-default-features --features dasp
cargo check --example mic_denoise --features mic-example
cargo doc --no-deps --all-features
cargo build --release
```

To verify the CLI without a microphone, create a synthetic WAV with SoX:

```bash
sox -n -r 48000 -c 1 -b 16 /tmp/nnnoiseless-input.wav synth 1 sine 440
cargo run --release -- \
  /tmp/nnnoiseless-input.wav \
  /tmp/nnnoiseless-output.wav
sox --i /tmp/nnnoiseless-output.wav
```

The output should report a 48 kHz, 16-bit WAV. Its duration is one 10 ms
frame shorter than the input because the first warm-up frame is discarded.

## Feature flags

| Feature | Purpose |
| --- | --- |
| `bin` | Builds the WAV/RAW `nnnoiseless` command-line program. |
| `dasp` | Enables the DASP streaming adapter. |
| `mic-example` | Builds the CPAL microphone-recording example. |

The default feature set is `bin,dasp`. The microphone example is deliberately
opt-in because it adds a platform audio backend.

## Source layout

- `src/lib.rs` — shared constants, FFT windowing, Bark-band aggregation, and
  public exports;
- `src/util.rs` — high-pass filter and activation approximations;
- `src/features.rs` — spectral, cepstral, pitch-filter, and synthesis state;
- `src/pitch.rs` — multi-resolution pitch search;
- `src/rnn.rs` — dense/GRU layers, model parser, and recurrent inference;
- `src/denoise.rs` — frame-level orchestration;
- `src/nnnoiseless.rs` — WAV/RAW command-line interface;
- `examples/mic_denoise.rs` — microphone recording and WAV denoising workflow.

## License

BSD-3-Clause. See [COPYING](COPYING).
