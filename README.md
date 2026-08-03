# nnnoiseless

`nnnoiseless` is a self-contained, pure-Rust implementation of the RNNoise
signal path. It has no C library, C headers, FFI bridge, or native build step.

The core processes 48 kHz signed PCM in 10 ms frames (`480` samples). It
contains the RNNoise-style high-pass filter, windowed FFT feature extraction,
22 Bark bands, cepstral/delta features, pitch search and filtering, GRU
inference, overlap-add synthesis, and an optional WAV/RAW command-line tool.

## Library

```rust
use nnnoiseless::{DenoiseState, FRAME_SIZE};

let mut denoise = DenoiseState::new();
let input = [0.0f32; FRAME_SIZE];
let mut output = [0.0f32; FRAME_SIZE];
let vad_probability = denoise.process_frame(&mut output, &input);
assert!((0.0..=1.0).contains(&vad_probability));
```

Samples use the denoiser's RNNoise convention: `f32` values representing
16-bit PCM (`-32768.0..=32767.0`), not normalized `-1.0..=1.0` samples. The
first synthesized frame is a warm-up frame and should normally be discarded.

Custom binary models can be loaded with `RnnModel::from_bytes` or embedded
without allocation with `RnnModel::from_static_bytes`. The six-layer binary
format is the concatenation of the input dense layer, VAD GRU, noise GRU,
denoise GRU, denoise output dense layer, and VAD output dense layer.

## Command line

```text
cargo run --release -- input.wav output.wav
cargo run --release -- --model weights.rnn input.raw output.raw
```

The CLI accepts WAV input/output by extension (or `--wav-in`/`--wav-out`) and
16-bit little-endian interleaved RAW PCM. RAW input supports `--sample-rate`
and `--channels`; input is resampled to the algorithm's 48 kHz rate.
