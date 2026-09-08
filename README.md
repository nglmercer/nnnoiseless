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

## Recommended next improvements

The current implementation already covers the obvious baseline optimizations:
runtime SIMD dispatch, cached FFT plans, widened model weights, amortized input
history, and pitch-search decimation. The next changes should be measured against
both output quality and the included benchmark. Keep
`DenoiseParams::default()` compatible with the original RNNoise path; expose
experimental behaviour as an opt-in preset or parameter until it has been
validated on real recordings.

### Improve denoising quality

1. **Evaluate and train against real recordings.** `tests/quality.rs` is a fast,
   deterministic smoke test built from synthetic speech-like signals. Add an
   opt-in evaluation harness with licensed speech, stationary and changing noise,
   reverberation, music, transients, and clean speech. Track noise reduction,
   speech loss, SI-SDR/segmental SNR, onset preservation, VAD accuracy, and
   latency. Use the resulting train/validation split to fine-tune a custom
   `RnnModel` for the target microphone or application; structural model parsing
   alone cannot guarantee that a model is well calibrated.

2. **Add an explicit noise-only strategy.** The RNNoise-style model is trained to
   separate speech from noise, so its VAD can be fooled by noise without speech.
   A noise profile/calibration mode, a conservative stationary-noise estimator,
   or a separately trained non-speech detector would address this case. Raising
   `vad_threshold` is not a sufficient fix because it relies on the same VAD.

3. **Protect clean speech and transients.** Add an opt-in adaptive bypass or
   dry/wet mix when the input is already clean, and validate it on high-SNR speech
   rather than assuming more suppression is better. For live audio, offer a
   bounded onset hangover or one/two-frame lookahead mode when the application can
   afford the extra latency; the existing `lookahead` path is mainly intended for
   offline processing.

4. **Make multichannel decisions spatially coherent.** `ChannelLink::Max` is a
   safe stereo baseline, but every channel still performs its own analysis and
   recurrent inference. A shared reference (for example, a validated mid/beam
   signal) with linked gains could reduce wandering and preserve stereo imaging;
   compare it against `Independent`, `Max`, and `Mean` on phase, localization,
   speech loss, and noise reduction before changing the default.

### Improve CPU and browser performance

1. **Skip expensive analysis on silence.** `compute_frame_features_with` currently
   performs the pitch search and a second FFT before checking the low-energy
   condition that ultimately marks a frame silent. Move the early-silence decision
   immediately after the first FFT's band energies, while preserving frame-count
   and state semantics. This should help voice-chat workloads with long pauses;
   add a silence-heavy benchmark and regression test.

2. **Fuse the GRU matrix-vector products.** `src/rnn.rs` currently evaluates the
   three gates with separate input and recurrent matvec calls. Pack the gate
   weights in the hot layout and use fused kernels for the shared input gates and
   the update/reset recurrent gates, leaving the candidate recurrent product after
   the reset activation. Preserve a scalar/reference path and compare numerical
   drift, quality metrics, and per-frame latency before accepting the change.

3. **Remove per-frame multichannel and CLI overhead.** The CLI builds temporary
   `Vec<&[f32]>` and `Vec<&mut [f32]>` values for every frame, then deinterleaves
   and reinterleaves samples. Reuse those views or add an interleaved-frame API to
   `MultiDenoiser`. Also replace the byte-at-a-time RAW decoder with block reads;
   these changes improve throughput without changing the DSP result.

4. **Make resampling a specialized fast path.** The sinc resampler is much more
   expensive than the 48 kHz denoiser for non-48 kHz input. Add a true identity
   copy path, replace `Vec::drain` history movement with a ring buffer, and use
   precomputed polyphase coefficients for common ratios such as 16↔48 kHz. Keep
   alias rejection, passband level, output length, and channel-isolation tests.

5. **Make the WebAssembly streaming path allocation-free.** `Denoiser::push`
   shifts `pending`, drains `ready`, and allocates a returned `Vec` for each
   AudioWorklet block. Add a ring-buffer implementation and a caller-provided
   `push_into`/output-buffer API so the real-time worklet does not create garbage
   or risk a GC pause on the audio thread. Keep `simd128` enabled and compare
   bundle size, worklet underruns, and realtime factor.

### Rollout and measurement

For every quality change, keep the regression fixtures and add real-recording
results before changing defaults. For every performance change, run
`cargo bench` with the active ISA recorded, compare 48 kHz denoising separately
from resampling and multichannel scaling, and check the `reference`,
`low-memory`, and WebAssembly builds. A change is ready when it improves its
target metric without exceeding the agreed speech-loss, latency, memory, or
numerical-drift budget.

## Denoiser model survey (2026)

Survey date: 2026-08-03. No public catalogue is literally exhaustive, so this
is an implementation-oriented shortlist of models and architectures with public
papers, code, or weights. Published quality scores are not directly comparable
across datasets. The existing RNNoise model should remain the default until a
replacement is measured on the same recordings and latency budget.

### Candidates worth implementing

| Priority | Candidate | Why it is interesting | Possible integration | Main risk |
| --- | --- | --- | --- | --- |
| 1 | [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) | Very small grouped temporal-convolutional recurrent model: 48.2K parameters and 33.0 MMAC/s in the official repository. It includes streaming code and pretrained checkpoints. | Add a native Rust backend with an explicit feature extractor, recurrent state, and model metadata. It is the best first candidate for a small CPU/WASM model after verifying the checkpoint sample rate and tensor layout. | It is not a drop-in RNNoise replacement; the ERB/grouped temporal frontend and checkpoint conversion need to be implemented and validated. |
| 2 | [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) | Full-band 48 kHz enhancement with a Rust `libDF` runtime, deep filtering, pretrained models, and permissive MIT/Apache-2.0 code licensing. | Reuse or port the Rust frontend/runtime behind a feature-gated backend. This is the strongest quality-oriented fit for this crate's native 48 kHz API. | Larger feature/model pipeline and more state than RNNoise; WASM size and allocations need dedicated work. |
| 3 | [Time-Varying Filtering (TVF)](https://arxiv.org/abs/2603.02794) | A 2026 low-latency design in which a neural controller predicts coefficients for an interpretable 35-band IIR filter cascade. The paper reports a compact 24K-parameter realization and 10.7 ms latency. | Implement the 35-band filterbank and a small causal controller, then train/export a model in a new compact format. It matches the existing DSP-oriented design better than a large spectrogram network. | The paper does not provide a drop-in implementation or weights; training and reproduction are required. |
| 4 | [DTLN](https://huggingface.co/alekya/DTLN) | Small dual-signal-transform LSTM with SavedModel, TFLite, and ONNX exports; the model card describes real-time, one-frame-in/one-frame-out operation and fewer than one million parameters. | Add it as an optional 16 kHz ONNX/native backend for voice applications, with an explicit resampling boundary. | It is generally used at 16 kHz, so it is not a native replacement for the current 48 kHz path. |
| 5 | [LiSenNet](https://huggingface.co/claroche1/LiSenNet) | Ultra-compact causal 16 kHz enhancer with ONNX FP32/static INT8 graphs and an explicit streaming state graph. | Good embedded/INT8 experiment after adding a 16 kHz backend and caller-owned state buffers. | Magnitude-only masking with noisy phase limits quality, and it needs a separate 16 kHz API path. |
| 6 | [DPDFNet](https://github.com/ceva-ip/DPDFNet) | Stateful streaming models for 8/16/48 kHz with ONNX/TFLite exports. The repository lists a 48 kHz high-resolution model at about 2.58M parameters and 2.42G MACs. | Use as an optional ONNX/TFLite quality backend and benchmark against native models before attempting a Rust port. | It is far heavier than the current denoiser and is unlikely to be a good first pure-Rust or browser implementation. |
| 7 | [Hush](https://huggingface.co/weya-ai/hush) | Apache-2.0, 16 kHz, about 8 MB, causal, and aimed at suppressing competing speakers in voice-agent audio; its model card reports sub-millisecond CPU processing per 10 ms frame. | Offer it as a voice-agent-specific model/backend, not as the general-purpose default. | It is specialized for interfering speech and 16 kHz input; validate the weights and redistribution terms before bundling them. |

### 2026 research and quality references

These are useful for future backends or design ideas, but are not the first
models to port into a small, allocation-free Rust/WASM library.

| Model or direction | Useful idea | Implementation decision |
| --- | --- | --- |
| [NVIDIA Real-time RE-USE](https://huggingface.co/nvidia/Real-time_RE-USE) | 2026 Mamba-based enhancement with one-frame online inference, 30 latency configurations, and sample rates from 8 to 48 kHz. | Keep as an optional GPU/ONNX benchmark. The NVIDIA noncommercial license and multi-million-parameter runtime make it unsuitable as the bundled default. |
| [Shell-Core Mamba](https://research.nvidia.com/labs/twn/publication/chime_2026_shellcoremamba/) | 2026 multichannel design that separates local spectral-spatial processing from a causal Mamba core. | Use as a roadmap for spatially coherent stereo/multichannel denoising; wait for portable public weights/code before implementing. |
| [RT-Tango](https://arxiv.org/abs/2607.01834) | 2026 low-latency binaural design with asymmetric STFTs, online spatial statistics, grouped recurrent masks, and temporal sparsification. | Borrow the latency and spatial-state ideas for a future `MultiDenoiser` backend; the paper is not a drop-in model. |
| [PercepNet+](https://arxiv.org/abs/2203.02263) | Hybrid DSP/RNN design with an SNR estimator and SNR-switched postprocessing to reduce over-attenuation on clean speech. | High-value design reference for fixing this crate's nearly-clean-input regression before changing the core model. |
| [Fast FullSubNet+](https://arxiv.org/abs/2212.09019) | Full-band/sub-band fusion with better quality than very small RNNs, while reducing the original FullSubNet cost. | Consider for an optional native/ONNX quality backend; sub-band batching, complex STFT state, and memory traffic are substantial. |
| [DCCRN](https://arxiv.org/abs/2008.00264) and [FRCRN](https://arxiv.org/abs/2206.07293) | Phase-aware complex or frequency-recurrent mask estimation. | Good quality baselines, but their complex spectrogram pipelines and millions of parameters are too heavy for the first implementation. |
| [PASE](https://huggingface.co/cisco-ai/pase) and [Clear](https://huggingface.co/desert-ant-labs/clear) | Recent high-quality 16 kHz generative or 48 kHz on-device enhancement options. | Use only as offline/mobile quality references initially; PASE is heavyweight and Clear uses a source-available license that is not equivalent to a permissive model license. |
| [TokenSE](https://arxiv.org/abs/2604.12246) | 2026 codec-token/Mamba enhancement for cochlear-implant intelligibility. | Research-only for now: a neural codec/token pipeline is outside the current streaming DSP scope. |

### Recommended implementation sequence

1. Add a backend abstraction rather than teaching `RnnModel` every topology. Each
   backend should declare sample rate, frame/hop size, lookahead, channel mode,
   state size, input feature layout, output semantics, and quantization format.
2. Build a native **GTCRN** prototype for the smallest real-time alternative,
   and a 48 kHz **DeepFilterNet3** prototype for the quality alternative. Keep
   both behind opt-in features while RNNoise remains the compatibility path.
3. Add an optional ONNX backend for heavier models such as DPDFNet and RE-USE;
   do not put their weights in the base crate. Model code and weights have
   separate licenses and must be reviewed before redistribution.
4. Prototype TVF only after the evaluation harness exists, because its public
   paper is an architecture specification rather than a ready-to-run model.
5. Compare every backend on matched sample rates using speech-plus-stationary
   noise, nonstationary noise, interfering speech, reverberation, clean speech,
   noise-only input, realtime factor, allocations, peak memory, and end-to-end
   latency. The [DNS Challenge tools](https://github.com/microsoft/DNS-Challenge)
   are a useful starting point for reproducible noisy-speech evaluation.

### Hush backend (opt-in)

This branch includes a Hush backend built on the released DeepFilterNet 0.5.3
Tract runtime. It loads Hush's `advanced_dfnet16k_model_best_onnx.tar.gz`
bundle at runtime and keeps the existing RNNoise path unchanged. Hush accepts
normalized mono `f32` samples at 16 kHz in 160-sample frames, while the
RNNoise API continues to use the crate's 48 kHz, signed-16-bit-scaled contract.
Hush reports 320 samples of model algorithmic latency, while its 160-sample
overlap-add synthesis delay is the alignment delay used by the streaming and
complete-buffer adapters.

```rust
use nnnoiseless::{HushModel, HUSH_FRAME_SIZE};

let model = HushModel::from_path("advanced_dfnet16k_model_best_onnx.tar.gz")?;
let mut hush = model.denoiser()?;
let input = [0.0f32; HUSH_FRAME_SIZE];
let mut output = [0.0f32; HUSH_FRAME_SIZE];
let lsnr_db = hush.process_frame(&mut output, &input)?;
# let _ = lsnr_db;
```

Verify the native backend and measure its streaming cost with the released
bundle:

```bash
HUSH_MODEL=/path/to/advanced_dfnet16k_model_best_onnx.tar.gz \
  cargo test --features hush --test hush
HUSH_MODEL=/path/to/advanced_dfnet16k_model_best_onnx.tar.gz \
  cargo bench --features hush --bench hush
```

The Hush WebAssembly bindings expose `HushDenoiser.fromModelBytes`, its
streaming `push`/`finish` API, and `denoiseHushBuffer`. The browser build
includes the Tract runtime but not the 8 MB model; the Vite demo has a
model-bundle picker and a separate Hush microphone path that requests a 16 kHz
`AudioContext`:

```bash
cd web
npm run wasm
HUSH_MODEL=/path/to/advanced_dfnet16k_model_best_onnx.tar.gz npm test
```

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
