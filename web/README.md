# nnnoiseless in the browser

A [Vite](https://vite.dev/) demo of the `nnnoiseless` speech denoiser compiled
to WebAssembly. It shows both ways audio usually reaches a page:

- **a decoded clip**, denoised in one call with `denoiseBuffer`, with the
  original and the result drawn side by side and playable A/B;
- **live microphone input**, denoised by the streaming `Denoiser` running inside
  an `AudioWorklet`, with a speech-probability meter.
- **Hush**, an optional 16 kHz DeepFilterNet-SE backend. Load the released
  `advanced_dfnet16k_model_best_onnx.tar.gz` bundle with the model picker to
  test decoded clips and live microphone input.

The denoising parameters (`max_attenuation_db`, `vad_threshold`, `lookahead`)
are wired to sliders and take effect immediately on both paths.

## Requirements

- Rust with the `wasm32-unknown-unknown` target
  (`rustup target add wasm32-unknown-unknown`);
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) on `PATH`;
- Node 18 or newer.
- The Hush model bundle is optional and is not bundled in this repository.

## Build and run

```bash
npm install
npm run wasm     # compiles the Rust crate to WebAssembly (both targets)
npm run dev      # http://localhost:5173
```

Select **Hush — 16 kHz**, choose the model bundle, and then load a clip. The
Hush live microphone path requests a 16 kHz `AudioContext`; browsers that do
not honor that requested rate report an error instead of silently running the
model at the wrong rate.

For a production bundle, which also runs the WebAssembly smoke test:

```bash
npm run build
npm run preview
```

To check the WebAssembly build on its own, without a browser:

```bash
npm test
```

It denoises a synthetic noisy clip and asserts that noise actually went down
and speech did not, then exercises the streaming API with the 128-sample blocks
an `AudioWorklet` delivers.

For a finite recording passed through `HushDenoiser.push`, call
`HushDenoiser.finish()` after the last block. It pads any partial frame,
returns the delayed tail, and resets the session for the next recording. The
live microphone path remains open-ended and does not call `finish()`.

To exercise Hush in the same smoke test and print its load time, microseconds
per 10 ms frame, and realtime factor:

```bash
HUSH_MODEL=/path/to/advanced_dfnet16k_model_best_onnx.tar.gz npm test
```

## Why there are two wasm builds

`npm run wasm` produces the same `.wasm` binary twice, with different
JavaScript glue:

| Output | wasm-pack target | Used by |
| --- | --- | --- |
| `src/pkg/` | `web` | the main thread, as an ES module |
| `src/pkg-worklet/` | `no-modules` | the `AudioWorklet` |

An `AudioWorkletGlobalScope` has no `fetch`, so it cannot load a wasm module by
URL itself. The `no-modules` glue and the processor source are concatenated into
a single blob and registered with `addModule`, and the main thread compiles the
wasm and hands the `WebAssembly.Module` over by `postMessage` for
`initSync`. The two `.wasm` files are byte-identical, so Vite emits one asset
and both paths share the same Wasm download. Enabling Hush makes this binary
substantially larger because it includes the Tract inference runtime; the Hush
weights remain an external model bundle.

## Notes

- The page requests a 48 kHz `AudioContext` for RNNoise and a 16 kHz context for
  Hush, so the live paths do not resample inside the worklet. The decoded-clip
  Hush path resamples other rates to 16 kHz and back.
- The microphone is opened with `echoCancellation`, `noiseSuppression` and
  `autoGainControl` all disabled — otherwise the browser's own noise suppression
  would be doing the job this demo is meant to show.
- Use headphones for the live path, or the output will feed back into the input.
- Output lags input by one 10 ms frame, plus one frame per unit of lookahead.
  `denoiseBuffer` compensates for this; the live path does not, since latency is
  the point there.
- Hush's `latencySamples` reports its 20 ms algorithmic latency; its streaming
  frame alignment uses the 10 ms overlap-add synthesis delay.
- The synthetic sample is deterministic so the demo works offline and always
  sounds the same. Being harmonic, it flatters a pitch-driven model — load a
  real recording for a fair impression.

## Layout

- `index.html` — page structure;
- `src/main.ts` — application wiring and initialization;
- `src/app/` — focused modules for UI/state, settings, clips, waveforms,
  playback, and microphone setup;
- `src/denoise-worklet.ts` — the typed `AudioWorkletProcessor`;
- `src/style.css` — styling, light and dark;
- `src/audio-worklet.d.ts` — the missing Web Audio worklet globals;
- `tsconfig.json` — strict browser-side type checking;
- `vite.config.ts` — Vite configuration and raw-worklet transpilation;
- `smoke-test.mjs` — headless verification of the wasm build;
- `src/pkg/`, `src/pkg-worklet/` — generated, not checked in.
