# nnnoiseless in the browser

A [Vite](https://vite.dev/) demo of the `nnnoiseless` speech denoiser compiled
to WebAssembly. It shows both ways audio usually reaches a page:

- **a decoded clip**, denoised in one call with `denoiseBuffer`, with the
  original and the result drawn side by side and playable A/B;
- **live microphone input**, denoised by the streaming `Denoiser` running inside
  an `AudioWorklet`, with a speech-probability meter.

The denoising parameters (`max_attenuation_db`, `vad_threshold`, `lookahead`)
are wired to sliders and take effect immediately on both paths.

## Requirements

- Rust with the `wasm32-unknown-unknown` target
  (`rustup target add wasm32-unknown-unknown`);
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) on `PATH`;
- Node 18 or newer.

## Build and run

```bash
npm install
npm run wasm     # compiles the Rust crate to WebAssembly (both targets)
npm run dev      # http://localhost:5173
```

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
and both paths share the same ~441 kB download (~208 kB gzipped).

## Notes

- The page requests a 48 kHz `AudioContext` for the live path, so no resampling
  is needed there. `denoiseBuffer` resamples other rates to 48 kHz and back.
- The microphone is opened with `echoCancellation`, `noiseSuppression` and
  `autoGainControl` all disabled — otherwise the browser's own noise suppression
  would be doing the job this demo is meant to show.
- Use headphones for the live path, or the output will feed back into the input.
- Output lags input by one 10 ms frame, plus one frame per unit of lookahead.
  `denoiseBuffer` compensates for this; the live path does not, since latency is
  the point there.
- The synthetic sample is deterministic so the demo works offline and always
  sounds the same. Being harmonic, it flatters a pitch-driven model — load a
  real recording for a fair impression.

## Layout

- `index.html` — page structure;
- `src/main.js` — wiring, clip processing, waveform drawing, mic setup;
- `src/denoise-worklet.js` — the `AudioWorkletProcessor`;
- `src/style.css` — styling, light and dark;
- `smoke-test.mjs` — headless verification of the wasm build;
- `src/pkg/`, `src/pkg-worklet/` — generated, not checked in.
