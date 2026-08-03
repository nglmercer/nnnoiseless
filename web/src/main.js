// Browser demo for nnnoiseless.
//
// Two paths are shown, because audio reaches a page in two quite different
// ways:
//
//   * a decoded clip, handled in one shot by `denoiseBuffer`;
//   * live microphone input, handled by the streaming `Denoiser` running in an
//     AudioWorklet.
//
// The clip path uses the ES-module wasm build. The worklet path needs the
// `no-modules` build, for the reasons explained in `denoise-worklet.js`.

import init, { activeIsa, denoiseBuffer, version } from './pkg/nnnoiseless.js';

import workletSource from './denoise-worklet.js?raw';
import workletGlue from './pkg-worklet/nnnoiseless.js?raw';
import workletWasmUrl from './pkg-worklet/nnnoiseless_bg.wasm?url';

import './style.css';

const DEMO_SECONDS = 4;
const SAMPLE_RATE = 48_000;

const el = (id) => document.getElementById(id);

const ui = {
  status: el('badge-status'),
  version: el('badge-version'),
  isa: el('badge-isa'),
  attenuation: el('ctl-attenuation'),
  attenuationOut: el('out-attenuation'),
  vad: el('ctl-vad'),
  vadOut: el('out-vad'),
  lookahead: el('ctl-lookahead'),
  lookaheadOut: el('out-lookahead'),
  demo: el('btn-demo'),
  file: el('file-input'),
  clipInfo: el('clip-info'),
  waveforms: el('waveforms'),
  waveBefore: el('wave-before'),
  waveAfter: el('wave-after'),
  playBefore: el('play-before'),
  playAfter: el('play-after'),
  processTime: el('process-time'),
  mic: el('btn-mic'),
  bypass: el('ctl-bypass'),
  vadMeter: el('vad-meter'),
  vadValue: el('vad-value'),
  micError: el('mic-error'),
};

/** Everything that survives between interactions. */
const state = {
  clip: null, // { samples: Float32Array, sampleRate: number, name: string }
  denoised: null, // Float32Array
  audioCtx: null,
  playing: null, // currently playing AudioBufferSourceNode
  mic: null, // { ctx, stream, node, source }
};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

function settings() {
  return {
    attenuationDb: Number(ui.attenuation.value),
    vadThreshold: Number(ui.vad.value),
    lookahead: Number(ui.lookahead.value),
  };
}

function renderSettings() {
  const s = settings();
  ui.attenuationOut.textContent = s.attenuationDb > 0 ? `${s.attenuationDb} dB` : 'off';
  ui.vadOut.textContent = s.vadThreshold > 0 ? s.vadThreshold.toFixed(2) : 'off';
  ui.lookaheadOut.textContent = `${s.lookahead} frame${s.lookahead === 1 ? '' : 's'}`;
}

// ---------------------------------------------------------------------------
// Clip processing
// ---------------------------------------------------------------------------

/**
 * A deterministic, speech-like test signal: a harmonic stack with a drifting
 * pitch and syllable-rate envelope, buried in coloured noise.
 *
 * It is synthetic on purpose — it makes the demo work offline and always sound
 * the same — but it does flatter a pitch-driven model, so treat it as an
 * illustration rather than a benchmark.
 */
function makeNoisyDemo(seconds, sampleRate) {
  const n = Math.floor(seconds * sampleRate);
  const out = new Float32Array(n);
  let seed = 0x2545f491;
  let lp = 0;

  for (let i = 0; i < n; i += 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const white = (seed >>> 16) / 32768 - 1;
    lp = 0.85 * lp + 0.15 * white;

    const t = i / sampleRate;
    const syllable = Math.sin(2 * Math.PI * 3.2 * t);
    const envelope = syllable > 0 ? Math.pow(syllable, 0.6) : 0;
    const f0 = 150 + 25 * Math.sin(2 * Math.PI * 0.7 * t);

    let voice = 0;
    for (let h = 1; h <= 14; h += 1) {
      voice += Math.sin(2 * Math.PI * f0 * h * t) / Math.pow(h, 1.15);
    }

    out[i] = voice * envelope * 0.16 + (white * 0.5 + lp * 2.0) * 0.05;
  }
  return out;
}

function audioContext() {
  if (!state.audioCtx || state.audioCtx.state === 'closed') {
    state.audioCtx = new AudioContext();
  }
  return state.audioCtx;
}

async function loadFile(file) {
  const bytes = await file.arrayBuffer();
  const ctx = audioContext();
  const decoded = await ctx.decodeAudioData(bytes);

  // Downmix to mono: the denoiser is a mono algorithm, and mixing here keeps
  // the demo's comparison honest.
  const channels = decoded.numberOfChannels;
  const mono = new Float32Array(decoded.length);
  for (let c = 0; c < channels; c += 1) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < mono.length; i += 1) mono[i] += data[i] / channels;
  }

  setClip({ samples: mono, sampleRate: decoded.sampleRate, name: file.name });
}

function setClip(clip) {
  state.clip = clip;
  const seconds = clip.samples.length / clip.sampleRate;
  ui.clipInfo.textContent = `${clip.name} — ${seconds.toFixed(1)}s at ${(
    clip.sampleRate / 1000
  ).toFixed(1)} kHz`;
  ui.waveforms.hidden = false;
  processClip();
}

function processClip() {
  if (!state.clip) return;
  const { samples, sampleRate } = state.clip;
  const s = settings();

  const started = performance.now();
  state.denoised = denoiseBuffer(
    samples,
    sampleRate,
    s.attenuationDb,
    s.vadThreshold,
    s.lookahead,
  );
  const elapsed = performance.now() - started;

  const audioSeconds = samples.length / sampleRate;
  ui.processTime.textContent = `${elapsed.toFixed(0)} ms for ${audioSeconds.toFixed(
    1,
  )}s (${(audioSeconds / (elapsed / 1000)).toFixed(0)}x realtime)`;

  drawWave(ui.waveBefore, samples, 'before');
  drawWave(ui.waveAfter, state.denoised, 'after');
}

// ---------------------------------------------------------------------------
// Waveform rendering
// ---------------------------------------------------------------------------

function drawWave(canvas, samples, which) {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 600;
  const cssHeight = Number(canvas.getAttribute('height'));
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const styles = getComputedStyle(document.documentElement);
  const stroke = styles.getPropertyValue(
    which === 'before' ? '--wave-before' : '--wave-after',
  );
  const mid = cssHeight / 2;

  // One vertical bar per pixel column, spanning that column's min and max. This
  // is the honest way to draw a waveform that has far more samples than pixels.
  const perPixel = Math.max(1, Math.floor(samples.length / cssWidth));
  ctx.strokeStyle = stroke.trim() || '#888';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < cssWidth; x += 1) {
    const start = x * perPixel;
    const end = Math.min(samples.length, start + perPixel);
    if (start >= samples.length) break;
    let min = 1;
    let max = -1;
    for (let i = start; i < end; i += 1) {
      const v = samples[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x + 0.5, mid - max * mid * 0.95);
    ctx.lineTo(x + 0.5, mid - min * mid * 0.95);
  }
  ctx.stroke();

  // Centre line.
  ctx.strokeStyle = styles.getPropertyValue('--grid').trim() || '#ccc';
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(cssWidth, mid);
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

function stopPlayback() {
  if (state.playing) {
    try {
      state.playing.stop();
    } catch {
      // Already stopped; nothing to do.
    }
    state.playing = null;
  }
  ui.playBefore.textContent = '▶ Play';
  ui.playAfter.textContent = '▶ Play';
}

async function play(samples, button) {
  const wasPlaying = state.playing;
  stopPlayback();
  if (wasPlaying && button.dataset.active === 'true') {
    button.dataset.active = 'false';
    return;
  }

  const ctx = audioContext();
  await ctx.resume();

  const buffer = ctx.createBuffer(1, samples.length, state.clip.sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.onended = () => {
    button.dataset.active = 'false';
    stopPlayback();
  };
  source.start();

  state.playing = source;
  ui.playBefore.dataset.active = 'false';
  ui.playAfter.dataset.active = 'false';
  button.dataset.active = 'true';
  button.textContent = '■ Stop';
}

// ---------------------------------------------------------------------------
// Live microphone
// ---------------------------------------------------------------------------

/**
 * Builds the worklet module.
 *
 * The `no-modules` wasm-bindgen glue and the processor source are concatenated
 * into a single blob, so that the processor can reach the `wasm_bindgen` symbol
 * the glue defines.
 */
async function registerWorklet(ctx) {
  const blob = new Blob([workletGlue, '\n', workletSource], {
    type: 'application/javascript',
  });
  const url = URL.createObjectURL(blob);
  try {
    await ctx.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function startMic() {
  ui.micError.hidden = true;
  ui.mic.disabled = true;
  ui.mic.textContent = 'Starting…';

  try {
    // Ask for raw audio: the browser's own noise suppression would otherwise be
    // doing the very job we are trying to demonstrate.
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    // 48kHz keeps the live path free of any resampling.
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    await ctx.resume();
    await registerWorklet(ctx);

    const node = new AudioWorkletNode(ctx, 'nnnoiseless-denoiser', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });

    node.port.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'vad') {
        const pct = Math.max(0, Math.min(1, msg.value));
        ui.vadMeter.style.width = `${(pct * 100).toFixed(1)}%`;
        ui.vadValue.textContent = pct.toFixed(2);
      } else if (msg.type === 'ready') {
        ui.mic.disabled = false;
        ui.mic.textContent = 'Stop microphone';
        ui.mic.classList.add('danger');
      } else if (msg.type === 'error') {
        showMicError(msg.message);
      }
    };

    // A worklet cannot fetch, so compile here and hand the module over.
    const wasmBytes = await (await fetch(workletWasmUrl)).arrayBuffer();
    const module = await WebAssembly.compile(wasmBytes);
    const s = settings();
    node.port.postMessage({
      type: 'init',
      module,
      attenuationDb: s.attenuationDb,
      vadThreshold: s.vadThreshold,
    });

    const source = ctx.createMediaStreamSource(stream);
    source.connect(node);
    node.connect(ctx.destination);

    node.port.postMessage({ type: 'bypass', value: ui.bypass.checked });
    state.mic = { ctx, stream, node, source };
  } catch (err) {
    showMicError(err && err.message ? err.message : String(err));
    ui.mic.disabled = false;
    ui.mic.textContent = 'Start microphone';
  }
}

function stopMic() {
  if (!state.mic) return;
  const { ctx, stream, node, source } = state.mic;
  source.disconnect();
  node.disconnect();
  stream.getTracks().forEach((track) => track.stop());
  ctx.close();
  state.mic = null;

  ui.mic.textContent = 'Start microphone';
  ui.mic.classList.remove('danger');
  ui.vadMeter.style.width = '0%';
  ui.vadValue.textContent = '0.00';
}

function showMicError(message) {
  ui.micError.textContent = `Microphone unavailable: ${message}`;
  ui.micError.hidden = false;
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

async function main() {
  renderSettings();

  await init();
  ui.status.textContent = 'ready';
  ui.status.classList.add('ok');
  ui.version.textContent = version();
  ui.isa.textContent = activeIsa();

  for (const control of [ui.attenuation, ui.vad, ui.lookahead]) {
    control.addEventListener('input', () => {
      renderSettings();
      processClip();
      // Keep a running mic in step with the sliders.
      if (state.mic) {
        const s = settings();
        state.mic.node.port.postMessage({ type: 'attenuation', value: s.attenuationDb });
        state.mic.node.port.postMessage({ type: 'vadThreshold', value: s.vadThreshold });
      }
    });
  }

  ui.demo.addEventListener('click', () => {
    stopPlayback();
    setClip({
      samples: makeNoisyDemo(DEMO_SECONDS, SAMPLE_RATE),
      sampleRate: SAMPLE_RATE,
      name: 'synthetic noisy speech',
    });
  });

  ui.file.addEventListener('change', async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    stopPlayback();
    ui.clipInfo.textContent = 'decoding…';
    try {
      await loadFile(file);
    } catch (err) {
      ui.clipInfo.textContent = `could not decode: ${err.message ?? err}`;
    }
  });

  ui.playBefore.addEventListener('click', () => state.clip && play(state.clip.samples, ui.playBefore));
  ui.playAfter.addEventListener('click', () => state.denoised && play(state.denoised, ui.playAfter));

  ui.mic.addEventListener('click', () => (state.mic ? stopMic() : startMic()));
  ui.bypass.addEventListener('change', () => {
    state.mic?.node.port.postMessage({ type: 'bypass', value: ui.bypass.checked });
  });

  window.addEventListener('resize', () => {
    if (state.clip) {
      drawWave(ui.waveBefore, state.clip.samples, 'before');
      drawWave(ui.waveAfter, state.denoised, 'after');
    }
  });
}

main().catch((err) => {
  ui.status.textContent = `failed: ${err.message ?? err}`;
  ui.status.classList.add('bad');
});
