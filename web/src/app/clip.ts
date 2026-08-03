import { denoiseBuffer } from '../pkg/nnnoiseless.js';
import { audioContext } from './audio';
import { settings } from './settings';
import { state } from './state';
import type { Clip } from './types';
import { ui } from './ui';
import { drawWave } from './waveform';

/**
 * A deterministic, speech-like test signal: a harmonic stack with a drifting
 * pitch and syllable-rate envelope, buried in coloured noise.
 */
export function makeNoisyDemo(seconds: number, sampleRate: number): Float32Array {
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

export async function loadFile(file: File): Promise<void> {
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

export function setClip(clip: Clip): void {
  state.clip = clip;
  const seconds = clip.samples.length / clip.sampleRate;
  ui.clipInfo.textContent = `${clip.name} — ${seconds.toFixed(1)}s at ${(
    clip.sampleRate / 1000
  ).toFixed(1)} kHz`;
  ui.waveforms.hidden = false;
  processClip();
}

export function processClip(): void {
  if (!state.clip) return;
  const { samples, sampleRate } = state.clip;
  const current = settings();

  const started = performance.now();
  state.denoised = denoiseBuffer(
    samples,
    sampleRate,
    current.attenuationDb,
    current.vadThreshold,
    current.lookahead,
  );
  const elapsed = performance.now() - started;

  const audioSeconds = samples.length / sampleRate;
  ui.processTime.textContent = `${elapsed.toFixed(0)} ms for ${audioSeconds.toFixed(
    1,
  )}s (${(audioSeconds / (elapsed / 1000)).toFixed(0)}x realtime)`;

  drawWave(ui.waveBefore, samples, 'before');
  drawWave(ui.waveAfter, state.denoised, 'after');
}
