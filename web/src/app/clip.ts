import { denoiseBuffer } from '../pkg/nnnoiseless.js';
import { audioContext } from './audio';
import { settings } from './settings';
import { state } from './state';
import type { Clip } from './types';
import { ui } from './ui';
import { drawWave } from './waveform';

/**
 * Deterministic, speech-like test signal.
 *
 * Improvements over a naive harmonic-stack + noise generator:
 *  - phase-accumulated glottal source, so a drifting f0 stays phase-coherent
 *  - harmonics are band-limited to Nyquist with a soft roll-off (no aliasing,
 *    no clicks when the harmonic count changes)
 *  - a three-formant resonator chain gives a vowel-like spectral envelope that
 *    glides between targets, instead of a flat 1/h^k tilt
 *  - a syllable plan with voiced / fricative / silent segments and raised-cosine
 *    amplitude contours, so the temporal modulation looks like real speech
 *  - pink (1/f) background noise mixed at an exact, requested SNR
 *  - everything driven by one seeded PRNG => bit-identical output per seed
 */

export interface SpeechLikeOptions {
  /** Length in seconds. */
  seconds: number;
  sampleRate: number;
  /** PRNG seed; the same seed always yields the same samples. */
  seed?: number;
  /** Voice-to-noise ratio of the mix, in dB. Use e.g. 0 for a hard test. */
  snrDb?: number;
  /** Mean syllables per second (natural speech is ~3-6). */
  syllableRate?: number;
  /** Mean fundamental frequency in Hz (≈120 male, ≈210 female). */
  f0?: number;
  /** Peak the finished buffer is normalised to. */
  peak?: number;
}

interface Vowel {
  /** Formant centre frequencies, Hz. */
  f: [number, number, number];
  /** Formant bandwidths, Hz. */
  bw: [number, number, number];
}

const VOWELS: Vowel[] = [
  { f: [730, 1090, 2440], bw: [80, 90, 120] }, // /a/
  { f: [270, 2290, 3010], bw: [60, 100, 140] }, // /i/
  { f: [300, 870, 2240], bw: [60, 90, 120] }, // /u/
  { f: [530, 1840, 2480], bw: [70, 100, 130] }, // /e/
  { f: [570, 840, 2410], bw: [80, 90, 120] }, // /o/
];

/** Small, fast, seedable PRNG (mulberry32) — uniform in [0, 1). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Direct-form-1 biquad with settable RBJ band-pass coefficients. */
class Biquad {
  private b0 = 1;
  private b1 = 0;
  private b2 = 0;
  private a1 = 0;
  private a2 = 0;
  private x1 = 0;
  private x2 = 0;
  private y1 = 0;
  private y2 = 0;

  setBandpass(freq: number, bandwidth: number, sampleRate: number): void {
    const nyquist = sampleRate / 2;
    const f = Math.min(Math.max(freq, 20), nyquist * 0.98);
    const q = Math.max(f / Math.max(bandwidth, 1), 0.5);
    const w0 = (2 * Math.PI * f) / sampleRate;
    const alpha = Math.sin(w0) / (2 * q);
    const a0 = 1 + alpha;
    this.b0 = alpha / a0;
    this.b1 = 0;
    this.b2 = -alpha / a0;
    this.a1 = (-2 * Math.cos(w0)) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  process(x: number): number {
    const y =
      this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1;
    this.x1 = x;
    this.y2 = this.y1;
    this.y1 = y;
    return y;
  }
}

type SegmentKind = 'voiced' | 'fricative' | 'silence';

interface Segment {
  kind: SegmentKind;
  /** Sample index of the first sample of the segment. */
  start: number;
  /** Length in samples. */
  length: number;
  vowel: Vowel;
  /** Multiplicative f0 offset for this syllable (pitch accent). */
  f0Scale: number;
  gain: number;
}

/** Raised-cosine attack / decay; 1 in the middle, 0 at the edges. */
function syllableEnvelope(phase: number): number {
  const attack = 0.18;
  const release = 0.35;
  if (phase < attack) return 0.5 - 0.5 * Math.cos((Math.PI * phase) / attack);
  if (phase > 1 - release) {
    return 0.5 - 0.5 * Math.cos((Math.PI * (1 - phase)) / release);
  }
  return 1;
}

function planSegments(
  totalSamples: number,
  sampleRate: number,
  syllableRate: number,
  rand: () => number,
): Segment[] {
  const segments: Segment[] = [];
  let cursor = 0;
  while (cursor < totalSamples) {
    const r = rand();
    const kind: SegmentKind = r < 0.68 ? 'voiced' : r < 0.85 ? 'fricative' : 'silence';
    // Syllable duration jitters ±35 % around the mean rate.
    const jitter = 0.65 + 0.7 * rand();
    const seconds = jitter / syllableRate;
    const length = Math.max(1, Math.round(seconds * sampleRate));
    segments.push({
      kind,
      start: cursor,
      length: Math.min(length, totalSamples - cursor),
      vowel: VOWELS[Math.floor(rand() * VOWELS.length)],
      f0Scale: 0.88 + 0.3 * rand(),
      gain: kind === 'fricative' ? 0.35 + 0.25 * rand() : 0.7 + 0.3 * rand(),
    });
    cursor += length;
  }
  return segments;
}

function rms(buffer: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < buffer.length; i += 1) sum += buffer[i] * buffer[i];
  return Math.sqrt(sum / Math.max(buffer.length, 1));
}

export function makeSpeechLikeSignal(options: SpeechLikeOptions): Float32Array {
  const {
    seconds,
    sampleRate,
    seed = 0x2545f491,
    snrDb = 6,
    syllableRate = 4,
    f0: f0Mean = 130,
    peak = 0.9,
  } = options;

  const n = Math.floor(seconds * sampleRate);
  const voice = new Float32Array(n);
  const noise = new Float32Array(n);
  const rand = mulberry32(seed);
  const nyquist = sampleRate / 2;

  // ---- voice ------------------------------------------------------------
  const segments = planSegments(n, sampleRate, syllableRate, rand);
  const formants = [new Biquad(), new Biquad(), new Biquad()];
  const fricative = new Biquad();
  fricative.setBandpass(4800, 3000, sampleRate);

  // Formant targets are interpolated towards over ~25 ms, which is what makes
  // the signal glide between vowels instead of stepping.
  const glide = Math.exp(-1 / (0.025 * sampleRate));
  const current: [number, number, number] = [...segments[0].vowel.f];
  let phase = 0; // glottal phase, radians
  let f0Smoothed = f0Mean;
  let jitterState = 0;

  for (const seg of segments) {
    const target = seg.vowel;
    for (let k = 0; k < seg.length; k += 1) {
      const i = seg.start + k;
      const t = i / sampleRate;
      const env = syllableEnvelope(k / seg.length) * seg.gain;

      for (let b = 0; b < 3; b += 1) {
        current[b] = glide * current[b] + (1 - glide) * target.f[b];
        formants[b].setBandpass(current[b], target.bw[b], sampleRate);
      }

      let sample = 0;
      if (seg.kind === 'voiced') {
        // Declination across the utterance + intonation + micro-jitter.
        jitterState = 0.995 * jitterState + 0.005 * (rand() * 2 - 1);
        const declination = 1 - 0.12 * (t / Math.max(seconds, 1e-6));
        const intonation = 1 + 0.05 * Math.sin(2 * Math.PI * 0.6 * t);
        const f0Target = f0Mean * seg.f0Scale * declination * intonation * (1 + 0.03 * jitterState);
        f0Smoothed += (f0Target - f0Smoothed) * 0.002;
        phase += (2 * Math.PI * f0Smoothed) / sampleRate;
        if (phase > 2 * Math.PI) phase -= 2 * Math.PI;

        // Band-limited glottal pulse: harmonics only below Nyquist, faded out
        // near it so the count can change without a click.
        let source = 0;
        const maxH = Math.floor(nyquist / f0Smoothed);
        for (let h = 1; h <= maxH; h += 1) {
          const fade = Math.min(1, ((nyquist - h * f0Smoothed) / (0.1 * nyquist)) ** 2);
          source += (fade * Math.sin(h * phase)) / h;
        }
        // A little aspiration keeps voiced frames from sounding synthetic.
        source += (rand() * 2 - 1) * 0.06;
        sample =
          formants[0].process(source) * 1.0 +
          formants[1].process(source) * 0.55 +
          formants[2].process(source) * 0.3;
      } else if (seg.kind === 'fricative') {
        sample = fricative.process(rand() * 2 - 1) * 1.4;
      }

      voice[i] = sample * env;
    }
  }

  // ---- background: pink noise (Paul Kellet's economy filter) -------------
  let b0 = 0;
  let b1 = 0;
  let b2 = 0;
  for (let i = 0; i < n; i += 1) {
    const white = rand() * 2 - 1;
    b0 = 0.99765 * b0 + white * 0.099046;
    b1 = 0.963 * b1 + white * 0.2965164;
    b2 = 0.57 * b2 + white * 1.0526913;
    noise[i] = (b0 + b1 + b2 + white * 0.1848) * 0.2;
  }

  // ---- mix at the requested SNR and normalise ---------------------------
  const voiceRms = rms(voice) || 1e-9;
  const noiseRms = rms(noise) || 1e-9;
  const noiseGain = (voiceRms / noiseRms) * 10 ** (-snrDb / 20);

  let maxAbs = 0;
  for (let i = 0; i < n; i += 1) {
    const mixed = voice[i] + noise[i] * noiseGain;
    voice[i] = mixed;
    const a = Math.abs(mixed);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = peak / (maxAbs || 1);
  for (let i = 0; i < n; i += 1) voice[i] *= scale;

  return voice;
}

/** Drop-in replacement for the original helper. */
export function makeNoisyDemo(seconds: number, sampleRate: number): Float32Array {
  return makeSpeechLikeSignal({ seconds, sampleRate, snrDb: 6 });
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
