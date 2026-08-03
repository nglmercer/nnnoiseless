// Smoke test for the WebAssembly build.
//
// Verifies that the compiled module actually denoises, rather than merely
// linking. Run with `npm test` after `npm run wasm`.
//
// This exercises the same entry points the page uses: `denoiseBuffer` for a
// whole clip, and the streaming `Denoiser` fed in the small blocks an
// AudioWorklet would deliver.

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import init, {
  Denoiser,
  HushDenoiser,
  activeIsa,
  denoiseBuffer,
  version,
} from './src/pkg/nnnoiseless.js';

const wasmPath = fileURLToPath(new URL('./src/pkg/nnnoiseless_bg.wasm', import.meta.url));
await init({ module_or_path: await readFile(wasmPath) });

let failures = 0;
function check(name, condition, detail = '') {
  const mark = condition ? 'ok  ' : 'FAIL';
  console.log(`${mark} ${name}${detail ? ` — ${detail}` : ''}`);
  if (!condition) failures += 1;
}

const SAMPLE_RATE = 48_000;

/** Speech-like harmonics with a syllable envelope, plus coloured noise. */
function makeSignal(seconds, { withSpeech = true, withNoise = true } = {}) {
  const n = Math.floor(seconds * SAMPLE_RATE);
  const speech = new Float32Array(n);
  const noise = new Float32Array(n);
  let seed = 0x2545f491;
  let lp = 0;

  for (let i = 0; i < n; i += 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const white = (seed >>> 16) / 32768 - 1;
    lp = 0.85 * lp + 0.15 * white;

    const t = i / SAMPLE_RATE;
    const syl = Math.sin(2 * Math.PI * 3.2 * t);
    const env = syl > 0 ? Math.pow(syl, 0.6) : 0;
    const f0 = 150 + 25 * Math.sin(2 * Math.PI * 0.7 * t);
    let voice = 0;
    for (let h = 1; h <= 14; h += 1) {
      voice += Math.sin(2 * Math.PI * f0 * h * t) / Math.pow(h, 1.15);
    }
    speech[i] = withSpeech ? voice * env * 0.16 : 0;
    noise[i] = withNoise ? (white * 0.5 + lp * 2.0) * 0.05 : 0;
  }
  const mixed = new Float32Array(n);
  for (let i = 0; i < n; i += 1) mixed[i] = speech[i] + noise[i];
  return { mixed, speech, noise };
}

const rms = (a, from = 0, to = a.length) => {
  let sum = 0;
  for (let i = from; i < to; i += 1) sum += a[i] * a[i];
  return Math.sqrt(sum / Math.max(1, to - from));
};
const db = (x) => 20 * Math.log10(Math.max(x, 1e-12));

console.log(`nnnoiseless ${version()} — kernels: ${activeIsa()}\n`);
check('version is reported', /^\d+\.\d+\.\d+$/.test(version()), version());
check('kernels are reported', activeIsa().length > 0, activeIsa());

// --- whole-clip path -------------------------------------------------------

const { mixed, speech } = makeSignal(4);
const started = Date.now();
const denoised = denoiseBuffer(mixed, SAMPLE_RATE, 0, 0, 0);
const elapsed = Date.now() - started;

check('denoiseBuffer preserves length', denoised.length === mixed.length,
  `${denoised.length} vs ${mixed.length}`);
check('output is finite', denoised.every(Number.isFinite));

// Measure noise removal in the gaps between "syllables", where the clean
// reference is silent, and speech retention where it is loud.
const FRAME = 480;
const speechFloor = rms(speech) * 0.15;
let nIn = 0, nOut = 0, nCount = 0, sIn = 0, sOut = 0, sCount = 0;
for (let start = 0; start + FRAME <= mixed.length; start += FRAME) {
  const clean = rms(speech, start, start + FRAME);
  if (clean < speechFloor * 0.2) {
    nIn += rms(mixed, start, start + FRAME);
    nOut += rms(denoised, start, start + FRAME);
    nCount += 1;
  } else if (clean > speechFloor) {
    sIn += clean;
    sOut += rms(denoised, start, start + FRAME);
    sCount += 1;
  }
}
const noiseReduction = db(nIn / nCount) - db(nOut / nCount);
const speechLoss = db(sIn / sCount) - db(sOut / sCount);

console.log(`     noise reduction: ${noiseReduction.toFixed(1)} dB`);
console.log(`     speech loss:     ${speechLoss.toFixed(1)} dB`);
console.log(`     speed:           ${(4 / (elapsed / 1000)).toFixed(0)}x realtime`);
check('removes a useful amount of noise', noiseReduction > 8, `${noiseReduction.toFixed(1)} dB`);
check('keeps the speech', speechLoss < 4, `${speechLoss.toFixed(1)} dB`);

// --- attenuation limit -----------------------------------------------------

const capped = denoiseBuffer(mixed, SAMPLE_RATE, 6, 0, 0);
let cIn = 0, cOut = 0, cCount = 0;
for (let start = 0; start + FRAME <= mixed.length; start += FRAME) {
  if (rms(speech, start, start + FRAME) < speechFloor * 0.2) {
    cIn += rms(mixed, start, start + FRAME);
    cOut += rms(capped, start, start + FRAME);
    cCount += 1;
  }
}
const cappedReduction = db(cIn / cCount) - db(cOut / cCount);
console.log(`     with 6 dB cap:   ${cappedReduction.toFixed(1)} dB reduction`);
check('attenuation limit leaves a noise floor', cappedReduction < noiseReduction,
  `${cappedReduction.toFixed(1)} < ${noiseReduction.toFixed(1)} dB`);

// --- resampling path -------------------------------------------------------

const at44k = denoiseBuffer(mixed, 44_100, 0, 0, 0);
check('resampled input keeps its length', at44k.length === mixed.length);
check('resampled output is finite', at44k.every(Number.isFinite));

// --- streaming path (what the AudioWorklet does) ---------------------------

const denoiser = new Denoiser();
check('frame size is 480', denoiser.frameSize === 480, String(denoiser.frameSize));
check('latency is reported', denoiser.latencySamples === 480, String(denoiser.latencySamples));

const BLOCK = 128; // what an AudioWorklet hands over
let produced = 0;
let sawVad = false;
for (let start = 0; start + BLOCK <= mixed.length; start += BLOCK) {
  const out = denoiser.push(mixed.subarray(start, start + BLOCK));
  produced += out.length;
  if (out.length > BLOCK) {
    check('streaming never returns more than it was given', false, `${out.length} > ${BLOCK}`);
    break;
  }
  if (denoiser.vad > 0) sawVad = true;
}
const consumed = Math.floor(mixed.length / BLOCK) * BLOCK;
check('streaming keeps up with its input', produced >= consumed - 3 * 480,
  `produced ${produced} of ${consumed}`);
check('streaming reports voice activity', sawVad);

denoiser.setAttenuationLimitDb(12);
const afterChange = denoiser.push(mixed.subarray(0, BLOCK));
check('settings can change mid-stream', afterChange.every(Number.isFinite));
denoiser.reset();
denoiser.free();

// --- Hush backend ----------------------------------------------------------

const hushModelPath = process.env.HUSH_MODEL;
if (hushModelPath) {
  const hushModel = new Uint8Array(await readFile(hushModelPath));
  const loadStarted = performance.now();
  const hush = HushDenoiser.fromModelBytes(hushModel, 0);
  const loadMs = performance.now() - loadStarted;
  check('Hush reports its native sample rate', hush.sampleRate === 16_000, String(hush.sampleRate));
  check('Hush reports its frame size', hush.frameSize === 160, String(hush.frameSize));
  check('Hush reports finite latency', Number.isFinite(hush.latencySamples));

  const hushFrame = new Float32Array(hush.frameSize);
  let phase = 0;
  const hushFrames = 2_000; // 20 seconds at 16 kHz
  let hushProduced = 0;
  let hushOutputFinite = true;
  let hushLsnrFinite = true;
  const processStarted = performance.now();
  for (let frame = 0; frame < hushFrames; frame += 1) {
    for (let i = 0; i < hushFrame.length; i += 1) {
      phase += (2 * Math.PI * 180) / 16_000;
      hushFrame[i] = Math.sin(phase) * 0.1;
    }
    const out = hush.push(hushFrame);
    hushProduced += out.length;
    hushOutputFinite &&= out.every(Number.isFinite);
    hushLsnrFinite &&= Number.isFinite(hush.lsnrDb);
  }
  const processMs = performance.now() - processStarted;
  const audioSeconds = (hushFrames * hush.frameSize) / hush.sampleRate;
  console.log(
    `     Hush load: ${loadMs.toFixed(0)} ms; process: ${processMs.toFixed(0)} ms ` +
      `(${((processMs * 1000) / hushFrames).toFixed(1)} us/frame, ` +
      `${(audioSeconds / (processMs / 1000)).toFixed(2)}x realtime)`,
  );
  check('Hush output is finite', hushOutputFinite);
  check('Hush L-SNR is finite', hushLsnrFinite);
  check('Hush streaming produces delayed output', hushProduced > 0, `${hushProduced} samples`);
  hush.reset();
  hush.free();
} else {
  console.log('     Hush checks skipped — set HUSH_MODEL to the released ONNX bundle');
}

// --- edge cases ------------------------------------------------------------

check('empty input is handled', denoiseBuffer(new Float32Array(0), SAMPLE_RATE, 0, 0, 0).length === 0);
const silence = denoiseBuffer(new Float32Array(48_000), SAMPLE_RATE, 0, 0, 0);
check('silence stays silent', silence.every((x) => x === 0));

console.log(`\n${failures === 0 ? 'all checks passed' : `${failures} check(s) failed`}`);
process.exit(failures === 0 ? 0 : 1);
