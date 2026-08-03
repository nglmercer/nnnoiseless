// Browser demo entrypoint. Feature-specific behavior lives under `app/`.

import init, { activeIsa, version } from './pkg/nnnoiseless.js';
import './style.css';

import { loadFile, makeNoisyDemo, processClip, setClip } from './app/clip';
import { stopPlayback } from './app/audio';
import { errorMessage } from './app/errors';
import { play } from './app/playback';
import { startMic, stopMic } from './app/microphone';
import { renderSettings, settings } from './app/settings';
import { state } from './app/state';
import { ui } from './app/ui';
import { drawWave } from './app/waveform';

const DEMO_SECONDS = 4;
const SAMPLE_RATE = 48_000;

async function main(): Promise<void> {
  renderSettings();

  await init();
  ui.status.textContent = 'ready';
  ui.status.classList.add('ok');
  ui.version.textContent = version();
  ui.isa.textContent = activeIsa();

  ui.backend.addEventListener('change', () => {
    renderSettings();
    processClip();
    if (state.mic) {
      stopMic();
      ui.micError.textContent = 'Microphone stopped; restart it to change backend.';
      ui.micError.hidden = false;
    }
  });

  ui.hushModel.addEventListener('change', async () => {
    const file = ui.hushModel.files?.[0];
    if (!file) return;
    ui.hushModelInfo.textContent = 'loading…';
    try {
      state.hushModel = new Uint8Array(await file.arrayBuffer());
      ui.hushModelInfo.textContent = `Hush ${(state.hushModel.byteLength / 1_000_000).toFixed(1)} MB`;
      processClip();
    } catch (err: unknown) {
      state.hushModel = null;
      ui.hushModelInfo.textContent = 'load failed';
      ui.clipInfo.textContent = `could not load Hush model: ${errorMessage(err)}`;
    }
  });

  for (const control of [ui.attenuation, ui.vad, ui.lookahead]) {
    control.addEventListener('input', () => {
      renderSettings();
      processClip();
      if (state.mic) {
        const current = settings();
        state.mic.node.port.postMessage({
          type: 'attenuation',
          value: current.attenuationDb,
        });
        state.mic.node.port.postMessage({
          type: 'vadThreshold',
          value: current.vadThreshold,
        });
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

  ui.file.addEventListener('change', async (event: Event) => {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    stopPlayback();
    ui.clipInfo.textContent = 'decoding…';
    try {
      await loadFile(file);
    } catch (err: unknown) {
      ui.clipInfo.textContent = `could not decode: ${errorMessage(err)}`;
    }
  });

  ui.playBefore.addEventListener('click', () => {
    if (state.clip) void play(state.clip.samples, ui.playBefore);
  });
  ui.playAfter.addEventListener('click', () => {
    if (state.denoised) void play(state.denoised, ui.playAfter);
  });

  ui.mic.addEventListener('click', () => {
    if (state.mic) stopMic();
    else void startMic();
  });
  ui.bypass.addEventListener('change', () => {
    state.mic?.node.port.postMessage({ type: 'bypass', value: ui.bypass.checked });
  });

  window.addEventListener('resize', () => {
    if (state.clip && state.denoised) {
      drawWave(ui.waveBefore, state.clip.samples, 'before');
      drawWave(ui.waveAfter, state.denoised, 'after');
    }
  });
}

main().catch((err: unknown) => {
  ui.status.textContent = `failed: ${errorMessage(err)}`;
  ui.status.classList.add('bad');
});
