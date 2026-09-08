import type { Settings } from './types';
import { ui } from './ui';

export function settings(): Settings {
  return {
    backend: ui.backend.value === 'hush' ? 'hush' : 'rnnoise',
    attenuationDb: Number(ui.attenuation.value),
    vadThreshold: Number(ui.vad.value),
    lookahead: Number(ui.lookahead.value),
  };
}

export function renderSettings(): void {
  const current = settings();
  ui.attenuationOut.textContent =
    current.attenuationDb > 0 ? `${current.attenuationDb} dB` : 'off';
  if (current.backend === 'hush') {
    ui.vadOut.textContent = 'Hush SNR';
    ui.lookaheadOut.textContent = 'causal';
    ui.vad.disabled = true;
    ui.lookahead.disabled = true;
  } else {
    ui.vadOut.textContent = current.vadThreshold > 0 ? current.vadThreshold.toFixed(2) : 'off';
    ui.lookaheadOut.textContent = `${current.lookahead} frame${
      current.lookahead === 1 ? '' : 's'
    }`;
    ui.vad.disabled = false;
    ui.lookahead.disabled = false;
  }
}
