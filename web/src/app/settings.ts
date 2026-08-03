import type { Settings } from './types';
import { ui } from './ui';

export function settings(): Settings {
  return {
    attenuationDb: Number(ui.attenuation.value),
    vadThreshold: Number(ui.vad.value),
    lookahead: Number(ui.lookahead.value),
  };
}

export function renderSettings(): void {
  const current = settings();
  ui.attenuationOut.textContent =
    current.attenuationDb > 0 ? `${current.attenuationDb} dB` : 'off';
  ui.vadOut.textContent = current.vadThreshold > 0 ? current.vadThreshold.toFixed(2) : 'off';
  ui.lookaheadOut.textContent = `${current.lookahead} frame${
    current.lookahead === 1 ? '' : 's'
  }`;
}
