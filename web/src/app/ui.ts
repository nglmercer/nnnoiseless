export const el = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing UI element: #${id}`);
  return element as T;
};

export const ui = {
  status: el<HTMLElement>('badge-status'),
  version: el<HTMLElement>('badge-version'),
  isa: el<HTMLElement>('badge-isa'),
  attenuation: el<HTMLInputElement>('ctl-attenuation'),
  attenuationOut: el<HTMLOutputElement>('out-attenuation'),
  vad: el<HTMLInputElement>('ctl-vad'),
  vadOut: el<HTMLOutputElement>('out-vad'),
  lookahead: el<HTMLInputElement>('ctl-lookahead'),
  lookaheadOut: el<HTMLOutputElement>('out-lookahead'),
  demo: el<HTMLButtonElement>('btn-demo'),
  file: el<HTMLInputElement>('file-input'),
  clipInfo: el<HTMLElement>('clip-info'),
  waveforms: el<HTMLElement>('waveforms'),
  waveBefore: el<HTMLCanvasElement>('wave-before'),
  waveAfter: el<HTMLCanvasElement>('wave-after'),
  playBefore: el<HTMLButtonElement>('play-before'),
  playAfter: el<HTMLButtonElement>('play-after'),
  processTime: el<HTMLElement>('process-time'),
  mic: el<HTMLButtonElement>('btn-mic'),
  bypass: el<HTMLInputElement>('ctl-bypass'),
  vadMeter: el<HTMLElement>('vad-meter'),
  vadValue: el<HTMLOutputElement>('vad-value'),
  micError: el<HTMLElement>('mic-error'),
};
