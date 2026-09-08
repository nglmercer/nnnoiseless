export type Clip = {
  samples: Float32Array;
  sampleRate: number;
  name: string;
};

export type Backend = 'rnnoise' | 'hush';

export type MicState = {
  ctx: AudioContext;
  stream: MediaStream;
  node: AudioWorkletNode;
  source: MediaStreamAudioSourceNode;
};

export type AppState = {
  clip: Clip | null;
  denoised: Float32Array | null;
  audioCtx: AudioContext | null;
  playing: AudioBufferSourceNode | null;
  mic: MicState | null;
  hushModel: Uint8Array | null;
};

export type Settings = {
  backend: Backend;
  attenuationDb: number;
  vadThreshold: number;
  lookahead: number;
};
