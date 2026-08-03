export type Clip = {
  samples: Float32Array;
  sampleRate: number;
  name: string;
};

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
};

export type Settings = {
  attenuationDb: number;
  vadThreshold: number;
  lookahead: number;
};
