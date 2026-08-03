interface AudioWorkletProcessor {
  readonly port: MessagePort;
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters?: Record<string, Float32Array>,
  ): boolean;
}

declare const AudioWorkletProcessor: {
  prototype: AudioWorkletProcessor;
  new (): AudioWorkletProcessor;
};

declare function registerProcessor(
  name: string,
  processor: new () => AudioWorkletProcessor,
): void;

declare const sampleRate: number;
