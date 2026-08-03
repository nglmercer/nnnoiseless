// AudioWorklet processor for the streaming denoiser.
//
// This file is never loaded on its own. `main.ts` concatenates it with the
// `no-modules` wasm-bindgen glue and registers the result via a blob URL,
// because an AudioWorkletGlobalScope has no `fetch` and so cannot load a wasm
// module by URL itself. The compiled `WebAssembly.Module` is handed over from
// the main thread instead, and instantiated synchronously here.
//
// The `wasm_bindgen` binding below comes from that prepended glue.

/* global wasm_bindgen, registerProcessor, AudioWorkletProcessor, sampleRate */

type WorkletWasm = typeof wasm_bindgen & {
  initSync(module: { module: WebAssembly.Module }): InitOutput;
};

const wasm = wasm_bindgen as WorkletWasm;

class NnnoiselessProcessor extends AudioWorkletProcessor {
  private denoiser: wasm_bindgen.Denoiser | null = null;
  private bypass = false;
  private blockCount = 0;

  constructor() {
    super();

    this.port.onmessage = (event: MessageEvent<any>) => {
      const msg = event.data;
      try {
        switch (msg.type) {
          case 'init': {
            wasm.initSync({ module: msg.module });
            this.denoiser = wasm.Denoiser.withSettings(
              msg.attenuationDb ?? 0,
              msg.vadThreshold ?? 0,
              0, // lookahead adds latency; keep the live path as tight as possible
            );
            this.port.postMessage({
              type: 'ready',
              isa: this.denoiser.activeIsa,
              latencySamples: this.denoiser.latencySamples,
              // `sampleRate` is a global in the worklet scope.
              sampleRate,
            });
            break;
          }
          case 'bypass':
            this.bypass = Boolean(msg.value);
            break;
          case 'attenuation':
            this.denoiser?.setAttenuationLimitDb(msg.value);
            break;
          case 'vadThreshold':
            this.denoiser?.setVadThreshold(msg.value);
            break;
          case 'reset':
            this.denoiser?.reset();
            break;
          default:
            break;
        }
      } catch (err) {
        this.port.postMessage({ type: 'error', message: String(err) });
      }
    };
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || output.length === 0) return true;

    const outChannel = output[0];
    const inChannel = input && input.length > 0 ? input[0] : null;

    // No input connected yet: emit silence but keep the node alive.
    if (!inChannel) {
      outChannel.fill(0);
      return true;
    }

    if (!this.denoiser || this.bypass) {
      outChannel.set(inChannel);
      return true;
    }

    // `push` buffers internally: it accepts the worklet's 128-sample blocks even
    // though the algorithm works in 480-sample frames, and returns at most as
    // many samples as it was given. It returns fewer only while filling its
    // delay line, which is why the tail is zeroed.
    const denoised = this.denoiser.push(inChannel);
    if (denoised.length < outChannel.length) {
      outChannel.fill(0);
    }
    outChannel.set(denoised.subarray(0, outChannel.length));

    // Copy to any further output channels so stereo destinations stay centred.
    for (let ch = 1; ch < output.length; ch += 1) {
      output[ch].set(outChannel);
    }

    // Reporting every block would flood the message port for no benefit; ~11ms
    // is plenty for a meter.
    this.blockCount += 1;
    if ((this.blockCount & 3) === 0) {
      this.port.postMessage({ type: 'vad', value: this.denoiser.vad });
    }
    return true;
  }
}

registerProcessor('nnnoiseless-denoiser', NnnoiselessProcessor);
