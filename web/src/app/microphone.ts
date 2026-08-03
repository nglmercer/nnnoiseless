import workletSource from '../denoise-worklet.ts?raw';
import workletGlue from '../pkg-worklet/nnnoiseless.js?raw';
import workletWasmUrl from '../pkg-worklet/nnnoiseless_bg.wasm?url';
import { settings } from './settings';
import { state } from './state';
import { ui } from './ui';
import { errorMessage } from './errors';

const SAMPLE_RATE = 48_000;

// Some browsers expose a smaller global inside AudioWorkletGlobalScope and do
// not provide TextDecoder there. wasm-bindgen's no-modules glue uses it while
// the worklet module is being evaluated, so provide a small UTF-8 fallback
// before concatenating that glue with the processor source.
const WORKLET_TEXT_DECODER = `
const TextDecoder = globalThis.TextDecoder || class TextDecoder {
  decode(input = new Uint8Array()) {
    let text = '';
    for (let i = 0; i < input.length;) {
      const first = input[i++];
      let codePoint;
      let length;

      if (first < 0x80) {
        codePoint = first;
      } else if ((first & 0xe0) === 0xc0) {
        codePoint = first & 0x1f;
        length = 1;
      } else if ((first & 0xf0) === 0xe0) {
        codePoint = first & 0x0f;
        length = 2;
      } else if ((first & 0xf8) === 0xf0) {
        codePoint = first & 0x07;
        length = 3;
      } else {
        text += '\\ufffd';
        continue;
      }

      if (i + length > input.length) {
        text += '\\ufffd';
        break;
      }
      let valid = true;
      for (let j = 0; j < length; j += 1) {
        const next = input[i++];
        if ((next & 0xc0) !== 0x80) valid = false;
        codePoint = (codePoint << 6) | (next & 0x3f);
      }

      if (
        !valid ||
        (length === 1 && codePoint < 0x80) ||
        (length === 2 && codePoint < 0x800) ||
        (length === 3 && codePoint < 0x10000) ||
        codePoint > 0x10ffff ||
        (codePoint >= 0xd800 && codePoint <= 0xdfff)
      ) {
        text += '\\ufffd';
      } else {
        text += String.fromCodePoint(codePoint);
      }
    }
    return text;
  }
};
`;

async function registerWorklet(ctx: AudioContext): Promise<void> {
  const blob = new Blob([WORKLET_TEXT_DECODER, workletGlue, '\n', workletSource], {
    type: 'application/javascript',
  });
  const url = URL.createObjectURL(blob);
  try {
    await ctx.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function startMic(): Promise<void> {
  ui.micError.hidden = true;
  ui.mic.disabled = true;
  ui.mic.textContent = 'Starting…';

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    await ctx.resume();
    await registerWorklet(ctx);

    const node = new AudioWorkletNode(ctx, 'nnnoiseless-denoiser', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });

    node.port.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'vad') {
        const pct = Math.max(0, Math.min(1, msg.value));
        ui.vadMeter.style.width = `${(pct * 100).toFixed(1)}%`;
        ui.vadMeter.parentElement?.setAttribute('aria-valuenow', pct.toFixed(2));
        ui.vadValue.textContent = pct.toFixed(2);
      } else if (msg.type === 'ready') {
        ui.mic.disabled = false;
        ui.mic.textContent = 'Stop microphone';
        ui.mic.classList.add('danger');
      } else if (msg.type === 'error') {
        showMicError(msg.message);
      }
    };

    const wasmBytes = await (await fetch(workletWasmUrl)).arrayBuffer();
    const module = await WebAssembly.compile(wasmBytes);
    const current = settings();
    node.port.postMessage({
      type: 'init',
      module,
      attenuationDb: current.attenuationDb,
      vadThreshold: current.vadThreshold,
    });

    const source = ctx.createMediaStreamSource(stream);
    source.connect(node);
    node.connect(ctx.destination);
    node.port.postMessage({ type: 'bypass', value: ui.bypass.checked });
    state.mic = { ctx, stream, node, source };
  } catch (err: unknown) {
    showMicError(errorMessage(err));
    ui.mic.disabled = false;
    ui.mic.textContent = 'Start microphone';
  }
}

export function stopMic(): void {
  if (!state.mic) return;
  const { ctx, stream, node, source } = state.mic;
  source.disconnect();
  node.disconnect();
  stream.getTracks().forEach((track) => track.stop());
  ctx.close();
  state.mic = null;

  ui.mic.textContent = 'Start microphone';
  ui.mic.classList.remove('danger');
  ui.vadMeter.style.width = '0%';
  ui.vadMeter.parentElement?.setAttribute('aria-valuenow', '0');
  ui.vadValue.textContent = '0.00';
}

function showMicError(message: string): void {
  ui.micError.textContent = `Microphone unavailable: ${message}`;
  ui.micError.hidden = false;
}
