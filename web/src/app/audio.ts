import { state } from './state';
import { ui } from './ui';

export function audioContext(): AudioContext {
  if (!state.audioCtx || state.audioCtx.state === 'closed') {
    state.audioCtx = new AudioContext();
  }
  return state.audioCtx;
}

export function stopPlayback(): void {
  if (state.playing) {
    try {
      state.playing.stop();
    } catch {
      // Already stopped; nothing to do.
    }
    state.playing = null;
  }
  ui.playBefore.textContent = '▶ Play';
  ui.playAfter.textContent = '▶ Play';
}
