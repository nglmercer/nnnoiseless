import { audioContext, stopPlayback } from './audio';
import { state } from './state';
import { ui } from './ui';

export async function play(
  samples: Float32Array,
  button: HTMLButtonElement,
): Promise<void> {
  const wasPlaying = state.playing;
  stopPlayback();
  if (wasPlaying && button.dataset.active === 'true') {
    button.dataset.active = 'false';
    return;
  }
  const clip = state.clip;
  if (!clip) return;

  const ctx = audioContext();
  await ctx.resume();

  const buffer = ctx.createBuffer(1, samples.length, clip.sampleRate);
  buffer.copyToChannel(new Float32Array(samples), 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.onended = () => {
    button.dataset.active = 'false';
    stopPlayback();
  };
  source.start();

  state.playing = source;
  ui.playBefore.dataset.active = 'false';
  ui.playAfter.dataset.active = 'false';
  button.dataset.active = 'true';
  button.textContent = '■ Stop';
}
