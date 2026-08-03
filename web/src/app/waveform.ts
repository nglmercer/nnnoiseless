export function drawWave(
  canvas: HTMLCanvasElement,
  samples: Float32Array,
  which: 'before' | 'after',
): void {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 600;
  const cssHeight = Number(canvas.getAttribute('height'));
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);

  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const styles = getComputedStyle(document.documentElement);
  const stroke = styles.getPropertyValue(
    which === 'before' ? '--wave-before' : '--wave-after',
  );
  const mid = cssHeight / 2;

  // One vertical bar per pixel column, spanning that column's min and max. This
  // is the honest way to draw a waveform that has far more samples than pixels.
  const perPixel = Math.max(1, Math.floor(samples.length / cssWidth));
  ctx.strokeStyle = stroke.trim() || '#888';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < cssWidth; x += 1) {
    const start = x * perPixel;
    const end = Math.min(samples.length, start + perPixel);
    if (start >= samples.length) break;
    let min = 1;
    let max = -1;
    for (let i = start; i < end; i += 1) {
      const v = samples[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x + 0.5, mid - max * mid * 0.95);
    ctx.lineTo(x + 0.5, mid - min * mid * 0.95);
  }
  ctx.stroke();

  // Centre line.
  ctx.strokeStyle = styles.getPropertyValue('--grid').trim() || '#ccc';
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(cssWidth, mid);
  ctx.stroke();
}
