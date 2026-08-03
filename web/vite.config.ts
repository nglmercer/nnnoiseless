import { defineConfig, transformWithEsbuild, type Plugin } from 'vite';

const audioWorkletRaw: Plugin = {
  name: 'nnnoiseless-audio-worklet-raw',
  enforce: 'post',
  async transform(code, id) {
    if (!id.endsWith('denoise-worklet.ts?raw')) return;

    // Vite's built-in raw plugin has already turned the source into an
    // `export default "..."` module by this point. Decode that payload,
    // transpile the TypeScript, and expose the resulting JavaScript string.
    const rawSource = JSON.parse(code.slice('export default '.length).replace(/;\s*$/, '')) as string;
    const transformed = await transformWithEsbuild(rawSource, id, {
      loader: 'ts',
      target: 'es2022',
      format: 'iife',
      sourcemap: false,
    });
    return {
      code: `export default ${JSON.stringify(transformed.code)};`,
      map: null,
    };
  },
};

export default defineConfig({
  plugins: [audioWorkletRaw],
  // Relative base so the built site works from any subdirectory, including
  // `file://` previews and GitHub Pages.
  base: './',
  build: {
    target: 'es2022',
    // The wasm module must stay a separate file: the AudioWorklet compiles it
    // from a URL, so it cannot be inlined as a data URI.
    assetsInlineLimit: 0,
    outDir: 'dist',
  },
  server: {
    port: 5173,
  },
  // wasm-pack writes its output into src/, and Vite should not try to optimise
  // the generated glue as a dependency.
  optimizeDeps: {
    exclude: ['./src/pkg/nnnoiseless.js'],
  },
});
