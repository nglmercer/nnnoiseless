import { defineConfig } from 'vite';

export default defineConfig({
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
