import { defineConfig } from 'vite';

export default defineConfig({
  // VITE_* vars are inlined at build time. Set VITE_HF_TOKEN in Vercel project
  // settings (Environment Variables) or in a local .env file for dev builds.
  // The token is forwarded to BrutScript's model blocks for HuggingFace inference.
  envPrefix: 'VITE_',
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
  assetsInclude: ['**/*.wgsl'],
});
