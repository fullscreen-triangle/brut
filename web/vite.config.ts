import { defineConfig } from 'vite';
import { resolve } from 'path';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
  // VITE_* vars are inlined at build time. Set VITE_HF_TOKEN in Vercel project
  // settings (Environment Variables) or in a local .env file for dev builds.
  // The token is forwarded to BrutScript's model blocks for HuggingFace inference.
  envPrefix: 'VITE_',
  // basic-ssl gives the dev server a self-signed HTTPS cert. Camera + mic
  // (getUserMedia) require a secure context on any non-localhost address, so a
  // phone on the LAN can only use them over https://. Accept the one-time cert
  // warning on the phone the first time.
  plugins: [basicSsl()],
  server: {
    // Bind to all interfaces so a phone on the same Wi-Fi / tailnet can reach it.
    host: true,
    port: 5173,
  },
  build: {
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      input: {
        // Desktop observatory + the mobile-targeted training instance.
        main: resolve(__dirname, 'index.html'),
        train: resolve(__dirname, 'train.html'),
      },
    },
  },
  assetsInclude: ['**/*.wgsl'],
});
