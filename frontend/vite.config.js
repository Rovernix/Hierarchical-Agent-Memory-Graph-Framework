import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// VITE_HAMGF_API_URL belongs to the original /v1 API; Studio uses its own service.
const target = process.env.HAMGF_STUDIO_API_URL || 'http://127.0.0.1:8765';
export default defineConfig({
  plugins: [react()],
  build: { target: 'es2022' },
  server: {
    host: '127.0.0.1', port: 5173,
    proxy: {
      '/api/studio': {
        target, changeOrigin: true,
        configure(proxy) {
          proxy.on('proxyReq', request => request.setHeader('Origin', new URL(target).origin));
        },
      },
    },
  },
});
