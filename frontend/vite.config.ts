import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/health': 'http://127.0.0.1:5000',
      '/analyze': 'http://127.0.0.1:5000',
      '/results': 'http://127.0.0.1:5000',
      '/outputs': 'http://127.0.0.1:5000',
    },
  },
});
