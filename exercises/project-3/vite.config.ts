import { resolve } from 'path';
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        // "head-tilt": resolve(__dirname, 'head-tilt/index.html'),
      },
    },
  },
});
