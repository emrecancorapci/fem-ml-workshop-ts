import { resolve } from 'path';
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        image: resolve(__dirname, 'image/index.html'),
        faceDetection: resolve(__dirname, 'face-detection/index.html'),
        webcam: resolve(__dirname, 'webcam/index.html'),
      },
    },
  },
});
