import { defineConfig } from 'vite'

// Vanilla Three.js viewer. `.glsl` files are imported as raw strings below.
export default defineConfig({
  assetsInclude: ['**/*.glsl'],
  server: { open: true },
})
