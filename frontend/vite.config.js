import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0', // Allow external access on EC2
    // No proxy - frontend calls backend directly with full URL
  },
  // Environment variable prefix
  envPrefix: 'VITE_',
  // Force cache busting in development
  optimizeDeps: {
    force: true
  }
})
