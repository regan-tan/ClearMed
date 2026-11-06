import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'CLEARMED_')
  const target = env.CLEARMED_BACKEND_URL || 'http://127.0.0.1:8000'
  const backendUrl = new URL(target)
  if (
    !['http:', 'https:'].includes(backendUrl.protocol) ||
    backendUrl.pathname !== '/' || backendUrl.search || backendUrl.hash ||
    backendUrl.username || backendUrl.password
  ) {
    throw new Error('CLEARMED_BACKEND_URL must be an HTTP(S) origin, e.g. http://127.0.0.1:8000.')
  }
  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        '/api': target,
      },
    },
  }
})
