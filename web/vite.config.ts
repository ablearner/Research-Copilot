import { defineConfig, Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'

const MIME: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf',
}

function serveStorageFiles(): Plugin {
  return {
    name: 'serve-storage-files',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.startsWith('/files/')) return next()
        const filePath = req.url.replace(/^\/files\//, '')
        const absPath = path.resolve(__dirname, '..', '.data/storage', filePath)
        if (fs.existsSync(absPath)) {
          const ext = path.extname(absPath).toLowerCase()
          res.setHeader('Content-Type', MIME[ext] || 'application/octet-stream')
          fs.createReadStream(absPath).pipe(res)
        } else {
          res.statusCode = 404
          res.end('Not found')
        }
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), serveStorageFiles()],
  server: {
    port: 3000,
  },
})
