/// <reference types="vitest/config" />
import { defineConfig } from "vite";

// The Python server (`uv run nvs serve --dev`). `changeOrigin` rewrites the Host header to the target's:
// the server only accepts its own host names (DNS-rebinding guard); the browser's Origin
// (http://localhost:5173) is allowed by `--dev`.
const API = "http://127.0.0.1:8765";

export default defineConfig({
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: API, changeOrigin: true, ws: true },
    },
  },
  // Only index.html is built; developer pages under dev/ and fixtures/ are served by the dev server only.
  build: { outDir: "dist", emptyOutDir: true },
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
