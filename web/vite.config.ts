/// <reference types="vitest/config" />
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

// Dev-only helper for the M0 spike: POST /__save?name=x.png with a PNG body writes data/spikes/t07/x.png,
// so canvas captures can be collected for the result file. Not part of any build. (Deleted with the spike.)
function saveCaptures(): Plugin {
  return {
    name: "spike-save-captures",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/__save", (req, res) => {
        const name = new URL(req.url ?? "", "http://x").searchParams.get("name") ?? "";
        if (req.method !== "POST" || !/^[\w.-]+\.png$/.test(name)) {
          res.statusCode = 400;
          res.end("bad request");
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
          const dir = resolve(import.meta.dirname, "../data/spikes/t07");
          mkdirSync(dir, { recursive: true });
          writeFileSync(resolve(dir, name), Buffer.concat(chunks));
          res.end("ok");
        });
      });
    },
  };
}

// The Python server (`uv run nvs serve --dev`). `changeOrigin` rewrites the Host header to the target's:
// the server only accepts its own host names (DNS-rebinding guard); the browser's Origin
// (http://localhost:5173) is allowed by `--dev`.
const API = "http://127.0.0.1:8765";

export default defineConfig(({ command }) => ({
  // Dev only: the spike's git-ignored scene assets. Never copied into the production build.
  publicDir: command === "serve" ? "spike/public" : false,
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: API, changeOrigin: true, ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
  plugins: [saveCaptures()],
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
}));
