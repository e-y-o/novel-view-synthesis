import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

// Dev-only helper for the spike: POST /__save?name=x.png with a PNG body writes data/spikes/t07/x.png,
// so canvas captures can be collected for the result file. Not part of any build.
function saveCaptures(): Plugin {
  return {
    name: "spike-save-captures",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/__save", (req, res) => {
        const name = new URL(req.url ?? "", "http://x").searchParams.get("name") ?? "";
        if (req.method !== "POST" || !/^[\w.-]+\.png$/.test(name)) { res.statusCode = 400; res.end("bad request"); return; }
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
          const dir = resolve(__dirname, "../data/spikes/t07");
          mkdirSync(dir, { recursive: true });
          writeFileSync(resolve(dir, name), Buffer.concat(chunks));
          res.end("ok");
        });
      });
    },
  };
}

// The spike page is served at /spike/index.html; its assets (git-ignored) live in spike/public/<scene>/.
export default defineConfig({
  publicDir: "spike/public",
  server: { port: 5173, strictPort: true },
  plugins: [saveCaptures()],
});
