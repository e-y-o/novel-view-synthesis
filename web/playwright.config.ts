import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { defineConfig, devices, type Project } from "@playwright/test";

/**
 * Browser tests (`npm run e2e`, `npm run e2e:chromium`). They run against the *built* app served by
 * `nvs serve --stub --dev` on port 8799 (`npm run build` first; the npm scripts do it), with a fresh data dir.
 * The depth backend is the deterministic stub, so no GPU or model is needed; the real-model tests are opt-in
 * (`PLAYWRIGHT_REAL=1`) and start their own server.
 *
 * Chromium runs in the installed Google Chrome (`channel: "chrome"`), so no browser download is needed for it;
 * set `PW_CHANNEL=` (empty) to use Playwright's own Chromium after `npx playwright install chromium`. Firefox and
 * WebKit need `npx playwright install firefox webkit` (a download of a few hundred MB).
 */
const PORT = 8799;
const REPO = resolve(import.meta.dirname, "..");
const dataDir = mkdtempSync(join(tmpdir(), "nvs-e2e-"));
const channel = process.env.PW_CHANNEL ?? "chrome";
/** `PLAYWRIGHT_REAL=1`: start the server with the real depth model (opt-in tests). */
const REAL = Boolean(process.env.PLAYWRIGHT_REAL);

const projects: Project[] = [
  {
    name: "chromium",
    use: {
      ...devices["Desktop Chrome"],
      ...(channel ? { channel } : {}),
      // Real GPU where there is one: headless Chromium otherwise falls back to a software renderer.
      launchOptions: {
        args: ["--use-angle=default", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"],
      },
    },
  },
  { name: "firefox", use: { ...devices["Desktop Firefox"] } },
  { name: "webkit", use: { ...devices["Desktop Safari"] } },
];

export default defineConfig({
  testDir: "e2e",
  outputDir: "e2e-results/artifacts",
  reporter: [["list"], ["html", { outputFolder: "e2e-results/report", open: "never" }]],
  fullyParallel: false, // one server, one worker process: tests share the GPU queue
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 1,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects,
  webServer: {
    command: `uv run nvs serve ${REAL ? "--accept-license" : "--stub"} --dev --port ${PORT} --data-dir "${dataDir}"`,
    // the real model needs up to a minute to load; the health endpoint answers earlier, the tests wait for the dot
    cwd: REPO,
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: REAL ? 120_000 : 60_000,
    stdout: "ignore",
    stderr: "ignore",
  },
});
